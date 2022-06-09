from __future__ import print_function
import argparse
from cProfile import label
from dis import dis
import os
import random
from socket import MSG_DONTROUTE
from sklearn import cluster
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import LidarDataset, BoxDataset
from pointnet.box_model import BoxNet
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
from model_utils import BoxNetLoss, parse_output_to_tensors, get_box3d_corners_helper, get_box3d_corners
import open3d as o3d
from provider import angle2class, size2class, class2angle, class2size, compute_box3d_iou, size2class2, give_pred_box_corners, get_3d_box
#from viz_util import draw_lidar, draw_lidar_simple

Loss = BoxNetLoss()
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3 # one cluster for each type
NUM_OBJECT_POINT = 512

def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        corners3d: (N, 8, 3)
    """
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ]) / 2

    corners3d = boxes3d[:, None, 3:6] * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d, boxes3d[:, 6]).reshape(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]
    return corners3d

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3)
        angle: (B), angle along z-axis, angle increases x ==> y

    Returns:
    """
    cosa = np.cos(angle)
    sina = np.sin(angle)
    ones = np.ones_like(angle, dtype=np.float32)
    zeros = np.zeros_like(angle, dtype=np.float32)
    rot_matrix = np.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3)
    points_rot = np.matmul(points, rot_matrix)
    return points_rot

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=128, help='input size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=False, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='bbox', help="dataset type bbox|lidar")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'bbox':
    box_dataset = BoxDataset(
        #root=opt.dataset,
        root='train_unbbox_dataset',
        classification=True,
        npoints=opt.num_points,
        data_augmentation=True)

    test_box_dataset = BoxDataset(
        #root=opt.dataset,
        root='test_unbbox_dataset',
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)

    out_dataset = BoxDataset(
        #root='un_fov_lim_outlier_dataset_train',
        root='near_ID_dataset', 
        classification=True,
        split='train',
        npoints=opt.num_points,
        data_augmentation=True)
else:
    exit('wrong dataset type')

box_dataloader = torch.utils.data.DataLoader(
    box_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testboxdataloader = torch.utils.data.DataLoader(
    test_box_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

box_outdataloader = torch.utils.data.DataLoader(
    out_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(box_dataset), len(test_box_dataset))
num_classes = len(box_dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = BoxNet(n_classes=num_classes, n_channel=3)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))




optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999),eps=1e-08, weight_decay=0.0)

#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=20, gamma=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)



#optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) 
classifier.cuda()

num_batch = len(box_dataset) / opt.batchSize

plt.ion()
figure = plt.figure()
ax = figure.add_subplot(111)
idx = []
test_loss = []
train_loss = []
plot1, = ax.plot(idx, test_loss, label='test')
plot2, = ax.plot(idx, train_loss, label='train')
plt.ylim(0, 10)
plt.xlim(0, 158200)
plt.xlabel("i")
plt.ylabel("loss")
plt.legend(loc="lower left")
plt.title("loss-iteration")

for epoch in range(opt.nepoch):
    scheduler.step()
    i = 0
    for data, ood_data in zip(box_dataloader, box_outdataloader):

        points = torch.cat((data[0], ood_data[0]), 0)
        bbox_target = data[1]
        target = torch.cat((data[2], ood_data[2]), 0)
        dist = torch.cat((data[4], ood_data[4]), 0)
        #cluster_center = torch.cat((data[5], ood_data[5]), 0)
        cluster_center = data[5]
        voxel = torch.cat((data[6], ood_data[6]), 0)
        voxel = voxel[:, :, None]
        dist = dist[:, None]
        target = target[:, 0]
        points1 = points[:len(data[0])] + cluster_center[:, None]

        # transform target scalar to 3x one hot vector
        hot1 = torch.zeros(len(data[0])+len(ood_data[0]))
        hot1[target == 0] = 1
        hot2 = torch.zeros(len(data[0])+len(ood_data[0]))
        hot2[target == 2] = 1
        hot3 = torch.zeros(len(data[0])+len(ood_data[0]))
        hot3[target == 1] = 1
        one_hot = torch.vstack((hot1, hot2, hot3))
        one_hot = one_hot.transpose(1, 0)

        points = points.transpose(2, 1)
        points, target, bbox_target, one_hot, dist, cluster_center, voxel = points.cuda(), target.cuda(), bbox_target.cuda(), one_hot.cuda(), dist.cuda().float(), cluster_center.cuda(), voxel.cuda().float()
        optimizer.zero_grad()
        classifier = classifier.train()

        # NN
        box_pred, center_delta = classifier(points, one_hot, dist, voxel)
        
        center_boxnet, \
        heading_scores, heading_residual_normalized, heading_residual, \
        size_scores, size_residual_normalized, size_residual = \
                parse_output_to_tensors(box_pred)

        #box3d_center = center_boxnet + center_delta
        stage1_center = cluster_center[:len(data[0])] + center_delta[:len(data[0])] # original cluster center in the world
        box3d_center = center_boxnet[:len(data[0])] + stage1_center

        # compute GT
        bbox_target[:,:3] = bbox_target[:,:3] + cluster_center
        box3d_center_label = bbox_target[:,:3]
        angle = bbox_target[:, 6]
        heading_class_label, heading_residual_label = angle2class(angle, NUM_HEADING_BIN)
        size_class_label, size_residual_label = size2class2(bbox_target[:,3:6], target[:len(data[0])]) 
        
        # losses
        losses = Loss(box3d_center, box3d_center_label, stage1_center, \
                heading_scores[:len(data[0])], heading_residual_normalized[:len(data[0])], \
                heading_residual[:len(data[0])], \
                heading_class_label, heading_residual_label, \
                size_scores[:len(data[0])], size_residual_normalized[:len(data[0])], \
                size_residual[:len(data[0])], \
                size_class_label, size_residual_label)

        loss = losses['total_loss']

        # add energy loss
        m_in = -9 
        m_out = -6 
        Ec_out = -torch.logsumexp(heading_scores[len(data[0]):], dim=1)
        Ec_in = -torch.logsumexp(heading_scores[:len(data[0])], dim=1)
        loss += 0.05*(torch.pow(F.relu(Ec_in-m_in), 2).mean() + torch.pow(F.relu(m_out-Ec_out), 2).mean()) # heading
        m_in = -5
        m_out = -2
        Ec_out = -torch.logsumexp(size_scores[len(data[0]):], dim=1)
        Ec_in = -torch.logsumexp(size_scores[:len(data[0])], dim=1)
        loss += 0.05*(torch.pow(F.relu(Ec_in-m_in), 2).mean() + torch.pow(F.relu(m_out-Ec_out), 2).mean()) # size

        loss.backward()
        optimizer.step()
        
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))
        loss_train = loss.item()

        if i % 10 == 0:
            j, data = next(enumerate(testboxdataloader, 0))
            points, bbox_target, target, _, dist, cluster_center, voxel = data
            points1 = points + cluster_center[:, None]
            target = target[:, 0]
            dist = dist[:, None]
            voxel = voxel[:, :, None]

            # transform target scalar to 3x one hot vector
            hot1 = torch.zeros(len(data[0]))
            hot1[target == 0] = 1
            hot2 = torch.zeros(len(data[0]))
            hot2[target == 2] = 1
            hot3 = torch.zeros(len(data[0]))
            hot3[target == 1] = 1
            one_hot = torch.vstack((hot1, hot2, hot3))
            one_hot = one_hot.transpose(1, 0)

            points = points.transpose(2, 1)
            points, target, bbox_target, one_hot, dist, cluster_center, voxel = points.cuda(), target.cuda(), bbox_target.cuda(), one_hot.cuda(), dist.cuda().float(), cluster_center.cuda(), voxel.cuda().float()
            classifier = classifier.eval()

            # NN
            box_pred, center_delta = classifier(points, one_hot, dist, voxel)
            
            center_boxnet, \
            heading_scores, heading_residual_normalized, heading_residual, \
            size_scores, size_residual_normalized, size_residual = \
                    parse_output_to_tensors(box_pred)

            stage1_center = cluster_center + center_delta # original cluster center in the world
            box3d_center = center_boxnet + stage1_center

            # compute GT, probably wrong setup
            bbox_target[:,:3] = bbox_target[:,:3] + cluster_center
            box3d_center_label = bbox_target[:,:3]
            angle = bbox_target[:, 6] #+ 3/2*np.pi
            heading_class_label, heading_residual_label = angle2class(angle, NUM_HEADING_BIN)
            size_class_label, size_residual_label = size2class2(bbox_target[:,3:6], target) 

            # losses
            losses = Loss(box3d_center, box3d_center_label, stage1_center, \
                    heading_scores, heading_residual_normalized, \
                    heading_residual, \
                    heading_class_label, heading_residual_label, \
                    size_scores, size_residual_normalized, \
                    size_residual, \
                    size_class_label, size_residual_label)

            loss = losses['total_loss']

            # accuracy
            ioubev, iou3dbox = compute_box3d_iou(box3d_center.cpu().detach().numpy(), heading_scores.cpu().detach().numpy(), \
                        heading_residual.cpu().detach().numpy(), size_scores.cpu().detach().numpy(), size_residual.cpu().detach().numpy(), \
                        box3d_center_label.cpu().detach().numpy(), heading_class_label.cpu().detach().numpy(), \
                        heading_residual_label.cpu().detach().numpy(), size_class_label.cpu().detach().numpy(), \
                        size_residual_label.cpu().detach().numpy())

            # matplotlib viz
            pred_box_corners = give_pred_box_corners(box3d_center.cpu().detach().numpy(), heading_scores.cpu().detach().numpy(), \
                        heading_residual.cpu().detach().numpy(), size_scores.cpu().detach().numpy(), size_residual.cpu().detach().numpy())
            np_bbox_target = bbox_target.cpu().detach().numpy()
            gt_corners = boxes_to_corners_3d(np_bbox_target)

            if i > 0 and epoch == -1:
                for cc in range(32):
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')

                    np_points = points1.cpu().detach().numpy()
                    pts = np_points[cc]

                    gt_b = gt_corners[cc]  # (8, 3)
                    b = pred_box_corners[cc]

                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, c='b', lw=0, alpha=1)

                    for k in range(0, 4):
                        
                        xx = 0
                        yy = 1
                        zz = 2

                        # pred
                        i, j = k, (k + 1) % 4
                        ax.plot([b[i, xx], b[j, xx]], [b[i, yy], b[j, yy]], [b[i, zz], b[j, zz]],
                                color='r')

                        i, j = k + 4, (k + 1) % 4 + 4
                        ax.plot([b[i, xx], b[j, xx]], [b[i, yy], b[j, yy]], [b[i, zz], b[j, zz]],
                                color='r')

                        i, j = k, k + 4
                        ax.plot([b[i, xx], b[j, xx]], [b[i, yy], b[j, yy]], [b[i, zz], b[j, zz]],
                                color='r')

                        # gt
                        i, j = k, (k + 1) % 4
                        ax.plot([gt_b[i, xx], gt_b[j, xx]], [gt_b[i, yy], gt_b[j, yy]], [gt_b[i, zz], gt_b[j, zz]],
                                color='g')

                        i, j = k + 4, (k + 1) % 4 + 4
                        ax.plot([gt_b[i, xx], gt_b[j, xx]], [gt_b[i, yy], gt_b[j, yy]], [gt_b[i, zz], gt_b[j, zz]],
                                color='g')

                        i, j = k, k + 4
                        ax.plot([gt_b[i, xx], gt_b[j, xx]], [gt_b[i, yy], gt_b[j, yy]], [gt_b[i, zz], gt_b[j, zz]],
                                color='g')

                    #visual_right_scale(corners3d.reshape(-1, 3), ax)
                    ax.title.set_text('IOU: {}'.format(iou3dbox[cc]))
                    ax.view_init(elev=30., azim=-45)
                    ax.set_box_aspect([1,1,1])
                    #ax.set_xlim3d(-3, 3)
                    #ax.set_ylim3d(-3, 3)
                    #ax.set_zlim3d(-3, 3)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    plt.show()

            #print('[%d: %d/%d] %s loss: %f MIOU: %f' % (epoch, i, num_batch, blue('test'), loss.item(), np.mean(iou3dbox)))
            print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))

            test_loss.append(loss.item())
            train_loss.append(loss_train)
            #loss_list[epoch*791 + i] = loss.item()
            idx.append(epoch*791 + i)
            plot1.set_xdata(idx)
            plot1.set_ydata(test_loss)
            plot2.set_xdata(idx)
            plot2.set_ydata(train_loss)
            figure.canvas.draw()
            figure.canvas.flush_events()
            time.sleep(0.01)
        
        i = i + 1
    
    

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))



'''total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))'''