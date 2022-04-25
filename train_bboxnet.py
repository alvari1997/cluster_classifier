from __future__ import print_function
import argparse
import os
import random
from socket import MSG_DONTROUTE
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
from provider import angle2class, size2class, class2angle, class2size, from_prediction_to_label_format, compute_box3d_iou, size2class2
#from viz_util import draw_lidar, draw_lidar_simple

Loss = BoxNetLoss()
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3 #8 # one cluster for each type
NUM_OBJECT_POINT = 512

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=100, help='input size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='bbox', help="dataset type bbox|lidar")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'bbox':
    dataset = LidarDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_dataset = LidarDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)

    box_dataset = BoxDataset(
        root=opt.dataset,
        classification=True,
        npoints=opt.num_points)

    test_box_dataset = BoxDataset(
        root=opt.dataset,
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

box_dataloader = torch.utils.data.DataLoader(
    box_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

testboxdataloader = torch.utils.data.DataLoader(
    test_box_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(box_dataset), len(test_dataset), len(test_box_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = BoxNet(n_classes=num_classes, n_channel=3)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

for epoch in range(opt.nepoch):
    scheduler.step()
    i = 0
    for data, box_data in zip(dataloader, box_dataloader): #combine points, target, bbox to single dataloader
        points = data[0]
        target = data[1]
        bbox_target = box_data[0]
        target = target[:, 0]

        print(target)
        #print(bbox_target[:, 3:6])

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
        points, target, bbox_target, one_hot = points.cuda(), target.cuda(), bbox_target.cuda(), one_hot.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()

        # NN
        box_pred, center_delta = classifier(points, one_hot)
        
        center_boxnet, \
        heading_scores, heading_residual_normalized, heading_residual, \
        size_scores, size_residual_normalized, size_residual = \
                parse_output_to_tensors(box_pred)

        box3d_center = center_boxnet + center_delta
        #stage1_center = center_delta + center_delta 
        stage1_center = center_delta
        #stage1_center = 0

        # heading_scores (32, 12) which bin is the heading
        # heading_residual (32, 12) residual angle
        # size_scores (32, 3) which bin is the size
        # size_residual (32, 3, 3) residual size

        '''
        2.Center
        center: torch.Size([32, 3]) torch.float32
        stage1_center: torch.Size([32, 3]) torch.float32
        center_label:[32,3]
        3.Heading
        heading_scores: torch.Size([32, 12]) torch.float32
        heading_residual_normalized: torch.Size([32, 12]) torch.float32
        heading_residual: torch.Size([32, 12]) torch.float32
        heading_class_label:(32)
        heading_residual_label:(32)
        4.Size
        size_scores: torch.Size([32, 8]) torch.float32
        size_residual_normalized: torch.Size([32, 8, 3]) torch.float32
        size_residual: torch.Size([32, 8, 3]) torch.float32
        size_class_label:(32)
        size_residual_label:(32,3)'''

        # compute GT, probably wrong setup
        box3d_center_label = bbox_target[:,:3] # check if correct
        angle = bbox_target[:, 6] + 3/2*np.pi
        heading_class_label, heading_residual_label = angle2class(angle, NUM_HEADING_BIN)
        size_class_label, size_residual_label = size2class2(bbox_target[:,3:6], target) #sometimes residual size > 3m, target vector wrong, problem is that dataloader shuffles bbox and points differently

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

        '''gt_box_corners = get_box3d_corners_helper(bbox_target[:,:3], bbox_target[:,6], bbox_target[:,3:6])
        pred_box = get_box3d_corners(box3d_center, heading_residual, size_residual)
        #pred_box_corners = pred_box[:, 0, 0, :, :]
        pred_box_corners = pred_box[:, heading_scores.data.max(1)[1], size_scores.data.max(1)[1], :, :]

        print(pred_box_corners.shape)

        pred_heading_class = heading_scores.data.max(1)[1]
        pred_heading_residual = heading_residual[:, pred_heading_class][0]
        pred_size_class = size_scores.data.max(1)[1]
        pred_size_residual = size_residual[:, pred_size_class][0]

        #pred_box = from_prediction_to_label_format(box3d_center, pred_heading_class, pred_heading_residual, pred_size_class, pred_size_residual, 0)
        #pred_box = from_prediction_to_label_format(box3d_center, pred_heading_class, pred_heading_residual, pred_size_class, pred_size_residual, 0)

        pred_center = box3d_center
        pred_heading = class2angle(pred_heading_class, pred_heading_residual, NUM_HEADING_BIN)
        pred_size = class2size(pred_size_class, pred_size_residual)'''
        #print(pred_size)
        #print(pred_heading)

        #print(pred_center.shape)
        #print(pred_heading.shape)
        #print(pred_size.shape)
        
        #pred_box_corners = get_box3d_corners_helper(pred_center, pred_heading, pred_size)

        # switch to matplotlib viz
        '''if i > 710 and epoch == 5:
            np_pred_box = pred_box_corners.cpu().detach().numpy()
            np_gt_box = gt_box_corners.cpu().detach().numpy()
            # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
            lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                    [4, 5], [5, 6], [6, 7], [4, 7],
                    [0, 4], [1, 5], [2, 6], [3, 7]]
            # Use the same color for all lines
            colors = [[1, 0, 0] for _ in range(len(lines))]
            colors1 = [[0, 1, 0] for _ in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np_pred_box[0])
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_set1 = o3d.geometry.LineSet()
            line_set1.points = o3d.utility.Vector3dVector(np_gt_box[0])
            line_set1.lines = o3d.utility.Vector2iVector(lines)
            line_set1.colors = o3d.utility.Vector3dVector(colors1)
            # Create a visualization object and window
            #vis = o3d.visualization.Visualizer()
            #vis.create_window()
            # Display the bounding boxes:
            #vis.add_geometry(line_set)
            #o3d.visualization.draw_geometries([line_set,line_set1,pcd])
            #o3d.visualization.draw_geometries([line_set1])

            #np_points = points1.cpu().detach().numpy()
            #np_points = np.transpose(np_points)
            #pcd = o3d.geometry.PointCloud()
            #pcd.points = o3d.utility.Vector3dVector(np_points)
            #o3d.visualization.draw_geometries([pcd])

            o3d.visualization.draw_geometries([line_set, line_set1])'''

        loss.backward()
        optimizer.step()
        
        print('[%d: %d/%d] train loss: %f MIOU: %f' % (epoch, i, num_batch, loss.item(), np.mean(iou3dbox)))
        #print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

        if i % 10 == 0:
            data, boxdata = next(zip(testdataloader, testboxdataloader))
            points = data[0]
            target = data[1]
            bbox_target = box_data[0]
            target = target[:, 0]

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
            points, target, bbox_target, one_hot = points.cuda(), target.cuda(), bbox_target.cuda(), one_hot.cuda()
            
            # NN
            classifier = classifier.eval()
            box_pred, _ = classifier(points, one_hot)

            center_boxnet, \
            heading_scores, heading_residual_normalized, heading_residual, \
            size_scores, size_residual_normalized, size_residual = \
                    parse_output_to_tensors(box_pred)

            box3d_center = center_boxnet + center_delta # ???
            stage1_center = 0 # ???

            # compute GT
            box3d_center_label = bbox_target[:,:3]
            heading_class_label, heading_residual_label = angle2class(bbox_target[:, 6], NUM_HEADING_BIN)
            size_class_label, size_residual_label = size2class2(bbox_target[:,3:6], target)

            # loss
            losses = Loss(box3d_center, box3d_center_label, stage1_center, \
                heading_scores, heading_residual_normalized, \
                heading_residual, \
                heading_class_label, heading_residual_label, \
                size_scores, size_residual_normalized, \
                size_residual, \
                size_class_label, size_residual_label)

            loss = losses['total_loss']
            print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))
        i = i + 1

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

plt.show()

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