from __future__ import print_function
import argparse
from dis import dis
import os
import random
from socket import MSG_DONTROUTE
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset, LidarDataset, FilterDataset
from pointnet.custom_models import PointNetCls, feature_transform_regularizer, OutlierFilter
#from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=128, help='input size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=False, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='lidar', help="dataset type lidar|shapenet")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

blue = lambda x: '\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.dataset_type == 'lidar':
    dataset = LidarDataset(
        #root=opt.dataset,
        root='train_unbbox_dataset',
        classification=True,
        split='train',
        npoints=opt.num_points,
        data_augmentation=True)

    test_dataset = LidarDataset(
        #root=opt.dataset,
        root='test_unbbox_dataset',
        classification=True,
        split='test',
        npoints=opt.num_points,
        data_augmentation=False)

    out_dataset = LidarDataset(
        #root='un_fov_lim_outlier_dataset_train',
        root='train_unood_dataset',
        classification=True,
        split='train',
        npoints=opt.num_points,
        data_augmentation=True)
else:
    exit('wrong dataset type')


dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

outdataloader = torch.utils.data.DataLoader(
    out_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset), len(out_dataset))
num_classes = len(dataset.classes)
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)
#classifier = OutlierFilter(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize

plt.axis([0, 10, 0, 1])

for epoch in range(opt.nepoch):
    scheduler.step()
    i = 0
    for data, ood_data in zip(dataloader, outdataloader):
        points = torch.cat((data[0], ood_data[0]), 0)
        target = data[1]
        one_hot_w_pos = data[2]
        dist = torch.cat((data[3], ood_data[3]), 0)
        dist = dist[:, None]
        voxel = torch.cat((data[4], ood_data[4]), 0)
        voxel = voxel[:, :, None]
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target, one_hot_w_pos, dist, voxel = points.cuda(), target.cuda(), one_hot_w_pos.cuda(), dist.cuda().float(), voxel.cuda().float()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat, out = classifier(points, dist, voxel)
        loss = F.cross_entropy(pred[:len(data[0])], target)
        #loss = F.nll_loss(pred, target)
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        
        # add energy loss
        #m_in = -18 
        #m_out = -2 
        m_in = -5 #winner
        m_out = -2 #winner
        # tried
        #m_in = -0.1
        #m_out = -1
        #m_in = 5 
        #m_out = 2 
        Ec_out = -torch.logsumexp(out[len(data[0]):], dim=1)
        Ec_in = -torch.logsumexp(out[:len(data[0])], dim=1)
        loss += 0.1*(torch.pow(F.relu(Ec_in-m_in), 2).mean() + torch.pow(F.relu(m_out-Ec_out), 2).mean())

        # plot loss
        #plt.scatter(i, loss.cpu().detach().numpy())
        #plt.pause(0.005)

        loss.backward()
        optimizer.step()
        #pred_choice = pred.data.max(1)[1]
        pred_choice = pred[:len(data[0])].data.max(1)[1]
        #correct = pred_choice.eq(target.data).cpu().sum()
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
        #print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target, _, dist, voxel = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            dist = dist[:, None]
            voxel = voxel[:, :, None]
            points, target, dist, voxel = points.cuda(), target.cuda(), dist.cuda().float(), voxel.cuda().float()
            classifier = classifier.eval()
            pred, _, _, _ = classifier(points, dist, voxel)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            #correct = pred_choice[0].eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
            #print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))
        i = i + 1

    torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

#plt.show()

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(testdataloader, 0)):
    points, target, _, dist, voxel = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    dist = dist[:, None]
    voxel = voxel[:, :, None]
    points, target, dist, voxel = points.cuda(), target.cuda(), dist.cuda().float(), voxel.cuda().float()
    classifier = classifier.eval()
    pred, _, _, _ = classifier(points, dist, voxel)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))