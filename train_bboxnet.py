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
from model_utils import FrustumPointNetLoss, parse_output_to_tensors, huber_loss, get_box3d_corners, get_box3d_corners_helper

Loss = FrustumPointNetLoss()
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3 #8 # one cluster for each type
NUM_OBJECT_POINT = 512

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--num_points', type=int, default=100, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=True, help="dataset path")
parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

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
    shuffle=True,
    num_workers=int(opt.workers))

testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers))

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
    for data, box_data in zip(dataloader, box_dataloader):
        points = data[0]
        target = data[1]
        bbox_target = box_data[0]
        target = target[:, 0]

        # transform target scalar to 3x one hot vector
        hot1 = torch.zeros(opt.batchSize)
        hot1[target == 0] = 1
        hot2 = torch.zeros(opt.batchSize)
        hot2[target == 2] = 1
        hot3 = torch.zeros(opt.batchSize)
        hot3[target == 1] = 1
        one_hot = torch.vstack((hot1, hot2, hot3))
        one_hot = one_hot.transpose(1, 0)

        points = points.transpose(2, 1)
        points, target, bbox_target, one_hot = points.cuda(), target.cuda(), bbox_target.cuda(), one_hot.cuda()
        #points, target, bbox_target, one_hot = points.cpu(), target.cpu(), bbox_target.cpu(), one_hot.cpu()
        optimizer.zero_grad()
        classifier = classifier.train()

        # NN
        box_pred, center_delta = classifier(points, one_hot)
        
        # transform box_pred vector to box coordinates
        # (32, 39) -> (32, 7)
        center_boxnet, \
        heading_scores, heading_residual_normalized, heading_residual, \
        size_scores, size_residual_normalized, size_residual = \
                parse_output_to_tensors(box_pred)

        box3d_center = center_boxnet + center_delta
        stage1_center = center_delta

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
        heading_residual_snormalized: torch.Size([32, 12]) torch.float32
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
        box3d_center_label = bbox_target[:,:3]
        heading_class_label = torch.floor(((np.pi*3/2 + bbox_target[:,6]) * 12 / (2*np.pi))) # maybe  -pi*3/2 < float < pi*1/2   ->   0 < int < 12
        heading_residual_label = bbox_target[:,6] # residual of above conversion?
        size_class_label = target
        size_residual_label = bbox_target[:,3:6]

        #ttt = torch.floor(((np.pi*3/2 + bbox_target[:,6]) * 12 / (2*np.pi)))

        #print(target)
        #print(heading_scores)
        #print(torch.max(heading_residual_label))
        #print(size_scores)
        #print(size_class_label)

        #print(torch.eye(12)[heading_class_label.long()])
        #print(heading_class_label)
        #heading_class_loss = F.nll_loss(F.log_softmax(heading_scores,dim=1), heading_class_label.long())

        losses = Loss(box3d_center, box3d_center_label, stage1_center, \
                heading_scores, heading_residual_normalized, \
                heading_residual, \
                heading_class_label, heading_residual_label, \
                size_scores, size_residual_normalized, \
                size_residual, \
                size_class_label, size_residual_label)

        
        
        loss = losses['total_loss']

        

        loss.backward()
        optimizer.step()
        #pred_choice = pred.data.max(1)[1]
        #correct = pred_choice.eq(target.data).cpu().sum()
        #correct = pred_choice[0].eq(target.data).cpu().sum()
        #print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))
        print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

        '''if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            #correct = pred_choice.eq(target.data).cpu().sum()
            correct = pred_choice[0].eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
            #print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))'''
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