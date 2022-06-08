from __future__ import print_function
import argparse
#from cProfile import label
#from cv2 import IMWRITE_PAM_FORMAT_GRAYSCALE
import numpy as np
#from sklearn.metrics import accuracy_score
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pointnet.dataset import BoxDataset, LidarDataset
from pointnet.custom_models import PointNetCls
from pointnet.box_model import BoxNet
#from pointnet.model import PointNetCls
import torch.nn.functional as F
import time
#import open3d as o3d
import matplotlib.pyplot as plt
import scipy.special as sci
from model_utils import BoxNetLoss, parse_output_to_tensors_cpu
import random
import seaborn as sns

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=128, help='input batch size')


opt = parser.parse_args()
print(opt)

T = 1000
a = 1
n = 30

max_i = 1000

test_dataset = BoxDataset(
    root='train_unbbox_dataset',
    split='train',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=True)

outlier_dataset = BoxDataset(
    root='near_ID_dataset',
    split='train',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=True)

'''outlier_dataset = LidarDataset(
    root='train_unood_dataset',
    split='test',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=False)'''

testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=True)

# initialize classifier
boxnet = BoxNet(n_classes=3, n_channel=3)
#print(len(test_dataset.classes))
boxnet.cpu()
# load weights
boxnet.load_state_dict(torch.load(opt.model))
# set evaluation mode
boxnet.eval()

# initialize classifier
classifier = PointNetCls(k=3)
#print(len(test_dataset.classes))
classifier.cpu()
# load weights
classifier.load_state_dict(torch.load('cls_128_feat_3cls_energy_128p/cls_model_199.pth'))
# set evaluation mode
classifier.eval()

last_layer = []
all_size = []
all_heading = []
cars_s = []
cars_h = []
cycs_s = []
cycs_h = []
peds_s = []
peds_h = []

all_near_size = []
all_near_heading = []
near_cars_s = []
near_cars_h = []
near_cycs_s = []
near_cycs_h = []
near_peds_s = []
near_peds_h = []

for i, data in enumerate(testdataloader, 0):

    if i > max_i:
        break

    points, bbox_target, target, _, dist, cluster_center, voxel = data
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
    points, target, bbox_target, one_hot, dist, cluster_center, voxel = points.cpu(), target.cpu(), bbox_target.cpu(), one_hot.cpu(), dist.cpu().float(), cluster_center.cpu(), voxel.cpu().float()

    # NN
    box_pred, center_delta = boxnet(points, one_hot, dist, voxel)

    center_boxnet, \
        heading_scores, heading_residual_normalized, heading_residual, \
        size_scores, size_residual_normalized, size_residual = \
                parse_output_to_tensors_cpu(box_pred)

    size_energ = -sci.logsumexp(size_scores.cpu().detach().numpy(), axis=1)
    heading_energ = -sci.logsumexp(heading_scores.cpu().detach().numpy(), axis=1)
    
    all_size.append(size_energ)
    all_heading.append(heading_energ)
    
    if target == 0:
        cars_s.append(size_energ)
        cars_h.append(heading_energ)
    if target == 1:
        cycs_s.append(size_energ)
        cycs_h.append(heading_energ)
    if target == 2:
        peds_s.append(size_energ)
        peds_h.append(heading_energ)


testdataloader = torch.utils.data.DataLoader(
    outlier_dataset, batch_size=1, shuffle=True)

mAvgs_outliers = []
mAvgs_outliers_filtered = []
last_layer_outliers = []
j = 0
for i, data in enumerate(testdataloader, 0):

    if j > max_i:
        break

    points, bbox_target, target, _, dist, cluster_center, voxel = data
    target = target[:, 0]
    dist = dist[:, None]
    voxel = voxel[:, :, None]

    #print(j)
    #print(i)
    j = j + 1
    # transform target scalar to 3x one hot vector
    hot1 = torch.zeros(len(data[0]))
    hot1[target == 0] = 1
    hot2 = torch.zeros(len(data[0]))
    hot2[target == 2] = 1
    hot3 = torch.zeros(len(data[0]))
    hot3[target == 1] = 1
    one_hot = torch.vstack((hot1, hot2, hot3))
    one_hot = one_hot.transpose(1, 0)
    one_hot = one_hot.cpu()

    points = points.transpose(2, 1)
    points, target, bbox_target, one_hot, dist, cluster_center, voxel = points.cpu(), target.cpu(), bbox_target.cpu(), one_hot.cpu(), dist.cpu().float(), cluster_center.cpu(), voxel.cpu().float()


    # NN
    box_pred, center_delta = boxnet(points, one_hot, dist, voxel)

    center_boxnet, \
        heading_scores, heading_residual_normalized, heading_residual, \
        size_scores, size_residual_normalized, size_residual = \
                parse_output_to_tensors_cpu(box_pred)
    
    size_energ = -sci.logsumexp(size_scores.cpu().detach().numpy(), axis=1)
    heading_energ = -sci.logsumexp(heading_scores.cpu().detach().numpy(), axis=1)
    
    all_near_size.append(size_energ)
    all_near_heading.append(heading_energ)
    
    if target == 0:
        near_cars_s.append(size_energ)
        near_cars_h.append(heading_energ)
    if target == 1:
        near_cycs_s.append(size_energ)
        near_cycs_h.append(heading_energ)
    if target == 2:
        near_peds_s.append(size_energ)
        near_peds_h.append(heading_energ)


print("all size ID ", np.mean(all_size))
print("all near size ID ", np.mean(all_near_size))
print("all heading ID ", np.mean(all_heading))
print("all near heading ID ", np.mean(all_near_heading))
print(' ')
print("cars_s ", np.mean(cars_s))
print("near_cars_s ", np.mean(near_cars_s))
print('')
print("cars_h ", np.mean(cars_h))
print("near_cars_h ", np.mean(near_cars_h))
print('')
print("cycs_s ", np.mean(cycs_s))
print("near_cycs_s ", np.mean(near_cycs_s))
print('')
print("cycs_h ", np.mean(cycs_h))
print("near_cycs_h ", np.mean(near_cycs_h))
print('')
print("peds_s ", np.mean(peds_s))
print("near_peds_s ", np.mean(near_peds_s))
print('')
print("peds_h ", np.mean(peds_h))
print("near_peds_h ", np.mean(near_peds_h))
print('')

plt.rcParams.update({'font.family':'serif'})
plt.rcParams["figure.figsize"] = (4,3)

plt.subplot(2, 4, 1)
sns.distplot(-np.asarray(all_size), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='ID size', color='deepskyblue')
sns.distplot(-np.asarray(all_near_size), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Near OOD size', color='silver')
#plt.vlines(np.mean(all_size), 0, 5, colors="Black", linewidth=1, linestyles="--", label="ID mean")
#plt.vlines(np.mean(all_near_size), 0, 5, colors="Black", linewidth=1, linestyles="--", label="Near OOD mean")
plt.legend(loc="upper left")
plt.xlabel("Negative energy")
plt.ylabel("Frequency")
plt.xlim(0, 16)
plt.subplots_adjust(left=0.16, right=0.985, top=0.99, bottom=0.155)
#plt.figure()

plt.subplot(2, 4, 5)
sns.distplot(-np.asarray(all_heading), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='ID heading', color='deepskyblue')
sns.distplot(-np.asarray(all_near_heading), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Near OOD heading', color='silver')
plt.legend(loc="upper left")
plt.xlabel("Negative energy")
plt.ylabel("Frequency")
plt.xlim(0, 16)
plt.subplots_adjust(left=0.16, right=0.985, top=0.99, bottom=0.155)
#plt.figure()

plt.subplot(2, 4, 2)
sns.distplot(-np.asarray(cars_s), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Cars size', color='deepskyblue')
sns.distplot(-np.asarray(near_cars_s), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Near OOD cars size', color='silver')
plt.legend(loc="upper left")
plt.xlabel("Negative energy")
plt.ylabel("Frequency")
plt.xlim(0, 16)
plt.subplots_adjust(left=0.16, right=0.985, top=0.99, bottom=0.155)
#plt.figure()

plt.subplot(2, 4, 6)
sns.distplot(-np.asarray(cars_h), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Cars heading', color='deepskyblue')
sns.distplot(-np.asarray(near_cars_h), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Near OOD cars heading', color='silver')
plt.legend(loc="upper left")
plt.xlabel("Negative energy")
plt.ylabel("Frequency")
plt.xlim(0, 16)
plt.subplots_adjust(left=0.16, right=0.985, top=0.99, bottom=0.155)
#plt.figure()

plt.subplot(2, 4, 3)
sns.distplot(-np.asarray(cycs_s), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Cyclists size', color='deepskyblue')
sns.distplot(-np.asarray(near_cycs_s), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Near OOD cyclists size', color='silver')
plt.legend(loc="upper left")
plt.xlabel("Negative energy")
plt.ylabel("Frequency")
plt.xlim(0, 16)
plt.subplots_adjust(left=0.16, right=0.985, top=0.99, bottom=0.155)
#plt.figure()

plt.subplot(2, 4, 7)
sns.distplot(-np.asarray(cycs_h), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Cyclists heading', color='deepskyblue')
sns.distplot(-np.asarray(near_cycs_h), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Near OOD cyclists heading', color='silver')
plt.legend(loc="upper left")
plt.xlabel("Negative energy")
plt.ylabel("Frequency")
plt.xlim(0, 16)
plt.subplots_adjust(left=0.16, right=0.985, top=0.99, bottom=0.155)
#plt.figure()

plt.subplot(2, 4, 4)
sns.distplot(-np.asarray(peds_s), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Pedestrians size', color='deepskyblue')
sns.distplot(-np.asarray(near_peds_s), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Near OOD pedestrians size', color='silver')
plt.legend(loc="upper left")
plt.xlabel("Negative energy")
plt.ylabel("Frequency")
plt.xlim(0, 16)
plt.subplots_adjust(left=0.16, right=0.985, top=0.99, bottom=0.155)
#plt.figure()

plt.subplot(2, 4, 8)
sns.distplot(-np.asarray(peds_h), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Pedestrians heading', color='deepskyblue')
sns.distplot(-np.asarray(near_peds_h), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Near OOD pedestrians heading', color='silver')
plt.legend(loc="upper left")
plt.xlabel("Negative energy")
plt.ylabel("Frequency")
plt.xlim(0, 16)
plt.subplots_adjust(left=0.16, right=0.985, top=0.99, bottom=0.155)
#plt.figure()

plt.show()