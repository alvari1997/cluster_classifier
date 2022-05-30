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

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=128, help='input batch size')


opt = parser.parse_args()
print(opt)

T = 1000
a = 1
n = 30

max_i = 300

test_dataset = BoxDataset(
    root='unbbox_dataset',
    split='test',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=False)

outlier_dataset = LidarDataset(
    root='un_fov_lim_outlier_dataset_test',
    split='test',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=False)

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


accuracy_scores = []
times = []
max_scores = []
global_feats = []
mAvgs = []
class_and_confidence = [[]]
last_layer = []
correctVans = 0
totalVans = 0
correctPed = 0
totalPed = 0
correctCyc = 0
totalCyc = 0
correctCars = 0
totalCars = 0

for i, data in enumerate(testdataloader, 0):

    if i > max_i:
        break

    points, bbox_target, target, _, dist, cluster_center, voxel = data
    target = target[:, 0]
    dist = dist[:, None]
    voxel = voxel[:, :, None]

    if target == 0:
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

        '''np_out_head = heading_scores.cpu().detach().numpy()
        np_out_size = size_scores.cpu().detach().numpy()
        np_out = np.hstack((np_out_head, np_out_size))'''

        np_out = heading_scores.cpu().detach().numpy()
        
        #energ = -torch.logsumexp(np_out, dim=1)
        energ = -sci.logsumexp(np_out, axis=1)
        #energ = -T * sci.logsumexp(np.var(np_global_feat)/T)
        #energ = -T * sci.logsumexp(np.max(np_out)/T)
        #energ = -sci.logsumexp(np.mean(np_out))
        pot = (np.exp(a*energ))/(1 + np.exp(a*energ))
        last_layer.append(energ)
        #print(pred_choice[0])


testdataloader = torch.utils.data.DataLoader(
    outlier_dataset, batch_size=1, shuffle=True)

mAvgs_outliers = []
mAvgs_outliers_filtered = []
last_layer_outliers = []
j = 0
for i, data in enumerate(testdataloader, 0):

    if j > max_i:
        break

    points, _, _, dist, voxel = data
    dist = dist[:, None]
    voxel = voxel[:, :, None]

    points = points.transpose(2, 1)
    points, dist, voxel = points.cpu(), dist.cpu().float(), voxel.cpu().float()

    # classifier for target
    pred, global_feat, avg, out = classifier(points, dist, voxel)
    pred_choice = pred.data.max(1)[1]
    target = pred_choice #random.randint(0, 2)

    enrg = -sci.logsumexp(out.cpu().detach().numpy(), axis=1)

    if target == 0 and enrg < -4.3:
        print(j)
        print(i)
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

        # NN
        box_pred, center_delta = boxnet(points, one_hot, dist, voxel)

        center_boxnet, \
            heading_scores, heading_residual_normalized, heading_residual, \
            size_scores, size_residual_normalized, size_residual = \
                    parse_output_to_tensors_cpu(box_pred)

        '''np_out_head = heading_scores.cpu().detach().numpy()
        np_out_size = size_scores.cpu().detach().numpy()
        np_out = np.hstack((np_out_head, np_out_size))'''

        np_out = heading_scores.cpu().detach().numpy()
        
        #energ = -torch.logsumexp(np_out, dim=1)
        energ = -sci.logsumexp(np_out, axis=1)
        #energ = -T * sci.logsumexp(np.var(np_global_feat)/T)
        #energ = -T * sci.logsumexp(np.max(np_out)/T)
        #energ = -sci.logsumexp(np.mean(np_out))
        pot = (np.exp(a*energ))/(1 + np.exp(a*energ))
        last_layer_outliers.append(energ)
        #print(pred_choice[0])


print("ID ", np.mean(last_layer))
print("OoD ", np.mean(last_layer_outliers))

plt.plot(range(len(last_layer)), last_layer, "o", label="In distribution (Car, Person, Cyclist)")
plt.plot(range(len(last_layer_outliers)), last_layer_outliers, "o", label="Out of distribution (Random clusters)")
plt.hlines(-2.2, 0, i, colors="Black", linestyles="--", label="Threshold")
plt.legend(loc="lower left")
plt.xlabel("Cluster i")
plt.ylabel("Energy")
plt.title("Default training")
#plt.title("Trained with energy loss function")
plt.ylim(-20, 10)
plt.show()