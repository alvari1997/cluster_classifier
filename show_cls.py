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
from pointnet.dataset import ShapeNetDataset, LidarDataset
from pointnet.custom_models import PointNetCls
#from pointnet.model import PointNetCls
import torch.nn.functional as F
import time
#import open3d as o3d
import matplotlib.pyplot as plt
import scipy.special as sci

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

test_dataset = LidarDataset(
    #root='lidar_dataset5',
    root='test_unbbox_dataset',
    split='test',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=False)

outlier_dataset = LidarDataset(
    root='test_unood_dataset',
    split='test',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=True)

# initialize classifier
classifier = PointNetCls(k=len(test_dataset.classes))
#print(len(test_dataset.classes))
classifier.cpu()

# load weights
classifier.load_state_dict(torch.load(opt.model))

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

    points, target, _, dist, voxel = data
    points, target, dist = Variable(points), Variable(target[:, 0]), Variable(dist)
    points = points.transpose(2, 1)
    dist = dist[:, None]
    voxel = voxel[:, :, None]
    points, target, dist, voxel = points.cpu(), target.cpu(), dist.cpu().float(), voxel.cpu().float()

    # run PointNet
    start = time.time()
    pred, global_feat, avg, out = classifier(points, dist, voxel)
    stop = time.time()
    #print(stop-start)
    #print(pred)

    loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    #print(max(pred.data.max(1)[0]))
    max_scores.append(max(pred.data.max(1)[0]))
    #global_feats.append(max(global_feat))
    #print(avg.size())
    np_global_feat = global_feat.cpu().detach().numpy()
    np_avg = avg.cpu().detach().numpy()
    np_out = out.cpu().detach().numpy()
    np_pred = pred.cpu().detach().numpy()
    
    #print(np_avg.shape)
    #np_global_feat[np_global_feat < 1] = -10
    energy = -np.log(np.exp(np_global_feat))
    potential = (np.exp(energy))/(1 + np.exp(energy))
    mAvgs.append(np.max(np_global_feat))
    
    ind = np_global_feat[0].argsort()[-n:][::-1]
    big_values = np_global_feat[0][ind]

    #energ = -torch.logsumexp(np_out, dim=1)
    #energ = -T * sci.logsumexp(np.var(np_global_feat)/T)
    energ = -T * sci.logsumexp(np.max(np_out)/T)
    #energ = -sci.logsumexp(np.mean(np_out))
    pot = (np.exp(a*energ))/(1 + np.exp(a*energ))
    last_layer.append(energ)
    #print(pred_choice[0])
    #print(target[0])
    correct = pred_choice.eq(target.data).cpu().sum()

    #class_and_confidence.append([pred_choice[0], pred[pred_choice[0]]])

    #print(pred[0])
    #print(pred[0][0])

    if correct > 0:
        if pred_choice[0] == 3: #van
            correctVans = correctVans + 1
    if target[0] == 3:
        totalVans = totalVans + 1

    if correct > 0:
        if pred_choice[0] == 2: #ped
            correctPed = correctPed + 1
    if target[0] == 2:
        totalPed = totalPed + 1

    if correct > 0:
        if pred_choice[0] == 1: #cyc
            correctCyc = correctCyc + 1
    if target[0] == 1:
        totalCyc = totalCyc + 1

    if correct > 0:
        if pred_choice[0] == 0: #car
            correctCars = correctCars + 1
    if target[0] == 0:
        totalCars = totalCars + 1

    
    print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), correct / float(32)))

    accuracy_scores.append(correct/float(32))
    times.append(stop-start)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    '''if pred_choice[0] == 0:
        np_points = points.cpu().detach().numpy()
        np_points = np.transpose(np_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_points)
        o3d.visualization.draw_geometries([pcd])'''

'''print('accuracy ', np.mean(accuracy_scores))
print('time ', np.mean(times))
print('max score ', np.max(max_scores))
print('vans accuracy ', correctVans/totalVans)
print('pedestrian accuracy ', correctPed/totalPed)
print('cyclists accuracy ', correctCyc/totalCyc)
print('cars accuracy ', correctCars/totalCars)'''

testdataloader = torch.utils.data.DataLoader(
    outlier_dataset, batch_size=1, shuffle=True)

mAvgs_outliers = []
mAvgs_outliers_filtered = []
last_layer_outliers = []

for i, data in enumerate(testdataloader, 0):

    if i > max_i:
        break

    start = time.time()
    points, target, _, dist, voxel = data
    points, target, dist = Variable(points), Variable(target[:, 0]), Variable(dist)
    points = points.transpose(2, 1)
    dist = dist[:, None]
    voxel = voxel[:, :, None]
    points, target, dist, voxel = points.cpu(), target.cpu(), dist.cpu().float(), voxel.cpu().float()

    # run PointNet
    pred, global_feat, avg, out = classifier(points, dist, voxel)
    #print(pred)

    loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    #print(max(pred.data.max(1)[0]))
    max_scores.append(max(pred.data.max(1)[0]))
    #global_feats.append(max(global_feat))
    #print(avg.size())
    np_global_feat = global_feat.cpu().detach().numpy()
    np_avg = avg.cpu().detach().numpy()
    np_out = out.cpu().detach().numpy()
    np_pred = pred.cpu().detach().numpy()

    #print(np_global_feat[0])
    #np_global_feat[np_global_feat < 1] = -10
    energy = -np.log(np.exp(np_global_feat))
    potential = (np.exp(energy))/(1 + np.exp(energy))
    mAvgs_outliers.append(np.max(np_global_feat))

    ind = np_global_feat[0].argsort()[-n:][::-1]
    big_values = np_global_feat[0][ind]

    #energ = -torch.logsumexp(np_out, dim=1)
    #energ = -T * sci.logsumexp(np.var(np_global_feat)/T)
    energ = -T * sci.logsumexp(np.max(np_out)/T)
    #energ = -sci.logsumexp(np.mean(np_out))
    pot = (np.exp(a*energ))/(1 + np.exp(a*energ))
    last_layer_outliers.append(energ)
    #if np.max(np_global_feat) < 1.77:
    #    mAvgs_outliers_filtered.append(np.max(np_global_feat))
    #print(pred_choice[0])
    #print(target[0])
    correct = pred_choice.eq(target.data).cpu().sum()
    stop = time.time()
    print(stop-start)
    print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), correct / float(32)))

    accuracy_scores.append(correct/float(32))
    times.append(stop-start)

#print(len(mAvgs_outliers_filtered)/len(mAvgs_outliers))

#plt.plot(range(len(mAvgs)), mAvgs, range(len(mAvgs_outliers)), mAvgs_outliers)

print("ID ", np.mean(last_layer))
print("OoD ", np.mean(last_layer_outliers))

#np.save(last_layer)
# set the font globally
plt.rcParams.update({'font.family':'serif'})
plt.hlines(-3.5, 0, i, colors="Black", linestyles="--", label="Threshold")
plt.plot(range(len(last_layer)), last_layer, "o", label="In distribution (Car, Person, Cyclist)", color='deepskyblue', alpha=0.5)
plt.plot(range(len(last_layer_outliers)), last_layer_outliers, "^", label="Out-of-distribution (Random clusters)", color='fuchsia', alpha=0.5)
plt.legend(loc="lower left")
plt.xlabel("Cluster i")
plt.ylabel("Energy")
plt.xticks([0,300])
#plt.title("Default training")
#plt.title("Trained with energy loss function")
plt.ylim(-20, 10)
plt.show()