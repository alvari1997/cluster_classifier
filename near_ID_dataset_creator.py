from __future__ import print_function
import argparse
from math import fabs
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
import seaborn as sns
import open3d as o3d

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_points', type=int, default=128, help='input batch size')


opt = parser.parse_args()
print(opt)

T = 1000
a = 1
n = 30

max_i = 10000

outlier_dataset = LidarDataset(
    root='train_unood_dataset',
    split='train',
    classification=True,
    npoints=opt.num_points,
    data_augmentation=False)

testdataloader = torch.utils.data.DataLoader(
    outlier_dataset, batch_size=1, shuffle=True)

# initialize classifier
classifier = PointNetCls(k=3)
classifier.cuda()

# load weights
classifier.load_state_dict(torch.load(opt.model))

# set evaluation mode
classifier.eval()

cars = []
peds = []
cycs = []
last_layer_outliers = []

for i, data in enumerate(testdataloader, 0):

    #if i > max_i:
    #    break

    points, target, _, dist, voxel, original_points = data
    points, target, dist = Variable(points), Variable(target[:, 0]), Variable(dist)
    points = points.transpose(2, 1)
    dist = dist[:, None]
    voxel = voxel[:, :, None]
    points, target, dist, voxel = points.cuda(), target.cuda(), dist.cuda().float(), voxel.cuda().float()

    np_original_points = original_points.cpu().detach().numpy().squeeze()
    print(np_original_points.shape)

    # run PointNet
    pred, global_feat, avg, out = classifier(points, dist, voxel)

    loss = F.nll_loss(pred, target)

    pred_choice = pred.data.max(1)[1]
    np_out = out.cpu().detach().numpy()
    np_pred = pred.cpu().detach().numpy()

    energ = -sci.logsumexp(np_out, axis=1)
    last_layer_outliers.append(energ)
    correct = pred_choice.eq(target.data).cpu().sum()

    if energ < -3.5:
        if (pred_choice[0] == 0):
            # save unsampled original points into correct folder
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_original_points)
            o3d.io.write_point_cloud('/home/alvari/test_ws/train_dataset/near_cars/points/idx_{}.pts'.format(i), pcd)
            np.savetxt('/home/alvari/test_ws/train_dataset/near_cars/bbox/idx_{}.txt'.format(i), np.ones(7))
            cars.append(energ)
        if (pred_choice[0] == 1):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_original_points)
            o3d.io.write_point_cloud('/home/alvari/test_ws/train_dataset/near_cyclists/points/idx_{}.pts'.format(i), pcd)
            np.savetxt('/home/alvari/test_ws/train_dataset/near_cyclists/bbox/idx_{}.txt'.format(i), np.ones(7))
            cycs.append(energ)
        if (pred_choice[0] == 2):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np_original_points)
            o3d.io.write_point_cloud('/home/alvari/test_ws/train_dataset/near_pedestrians/points/idx_{}.pts'.format(i), pcd)
            np.savetxt('/home/alvari/test_ws/train_dataset/near_pedestrians/bbox/idx_{}.txt'.format(i), np.ones(7))
            peds.append(energ)

    print('i:%d  loss: %f accuracy: %f' % (i, loss.data.item(), correct / float(32)))

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    '''if pred_choice[0] == 0:
        np_points = points.cpu().detach().numpy()
        np_points = np.transpose(np_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_points)
        o3d.visualization.draw_geometries([pcd])'''

print("OoD ", np.mean(last_layer_outliers))

print("near cars ", len(cars))
print("near peds ", len(peds))
print("near cycs ", len(cycs))

# set the font globally
plt.rcParams.update({'font.family':'serif'})
plt.rcParams["figure.figsize"] = (4,3)
sns.distplot(-np.asarray(cars), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Cars', color='fuchsia')
sns.distplot(-np.asarray(peds), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Pedestrians', color='springgreen')
sns.distplot(-np.asarray(cycs), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Cyclists', color='orangered')
#sns.distplot(-np.asarray(last_layer), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='In-distribution', color='fuchsia')
sns.distplot(-np.asarray(last_layer_outliers), bins=100, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='Out-of-distribution', color='silver')
plt.vlines(3.5, 0, 5, colors="Black", linewidth=1, linestyles="--", label="Threshold")#deepskyblue
#plt.plot(range(len(last_layer)), last_layer, "o", label="In-distribution (Car, Person, Cyclist)", color='deepskyblue', alpha=0.5)
#plt.plot(range(len(last_layer_outliers)), last_layer_outliers, "^", label="Out-of-distribution (Random clusters)", color='fuchsia', alpha=0.5)
#plt.legend(loc="lower left")
plt.legend(loc="upper left")
plt.xlabel("Negative energy")
plt.ylabel("Frequency")
#plt.xticks([0,300])
#plt.title("Default training")
#plt.title("Trained with energy loss function")
plt.xlim(-2.5, 7.5)
plt.ylim(0, 1.4)
plt.subplots_adjust(left=0.16, right=0.985, top=0.99, bottom=0.155)
plt.show()