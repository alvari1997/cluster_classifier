from __future__ import print_function
from matplotlib.pyplot import box
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
#import ipbd

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3 #8 # one cluster for each type
#NUM_OBJECT_POINT = 512

class BoxEstimation(nn.Module):
    def __init__(self,n_classes=2):
        '''v1 Amodal 3D Box Estimation Pointnet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        '''
        super(BoxEstimation, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.n_classes = n_classes

        self.fc1 = nn.Linear(512+3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts, one_hot_vec): # bs,3,m
        '''
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residual,
            and size cluster scores and residual
        '''
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts))) # bs,128,n
        out2 = F.relu(self.bn2(self.conv2(out1))) # bs,128,n
        out3 = F.relu(self.bn3(self.conv3(out2))) # bs,256,n
        out4 = F.relu(self.bn4(self.conv4(out3)))# bs,512,n
        global_feat = torch.max(out4, 2, keepdim=False)[0] #bs,512

        expand_one_hot_vec = one_hot_vec.view(bs,-1)#bs,3
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec],1)#bs,515

        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))#bs,512
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,256
        box_pred = self.fc3(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        return box_pred

class STNxyz(nn.Module):
    def __init__(self,n_classes=3):
        super(STNxyz, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        #self.conv4 = torch.nn.Conv1d(256, 512, 1)
        self.fc1 = nn.Linear(256+n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        #init.zeros_(self.fc3.weight)
        #init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)
    def forward(self, pts, one_hot_vec):
        bs = pts.shape[0]
        x = F.relu(self.bn1(self.conv1(pts)))# bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))# bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))# bs,256,n
        x = torch.max(x, 2)[0]# bs,256
        expand_one_hot_vec = one_hot_vec.view(bs, -1)# bs,3
        x = torch.cat([x, expand_one_hot_vec],1)#bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))# bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))# bs,128
        x = self.fc3(x)# bs,
        return x

class BoxNet(nn.Module):
    def __init__(self,n_classes=3,n_channel=3):
        super(BoxNet, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.STN = STNxyz(n_classes=3)
        self.est = BoxEstimation(n_classes=3)

    def forward(self, points, one_hot):
        #dict_keys(['point_cloud', 'rot_angle', 'box3d_center', 'size_class', 'size_residual', 'angle_class', 'angle_residual', 'one_hot', 'seg'])

        #one_hot = data_dicts.get('one_hot') #torch.Size([32, 3])
        #point_cloud = data_dicts.get('point_cloud')#torch.Size([32, 4, 1024])
        #point_cloud = point_cloud[:,:self.n_channel,:]
        #object_pts_xyz = point_cloud

        # T-Net
        #object_pts_xyz = object_pts_xyz.cuda()
        center_delta = self.STN(points, one_hot) #(32,3)
        #stage1_center = center_delta + mask_xyz_mean #(32,3)
        #print(center_delta.shape)
        #print(points)
        #object_pts_xyz_new = points + center_delta

        #if(np.isnan(center_delta.cpu().detach().numpy()).any()):
        #    ipdb.set_trace()
        object_pts_xyz_new = points - \
                center_delta.view(center_delta.shape[0],-1,1).repeat(1,1,points.shape[-1])
        #object_pts_xyz_new = points

        #print(object_pts_xyz_new)

        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new, one_hot) #(32, 59)

        return box_pred, center_delta


if __name__ == '__main__':
    trans = STNxyz(n_classes=3)
    pointfeat = BoxEstimation(n_classes=3)
    cls = BoxNet(n_classes=3, n_channel=3)