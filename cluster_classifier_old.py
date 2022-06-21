#!/usr/bin/env python3

#from msilib import sequence
from math import dist
import torch
from torch.autograd import Variable
from pointnet.custom_models import PointNetCls
from pointnet.box_model import BoxNet
from utils.provider import write_detection_results_lidar
from utils.provider import write_detection_results_cam
from utils.model_utils import parse_output_to_tensors_cpu
from scripts.calibration import get_calib_from_file
from struct import pack
from tracemalloc import start
import numpy as np
import cv2
import time
#import matplotlib.pyplot as plt
import glob, os
from scripts import custom_functions
from scripts import plane_fit
from scripts import merge_labels
from scripts.laserscan import SemLaserScan, LaserScan
import argparse
from depth_cluster.build import Depth_Cluster
#from ScanLineRun_cluster.build import ScanLineRun_Cluster
import random
import open3d as o3d
import scipy.special as sci
#import mayavi.mlab as mlab

NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3

parser = argparse.ArgumentParser()
parser.add_argument('--sequence',dest= "sequence_in", default='00', help='')
parser.add_argument('--dataset',dest= "dataset", default='semanticKITTI', help='')
parser.add_argument('--root',  dest= "root", default='./Dataset/semanticKITTI/',help="./Dataset/semanticKITTI/")
parser.add_argument('--range_y', dest= "range_y", default=64, help="64")
parser.add_argument('--range_x', dest= "range_x", default=2048, help="2048")
parser.add_argument('--minimum_points', dest= "minimum_points", default=40, help="minimum_points of each class")
parser.add_argument('--which_cluster', dest= "which_cluster", default=1, help="2: ScanLineRun clustering; 3: superVoxel clustering; 4: euclidean; 1: depth_cluster; ")
parser.add_argument('--mode', dest= "mode", default='val', help="val or test; ")
args = parser.parse_args()

sequence_in = args.sequence_in

# ground segmentation setup
plane = plane_fit.Plane()

# cluster setup
if args.which_cluster == 1:
	cluster = Depth_Cluster.Depth_Cluster(0.3, 9) #angle threshold 0.15 (smaller th less clusters), search steps 9
    # 0.3 is good

#if args.which_cluster == 2:
#	cluster = ScanLineRun_Cluster.ScanLineRun_Cluster(0.5, 1)

# points per cluster
n_points = 128

# classifier setup
classifier = PointNetCls(k=3)
classifier.cpu()
#classifier.load_state_dict(torch.load('cls_weights_1_6.pth'))
classifier.load_state_dict(torch.load('cls_no_data_aug.pth'))
classifier.eval()

# box estimator setup
box_estimator = BoxNet(n_classes=3, n_channel=3)
box_estimator.cpu()
#box_estimator.load_state_dict(torch.load('box_weights_1_6.pth'))
box_estimator.load_state_dict(torch.load('box_no_data_aug.pth'))
box_estimator.eval()

def key_func(x):
        return os.path.split(x)[-1]

def appendCylindrical_np(xyz):
    ptsnew = np.dstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,:,0]**2 + xyz[:,:,1]**2
    ptsnew[:,:,3] = np.sqrt(xy) #radial
    ptsnew[:,:,4] = np.arctan2(xyz[:,:,1], xyz[:,:,0]) #azimuth, horizontal
    ptsnew[:,:,5] = xyz[:,:,2] #z, height
    return ptsnew

def full_scan():
    Scan = LaserScan(project=True, flip_sign=False, H=args.range_y, W=args.range_x, fov_up=3.0, fov_down=-25.0)

    #load data
    #lidar_data = sorted(glob.glob('scripts/data3/*.bin'), key=key_func)
    #lidar_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/velodyne/*.bin'.format(sequence_in)), key=key_func)
    lidar_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/velodyne/*.bin'), key=key_func)
    #lidar_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/testing/velodyne/*.bin'), key=key_func)
    #lidar_data = sorted(glob.glob('/home/alvari/Desktop/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data/*.bin'), key=key_func)
    #lidar_data = sorted(glob.glob('/home/alvari/Desktop/2011_09_28/2011_09_28_drive_0016_sync/velodyne_points/data/*.bin'), key=key_func)
    #label_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/labels/*.label'.format(sequence_in)), key=key_func)
    #calib_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/calib/*.txt'), key=key_func)
    calib_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/calib/*.txt'), key=key_func)

    result_dir = '/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/'
    #result_dir = '/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/testing/'
    #result_dir = '/home/alvari/Desktop/2011_09_26/2011_09_26_drive_0009_sync/'
    #result_dir = '/home/alvari/Desktop/2011_09_28/2011_09_28_drive_0016_sync/'
    #result_dir = '/home/alvari/Desktop/'

    pixel_accuracies = []
    IoUs = []
    total_times = []

    for i in range(len(lidar_data)-5000 , len(lidar_data)):
        total_time = 0

        Scan.open_scan(lidar_data[i])

        # organize points
        xyz_list = Scan.points
        range_img_pre = Scan.proj_range
        xyz = Scan.proj_xyz #runtime 0.001 ms
        orig_xyz = np.copy(xyz)

        # add cylindrical coordinates
        xyz_cyl = appendCylindrical_np(xyz) #runtime 9 ms

        # add label channel
        xyz_cyl = np.append(xyz_cyl, np.zeros((64, 2048, 1)), axis=2) #runtime 3 ms

        '''# setup label projection
        semantic_label = np.fromfile(label_data[i], dtype=np.uint32)
        semantic_label = semantic_label.reshape((-1))
        semantic_label = semantic_label & 0xFFFF
        semantic_label_img = np.zeros((64,2048))
        for jj in range(len(Scan.proj_x)):
            y_range, x_range = Scan.proj_y[jj], Scan.proj_x[jj]
            if (semantic_label_img[y_range, x_range] == 0):
                semantic_label_img[y_range, x_range] = semantic_label[jj]

        # create gt ground plane mask #label numbers for ground plane: 40,44,48,49,60,72
        gt_i = np.zeros((64, 2048))
        gt_i[semantic_label_img == 40] = 1
        gt_i[semantic_label_img == 44] = 1
        gt_i[semantic_label_img == 48] = 1
        gt_i[semantic_label_img == 49] = 1
        gt_i[semantic_label_img == 60] = 1
        gt_i[semantic_label_img == 72] = 1
        gt_mask = gt_i > 0'''

        # crop
        xyz_cyl = xyz_cyl[0:64, 768:1280]
        '''gt_i = gt_i[0:64, 768:1280]
        semantic_label_img = semantic_label_img[0:64, 768:1280]'''
        xyz = xyz[0:64, 768:1280]
        range_img_pre = range_img_pre[0:64, 768:1280]
        orig_xyz = orig_xyz[0:64, 768:1280]

        # sample ground points
        start = time.time()
        ground_i = custom_functions.groundRemoval(xyz_cyl)[:,:,6] #runtime 3 ms
        stop = time.time()
        total_time += (stop-start)
        ground_mask = ground_i > 0

        # fit plane with RANSAC
        ground_points = xyz[ground_mask]
        all_points = orig_xyz.reshape(-1, 3)
        start = time.time()
        best_eq, best_inliers = plane.fit(pts=ground_points, all_pts=all_points, thresh=0.2, minPoints=100, maxIteration=100)
        stop = time.time()
        total_time += (stop-start)
        ground_points_ransac = all_points[best_inliers]
        Scan.set_points(ground_points_ransac)
        ground_i_ransac = Scan.proj_mask
        ground_i_ransac = ground_i_ransac[0:64, 768:1280]
        range_unprojected = range_img_pre.reshape(-1)
        range_unprojected[best_inliers] = 0
        range_projected = range_unprojected.reshape(64, 512)
        ground_mask_ransac = ground_i_ransac > 0

        # clustering
        start = time.time()
        instance_label = cluster.Depth_cluster(range_projected.reshape(-1))
        stop = time.time()
        total_time += (stop-start)
        #print('clustering:', stop-start)
        instance_label = np.asarray(instance_label).reshape(64,512)

        # check ground accuracy
        '''TP = np.sum(np.logical_and(ground_i_ransac == 1, gt_i == 1))
        TN = np.sum(np.logical_and(ground_i_ransac == 0, gt_i == 0))
        FP = np.sum(np.logical_and(ground_i_ransac == 1, gt_i == 0))
        FN = np.sum(np.logical_and(ground_i_ransac == 0, gt_i == 1))
        pixel_accuracy = (TP + TN)/(TP + TN + FP + FN)
        IoU = TP/(TP + FP + FN)
        pixel_accuracies.append(pixel_accuracy)
        IoUs.append(IoU)'''

        # build NN input tensor, note: this should be done parallel with clustering, since now it's pretty slow (10-30 ms)
        nn_input_points = np.zeros((n_points, 3, 0))
        nn_input_dist = np.zeros((0))
        cluster_centers = np.zeros((1, 3, 0))
        nn_input_voxel = np.zeros((1, 3, 0))
        cluster_instance_ids = set(instance_label[instance_label != 0])
        for c in cluster_instance_ids:
            cluster_i = xyz[instance_label == c].reshape(-1, 3)
            #target = np.bincount(semantic_label_img[instance_label == c].astype(int)).argmax()
            if len(cluster_i) < 10:
                continue

            original_len = len(cluster_i)
            
            # remove big cluster
            if abs(np.max(cluster_i[:, 0]) - np.min(cluster_i[:, 0])) > 5:
                continue
            if abs(np.max(cluster_i[:, 1]) - np.min(cluster_i[:, 1])) > 5:
                continue
            if abs(np.max(cluster_i[:, 2]) - np.min(cluster_i[:, 2])) > 5:
                continue

            # sample cluster
            choice = np.random.choice(len(cluster_i), n_points, replace=True)
            cluster_i = cluster_i[choice, :]
            
            # origin to cluster center
            center = np.expand_dims(np.mean(cluster_i, axis = 0), 0)
            cluster_i = cluster_i - center

            # world position vector
            dist_from_orig = 20.0
            if original_len < n_points:
                dist_from_orig = np.linalg.norm(center)

            # spherical voxel position
            center_in_spherical = np.zeros((1, 3))
            center_in_spherical[:,0] = np.sqrt(center[:,0]**2 + center[:,1]**2 + center[:,2]**2) # euclidean from origin 
            center_in_spherical[:,1] = np.arctan2(center[:,1], center[:,0]) # azimuth, horizontal angle
            center_in_spherical[:,2] = np.arctan2(np.sqrt(center[:,0]**2 + center[:,1]**2), center[:,2]) # for elevation angle defined from Z-axis down

            # define voxel grid coordinate
            r_bins = np.linspace(0, 75, 10)
            a_bins = np.linspace(-(np.pi/4+0.2), np.pi/4+0.2, 10)
            e_bins = np.linspace(1*np.pi/3, 2*np.pi/3, 10)
            voxel_coord = np.zeros((1, 3))
            voxel_coord[:,0] = np.digitize(center_in_spherical[:,0], r_bins).squeeze()
            voxel_coord[:,1] = np.digitize(center_in_spherical[:,1], a_bins).squeeze()
            voxel_coord[:,2] = np.digitize(center_in_spherical[:,2], e_bins).squeeze()

            nn_input_points = np.dstack((nn_input_points, cluster_i))
            nn_input_dist = np.hstack((nn_input_dist, dist_from_orig))
            nn_input_voxel = np.dstack((nn_input_voxel, voxel_coord))
            cluster_centers = np.dstack((cluster_centers, center))

            #check if correct and visualize
            '''if target == 10 and i > 20:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(cluster_i)
                #o3d.visualization.draw_geometries([pcd])'''

        nn_input_dist = np.expand_dims(nn_input_dist, axis=0)
        nn_input_dist = torch.from_numpy(nn_input_dist)
        nn_input_points = torch.from_numpy(nn_input_points)
        nn_input_voxel = torch.from_numpy(nn_input_voxel)
        cluster_centers = torch.from_numpy(cluster_centers)
        nn_input_points = torch.swapaxes(nn_input_points, 0, 2)
        nn_input_dist = torch.swapaxes(nn_input_dist, 0, 1)
        nn_input_voxel = torch.swapaxes(nn_input_voxel, 0, 2)
        cluster_centers = torch.squeeze(torch.swapaxes(cluster_centers, 1, 2))
        nn_input_points, nn_input_dist, nn_input_voxel = Variable(nn_input_points), Variable(nn_input_dist), Variable(nn_input_voxel)
        nn_input_points, nn_input_dist, nn_input_voxel, cluster_centers = nn_input_points.cpu().float(), nn_input_dist.cpu().float(), nn_input_voxel.cpu().float(), cluster_centers.cpu().float()

        # classifier
        start = time.time()
        pred, global_feat, avg, out = classifier(nn_input_points, nn_input_dist, nn_input_voxel)
        stop = time.time()
        total_time += (stop-start)
        #print(stop-start)
        pred_choice = pred.data.max(1)[1]
        
        # energy score
        np_out = out.cpu().detach().numpy()
        #T = 1000
        #a = 1
        #energy = -T * sci.logsumexp(np.max(np_out)/T)
        #Ec = -torch.logsumexp(out, dim=1)
        energy = -sci.logsumexp(np_out, axis=1)

        # OOD dropout
        energy_threshold = -4.9 # -4.7
        nn_input_points = nn_input_points[energy < energy_threshold, :, :]
        nn_input_dist = nn_input_dist[energy < energy_threshold, :]
        nn_input_voxel = nn_input_voxel[energy < energy_threshold, :]
        cluster_centers = cluster_centers[energy < energy_threshold, :]
        pred_choice = pred_choice[energy < energy_threshold]
        energy = energy[energy < energy_threshold]

        # transform target scalar to 3x one hot vector
        hot1 = torch.zeros(np.count_nonzero(energy < energy_threshold))
        hot1[pred_choice == 0] = 1
        hot2 = torch.zeros(np.count_nonzero(energy < energy_threshold))
        hot2[pred_choice == 2] = 1
        hot3 = torch.zeros(np.count_nonzero(energy < energy_threshold))
        hot3[pred_choice == 1] = 1
        one_hot = torch.vstack((hot1, hot2, hot3))
        one_hot = one_hot.transpose(1, 0)

        if (nn_input_points.shape[0] > 0):
            # boxnet (bug: if batch size = 0 -> boxnet crashes)
            start = time.time()
            box_pred, center_delta = box_estimator(nn_input_points, one_hot, nn_input_dist, nn_input_voxel)
            stop = time.time()
            total_time += (stop-start)

            # parse output
            center_boxnet, \
            heading_scores, heading_residual_normalized, heading_residual, \
            size_scores, size_residual_normalized, size_residual = \
                    parse_output_to_tensors_cpu(box_pred)

            # move boxes to real positions using cluster center 
            stage1_center = cluster_centers + center_delta
            box3d_center = center_boxnet + stage1_center

            # to correct format
            class_to_string = {0:'Car', 1:'Cyclist', 2:'Pedestrian'}
            type_list = [class_to_string[class_i] for class_i in pred_choice.detach().numpy()]
            id_list = i * np.ones((len(type_list))) # id of the scan
            box2d_list = np.zeros((len(type_list), 4)) # leave to zero
            center_list = box3d_center.detach().numpy()
            heading_cls_list = heading_scores.data.max(1)[1].detach().numpy()
            hcls_onehot = np.eye(NUM_HEADING_BIN)[heading_cls_list]
            heading_res_list = np.sum(heading_residual.detach().numpy() * hcls_onehot, axis=1)
            size_cls_list = size_scores.data.max(1)[1].detach().numpy()
            scls_onehot = np.eye(NUM_SIZE_CLUSTER)[size_cls_list]
            scls_onehot_repeat = scls_onehot.reshape(-1, NUM_SIZE_CLUSTER, 1).repeat(1, 2)
            size_res_list = np.sum(size_residual.detach().numpy() * scls_onehot_repeat, axis=1)
            score_list1 = np.clip(-energy+energy_threshold, 0.01, 3) #np.ones((len(type_list)))
            score_list = (score_list1-0.01)/3 #np.random.uniform(low=0.01, high=1.00, size=(len(type_list),))
            #score_list = np.ones((len(type_list))) 

            # in lidar coordinates
            write_predictions = write_detection_results_lidar(result_dir, id_list, type_list, box2d_list, center_list, \
                                heading_cls_list, heading_res_list, size_cls_list, size_res_list, score_list)

            # in camera coordinates
            calib_file = get_calib_from_file(calib_data[i])
            V2C = calib_file['Tr_velo2cam']
            R0 = calib_file['R0']
            write_predictions = write_detection_results_cam(result_dir, id_list, type_list, box2d_list, center_list, \
                                heading_cls_list, heading_res_list, size_cls_list, size_res_list, score_list, V2C, R0)
        else:
            if result_dir is None: return
            results = {} # map from idx to list of strings, each string is a line (without \n)
            idx = i # idx is the number of the scan
            output_str = "DontCare -1.00 -1 -10.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 1.00"
            if idx not in results: results[idx] = []
            results[idx].append(output_str)

            # Write TXT files
            if not os.path.exists(result_dir): os.mkdir(result_dir)
            output_dir = os.path.join(result_dir, 'pred_lidar')
            if not os.path.exists(output_dir): os.mkdir(output_dir)
            for idx in results:
                pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
                fout = open(pred_filename, 'w')
                for line in results[idx]:
                    fout.write(line+'\n')
                fout.close() 

            # Write TXT files
            if not os.path.exists(result_dir): os.mkdir(result_dir)
            output_dir = os.path.join(result_dir, 'pred')
            if not os.path.exists(output_dir): os.mkdir(output_dir)
            for idx in results:
                pred_filename = os.path.join(output_dir, '%06d.txt'%(idx))
                fout = open(pred_filename, 'w')
                for line in results[idx]:
                    fout.write(line+'\n')
                fout.close() 
        
        #print('mean pixel accuracy:', np.mean(pixel_accuracies))
        #print('mIoU:', np.mean(IoUs))
        #print(int(i/len(lidar_data)*100), '%')
        print('amount of IDs: ', np.count_nonzero(energy < energy_threshold))
        print('index: ', i)
        #total_times.append(1/total_time)
        #print('FPS: ', np.mean(total_times))

        # visualize
        '''normed2 = cv2.normalize(instance_label, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color2 = cv2.applyColorMap(normed2, cv2.COLORMAP_JET)
        scale_percent = 300 # percent of original size
        width = int(color2.shape[1] * scale_percent / 100)
        height = int(color2.shape[0] * scale_percent / 100)
        dim = (width, height)
        color2 = cv2.resize(color2, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("test", color2)
        cv2.waitKey(1000)'''

def packets():
    Scan = LaserScan(project=True, flip_sign=False, H=args.range_y, W=args.range_x, fov_up=3.0, fov_down=-25.0)

    # load data
    #lidar_data = sorted(glob.glob('scripts/data3/*.bin'), key=key_func)
    #lidar_data = sorted(glob.glob('/home/alvari//Desktop/KITTI/2011_09_28/2011_09_28_drive_0021_sync/velodyne_points/data/*.bin'), key=key_func)
    #lidar_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/velodyne/*.bin'), key=key_func)
    lidar_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/velodyne/*.bin'.format(sequence_in)), key=key_func)
    label_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/labels/*.label'.format(sequence_in)), key=key_func)

    pixel_accuracies = []
    IoUs = []
    recalls = []
    F1scores = []
    precisions = []

    plane1 = plane_fit.Plane()

    #merge = merge_labels.Merge()

    for i in range(len(lidar_data)):

        # open lidar data
        Scan.open_scan(lidar_data[i])

        # organize pc
        range_img_pre = Scan.proj_range
        xyz = Scan.proj_xyz
        #orig_range_img = np.copy(range_img_pre)
        #test_range_img = np.copy(range_img_pre)

        # add cylindrical coordinates
        xyz_cyl = appendCylindrical_np(xyz) #runtime 9 ms

        # add label channel
        xyz_cyl = np.append(xyz_cyl, np.zeros((64, 2048, 1)), axis=2) 

        # open label data
        panoptic_label = np.fromfile(label_data[i], dtype=np.uint32)
        panoptic_label = panoptic_label.reshape((-1))
        semantic_label = panoptic_label & 0xFFFF
        #semantic_label_inv = [inv_label_dict_reverse[mm] for mm in semantic_label]
        semantic_label_img = np.zeros((64,2048))
        panoptic_label_img = np.zeros((64,2048))
        #depth_img = np.zeros((64,2048))
        #covered_points = []
        for jj in range(len(Scan.proj_x)):
            y_range, x_range = Scan.proj_y[jj], Scan.proj_x[jj]
            if (semantic_label_img[y_range, x_range] == 0):
                semantic_label_img[y_range, x_range] = semantic_label[jj]
                panoptic_label_img[y_range, x_range] = panoptic_label[jj]
                #semantic_label_img[y_range,x_range] = semantic_label_inv[jj]
                #depth_img[y_range,x_range] = Scan.unproj_range[jj]

        # create gt ground plane mask, label numbers for ground plane: 40,44,48,49,60,72
        gt_i = np.zeros((64, 2048))
        gt_i[semantic_label_img == 40] = 1
        gt_i[semantic_label_img == 44] = 1
        gt_i[semantic_label_img == 48] = 1
        gt_i[semantic_label_img == 49] = 1
        gt_i[semantic_label_img == 60] = 1
        gt_i[semantic_label_img == 72] = 1
        gt_mask = gt_i > 0
        
        # create data packets
        packet_w = 64
        #packet_w = 128
        range_packet = range_img_pre[0:64, 0:packet_w]
        range_processed = np.zeros((64,0))
        xyz_packet = xyz[0:64, 0:packet_w]
        xyz_processed = np.zeros((64,0,3))
        xyz_cyl_packet = xyz_cyl[0:64, 0:packet_w]
        xyz_cyl_processed = np.zeros((64,0,7))
        ground_prediction_packet = np.zeros((64, packet_w))
        ground_prediction_processed = np.zeros((64,0))
        cluster_prediction_packet = np.zeros((64, packet_w))
        cluster_prediction_processed = np.zeros((64,0))

        # timers
        clustering_time = 0
        ground_time = 0

        # simulate packet streaming
        for p in range(1, int(range_img_pre.shape[1]/packet_w+1)):

            # sample ground points with sobel filter
            start = time.time()
            ground_i = custom_functions.groundRemoval(xyz_cyl_packet)[:,:,6] #runtime - ms
            stop = time.time()
            ground_time = ground_time + stop - start
            ground_mask = ground_i > 0

            # fit plane with RANSAC
            all_points = xyz_packet.reshape(-1, 3)
            ground_points = xyz_packet[ground_mask]
            if len(ground_points) > 10:
                start = time.time()
                best_eq, best_inliers = plane1.fit(pts=ground_points, all_pts=all_points, thresh=0.15, minPoints=100, maxIteration=10)#th=0.15
                stop = time.time()
                ground_time = ground_time + stop - start
                range_unprojected = range_packet.reshape(-1)
                range_unprojected[best_inliers] = 0
                range_packet = range_unprojected.reshape(range_packet.shape[0], range_packet.shape[1])
                ground_prediction_unprojected = ground_prediction_packet.reshape(-1)
                ground_prediction_unprojected[best_inliers] = 1
                ground_prediction_packet = ground_prediction_unprojected.reshape(ground_prediction_packet.shape[0], ground_prediction_packet.shape[1])

            # add previous columns to the packet
            if p == 1:
                cluster_input_range = np.hstack((np.zeros((64,packet_w)), range_packet))
                cluster_input_prediction = np.hstack((np.zeros((64,packet_w)), cluster_prediction_packet))
            else:
                cluster_input_range = np.hstack((range_processed[0:64, range_processed.shape[1]-packet_w:range_processed.shape[1]], range_packet))
                cluster_input_prediction = np.hstack((cluster_prediction_processed[0:64, cluster_prediction_processed.shape[1]-packet_w:cluster_prediction_processed.shape[1]], cluster_prediction_packet))
                
            # fetch current label
            current_label = 0
            if p > 1:
                current_label = np.max(cluster_prediction_processed)

            # local clustering
            ranges = cluster_input_range.reshape(-1)
            cluster_predictions = cluster_input_prediction.reshape(-1)
            start = time.time()
            instance_prediction = cluster.Packet_cluster(ranges, int(current_label))
            stop = time.time()
            clustering_time = clustering_time + stop-start

            # remove first columns
            cluster_output_prediction = np.asarray(instance_prediction).reshape(64, packet_w+packet_w)
            cluster_prediction_packet = cluster_output_prediction[0:64, cluster_output_prediction.shape[1]-packet_w:cluster_output_prediction.shape[1]]

            # assign boundary region
            boundary = cluster_output_prediction[0:64, cluster_output_prediction.shape[1]-packet_w-2:cluster_output_prediction.shape[1]-packet_w]

            # add processed packet to processed array
            range_processed = np.hstack((range_processed, range_packet))
            xyz_processed = np.hstack((xyz_processed, xyz_packet))
            xyz_cyl_processed = np.hstack((xyz_cyl_processed, xyz_cyl_packet))
            ground_prediction_processed = np.hstack((ground_prediction_processed, ground_prediction_packet))
            cluster_prediction_processed = np.hstack((cluster_prediction_processed, cluster_prediction_packet))

            # delete small clusters
            cluster_idx = set(cluster_prediction_processed[cluster_prediction_processed != 0])
            for c in cluster_idx:
                if np.count_nonzero(cluster_prediction_processed == c) < 10:
                    cluster_prediction_processed[cluster_prediction_processed == c] = 0
            
            # global clustering
            start = time.time()
            cluster_prediction_processed = custom_functions.mergeLabels(cluster_prediction_processed, boundary, packet_w)
            stop = time.time()
            clustering_time = clustering_time + stop-start

            # if right edge of cluster is not touching the right edge of image, append to cluster instance id a list, check if list len > batch_size
            # get points of each instance id, random.choice(100pts), centering
            # numpy array: points (batch_size, max_points(=100), 3)
            # NN input: points = torch.tensor([batch_size, max_points(=100), 3])
            # NN output: 2 x torch.tensor([batch_size, classes]) 
            # compute energy, threshold, if ID, argmax(pred)
            # match labels with instance ids
            # update precessed semantic img
            
            # update packet
            range_packet = range_img_pre[0:64, p*packet_w:p*packet_w+packet_w]
            xyz_packet = xyz[0:64, p*packet_w:p*packet_w+packet_w]
            xyz_cyl_packet = xyz_cyl[0:64, p*packet_w:p*packet_w+packet_w]
            ground_prediction_packet = np.zeros((64, packet_w))
            cluster_prediction_packet = np.zeros((64, packet_w))

        # save clusters, ~100 clusters per scan
        cluster_instance_ids = set(cluster_prediction_processed[cluster_prediction_processed != 0])
        points_batch = np.zeros((100,3,0))
        #start = time.time()
        for c in cluster_instance_ids:
            clusterr = xyz[cluster_prediction_processed == c].reshape(-1, 3)
            #target = np.mean(semantic_label_img[cluster_prediction_processed == c])
            target = np.bincount(semantic_label_img[cluster_prediction_processed == c].astype(int)).argmax()
            #print(semantic_label_img[cluster_prediction_processed == c])
            if clusterr.shape[0] < 20:
                continue
            
            # sample n points 
            choice = np.random.choice(len(clusterr), 100, replace=True) # constant output size

            # resample
            point_set = clusterr[choice, :]

            point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
            #dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
            #point_set = point_set / dist # scale
        
            #stop = time.time()
            #print('nomralization:', stop-start)

            #points_tensor = np.dstack((points_tensor, point_set))

            #point_set_t = torch.from_numpy(points_tensor)

            #print(point_set_t.shape)
            #points.shape == 1,100,3

            points = torch.from_numpy(point_set)
            points = points[None, :, :]
            #print(point_set.shape)
            points = Variable(points)
            points = points.transpose(2, 1)
            points = points.cpu()

            #points.shape == 1,3,100

            # run PointNet
            pred, global_feat, avg, out = classifier(points)
            #print(pred)
            pred_choice = pred.data.max(1)[1]
            #print(pred_choice[0])

            np_out = out.cpu().detach().numpy()
            T = 1000
            #energy = -T * sci.logsumexp(np.max(np_out)/T)

            #check if correct and visualize
            #if pred_choice[0] == 0 and target == 10 and energy < -2.5:
            #if i > 3 and pred_choice[0] == 2 and energy < -2.5:
                #pcd = o3d.geometry.PointCloud()
                #pcd.points = o3d.utility.Vector3dVector(clusterr)
                #o3d.visualization.draw_geometries([pcd])

        normed_packet = cv2.normalize(cluster_prediction_processed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_packet = cv2.applyColorMap(normed_packet, cv2.COLORMAP_HOT)
        cv2.imshow("test", color_packet)
        cv2.waitKey(100)

        print(' ')
        print("cluster: ", clustering_time)
        print("ground: ", ground_time)
        print("cluster+ground: ", clustering_time + ground_time)

        # check ground accuracy
        TP = np.sum(np.logical_and(ground_prediction_processed == 1, gt_i == 1))
        TN = np.sum(np.logical_and(ground_prediction_processed == 0, gt_i == 0))
        FP = np.sum(np.logical_and(ground_prediction_processed == 1, gt_i == 0))
        FN = np.sum(np.logical_and(ground_prediction_processed == 0, gt_i == 1))
        pixel_accuracy = (TP + TN)/(TP + TN + FP + FN)
        IoU = TP/(TP + FP + FN)
        recall = TP/(TP + FN)
        precision = TP/(TP + FP)
        F1score = 2*TP/(2*TP + FP + FN)
        pixel_accuracies.append(pixel_accuracy)
        IoUs.append(IoU)
        recalls.append(recall)
        F1scores.append(F1score)
        precisions.append(precision)
        print(' ')
        print(int(i/len(lidar_data)*100), '%')
        print('mIoU:', np.mean(IoUs))
        print('recall:', np.mean(recalls))
        print('mean pixel accuracy:', np.mean(pixel_accuracies))
        print('F1-score:', np.mean(F1scores))
        print('Precision:', np.mean(precisions))

    # s.o.t.a.
    # mIoU: 0.836, 0.817
    # recall: 0.993
    # pixel accuracy: 0.871 
    # F1-score: 0.798 
    # precision: 0.841 

def main():
    '''lidar_data = sorted(glob.glob('scripts/test_data/*.bin'), key=key_func)
    Scan = LaserScan(project=True, flip_sign=False, H=16, W=2048, fov_up=15.0, fov_down=-15.0)
    for i in range(len(lidar_data)):
        Scan.open_scan(lidar_data[i])
        print(Scan.points.shape)
        range_img = Scan.proj_range
        normed2 = cv2.normalize(range_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color2 = cv2.applyColorMap(normed2, cv2.COLORMAP_JET)
        scale_percent = 1200 # percent of original size
        width = int(color2.shape[1]* 0.8)
        height = int(color2.shape[0] * scale_percent / 100)
        dim = (width, height)
        color2 = cv2.resize(color2, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("test", color2)
        cv2.waitKey(100)'''

    full_scan()
    #packets()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
