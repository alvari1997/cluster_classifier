#!/usr/bin/env python3

#from msilib import sequence
import torch
from torch.autograd import Variable
from pointnet.custom_models import PointNetCls
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
import random
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--sequence',dest= "sequence_in", default='00', help='')
parser.add_argument('--dataset',dest= "dataset", default='semanticKITTI', help='')
parser.add_argument('--root',  dest= "root", default='./Dataset/semanticKITTI/',help="./Dataset/semanticKITTI/")
parser.add_argument('--range_y', dest= "range_y", default=64, help="64")
parser.add_argument('--range_x', dest= "range_x", default=2048, help="2048")
parser.add_argument('--minimum_points', dest= "minimum_points", default=40, help="minimum_points of each class")
parser.add_argument('--which_cluster', dest= "which_cluster", default=1, help="4: ScanLineRun clustering; 3: superVoxel clustering; 2: euclidean; 1: depth_cluster; ")
parser.add_argument('--mode', dest= "mode", default='val', help="val or test; ")
args = parser.parse_args()

#inv_label_dict_reverse = {0:0,10:1,11:2,15:3,18:4,20:5,30:6,31:7,32:8,40:9,44:10,48:11,49:12,50:13,51:14,70:15,71:16,72:17,80:18,81:19}
#label numbers for ground plane: 40,44,48,49,60,72

sequence_in = args.sequence_in

# cluster setup
if args.which_cluster == 1:
	cluster = Depth_Cluster.Depth_Cluster(0.15,9) #angle threshold 0.15 (smaller th less clusters), search steps 9


# classifier setup
classifier = PointNetCls(k=4)
classifier.cpu()
# load weights
classifier.load_state_dict(torch.load('cls_model_99.pth'))
# set evaluation mode
classifier.eval()


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
    #Label = SemLaserScan(nclasses=20, sem_color_dict=None, project=False, flip_sign=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0)

    #load data
    #lidar_data = sorted(glob.glob('scripts/data3/*.bin'), key=key_func)
    lidar_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/velodyne/*.bin'.format(sequence_in)), key=key_func)
    label_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/labels/*.label'.format(sequence_in)), key=key_func)

    pixel_accuracies = []
    IoUs = []

    #plane1 = pyrsc.Plane()
    plane1 = plane_fit.Plane()

    for i in range(len(lidar_data)):
        Scan.open_scan(lidar_data[i])
        #Label.open_label(label_data[i])

        xyz_list = Scan.points

        #organize pc
        range_img_pre = Scan.proj_range
        orig_range_img = np.copy(range_img_pre)
        test_range_img = np.copy(range_img_pre)
        xyz = Scan.proj_xyz #runtime 0.001 ms
        orig_xyz = np.copy(xyz)

        #add cylindrical coordinates
        xyz_cyl = appendCylindrical_np(xyz) #runtime 9 ms

        #add label channel
        xyz_cyl = np.append(xyz_cyl, np.zeros((64, 2048, 1)), axis=2) #runtime 3 ms

        #organize labels
        #semantic_label_img_pre = Label.do_label_projection
        #semantic_label_img_pre = np.fromfile(label_data[i], dtype=np.uint32)
        #print(semantic_label_img_pre.shape)

        semantic_label = np.fromfile(label_data[i], dtype=np.uint32)
        semantic_label = semantic_label.reshape((-1))
        semantic_label = semantic_label & 0xFFFF
        #semantic_label_inv = [inv_label_dict_reverse[mm] for mm in semantic_label]
        semantic_label_img = np.zeros((64,2048))
        #depth_img = np.zeros((64,2048))
        #covered_points = []
        for jj in range(len(Scan.proj_x)):
            y_range, x_range = Scan.proj_y[jj], Scan.proj_x[jj]
            if (semantic_label_img[y_range, x_range] == 0):
                semantic_label_img[y_range, x_range] = semantic_label[jj]
                #semantic_label_img[y_range,x_range] = semantic_label_inv[jj]
                #depth_img[y_range,x_range] = Scan.unproj_range[jj]

        #create gt ground plane mask #label numbers for ground plane: 40,44,48,49,60,72
        gt_i = np.zeros((64, 2048))
        gt_i[semantic_label_img == 40] = 1
        gt_i[semantic_label_img == 44] = 1
        gt_i[semantic_label_img == 48] = 1
        gt_i[semantic_label_img == 49] = 1
        gt_i[semantic_label_img == 60] = 1
        gt_i[semantic_label_img == 72] = 1
        gt_mask = gt_i > 0

        '''#remove ground PyBind11
        z_img_pre = xyz[:,:,2]
        z_img = z_img_pre.reshape(-1)
        start = time.time()
        ground_mask1 = groundremoval.Ground_removal(z_img)
        stop = time.time()
        print('ground removal PyBind11:', stop-start)'''

        #remove ground Cython
        start = time.time()
        ground_i = custom_functions.groundRemoval(xyz_cyl)[:,:,6] #runtime 3 ms
        stop = time.time()
        #print('ground removal Cython:', stop-start)
        ground_mask = ground_i > 0
        #range_img_pre[ground_mask] = 0

        #fit plane with RANSAC
        ground_points = xyz[ground_mask]
        #all_points = xyz[orig_range_img > 0.0001]
        all_points = orig_xyz.reshape(-1, 3)
        start = time.time()
        #print(ground_points.shape)
        #print(all_points.shape)
        best_eq, best_inliers = plane1.fit(pts=ground_points, all_pts=all_points, thresh=0.2, minPoints=100, maxIteration=10)
        #print(best_inliers.shape)
        stop = time.time()
        #print(stop-start)
        #print(best_eq)
        ground_points_ransac = all_points[best_inliers]
        Scan.set_points(ground_points_ransac)
        ground_i_ransac = Scan.proj_mask
        range_unprojected = range_img_pre.reshape(-1)
        range_unprojected[best_inliers] = 0
        range_projected = range_unprojected.reshape(64, 2048)
        ground_mask_ransac = ground_i_ransac > 0
        #range_img_pre[ground_mask_ransac] = 0

        test_img = np.vstack((orig_range_img, range_projected))

        '''normed = cv2.normalize(test_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color = cv2.applyColorMap(normed, cv2.COLORMAP_HOT)
        cv2.imshow("test", color)
        cv2.waitKey(1000)'''

        range_img_crop = range_img_pre[0:64, 0:20]
        range_img_crop_reshaped = range_img_crop.reshape(-1)

        copied = np.copy(range_img_pre)
        range_img = copied.reshape(-1)

        #clustering
        start = time.time()
        instance_label = cluster.Depth_cluster(range_img)
        #instance_label = cluster.Packet_cluster(range_img, range_img)
        stop = time.time()
        print('clustering:', stop-start)
        print(np.max(instance_label))

        #instance_label=np.asarray(instance_label).reshape(64,20)
        instance_label=np.asarray(instance_label).reshape(64,2048)

        #check ground accuracy
        TP = np.sum(np.logical_and(ground_i_ransac == 1, gt_i == 1))
        TN = np.sum(np.logical_and(ground_i_ransac == 0, gt_i == 0))
        FP = np.sum(np.logical_and(ground_i_ransac == 1, gt_i == 0))
        FN = np.sum(np.logical_and(ground_i_ransac == 0, gt_i == 1))
        pixel_accuracy = (TP + TN)/(TP + TN + FP + FN)
        IoU = TP/(TP + FP + FN)
        pixel_accuracies.append(pixel_accuracy)
        IoUs.append(IoU)
        #print('pixel accuracy:', pixel_accuracy)
        #print('IoU:', IoU)
        #print(int(i/len(lidar_data)*100), '%')

        '''normed2 = cv2.normalize(instance_label, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color2 = cv2.applyColorMap(normed2, cv2.COLORMAP_JET)
        test_img2 = np.vstack((color, color2))

        cv2.imshow("test", test_img2)
        cv2.waitKey(1000)'''
    
    print('mean pixel accuracy:', np.mean(pixel_accuracies))
    print('mIoU:', np.mean(IoUs))

def packets():
    Scan = LaserScan(project=True, flip_sign=False, H=args.range_y, W=args.range_x, fov_up=3.0, fov_down=-25.0)

    # load data
    #lidar_data = sorted(glob.glob('scripts/data3/*.bin'), key=key_func)
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

        '''# save clusters, ~100 clusters per scan
        cluster_idx = set(cluster_prediction_processed[cluster_prediction_processed != 0])
        for c in cluster_idx:
            clusterr = xyz[cluster_prediction_processed == c].reshape(-1, 3)
            if clusterr.shape[0] < 20:
                continue
            
            # sample n points 
            choice = np.random.choice(len(clusterr), 100, replace=True)

            # resample
            point_set = clusterr[choice, :]

            point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
            point_set = point_set / dist #scale

            point_set = torch.from_numpy(point_set)

            torch.unsqueeze(point_set, 0)

            print(point_set.shape)
            #points.shape == 1,100,3

            points = point_set
            points = Variable(points)
            points = points.transpose(2, 1)
            points = points.cpu()

            #points.shape == 1,3,100

            # run PointNet
            pred, global_feat, avg, out = classifier(points)'''
            
    
        # visualize clusters
        cluster_idx = set(cluster_prediction_processed[cluster_prediction_processed != 0])
        for c in cluster_idx:
            clusterr = xyz[cluster_prediction_processed == c].reshape(-1, 3)
            if clusterr.shape[0] < 20:
                continue

            # normalize
            #points_canonical = points_to_center(clusterr)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(clusterr)
            #o3d.io.write_point_cloud('outlier_cluster_dataset/{}_seq_{}_scan_{}_cluster.pts'.format(sequence_in, i, c), pcd)
            #o3d.io.write_point_cloud('outlier_vis/{}_scan_{}_cluster.ply'.format(i, c), pcd)

            # visualize
            #pcd_load = o3d.io.read_point_cloud('outlier_vis/{}_scan_{}_cluster.ply'.format(i, c))
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
    #full_scan()
    packets()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()