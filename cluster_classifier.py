#!/usr/bin/env python3

#from msilib import sequence
from struct import pack
from tracemalloc import start
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import glob, os
import ctypes
from scripts import custom_functions
#import clustering_function_py
from scripts.laserscan import SemLaserScan, LaserScan
import argparse
from depth_cluster.build import Depth_Cluster
from ground_removal.build import Ground_Removal
import pyransac3d as pyrsc
import random

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

inv_label_dict_reverse = {0:0,10:1,11:2,15:3,18:4,20:5,30:6,31:7,32:8,40:9,44:10,48:11,49:12,50:13,51:14,70:15,71:16,72:17,80:18,81:19}
#label numbers for ground plane: 40,44,48,49,60,72

sequence_in = args.sequence_in

if args.which_cluster == 1:
	cluster = Depth_Cluster.Depth_Cluster(0.15,9) #angle threshold, search steps

#if args.which_cluster==1:
groundremoval = Ground_Removal.Ground_Removal(0.15,9) #angle threshold, search steps

def key_func(x):
        return os.path.split(x)[-1]

def cluster_classifier():
    #for filename in sorted(glob.glob('org_pointclouds/*.npy'), key=key_func):
    #data = np.load(filename)
    data = np.load('org_pointclouds/0000000035.npy')

    #append label channel
    data = np.append(data, np.zeros((64, 1900, 1)), axis=2)

    packet = data[0:64, 0:19] #10 columns
    original = np.copy(packet)
    original2 = np.copy(packet)
    original3 = np.copy(packet)
    original4 = np.copy(packet)
    tail = data[0:64, 0:1003] #1003 columns

    '''normed = cv2.normalize(packet[:,:,3], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
    cv2.imshow("test1", color)
    cv2.waitKey(0)'''

    start = time.time()
    custom_functions.groundRemoval(packet)
    stop = time.time()
    #print(new)
    print(1000*(stop-start)/(100/1900))
        
    '''plt.plot(np.arange(0,64,1), original[:,0,5], label="old")
    plt.plot(np.arange(0,64,1), packet[:,0,5], label="new")
    plt.legend()
    plt.show()'''

    '''normed = cv2.normalize(packet[:,:,3], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
    cv2.imshow("test", color)
    cv2.waitKey(0)'''

    start = time.time()
    custom_functions.clustering(packet)
    stop = time.time()
    #print(new)
    print(1000*(stop-start)/(100/1900))

    '''normed = cv2.normalize(packet[:,:,6], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
    cv2.imshow("test3", color)
    cv2.waitKey(0)'''

    '''height_map = original4[:,:,5]
    distance_map = original4[:,:,3]
    label_map = original4[:,:,6]
    start = time.time()
    #ground gradient
    dhv = cv2.Sobel(height_map,cv2.CV_64F,0,1,ksize=3)
    dhh = cv2.Sobel(height_map,cv2.CV_64F,1,0,ksize=3)
    ddv = cv2.Sobel(distance_map,cv2.CV_64F,0,1,ksize=3)
    ground_angle = dhv/ddv
    #ground threshold, bool
    h_i = np.logical_and(height_map > -4, height_map < -1.3)
    x_i = np.logical_and(dhh > -5, dhh < 5)
    y_i = np.logical_and(ground_angle > -0.2, ground_angle < 0.2)
    #combine to get ground indicies
    ground_mask = x_i * y_i * h_i #all pixels that are part of the ground
    #delete ground
    distance_map[ground_mask] = 0
    label_map[ground_mask] = -1
    stop = time.time()
    print(1000*(stop-start)/(100/1900))'''

def appendCylindrical_np(xyz):
    ptsnew = np.dstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,:,0]**2 + xyz[:,:,1]**2
    ptsnew[:,:,3] = np.sqrt(xy) #radial
    ptsnew[:,:,4] = np.arctan2(xyz[:,:,1], xyz[:,:,0]) #azimuth, horizontal
    ptsnew[:,:,5] = xyz[:,:,2] #z, height
    return ptsnew

def test():
    Scan = LaserScan(project=True, flip_sign=False, H=args.range_y, W=args.range_x, fov_up=3.0, fov_down=-25.0)
    #Label = SemLaserScan(nclasses=20, sem_color_dict=None, project=False, flip_sign=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0)

    #load data
    #lidar_data = sorted(glob.glob('scripts/data3/*.bin'), key=key_func)
    lidar_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/velodyne/*.bin'.format(sequence_in)), key=key_func)
    label_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/labels/*.label'.format(sequence_in)), key=key_func)

    pixel_accuracies = []
    IoUs = []

    plane1 = pyrsc.Plane()

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
        #label_img_pre = Label.do_label_projection
        #label_img_pre = np.fromfile(label_data[i], dtype=np.uint32)
        #print(label_img_pre.shape)

        semantic_label = np.fromfile(label_data[i], dtype=np.uint32)
        semantic_label = semantic_label.reshape((-1))
        semantic_label = semantic_label & 0xFFFF
        #semantic_label_inv = [inv_label_dict_reverse[mm] for mm in semantic_label]
        label_img = np.zeros((64,2048))
        #depth_img = np.zeros((64,2048))
        #covered_points = []
        for jj in range(len(Scan.proj_x)):
            y_range, x_range = Scan.proj_y[jj], Scan.proj_x[jj]
            if (label_img[y_range, x_range] == 0):
                label_img[y_range, x_range] = semantic_label[jj]
                #label_img[y_range,x_range] = semantic_label_inv[jj]
                #depth_img[y_range,x_range] = Scan.unproj_range[jj]

        #create gt ground plane mask #label numbers for ground plane: 40,44,48,49,60,72
        gt_i = np.zeros((64, 2048))
        gt_i[label_img == 40] = 1
        gt_i[label_img == 44] = 1
        gt_i[label_img == 48] = 1
        gt_i[label_img == 49] = 1
        gt_i[label_img == 60] = 1
        gt_i[label_img == 72] = 1
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
        print(stop-start)
        #print(best_eq)
        ground_points_ransac = all_points[best_inliers]
        Scan.set_points(ground_points_ransac)
        ground_i_ransac = Scan.proj_mask
        range_unprojected = range_img_pre.reshape(-1)
        range_unprojected[best_inliers] = 0
        range_projected = range_unprojected.reshape(64, 2048)
        #ground_mask_ransac = ground_i_ransac > 0
        #range_img_pre[ground_mask_ransac] = 0

        test_img = np.vstack((orig_range_img, range_projected))

        normed = cv2.normalize(test_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color = cv2.applyColorMap(normed, cv2.COLORMAP_HOT)
        '''cv2.imshow("test", color)
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
        #print('clustering:', stop-start)

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

        normed2 = cv2.normalize(instance_label, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color2 = cv2.applyColorMap(normed2, cv2.COLORMAP_JET)

        test_img2 = np.vstack((color, color2))

        cv2.imshow("test", test_img2)
        cv2.waitKey(1)
    
    print('mean pixel accuracy:', np.mean(pixel_accuracies))
    print('mIoU:', np.mean(IoUs))

def test2():
    Scan = LaserScan(project=True, flip_sign=False, H=args.range_y, W=args.range_x, fov_up=3.0, fov_down=-25.0)
    #Label = SemLaserScan(nclasses=20, sem_color_dict=None, project=False, flip_sign=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0)

    #load data
    #lidar_data = sorted(glob.glob('scripts/data3/*.bin'), key=key_func)
    lidar_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/velodyne/*.bin'.format(sequence_in)), key=key_func)
    label_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/labels/*.label'.format(sequence_in)), key=key_func)

    pixel_accuracies = []
    IoUs = []

    plane1 = pyrsc.Plane()

    for i in range(len(lidar_data)):
        Scan.open_scan(lidar_data[i])

        #organize pc
        range_img_pre = Scan.proj_range
        orig_range_img = np.copy(range_img_pre)
        test_range_img = np.copy(range_img_pre)
        xyz = Scan.proj_xyz #runtime 0.001 ms

        #add cylindrical coordinates
        xyz_cyl = appendCylindrical_np(xyz) #runtime 9 ms
        #add label channel
        xyz_cyl = np.append(xyz_cyl, np.zeros((64, 2048, 1)), axis=2) #runtime 3 ms

        semantic_label = np.fromfile(label_data[i], dtype=np.uint32)
        semantic_label = semantic_label.reshape((-1))
        semantic_label = semantic_label & 0xFFFF
        #semantic_label_inv = [inv_label_dict_reverse[mm] for mm in semantic_label]
        label_img = np.zeros((64,2048))
        #depth_img = np.zeros((64,2048))
        #covered_points = []
        for jj in range(len(Scan.proj_x)):
            y_range, x_range = Scan.proj_y[jj], Scan.proj_x[jj]
            if (label_img[y_range, x_range] == 0):
                label_img[y_range, x_range] = semantic_label[jj]
                #label_img[y_range,x_range] = semantic_label_inv[jj]
                #depth_img[y_range,x_range] = Scan.unproj_range[jj]

        #create gt ground plane mask #label numbers for ground plane: 40,44,48,49,60,72
        gt_i = np.zeros((64, 2048))
        gt_i[label_img == 40] = 1
        gt_i[label_img == 44] = 1
        gt_i[label_img == 48] = 1
        gt_i[label_img == 49] = 1
        gt_i[label_img == 60] = 1
        gt_i[label_img == 72] = 1
        gt_mask = gt_i > 0

        packet_w = 64
        range_packet = range_img_pre[0:64, 0:packet_w]
        range_processed = np.zeros((64,0))
        xyz_packet = xyz[0:64, 0:packet_w]
        xyz_processed = np.zeros((64,0,3))
        xyz_cyl_packet = xyz_cyl[0:64, 0:packet_w]
        xyz_cyl_processed = np.zeros((64,0,7))
        for p in range(0, int(range_img_pre.shape[1]/packet_w)):
            # do stuff to packet ...
            #remove ground Cython
            start = time.time()
            ground_i = custom_functions.groundRemoval(xyz_cyl_packet)[:,:,6] #runtime 3 ms
            stop = time.time()
            #print('ground removal Cython:', stop-start)
            ground_mask = ground_i > 0
            #range_packet[ground_mask] = 0

            #fit plane with RANSAC
            ground_points = xyz_packet[ground_mask]
            #all_points = xyz_packet[range_packet > 0.0001]
            all_points = xyz_packet.reshape(-1, 3)
            start = time.time()
            #print(ground_points.shape)
            #print(xyz_list.shape)
            best_eq, best_inliers = plane1.fit(pts=ground_points, all_pts=all_points, thresh=0.2, minPoints=100, maxIteration=1)
            #print(best_inliers.shape)
            stop = time.time()
            print(stop-start)
            #print(best_eq)
            range_unprojected = range_packet.reshape(-1)
            range_unprojected[best_inliers] = 0
            range_packet = range_unprojected.reshape(64, packet_w)

            # add processed packet to processed array
            range_processed = np.hstack((range_processed, range_packet))
            xyz_processed = np.hstack((xyz_processed, xyz_packet))
            xyz_cyl_processed = np.hstack((xyz_cyl_processed, xyz_cyl_packet))
            
            # update packet
            range_packet = np.hstack((range_processed[0:64, range_processed.shape[1]-1:range_processed.shape[1]], range_img_pre[0:64, p*packet_w+1:p*packet_w+packet_w]))
            xyz_packet = np.hstack((xyz_processed[0:64, xyz_processed.shape[1]-1:xyz_processed.shape[1]], xyz[0:64, p*packet_w+1:p*packet_w+packet_w]))
            xyz_cyl_packet = np.hstack((xyz_cyl_processed[0:64, xyz_cyl_processed.shape[1]-1:xyz_cyl_processed.shape[1]], xyz_cyl[0:64, p*packet_w+1:p*packet_w+packet_w]))

            normed_packet = cv2.normalize(range_processed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color_packet = cv2.applyColorMap(normed_packet, cv2.COLORMAP_HOT)
            cv2.imshow("test_packet", color_packet)
            cv2.waitKey(100)

        test_img = np.vstack((orig_range_img, range_img_pre))

        normed = cv2.normalize(test_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color = cv2.applyColorMap(normed, cv2.COLORMAP_HOT)
        '''cv2.imshow("test", color)
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
        #print('clustering:', stop-start)

        #instance_label=np.asarray(instance_label).reshape(64,20)
        instance_label=np.asarray(instance_label).reshape(64,2048)

        normed2 = cv2.normalize(instance_label, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color2 = cv2.applyColorMap(normed2, cv2.COLORMAP_JET)

        test_img2 = np.vstack((color, color2))

        #cv2.imshow("test", test_img2)
        #cv2.waitKey(1)

def main():
    #cluster_classifier()
    #test()
    test2()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()