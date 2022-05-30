#!/usr/bin/env python3

#from msilib import sequence
from math import dist
from matplotlib import projections
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
import random
import open3d as o3d
import scipy.special as sci
import scripts.box_extraction_utils as label_box_utils
from scripts.box_extraction_calib import Calibration
import matplotlib.pyplot as plt
#import mayavi.mlab as mlab

parser = argparse.ArgumentParser()
parser.add_argument('--sequence',dest= "sequence_in", default='00', help='')
parser.add_argument('--dataset',dest= "dataset", default='semanticKITTI', help='')
parser.add_argument('--root',  dest= "root", default='./Dataset/semanticKITTI/',help="./Dataset/semanticKITTI/")
parser.add_argument('--range_y', dest= "range_y", default=64, help="64")
parser.add_argument('--range_x', dest= "range_x", default=2048, help="2048")
parser.add_argument('--minimum_points', dest= "minimum_points", default=40, help="minimum_points of each class")
parser.add_argument('--which_cluster', dest= "which_cluster", default=1, help="4: ScanLineRun clustering; 3: superVoxel clustering; 2: euclidean; 1: depth_cluster; ")
parser.add_argument('--mode', dest= "mode", default='val', help="val or test; ")
parser.add_argument('--category', type=str, default='car',
                    help='specify the category to be extracted,' + 
                        '{ \
                            car, \
                            van, \
                            truck, \
                            pedestrian, \
                            person_sitting, \
                            cyclist, \
                            tram \
                        }')
args = parser.parse_args()

sequence_in = args.sequence_in

img_size = 472 # 472 -> 82 deg, 512 -> 90 deg
b = 0

min_points = 10

# ground segmentation setup
plane = plane_fit.Plane()

# cluster setup
if args.which_cluster == 1:
	cluster = Depth_Cluster.Depth_Cluster(0.4, 9) #angle threshold 0.15 (smaller th less clusters), search steps 9
    # 0.3 is good

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
    #label_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/labels/*.label'.format(sequence_in)), key=key_func)
    calib_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/calib/*.txt'), key=key_func)
    label_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/label_2/*.txt'), key=key_func)


    result_dir = '/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/'
    #result_dir = '/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/testing/'

    centers = np.zeros((1, 3))

    for i in range(len(lidar_data)-1000, len(lidar_data)):
        total_time = 0

        Scan.open_scan(lidar_data[i])
        calib = Calibration(calib_data[i])
        car_bboxes = label_box_utils.load_3d_boxes(label_data[i], 'car')
        ped_bboxes = label_box_utils.load_3d_boxes(label_data[i], 'pedestrian')
        cyc_bboxes = label_box_utils.load_3d_boxes(label_data[i], 'cyclist')

        #centers.append(car_bboxes[:,:3])
        if len(car_bboxes) != 0:
            centers = np.vstack((centers, car_bboxes[:,:3]))
        if len(ped_bboxes) != 0:
            centers = np.vstack((centers, ped_bboxes[:,:3]))
        if len(cyc_bboxes) != 0:
            centers = np.vstack((centers, cyc_bboxes[:,:3]))


        print(' ')
        print(i)
        print(' ')

        # organize points
        xyz_list = Scan.points
        range_img_pre = Scan.proj_range
        xyz = Scan.proj_xyz #runtime 0.001 ms
        orig_xyz = np.copy(xyz)

        # add cylindrical coordinates
        xyz_cyl = appendCylindrical_np(xyz) #runtime 9 ms

        # add label channel
        xyz_cyl = np.append(xyz_cyl, np.zeros((64, 2048, 1)), axis=2) #runtime 3 ms

        # crop
        xyz_cyl = xyz_cyl[0:64, int(2048/2-img_size/2)+b:int(2048/2+img_size/2)+b]
        xyz = xyz[0:64, int(2048/2-img_size/2)+b:int(2048/2+img_size/2)+b]
        range_img_pre = range_img_pre[0:64, int(2048/2-img_size/2)+b:int(2048/2+img_size/2)+b]
        orig_xyz = orig_xyz[0:64, int(2048/2-img_size/2)+b:int(2048/2+img_size/2)+b]

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
        best_eq, best_inliers = plane.fit(pts=ground_points, all_pts=all_points, thresh=0.15, minPoints=100, maxIteration=100)
        stop = time.time()
        total_time += (stop-start)
        ground_points_ransac = all_points[best_inliers]
        Scan.set_points(ground_points_ransac)
        ground_i_ransac = Scan.proj_mask
        ground_i_ransac = ground_i_ransac[0:64, int(2048/2-img_size/2)+b:int(2048/2+img_size/2)+b]
        range_unprojected = range_img_pre.reshape(-1)
        range_unprojected[best_inliers] = 0
        xyz_unprojected = xyz.reshape(-1, 3)
        xyz_unprojected[best_inliers] = [0, 0, 0]

        # clustering
        range_projected = range_unprojected.reshape(64, img_size)
        start = time.time()
        instance_label = cluster.Depth_cluster(range_projected.reshape(-1))
        stop = time.time()
        total_time += (stop-start)
        instance_label = np.asarray(instance_label)#.reshape(64, img_size)
        #instance_label = instance_label[:, None]

        #print(instance_label.shape)
        #print(xyz_unprojected.shape)
        #print(range_unprojected.shape)

        # collect IDs if they exist
        if len(car_bboxes != 0):
            car_bboxes = calib.bbox_rect_to_lidar(car_bboxes)
            corners3d = label_box_utils.boxes_to_corners_3d(car_bboxes)
            points_flag = label_box_utils.is_within_3d_box(xyz_unprojected, corners3d)

            for c_i in range(len(points_flag)):
                gt_p = xyz_unprojected[points_flag[c_i]]
                #xyz_unprojected[points_flag[c_i]] = [0, 0, 0]
                #range_unprojected[points_flag[c_i]] = 0
                overlapping_clusters = instance_label[points_flag[c_i]]
                #print(np.max(overlapping_cluster_idx))
                #print(np.min(overlapping_cluster_idx))
                if len(overlapping_clusters) == 0: continue
                biggest_cluster_inside_box = np.bincount(overlapping_clusters).argmax()
                idx_biggest_cluster_inside_box = np.where(overlapping_clusters == biggest_cluster_inside_box)
                pp = gt_p[idx_biggest_cluster_inside_box]
                xyz_unprojected[points_flag[c_i]] = [0, 0, 0]
                range_unprojected[points_flag[c_i]] = 0
                instance_label[points_flag[c_i]] = 0
                if len(pp) > min_points:
                    box = car_bboxes[c_i]
                    pts_name_pts = 'test_dataset/{}/points/{}_instance_{}.pts'.format('cars', i, c_i)
                    box_name_box = 'test_dataset/{}/bbox/{}_instance_{}.txt'.format('cars', i, c_i)
                    # write to pts file
                    label_box_utils.write_points_pts(pp, pts_name_pts)
                    label_box_utils.write_bboxes_box(box, box_name_box)
                    '''pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(gt_p)
                    o3d.visualization.draw_geometries([pcd])
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pp)
                    o3d.visualization.draw_geometries([pcd])'''

        # collect IDs if they exist
        if len(ped_bboxes != 0):
            ped_bboxes = calib.bbox_rect_to_lidar(ped_bboxes)
            corners3d = label_box_utils.boxes_to_corners_3d(ped_bboxes)
            points_flag = label_box_utils.is_within_3d_box(xyz_unprojected, corners3d)

            for c_i in range(len(points_flag)):
                gt_p = xyz_unprojected[points_flag[c_i]]
                #xyz_unprojected[points_flag[c_i]] = [0, 0, 0]
                #range_unprojected[points_flag[c_i]] = 0
                overlapping_clusters = instance_label[points_flag[c_i]]
                #print(np.max(overlapping_cluster_idx))
                #print(np.min(overlapping_cluster_idx))
                if len(overlapping_clusters) == 0: continue
                biggest_cluster_inside_box = np.bincount(overlapping_clusters).argmax()
                idx_biggest_cluster_inside_box = np.where(overlapping_clusters == biggest_cluster_inside_box)
                pp = gt_p[idx_biggest_cluster_inside_box]
                xyz_unprojected[points_flag[c_i]] = [0, 0, 0]
                range_unprojected[points_flag[c_i]] = 0
                instance_label[points_flag[c_i]] = 0
                if len(pp) > min_points:
                    box = ped_bboxes[c_i]
                    pts_name_pts = 'test_dataset/{}/points/{}_instance_{}.pts'.format('pedestrians', i, c_i)
                    box_name_box = 'test_dataset/{}/bbox/{}_instance_{}.txt'.format('pedestrians', i, c_i)
                    # write to pts file
                    label_box_utils.write_points_pts(pp, pts_name_pts)
                    label_box_utils.write_bboxes_box(box, box_name_box)

        # collect IDs if they exist
        if len(cyc_bboxes != 0):
            cyc_bboxes = calib.bbox_rect_to_lidar(cyc_bboxes)
            corners3d = label_box_utils.boxes_to_corners_3d(cyc_bboxes)
            points_flag = label_box_utils.is_within_3d_box(xyz_unprojected, corners3d)

            for c_i in range(len(points_flag)):
                gt_p = xyz_unprojected[points_flag[c_i]]
                #xyz_unprojected[points_flag[c_i]] = [0, 0, 0]
                #range_unprojected[points_flag[c_i]] = 0
                overlapping_clusters = instance_label[points_flag[c_i]]
                #print(np.max(overlapping_cluster_idx))
                #print(np.min(overlapping_cluster_idx))
                if len(overlapping_clusters) == 0: continue
                biggest_cluster_inside_box = np.bincount(overlapping_clusters).argmax()
                idx_biggest_cluster_inside_box = np.where(overlapping_clusters == biggest_cluster_inside_box)
                pp = gt_p[idx_biggest_cluster_inside_box]
                xyz_unprojected[points_flag[c_i]] = [0, 0, 0]
                range_unprojected[points_flag[c_i]] = 0
                instance_label[points_flag[c_i]] = 0
                if len(pp) > min_points:
                    box = cyc_bboxes[c_i]
                    pts_name_pts = 'test_dataset/{}/points/{}_instance_{}.pts'.format('cyclists', i, c_i)
                    box_name_box = 'test_dataset/{}/bbox/{}_instance_{}.txt'.format('cyclists', i, c_i)
                    # write to pts file
                    label_box_utils.write_points_pts(pp, pts_name_pts)
                    label_box_utils.write_bboxes_box(box, box_name_box)

        '''# clustering
        range_projected = range_unprojected.reshape(64, img_size)
        start = time.time()
        instance_label = cluster.Depth_cluster(range_projected.reshape(-1))
        stop = time.time()
        total_time += (stop-start)
        instance_label = np.asarray(instance_label)#.reshape(64, img_size)'''

        # collect OODs
        cluster_idx = set(instance_label[instance_label != 0])
        for c_i in cluster_idx:
            clu = xyz_unprojected[instance_label == c_i]
            if clu.shape[0] < min_points:
                continue
            p = clu
            pts_name_pts = 'test_dataset/{}/points/{}_instance_{}.pts'.format('oods', i, c_i)
            # write to pts file
            label_box_utils.write_points_pts(p, pts_name_pts)

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

    '''fig = plt.figure()
    ax = fig.add_subplot(111, projections='3d')
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=5, c='b', lw=0, alpha=1)
    plt.show()
    
    print(np.max(centers[:, 1]))
    print(np.min(centers[:, 1]))
    print(np.mean(centers[:, 1]))
    howmany = np.count_nonzero(centers[:, 1] < np.mean(centers[:, 1])-np.std(centers[:, 1]))
    print(howmany / np.count_nonzero(centers[:, 1]))'''

def main():
    full_scan()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()