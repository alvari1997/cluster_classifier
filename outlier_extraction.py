#!/usr/bin/env python3

import numpy as np
import time
import glob, os
from scripts.laserscan import SemLaserScan, LaserScan
import argparse
from depth_cluster.build import Depth_Cluster
import cv2
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument('--sequence',dest= "sequence_in", default='00', help='')
parser.add_argument('--dataset',dest= "dataset", default='semanticKITTI', help='')
parser.add_argument('--root',  dest= "root", default='./Dataset/semanticKITTI/',help="./Dataset/semanticKITTI/")
parser.add_argument('--range_y', dest= "range_y", default=64, help="64")
parser.add_argument('--range_x', dest= "range_x", default=2048, help="2048")
parser.add_argument('--minimum_points', dest= "minimum_points", default=40, help="minimum_points of each class")
parser.add_argument('--which_cluster', dest= "which_cluster", default=1, help="4: ScanLineRun clustering; 3: superVoxel clustering; 2: euclidean; 1: depth_cluster; ")
args = parser.parse_args()

sequence_in = args.sequence_in

if args.which_cluster == 1:
	cluster = Depth_Cluster.Depth_Cluster(0.15,9) #angle threshold 0.15 (smaller th less clusters), search steps 9

def key_func(x):
        return os.path.split(x)[-1]

def full_scan():
    Scan = LaserScan(project=True, flip_sign=False, H=args.range_y, W=args.range_x, fov_up=3.0, fov_down=-25.0)

    # load data
    lidar_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/velodyne/*.bin'.format(sequence_in)), key=key_func)
    label_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/labels/*.label'.format(sequence_in)), key=key_func)

    for i in range(len(lidar_data)):
        Scan.open_scan(lidar_data[i])

        xyz_list = Scan.points

        # organize pc
        range_img_pre = Scan.proj_range
        xyz = Scan.proj_xyz

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

        # remove ground plane
        gt_i[semantic_label_img == 40] = 1
        gt_i[semantic_label_img == 44] = 1
        gt_i[semantic_label_img == 48] = 1
        gt_i[semantic_label_img == 49] = 1
        gt_i[semantic_label_img == 60] = 1
        gt_i[semantic_label_img == 72] = 1

        # remove inliers
        gt_i[semantic_label_img == 10] = 1
        gt_i[semantic_label_img == 13] = 1
        gt_i[semantic_label_img == 15] = 1
        gt_i[semantic_label_img == 32] = 1
        gt_i[semantic_label_img == 18] = 1
        gt_i[semantic_label_img == 30] = 1

        gt_i[semantic_label_img == 11] = 1
        gt_i[semantic_label_img == 31] = 1
        gt_i[semantic_label_img == 252] = 1
        gt_i[semantic_label_img == 257] = 1
        gt_i[semantic_label_img == 255] = 1
        gt_i[semantic_label_img == 258] = 1
        gt_i[semantic_label_img == 254] = 1
        gt_i[semantic_label_img == 253] = 1
        
        gt_mask = gt_i > 0

        # remove ground points
        range_img_pre[gt_i == 1] = 0

        copied = np.copy(range_img_pre)
        range_img = copied.reshape(-1)

        # clustering
        start = time.time()
        instance_label = cluster.Depth_cluster(range_img)
        stop = time.time()
        #print(stop-start)
        instance_label=np.asarray(instance_label).reshape(64,2048)

        #print(np.max(instance_label))

        # save clusters
        #clu = 0
        cluster_idx = set(instance_label[instance_label != 0])
        for c in cluster_idx:
            clusterr = xyz[instance_label == c].reshape(-1, 3)
            if clusterr.shape[0] < 20:
                continue
            #print(clusterr)
            #print(clusterr.shape[0])
            #clu += 1

            # normalize
            points_canonical = points_to_center(clusterr)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_canonical)
            o3d.io.write_point_cloud('outlier_cluster_dataset/{}_seq_{}_scan_{}_cluster.pts'.format(sequence_in, i, c), pcd)
            #o3d.io.write_point_cloud('outlier_vis/{}_scan_{}_cluster.ply'.format(i, c), pcd)

            # vizualise
            #pcd_load = o3d.io.read_point_cloud('outlier_vis/{}_scan_{}_cluster.ply'.format(i, c))
            #o3d.visualization.draw_geometries([pcd_load])

        #print(clu)
        #print(np.max(instance_label))

        '''normed_packet = cv2.normalize(instance_label, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color_packet = cv2.applyColorMap(normed_packet, cv2.COLORMAP_HOT)
        cv2.imshow("test", color_packet)
        cv2.waitKey(100)'''

def points_to_center(points):
    max_p = np.max(points, axis=0)
    min_p = np.min(points, axis=0)
    center = np.mean([max_p, min_p], axis=0).reshape(1, 3)
    points_centered = (points - center).reshape(1, -1, 3)
    points_canonical = points_centered
    return points_canonical.squeeze()

def main():
    full_scan()
    
if __name__ == '__main__':
    main()