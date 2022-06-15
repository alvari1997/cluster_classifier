#!/usr/bin/env python3
import numpy as np
import cv2
import time
import glob, os
from scripts import custom_functions
from scripts import plane_fit
from scripts.laserscan import LaserScan
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sequence',dest= "sequence_in", default='00', help='')
parser.add_argument('--dataset',dest= "dataset", default='semanticKITTI', help='')
parser.add_argument('--root',  dest= "root", default='./Dataset/semanticKITTI/',help="./Dataset/semanticKITTI/")
parser.add_argument('--range_y', dest= "range_y", default=64, help="64")
parser.add_argument('--range_x', dest= "range_x", default=2048, help="2048")
args = parser.parse_args()
sequence_in = args.sequence_in

# ground segmentation setup
plane = plane_fit.Plane()

def key_func(x):
        return os.path.split(x)[-1]

def appendCylindrical_np(xyz):
    ptsnew = np.dstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,:,0]**2 + xyz[:,:,1]**2
    ptsnew[:,:,3] = np.sqrt(xy) #radial
    ptsnew[:,:,4] = np.arctan2(xyz[:,:,1], xyz[:,:,0]) #azimuth, horizontal
    ptsnew[:,:,5] = xyz[:,:,2] #z, height
    return ptsnew

def ground_seg_benchmark():
    Scan = LaserScan(project=True, flip_sign=False, H=args.range_y, W=args.range_x, fov_up=3.0, fov_down=-25.0)

    # load data
    lidar_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/velodyne/*.bin'.format(sequence_in)), key=key_func)
    label_data = sorted(glob.glob('/home/alvari/Desktop/semanticKITTI/dataset/sequences/{0}/labels/*.label'.format(sequence_in)), key=key_func)

    pixel_accuracies = []
    IoUs = []
    recalls = []
    F1scores = []
    precisions = []

    plane1 = plane_fit.Plane()

    for i in range(len(lidar_data)):

        # open lidar data
        Scan.open_scan(lidar_data[i])

        # organize pc
        range_img_pre = Scan.proj_range
        xyz = Scan.proj_xyz

        # add cylindrical coordinates
        xyz_cyl = appendCylindrical_np(xyz) #runtime 9 ms

        # add label channel
        xyz_cyl = np.append(xyz_cyl, np.zeros((64, 2048, 1)), axis=2) 

        # open label data
        panoptic_label = np.fromfile(label_data[i], dtype=np.uint32)
        panoptic_label = panoptic_label.reshape((-1))
        semantic_label = panoptic_label & 0xFFFF
        semantic_label_img = np.zeros((64,2048))
        for jj in range(len(Scan.proj_x)):
            y_range, x_range = Scan.proj_y[jj], Scan.proj_x[jj]
            if (semantic_label_img[y_range, x_range] == 0):
                semantic_label_img[y_range, x_range] = semantic_label[jj]

        # create gt ground plane mask, label numbers for ground plane: 40,44,48,49,60,72
        gt_i = np.zeros((64, 2048))
        gt_i[semantic_label_img == 40] = 1
        gt_i[semantic_label_img == 44] = 1
        gt_i[semantic_label_img == 48] = 1
        gt_i[semantic_label_img == 49] = 1
        gt_i[semantic_label_img == 60] = 1
        gt_i[semantic_label_img == 72] = 1
        
        # create sectors
        #packet_w = 64
        packet_w = 128
        range_packet = range_img_pre[0:64, 0:packet_w]
        range_processed = np.zeros((64,0))
        xyz_packet = xyz[0:64, 0:packet_w]
        xyz_processed = np.zeros((64,0,3))
        xyz_cyl_packet = xyz_cyl[0:64, 0:packet_w]
        xyz_cyl_processed = np.zeros((64,0,7))
        ground_prediction_packet = np.zeros((64, packet_w))
        ground_prediction_processed = np.zeros((64,0))
        ground_time = 0

        # process scan in sectors
        for p in range(1, int(range_img_pre.shape[1]/packet_w+1)):

            # sample ground points with sobel inspired filter
            start = time.time()
            ground_i = custom_functions.groundRemoval(xyz_cyl_packet)[:,:,6] 
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

            # add processed packet to processed array
            range_processed = np.hstack((range_processed, range_packet))
            xyz_processed = np.hstack((xyz_processed, xyz_packet))
            xyz_cyl_processed = np.hstack((xyz_cyl_processed, xyz_cyl_packet))
            ground_prediction_processed = np.hstack((ground_prediction_processed, ground_prediction_packet))
            
            # update packet
            range_packet = range_img_pre[0:64, p*packet_w:p*packet_w+packet_w]
            xyz_packet = xyz[0:64, p*packet_w:p*packet_w+packet_w]
            xyz_cyl_packet = xyz_cyl[0:64, p*packet_w:p*packet_w+packet_w]
            ground_prediction_packet = np.zeros((64, packet_w))

        # visualize
        '''normed = cv2.normalize(range_processed, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color = cv2.applyColorMap(normed, cv2.COLORMAP_HOT)
        scale_percent = 100 # percent of original size
        width = int(color.shape[1] * scale_percent / 100)
        height = int(color.shape[0] * scale_percent / 100)
        dim = (width, height)
        color = cv2.resize(color, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow("Range projection with predicted ground removed", color)
        cv2.imshow("Predicted ground mask", ground_prediction_processed)
        cv2.waitKey(1000)'''

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
        print('IOU:', np.mean(IoUs))
        print('Precision:', np.mean(precisions))
        print('Recall:', np.mean(recalls))
        print('Pixel accuracy:', np.mean(pixel_accuracies))
        print('F1-score:', np.mean(F1scores))
        print("Time: ", ground_time)

def main():
    ground_seg_benchmark()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()