#!/usr/bin/env python3

import multiprocessing as mp

#from cv2 import threshold
from matplotlib.cbook import index_of
if __name__ == '__main__':
  mp.set_start_method("spawn")

from cmath import nan
import os
import sys
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from zmq import NULL
from utils import load_poses, load_calib, load_files, load_vertex, load_vertex_label, load_image
from laserscan import SemLaserScan
#import cv2
#from points_visualize import draw_lidar
from utils import range_projection
import time
import random
from sklearn import linear_model
import math
import seaborn as sns

if __name__ == '__main__':
  # load config file
  config_filename = 'config/data_preparing.yaml'
  if len(sys.argv) > 1:
    config_filename = sys.argv[1]
  
  if yaml.__version__ >= '5.1':
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
  else:
    config = yaml.load(open(config_filename))
  
  # specify parameters
  num_frames = config['num_frames']
  debug = config['debug']
  normalize = config['normalize']
  num_last_n = config['num_last_n']
  visualize = config['visualize']
  visualization_folder = config['visualization_folder']
  
  # specify the output folders
  residual_image_folder = config['residual_image_folder']
  if not os.path.exists(residual_image_folder):
    os.makedirs(residual_image_folder)
  
  img_paths5 = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/residual_images_5')
  img_paths4 = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/residual_images_4')
  img_paths3 = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/residual_images_3')
  img_paths2 = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/residual_images_2')
  img_paths1 = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/residual_images_1')
  img_paths_1 = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/residual_images_-1')
  img_paths_2 = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/residual_images_-2')
  img_paths_3 = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/residual_images_-3')
  img_paths_4 = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/residual_images_-4')
  img_paths_5 = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/residual_images_-5')

  scan = SemLaserScan(nclasses=3, sem_color_dict={0:[0, 0, 0], 1:[255, 255, 255], 2:[0, 0, 0]}, 
                            project=True,
                            H=64,
                            W=2048,
                            fov_up=3,
                            fov_down=-25,
                            flip_sign=False)

  scan_files = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/snow_velodyne')
  label_files = load_files('/home/alvari/experiments/semanticKITTI/dataset/sequences/04/snow_labels')

  def distance(point,coef):
    return abs((coef[0]*point[0])-point[1]+coef[1])/math.sqrt((coef[0]*coef[0])+1)

  for frame_idx in range(len(img_paths1)):
    #frame_idx = 10

    scan.open_scan(scan_files[frame_idx])
    scan.open_label(label_files[frame_idx])
    label_proj = scan.proj_sem_label
    range_proj = scan.proj_range[:,:,None]

    img5 = load_image(img_paths5[frame_idx])[:,:,None]
    img4 = load_image(img_paths4[frame_idx])[:,:,None]
    img3 = load_image(img_paths3[frame_idx])[:,:,None]
    img2 = load_image(img_paths2[frame_idx])[:,:,None]
    img1 = load_image(img_paths1[frame_idx])[:,:,None]
    img_1 = load_image(img_paths_1[frame_idx])[:,:,None]
    img_2 = load_image(img_paths_2[frame_idx])[:,:,None]
    img_3 = load_image(img_paths_3[frame_idx])[:,:,None]
    img_4 = load_image(img_paths_4[frame_idx])[:,:,None]
    img_5 = load_image(img_paths_5[frame_idx])[:,:,None]
    
    stack = np.concatenate((img5, img4, img3, img2, img1, range_proj, img_1, img_2, img_3, img_4, img_5), axis=2)
    #stack = np.concatenate((img5, img4, img3, img2, img1, range_proj), axis=2)
    instances = stack.reshape(stack.shape[0]*stack.shape[1],stack.shape[2])[:,:,None]
    instances[instances < 1] = 0

    #instances = np.swapaxes(instances, 0,1)

    pseudo_labels = np.zeros((range_proj.shape[0], range_proj.shape[1]))

    #print(frame_idx)

    '''valid_instances = instances[:, np.count_nonzero(instances > 0, axis=0) > 5]
    #print(valid_instances.shape)
    
    X1,_ = np.mgrid[0:instances.shape[0]:1, 0:instances.shape[1]:1]
    

    X,Y = np.mgrid[0:6:1, 0:4:1]

    #print(X.shape)

    y1 = np.asarray([[0,0,5,1],
                     [1,4,4,1],
                     [2,4,3,1],
                     [0,4,2,1],
                     [0,4,1,0],
                     [0,0,0,0]])

    #y1 = np.asarray([[0,1,2,3,0,0], 
    #                 [0,4,4,4,4,0],
    #                 [5,4,3,2,1,0],
    #                 [1,1,0,0,1,1]])

    #print(y1)
    #print(X)

    #ss = y1[np.nonzero(y1 > 0)]
    #print(ss)
    min_samples = 2
    def is_data_valid(xx, yy):
      #print(xx)
      #print(yy)
      #return [False, False, False, np.count_nonzero(yy[:,3] > 0) < min_samples]
      #print(list(np.invert(np.count_nonzero(yy > 0, axis=0) < min_samples)))
      return list(np.count_nonzero(yy > 0, axis=0) < min_samples) # True if whole sample is > 0
      if np.count_nonzero(yy[:,0] > 0) < min_samples:
        #print('not valid')
        return False
      else:
        #print('valid')
        return True

    ransac = linear_model.RANSACRegressor(min_samples=min_samples, max_trials=100, is_data_valid=is_data_valid)
    ransac.fit(X, y1)
    b = ransac.estimator_.intercept_
    a = ransac.estimator_.coef_[:,0]
    #print(a)
    #print(b)
    
    
    print(instances.shape)
    print(times.shape)'''
    

    '''instances = np.asarray([[1,-2,-2,1,-2,1], 
                            [2,0,0,2,2,2],
                            [4,4,4,0,0,4],
                            [1,2,3,0,0,0]])[:,:,None]
    instances[instances < 1] = 0
    times = np.asarray([[1,2,3,4,5,6], 
                        [1,2,3,4,5,6],
                        [1,2,3,4,5,6],
                        [1,2,3,4,5,6]])[:,:,None]'''
    
    _,times = np.mgrid[0:instances.shape[0]:1, 1:instances.shape[1]+1:1]
    times = times[:,:,None]
    
    data = np.concatenate((instances, times), axis=2)
    inlier_table = np.zeros((0, data.shape[0]))
    samples_table = np.zeros((0, data.shape[0], 2, 2))
    n_iter = 50
    thresh = 1
    #start = time.time()
    for i in range(n_iter):
      id_samples = random.sample(range(0, data.shape[1]), 2)
      #print(id_samples)
      pt_samples = data[:, id_samples, :]
      #t_samples = times[:, id_samples]
      #print(pt_samples) # line fit
      # compute each points distance to line
      p1 = pt_samples[:, 0, :][:, None, :]
      p2 = pt_samples[:, 1, :][:, None, :]
      p3 = data
      d = np.abs(np.cross(p2-p1, p1-p3)/np.linalg.norm(p2-p1))
      #print(d)
      d[:,:,None][data[:,:,0] < 1] = np.inf # distances of missing data are changed to infinity
      #print(d)
      inlier_count = np.count_nonzero(d < thresh, axis=1)
      #inlier_count[] = 0
      pt_samples_flat = pt_samples.reshape(data.shape[0], -1)
      #print(pt_samples_flat)
      non_valid_samples = np.count_nonzero(pt_samples_flat, axis=1) < 4
      #print(inlier_count)
      inlier_count[non_valid_samples] = 0
      
      #print(inlier_count.shape)
      #print(pt_samples.shape)
      inlier_table = np.concatenate((inlier_table, inlier_count[None, :]), axis=0) # inlier table: ransac_iteration x pixel_count
      samples_table = np.concatenate((samples_table, pt_samples[None, :, :, :]), axis=0)
    #stop = time.time()
    #print(stop-start)
    #print(inlier_table.shape)
    #print(samples_table.shape)
    
    most_inliers_for_each_pixel = np.max(inlier_table, axis=0)
    #print(most_inliers_for_each_pixel)
    index_of_best_fit = np.argmax(inlier_table, axis=0)
    #print(index_of_best_fit)
    #print(samples_table[index_of_best_fit, range(0,data.shape[0]), :, :])
    best_points = samples_table[index_of_best_fit, range(0,data.shape[0]), :, :]
    print(np.count_nonzero(most_inliers_for_each_pixel > 4))
    #print(best_points[0,0,0])

    plt.figure()
    plt.plot(data[40000,:,1], data[40000,:,0])
    plt.plot(best_points[40000,0,1], best_points[40000,0,0], 'x')
    plt.plot(best_points[40000,1,1], best_points[40000,1,0], 'x')
    plt.show()

    # for each pixel: take index of best match, and get corresponding sample (2 points) from samples table
    # fit line and compute center point's distance to this line
    # pixel_count x 1 array, of distances

    '''non_snow_d = []
    snow_d = []
    start = time.time()
    x = np.arange(11)
    ransac = linear_model.RANSACRegressor(max_trials=2, min_samples=2)
    for j in tqdm(range(pseudo_labels.shape[0])):
      for i in range(pseudo_labels.shape[1]):

        #print(j)

        #if label_proj[5, 1024] == 0:
        #pseudo_label = 0
        y = stack[j, i, :]
        #x = np.arange(instance.shape[0])
        #params = np.polynomial.polynomial.polyfit(x[np.nonzero(y)], y[np.nonzero(y)], 1)
        #y_nonz = y[np.nonzero(y)]
        #sample = np.random.choice(y_nonz, 2)
        #print(y_nonz)
        #print(sample)

        samples = y[np.nonzero(y > 0)]
        xs = x[np.nonzero(y > 0)]
        #print(samples)
        if len(samples) < 5: 
          #pseudo_label = 9 # unlabelled
          continue
        
        ransac.fit(xs[:,None], samples[:,None])
        #inlier_mask = ransac.inlier_mask_
        #outlier_mask = np.logical_not(inlier_mask)
        #line_y_ransac = ransac.predict(x[np.nonzero(y)][:,None])
        
        #b = ransac.estimator_.intercept_[0]
        #a = ransac.estimator_.coef_[0][0]
        #d = distance([5, y[5]], [a, b])
        #if label_proj[j, i] == 0:
        #  non_snow_d.append(d)
        #if label_proj[j, i] == 1:
        #  snow_d.append(d)

      #yl = x*params[1] + params[0]
      yr = x*a + b
      plt.figure()
      plt.plot(x, y, yr)
      plt.plot(5, y[5], 'x')
      plt.show()

    stop = time.time()
    print('time: ', stop-start)

    print(len(snow_d))
    print(len(non_snow_d))
    print(np.min(snow_d), np.max(snow_d))
    print(np.min(non_snow_d), np.max(non_snow_d))'''
    
    '''plt.figure()
    sns.distplot(np.asarray(snow_d), bins=1000, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='snow', color='deepskyblue')
    sns.distplot(np.asarray(non_snow_d), bins=1000, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 1}, label='non-snow', color='silver')
    plt.legend()
    plt.show()'''

    '''normed = cv2.normalize(scan.proj_sem_label, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
    scale_percent = 100 # percent of original size
    width = int(color.shape[1] * scale_percent / 100)
    height = int(color.shape[0] * scale_percent / 100)
    dim = (width, height)
    #color = cv2.resize(binarized, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("test", color)
    cv2.waitKey(100)'''