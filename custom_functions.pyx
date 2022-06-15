import numpy as np
cimport numpy as np
import time
from cpython cimport array
import array

def groundRemoval(np.ndarray[np.float64_t, ndim=3] packet): # remove ground with convolutional sobel-like filter
    cdef Py_ssize_t i,j,k
    cdef float bias = 0
    cdef float dhv
    cdef float dhh
    cdef float ground_angle

    for i in range(packet.shape[0]-1):
        for j in range(2,packet.shape[1]-2):
            if packet[i,j,5] > -10 and packet[i,j,5] < -1.3: # first condition
                dhh = (2*packet[i,j,5] + packet[i,j-1,5]) - (2*packet[i,j+1,5] + packet[i,j+2,5]) # 1x4

                if dhh > -1 and dhh < 1: # second condition
                    ddv = (2*packet[i,j,3] + packet[i,j+1,3]) - (2*packet[i+1,j,3] + packet[i+1,j+1,3])
                    if ddv != 0:
                        ground_angle = ((2*packet[i,j,5] + packet[i,j+1,5]) - (2*packet[i+1,j,5] + packet[i+1,j+1,5]))/ddv
                    else:
                        ground_angle = 0
                    bias = 0.0
                    if ground_angle > -0.2+bias and ground_angle < 0.2+bias: # third condition
                        for k in range(6):
                            packet[i, j, k] = 0
                        packet[i,j,6] = 1 # label ground
    return packet

def mergeLabels(np.ndarray[np.double_t, ndim=2] cluster_prediction_processed, np.ndarray[np.long_t, ndim=2] new_boundary, int packet_w):
    old_boundary = cluster_prediction_processed[0:64, cluster_prediction_processed.shape[1]-packet_w-2:cluster_prediction_processed.shape[1]-packet_w]

    cdef Py_ssize_t i,j, ii, jj
    cdef int smaller_label = 0
    cdef int bigger_label = 0

    for j in range(old_boundary.shape[0]):
        for i in range(old_boundary.shape[1]):
            if old_boundary[j, i] != 0 and new_boundary[j, i] != 0:
                if old_boundary[j, i] < new_boundary[j, i]:
                    smaller_label = old_boundary[j, i]
                    bigger_label = new_boundary[j, i]
                else:
                    smaller_label = new_boundary[j, i]
                    bigger_label = old_boundary[j, i]
                
                # change labels of max to mins
                cluster_prediction_processed[cluster_prediction_processed == bigger_label] = smaller_label
                new_boundary[new_boundary == bigger_label] = smaller_label
                
    return cluster_prediction_processed