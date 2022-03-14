import numpy as np
import time

class Merge:

    def __init__(self):
        self.connections = []

    def mergeLabels(self, cluster_prediction_processed, new_boundary, packet_w):
        old_boundary = cluster_prediction_processed[0:64, cluster_prediction_processed.shape[1]-packet_w-2:cluster_prediction_processed.shape[1]-packet_w]

        '''if old_boundary.shape[1] < 3:
            old_boundary = np.zeros((64,3))'''

        start = 0
        stop = 0
        #start = time.time()
        for j in range(old_boundary.shape[0]):
            for i in range(old_boundary.shape[1]):
                if old_boundary[j, i] != 0 and new_boundary[j, i] != 0:
                    start = time.time()
                    smaller_label = np.min([old_boundary[j, i], new_boundary[j, i]])
                    bigger_label = np.max([old_boundary[j, i], new_boundary[j, i]])
                    # change labels of max to mins
                    cluster_prediction_processed[cluster_prediction_processed == bigger_label] = smaller_label
                    new_boundary[new_boundary == bigger_label] = smaller_label
                    stop = time.time()
        #print(stop-start)

        '''smaller_labels = np.fmin(new_boundary, old_boundary)
        bigger_labels = np.fmax(new_boundary, old_boundary)'''

        '''cluster_prediction_processed = np.where(cluster_prediction_processed == bigger_labels, cluster_prediction_processed, smaller_labels)
        new_boundary = np.where(new_boundary == bigger_labels, new_boundary, smaller_labels)'''

        return cluster_prediction_processed
