#from cython.view cimport array as cvarray
import numpy as np
cimport numpy as np
#import cv2
#from collections import deque

def groundRemoval(np.ndarray[np.float64_t, ndim=3] packet): #remove ground with sobel
    cdef Py_ssize_t i,j,k
    cdef int pass1 = 0
    cdef int pass2 = 0
    cdef int pass3 = 0
    cdef int pass4 = 0
    cdef float threshold = 0.1
    cdef float bias = 0
    cdef float dhv
    cdef float dhh
    cdef float ground_angle
    for i in range(packet.shape[0]-1):
        for j in range(2,packet.shape[1]-2):
            if packet[i,j,5] > -10 and packet[i,j,5] < -1.3: #first condition
                #packet[i,j,6] = 1 #label ground
                dhh = (2*packet[i,j,3] + packet[i,j-1,3]) - (2*packet[i,j+1,3] + packet[i,j+2,3]) #1x4
                #dhh = (2*packet[i,j,5] + packet[i+1,j,5]) - (2*packet[i,j+1,5] + packet[i+1,j+1,5]) #2x2

                '''temp_array = np.array([packet[i,j-1,3], packet[i,j,3], packet[i,j+1,3], packet[i,j+2,3]])
                temp_array = (temp_array > 0.01) * temp_array
                temp_min = np.min(temp_array[np.nonzero(temp_array)])
                temp_max = np.max(temp_array[np.nonzero(temp_array)])
                if abs(temp_max - temp_min) < threshold:'''


                '''if packet[i,j,3] > 0.001 and packet[i,j+1,3] > 0.001:
                    if abs(packet[i,j,3] - packet[i,j+1,3]) < threshold:
                        pass1 = 1
                    else:
                        pass1 = 0
                else:
                    pass1 = 1

                if packet[i,j,3] > 0.001 and packet[i,j-1,3] > 0.001:
                    if abs(packet[i,j,3] - packet[i,j-1,3]) < threshold:
                        pass2 = 1
                    else:
                        pass2 = 0
                else:
                    pass2 = 1

                if packet[i,j+2,3] > 0.001 and packet[i,j+1,3] > 0.001:
                    if abs(packet[i,j+2,3] - packet[i,j+1,3]) < threshold:
                        pass3 = 1
                    else:
                        pass3 = 0
                else:
                    pass3 = 1

                if packet[i,j-2,3] > 0.001 and packet[i,j-1,3] > 0.001:
                    if abs(packet[i,j-2,3] - packet[i,j-1,3]) < threshold:
                        pass4 = 1
                    else:
                        pass4 = 0
                else:
                    pass4 = 1

                if pass1 == 1 and pass2 == 1 and pass3 == 1 and pass4 == 1:'''
                if dhh > -10 and dhh < 10: #second condition, 10
                    #packet[i,j,6] = 1 #label ground

                    ddv = (2*packet[i,j,3] + packet[i,j+1,3]) - (2*packet[i+1,j,3] + packet[i+1,j+1,3])
                    if ddv != 0:
                        ground_angle = ((2*packet[i,j,5] + packet[i,j+1,5]) - (2*packet[i+1,j,5] + packet[i+1,j+1,5]))/ddv
                    else:
                        ground_angle = 0
                    bias = 0.0
                    if ground_angle > -0.1+bias and ground_angle < 0.1+bias: #third condition, 0.5
                        for k in range(6):
                            packet[i, j, k] = 0
                        packet[i,j,6] = 1 #label ground'''
    return packet

def clustering(np.ndarray[np.float64_t, ndim=3] packet):
    cdef Py_ssize_t i,j
    cdef int l = 1
    for i in range(packet.shape[0]):
        for j in range(packet.shape[1]):
            if packet[i,j,6] == 0: #unlabeled
                #print(i,j)
                bfs(packet,i,j,l)
                l = l + 1

    #bfs(packet,1,1,l)

def bfs(np.ndarray[np.float64_t, ndim=3] packet, int r, int c, int l):
    cdef Py_ssize_t i,g
    cdef list r_queue = []
    cdef list c_queue = []
    cdef int rr
    cdef int cc
    cdef list dr = [-1,1,0,0]
    cdef list dc = [0,0,1,-1]

    cdef int ri 
    cdef int ci 

    #cdef list queue = []
    #queue.append(r) # =push, put item on the right
    #queue.append(r) # =push, put item on the right
    #queue.append(r) # =push, put item on the right
    #cdef int gg = queue[-1] # top=, get the element on the right
    #print(gg)
    #queue.pop() # =pop, remove item on the right

    #push starting point on the queue
    r_queue.append(r)
    c_queue.append(c)
    while len(r_queue) > 0:
        #print(len(r_queue))
        ri = r_queue[-1]
        ci = c_queue[-1]
        #packet[ri,ci,6] = l #label a point
        for i in range(4): #get neighbours
            rr = ri + dr[i]
            cc = ci + dc[i]
            if rr < 0 or rr >= packet.shape[0]: continue
            if cc < 0 or cc >= packet.shape[1]: continue
            if packet[rr, cc, 6] != 0: continue #already labeled
            #if packet[rr, cc, 3] == 0: continue 

            if abs(packet[ri,ci,3] - packet[rr,cc,3]) > 0.1 and abs(packet[ri,ci,3] - packet[rr,cc,3]) < 0.6:
                r_queue.append(rr)
                c_queue.append(cc)
                packet[rr, cc, 6] = l 
        
        r_queue.pop()
        c_queue.pop()
        #print(len(r_queue))

