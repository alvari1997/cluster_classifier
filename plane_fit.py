import random
import numpy as np

class Plane:

    def __init__(self):
        self.inliers = []
        self.equation = []

    def fit(self, pts, all_pts, thresh=0.15, minPoints=100, maxIteration=10):
        
        n_points = pts.shape[0]
        best_eq = []
        best_inliers = []
        prev = 0

        if len(pts) < 3:
            return [0,0,0,0], [0]

        for it in range(maxIteration):
            for s in range(0, 1):
                # Samples 3 random points 
                id_samples = random.sample(range(0, n_points), 3)
                pt_samples = pts[id_samples]

                # try: sample point so that they are away from each other
                # ...

                # We have to find the plane equation described by those 3 points
                # We find first 2 vectors that are part of this plane
                # A = pt2 - pt1
                # B = pt3 - pt1

                vecA = pt_samples[1, :] - pt_samples[0, :]
                vecB = pt_samples[2, :] - pt_samples[0, :]

                # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
                vecC = np.cross(vecA, vecB)
                
                #print(s)
                #print(np.arccos(vecC[2]/(np.linalg.norm(vecC))))

                # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
                # We have to use a point to find k
                vecC = vecC / np.linalg.norm(vecC)
                k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
                plane_eq = [vecC[0], vecC[1], vecC[2], k]

                #print(s)
                #print(abs(np.dot(vecC, [0,0,1])/(np.linalg.norm(vecC)*np.linalg.norm([0,0,1]))))

                # try: iterate always n times, pick the one that is closest to the verical vector
                # mIoU: 0.7941365970129679
                # recall: 0.879087061492947
                # mean pixel accuracy: 0.9071446661579653
                # F1-score: 0.8846646984563402
                # inference time: 1.5 ms 10 iterations

                '''if np.abs(np.dot(vecC, [0,0,1])/(np.linalg.norm(vecC)*np.linalg.norm([0,0,1]))) > prev:
                    best_plane_eq = plane_eq
                    prev = abs(np.dot(vecC, [0,0,1])/(np.linalg.norm(vecC)*np.linalg.norm([0,0,1])))'''

                # try: iterate always n times, pick plane eq that has most inliers among pts, not all points
                # mIoU: 0.8414700733816313
                # recall: 0.9430048847009304
                # mean pixel accuracy: 0.9287129081923143
                # F1-score: 0.9133554693690086
                # inference time: 1.5 ms 10 iterations

                #if abs(np.arccos(vecC[2]/(np.linalg.norm(vecC))) - np.pi) < 0.05 or abs(np.arccos(vecC[2]/(np.linalg.norm(vecC)))) < 0.05:
                    #if k > -2 and k < -1.5:
                    #print(vecC)
                    #plane_eq = [vecC[0], vecC[1], vecC[2], k]
                    #thresh = 0.15
                    #plane_eq = [0, 0, 1, 1.8]
                    #break
                #else:
                    #plane_eq = [0, 0, 1, 1.8]
                    #thresh = 1.0

            #plane_eq = best_plane_eq
            # Distance from a point to a plane
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            '''pt_id_inliers = []  # list of inliers ids
            dist_pt = (
                plane_eq[0] * all_pts[:, 0] + plane_eq[1] * all_pts[:, 1] + plane_eq[2] * all_pts[:, 2] + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)'''

            pt_id_inliers = []  # list of inliers ids
            dist_pt = (
                plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if len(pt_id_inliers) > len(best_inliers):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
            #self.inliers = best_inliers
            #self.equation = best_eq

        # finally get inliers from all points
        dist_pt = (
            best_eq[0] * all_pts[:, 0] + best_eq[1] * all_pts[:, 1] + best_eq[2] * all_pts[:, 2] + best_eq[3]
        ) / np.sqrt(best_eq[0] ** 2 + best_eq[1] ** 2 + best_eq[2] ** 2)
        pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]

        # if plane_eq is shit, just return list of points that are below certain height
        if abs(np.dot(best_eq[0:3], [0,0,1])/(np.linalg.norm(best_eq[0:3])*np.linalg.norm([0,0,1]))) < 0.995:
            pt_id_inliers = np.where(all_pts[:,2] < -1.4)[0]
        
        self.inliers = pt_id_inliers
        self.equation = best_eq

        return self.equation, self.inliers
