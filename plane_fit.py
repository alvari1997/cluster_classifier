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

        if len(pts) < 3:
            return [0,0,0,0], [0]

        for it in range(maxIteration):
            for s in range(0, 1):
                # samples 3 random points 
                id_samples = random.sample(range(0, n_points), 3)
                pt_samples = pts[id_samples]

                vecA = pt_samples[1, :] - pt_samples[0, :]
                vecB = pt_samples[2, :] - pt_samples[0, :]
                vecC = np.cross(vecA, vecB)

                vecC = vecC / np.linalg.norm(vecC)
                k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
                plane_eq = [vecC[0], vecC[1], vecC[2], k]

            pt_id_inliers = []  # list of inliers ids
            dist_pt = (
                plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
            ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            # select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if len(pt_id_inliers) > len(best_inliers):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
            
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
