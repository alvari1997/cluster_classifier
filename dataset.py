from __future__ import print_function
from cmath import inf
import dis
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement

def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            #print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))
                #print(os.path.join(self.root, category, 'points', uuid+'.pts'))
                #print(os.path.join(self.root, category, 'points_label', uuid+'.seg'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        #print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)

class LidarDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=True,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category_no_van.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        #self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        # uuid example: 000001_instance_0

        for file in filelist:
            _, category, _, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid)))
                            #os.path.join(self.root, category, 'points', uuid)))
                #print(os.path.join(self.root, category, 'points', uuid))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
                #self.datapath.append((item, fn[0], fn[0]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        #with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
        #    for line in f:
        #        ls = line.strip().split()
        #        self.seg_classes[ls[0]] = int(ls[1])
        #self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        #print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]

        # load points
        point_set = np.loadtxt(fn[1], skiprows=1).astype(np.float32)
        original_len = len(point_set)

        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        point_set = point_set[choice, :] # resample
        center = np.expand_dims(np.mean(point_set, axis = 0), 0)

        point_set = point_set - center # center

        #dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        #point_set = point_set / dist #scale

        # world position vector
        if original_len < self.npoints:
            dist_from_orig = np.linalg.norm(center) # (1,1)
        else: 
            dist_from_orig = 20.0 # (1,1)
        bins = np.array([0, 30, 50, np.inf]) # m
        nbin = np.digitize(dist_from_orig, bins, right=True) - 1
        one_hot_w_pos = np.eye(3)[nbin] # (3,1)

        # cylinder voxel position
        '''center_in_cylindrical = np.zeros((1, 3))
        center_in_cylindrical[:,0] = np.sqrt(center[:,0]**2 + center[:,1]**2) #radial
        center_in_cylindrical[:,1] = np.arctan2(center[:,1], center[:,0]) #azimuth, horizontal
        center_in_cylindrical[:,2] = center[:,2] #z, height'''

        # spherical voxel position
        center_in_spherical = np.zeros((1, 3))
        center_in_spherical[:,0] = np.sqrt(center[:,0]**2 + center[:,1]**2 + center[:,2]**2) # euclidean from origin 
        center_in_spherical[:,1] = np.arctan2(center[:,1], center[:,0]) # azimuth, horizontal angle
        center_in_spherical[:,2] = np.arctan2(np.sqrt(center[:,0]**2 + center[:,1]**2), center[:,2]) # for elevation angle defined from Z-axis down

        # define voxel grid coordinate
        r_bins = np.linspace(0, 75, 10)
        a_bins = np.linspace(-(np.pi/4+0.2), np.pi/4+0.2, 10)
        e_bins = np.linspace(1*np.pi/3, 2*np.pi/3, 10)
        voxel_coord = np.zeros((1, 3))
        voxel_coord[:,0] = np.digitize(center_in_spherical[:,0], r_bins).squeeze()
        voxel_coord[:,1] = np.digitize(center_in_spherical[:,1], a_bins).squeeze()
        voxel_coord[:,2] = np.digitize(center_in_spherical[:,2], e_bins).squeeze()
        #print(voxel_coord)

        if self.data_augmentation:
            #theta = np.random.uniform(-np.pi/6,np.pi/6)
            #theta = np.random.uniform(0,np.pi*2)theta = np.random.uniform(-np.pi/6,np.pi/6)
            #cluster_observation_angle = np.arctan(center[:,1]/center[:,0])
            cluster_observation_angle = center_in_spherical[:,1]
            theta = np.random.uniform(-(cluster_observation_angle+np.pi/4), np.pi/4-cluster_observation_angle).squeeze()
            new_observation_angle = cluster_observation_angle + theta

            # rotate
            rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
            point_set[:,[0,1]] = point_set[:,[0,1]].dot(rotation_matrix) # random rotation on z-axis

            # flip on x-axis
            if np.random.random() > 0.5: # 50% chance flipping
                point_set[:,0] *= -1
                new_observation_angle *= -1

            # translate
            #xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
            xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
            dxy = np.random.uniform(low=-0.5, high=0.5, size=[2])
            dz = np.random.uniform(low=-0.2, high=0.2, size=[1])
            dxyz = np.hstack((dxy, dz))
            #point_set = np.add(np.multiply(point_set, xyz1), xyz2).astype('float32')
            point_set = np.add(point_set, dxyz).astype('float32')

            # jitter
            #point_set += np.random.normal(0, 0.02, size=point_set.shape)

            # update voxel coordinate, only azimuth
            voxel_coord[:,1] = np.digitize(new_observation_angle, a_bins)

        point_set = torch.from_numpy(point_set)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return point_set, cls, one_hot_w_pos, dist_from_orig, np.squeeze(voxel_coord)

    def __len__(self):
        return len(self.datapath)

class BoxDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=True,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'category_for_boxnet.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        # uuid example: 000001_instance_0.pts

        for file in filelist:
            _, category, _, uuid = file.split('/')
            i_name, _ = uuid.split('.')
            if category in self.cat.values():
                #self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'bbox', i_name+'.txt'))),

                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid),
                                        os.path.join(self.root, category, 'bbox', i_name+'.txt')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                #self.datapath.append((item, fn))
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]

        # load points
        point_set = np.loadtxt(fn[1], skiprows=1).astype(np.float32)
        original_len = len(point_set)
        choice = np.random.choice(len(point_set), self.npoints, replace=True) # sample
        point_set = point_set[choice, :]
        center = np.expand_dims(np.mean(point_set, axis = 0), 0)

        # load bbox
        bbox = np.loadtxt(fn[2]).astype(np.float32)

        # world position vector
        if original_len < self.npoints:
            dist_from_orig = np.linalg.norm(center) # (1,1)
        else: 
            dist_from_orig = 20.0 # (1,1)
        bins = np.array([0, 30, 50, np.inf]) # m
        nbin = np.digitize(dist_from_orig, bins, right=True) - 1
        one_hot_w_pos = np.eye(3)[nbin] # (3,1)

        # center
        point_set = point_set - center
        bbox[:3] = bbox[:3] - center

        # spherical voxel position
        center_in_spherical = np.zeros((1, 3))
        center_in_spherical[:,0] = np.sqrt(center[:,0]**2 + center[:,1]**2 + center[:,2]**2) # euclidean from origin 
        center_in_spherical[:,1] = np.arctan2(center[:,1], center[:,0]) # azimuth, horizontal angle
        center_in_spherical[:,2] = np.arctan2(np.sqrt(center[:,0]**2 + center[:,1]**2), center[:,2]) # for elevation angle defined from Z-axis down

        # define voxel grid coordinate
        r_bins = np.linspace(0, 75, 10)
        a_bins = np.linspace(-(np.pi/4+0.2), np.pi/4+0.2, 10)
        e_bins = np.linspace(1*np.pi/3, 2*np.pi/3, 10)
        voxel_coord = np.zeros((1, 3))
        voxel_coord[:,0] = np.digitize(center_in_spherical[:,0], r_bins).squeeze()
        voxel_coord[:,1] = np.digitize(center_in_spherical[:,1], a_bins).squeeze()
        voxel_coord[:,2] = np.digitize(center_in_spherical[:,2], e_bins).squeeze()
        #print(voxel_coord)

        if self.data_augmentation:
            #theta = np.random.uniform(-np.pi/6,np.pi/6)
            #theta = np.random.uniform(0,np.pi*2)
            #rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            #point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            cluster_observation_angle = center_in_spherical[:,1]
            theta = np.random.uniform(-(cluster_observation_angle+np.pi/4), np.pi/4-cluster_observation_angle).squeeze()
            new_observation_angle = cluster_observation_angle + theta

            # rotation on z-axis
            rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
            point_set = point_set - bbox[:3]
            point_set[:,[0,1]] = point_set[:,[0,1]].dot(rotation_matrix) # random rotation on z-axis
            bbox[6] = bbox[6] + theta
            point_set = point_set + bbox[:3]

            # flip on x-axis
            if np.random.random() > 0.5: # 50% chance flipping
                point_set[:,0] *= -1
                bbox[0] *= -1
                bbox[6] = np.pi - bbox[6]
                new_observation_angle *= -1

            # translate
            #xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
            #xyz2 = np.random.uniform(low=-1.5, high=1.5, size=[3]) # 0.2
            dxy = np.random.uniform(low=-0.5, high=0.5, size=[2])
            dz = np.random.uniform(low=-0.2, high=0.2, size=[1])
            dxyz = np.hstack((dxy, dz))
            #point_set = np.add(np.multiply(point_set, xyz1), xyz2).astype('float32')
            point_set = np.add(point_set, dxyz).astype('float32')
            bbox[:3] = np.add(bbox[:3], dxyz).astype('float32')

            # jitter
            #point_set += np.random.normal(0, 0.02, size=point_set.shape)

            # update voxel coordinate, only azimuth
            voxel_coord[:,1] = np.digitize(new_observation_angle, a_bins)
        
        point_set = torch.from_numpy(point_set)
        bbox = torch.from_numpy(bbox)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return point_set, bbox, cls, one_hot_w_pos, dist_from_orig, np.squeeze(center), np.squeeze(voxel_coord)

    def __len__(self):
        return len(self.datapath)

class FilterDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=True,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        #self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        # uuid example: 000001_instance_0
        # tee json filu joka sekoittaa luokat

        for file in filelist:
            _, category, _, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid)))
                            #os.path.join(self.root, category, 'points', uuid)))
                #print(os.path.join(self.root, category, 'points', uuid))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
                #self.datapath.append((item, fn[0], fn[0]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        #with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
        #    for line in f:
        #        ls = line.strip().split()
        #        self.seg_classes[ls[0]] = int(ls[1])
        #self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        #print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]

        # load points
        point_set = np.loadtxt(fn[1], skiprows=1).astype(np.float32)

        #seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        #choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        #seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        #seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, cls

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        #print(len(d))
        ps, seg = d[0]
        #print(ps.size(), ps.type(), seg.size(),seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        #print(len(d))
        ps, cls = d[0]
        #print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'lidar':
        d = LidarDataset(root = datapath, classification = True)
        ps, cls = d[0]

    if dataset == 'filter':
        d = FilterDataset(root = datapath, classification = True)
        ps, cls = d[0]

    if dataset == 'bbox':
        d = BoxDataset(root = datapath, classification = True)
        ps, bbox, cls = d[0]

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])

