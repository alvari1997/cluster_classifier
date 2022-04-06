import argparse
import numpy as np
import utils
from calibration import Calibration
import glob


parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--idx', type=str, default='000936',
                    help='specify data index: {idx}.bin')
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

once = False
if args.idx == 'everything':
    lidar_data = glob.glob('kitti/training/velodyne/*.bin')
    total_scans = len(lidar_data)
else:
    total_scans = 1
    once = True

for s in range(1, total_scans):
    if once:
        id = args.idx
    else:
        id = str(s).zfill(6)
    print(int(100*s/total_scans), '%')
    points_path = 'kitti/training/velodyne/{}.bin'.format(id)
    label_path = 'kitti/training/label_2/{}.txt'.format(id)
    calib_path = 'kitti/training/calib/{}.txt'.format(id)

    calib = Calibration(calib_path)
    points = utils.load_point_clouds(points_path)
    bboxes = utils.load_3d_boxes(label_path, args.category)

    # extract class clusters if they exist
    if len(bboxes != 0):
        bboxes = calib.bbox_rect_to_lidar(bboxes)
        corners3d = utils.boxes_to_corners_3d(bboxes)
        points_flag = utils.is_within_3d_box(points, corners3d)

        #points_is_within_3d_box = []
        for i in range(len(points_flag)):
            p = points[points_flag[i]]
            if len(p) > 20:
                #points_is_within_3d_box.append(p)
                box = bboxes[i]
                #points_canonical, box_canonical = p, box
                points_canonical, box_canonical = utils.points_to_center(p, box)
                #points_canonical, box_canonical = utils.points_to_canonical(p, box)
                #points_canonical, box_canonical = utils.lidar_to_shapenet(points_canonical, box_canonical)
                pts_name = 'output2/{}/{}_instance_{}'.format(args.category, id, i)
                pts_name_pts = 'output2/{}/{}_instance_{}.pts'.format(args.category, id, i)
                #pts_name_pts = 'output/outliers/{}_{}_instance_{}.pts'.format(args.category, id, i)
                box_name = 'output2/{}/{}_bbox_{}'.format(args.category, id, i)
                #utils.write_points(points_canonical, pts_name)
                #utils.write_bboxes(box_canonical, box_name)

                # write to pts file
                #print(len(p))
                utils.write_points_pts(points_canonical, pts_name_pts)

        #points_is_within_3d_box = np.concatenate(points_is_within_3d_box, axis=0)
        #points = points_is_within_3d_box

    #utils.write_points(points, 'output/points')
    #utils.write_bboxes(bboxes, 'output/bboxes')
