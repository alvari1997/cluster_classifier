""" Visualization code for point clouds and 3D bounding boxes with mayavi.
Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py
"""

import numpy as np
import mayavi.mlab as mlab
import glob, os
from laserscan import LaserScan
import kitti_util
import time
import argparse

'''parser = argparse.ArgumentParser()
parser.add_argument('--i',dest= "scan_idx", default=0, help='')
args = parser.parse_args()

i = args.scan_idx'''

raw_input = input

def normalize(vec):
    """normalizes an Nd list of vectors or a single vector
    to unit length.
    The vector is **not** changed in place.
    For zero-length vectors, the result will be np.nan.
    :param numpy.array vec: an Nd array with the final dimension
        being vectors
        ::
            numpy.array([ x, y, z ])
        Or an NxM array::
            numpy.array([
                [x1, y1, z1],
                [x2, y2, z2]
            ]).
    :rtype: A numpy.array the normalized value
    """
    # calculate the length
    # this is a duplicate of length(vec) because we
    # always want an array, even a 0-d array.
    return (vec.T / np.sqrt(np.sum(vec ** 2, axis=-1))).T


def rotation_matrix_numpy0(axis, theta, dtype=None):
    # dtype = dtype or axis.dtype
    # make sure the vector is normalized
    if not np.isclose(np.linalg.norm(axis), 1.0):
        axis = normalize(axis)

    thetaOver2 = theta * 0.5
    sinThetaOver2 = np.sin(thetaOver2)

    return np.array(
        [
            sinThetaOver2 * axis[0],
            sinThetaOver2 * axis[1],
            sinThetaOver2 * axis[2],
            np.cos(thetaOver2),
        ]
    )


def rotation_matrix_numpy(axis, theta):

    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)

    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )


def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def draw_lidar_simple(pc, color=None):
    """ Draw lidar points. simplest set up. """
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1600, 1000)
    )
    if color is None:
        color = pc[:, 2]
    # draw points
    mlab.points3d(
        pc[:, 0],
        pc[:, 1],
        pc[:, 2],
        color,
        color=None,
        mode="point",
        colormap="gnuplot",
        scale_factor=1,
        figure=fig,
    )
    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)
    # draw axis
    axes = np.array(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )
    mlab.view(
        azimuth=180,
        elevation=70,
        focalpoint=[12.0909996, -1.04700089, -2.03249991],
        distance=62.0,
        figure=fig,
    )
    return fig


# pts_mode='sphere'
def draw_lidar(
    pc,
    color=None,
    fig=None,
    bgcolor=(0, 0, 0),
    pts_scale=0.3,
    pts_mode="sphere",
    pts_color=(1,1,1),
    color_by_intensity=False,
    pc_label=False,
):
    """ Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    """
    # ind = (pc[:,2]< -1.65)
    # pc = pc[ind]
    pts_mode = "point"
    print("====================", pc.shape)
    if fig is None:
        fig = mlab.figure(
            figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000)
        )
    if color is None:
        color = pc[:, 2]
    if pc_label:
        color = pc[:, 4]
    if color_by_intensity:
        color = pc[:, 2]

    mlab.points3d(
        pc[:, 0],
        pc[:, 1],
        pc[:, 2],
        color,
        color=pts_color,
        mode=pts_mode,
        colormap="gnuplot",
        scale_factor=pts_scale,
        figure=fig,
    )

    # draw origin
    '''mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)

    # draw axis
    axes = np.array(
        [[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    mlab.plot3d(
        [0, axes[0, 0]],
        [0, axes[0, 1]],
        [0, axes[0, 2]],
        color=(1, 0, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[1, 0]],
        [0, axes[1, 1]],
        [0, axes[1, 2]],
        color=(0, 1, 0),
        tube_radius=None,
        figure=fig,
    )
    mlab.plot3d(
        [0, axes[2, 0]],
        [0, axes[2, 1]],
        [0, axes[2, 2]],
        color=(0, 0, 1),
        tube_radius=None,
        figure=fig,
    )'''

    '''# draw fov (todo: update to real sensor spec.)
    fov = np.array(
        [[20.0, 20.0, 0.0, 0.0], [20.0, -20.0, 0.0, 0.0]], dtype=np.float64  # 45 degree
    )

    mlab.plot3d(
        [0, fov[0, 0]],
        [0, fov[0, 1]],
        [0, fov[0, 2]],
        color=(1, 1, 1),
        tube_radius=None,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [0, fov[1, 0]],
        [0, fov[1, 1]],
        [0, fov[1, 2]],
        color=(1, 1, 1),
        tube_radius=None,
        line_width=1,
        figure=fig,
    )'''

    '''# draw square region
    TOP_Y_MIN = -20
    TOP_Y_MAX = 20
    TOP_X_MIN = 0
    TOP_X_MAX = 40
    #TOP_Z_MIN = -2.0
    #TOP_Z_MAX = 0.4

    x1 = TOP_X_MIN
    x2 = TOP_X_MAX
    y1 = TOP_Y_MIN
    y2 = TOP_Y_MAX
    mlab.plot3d(
        [x1, x1],
        [y1, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x2, x2],
        [y1, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x1, x2],
        [y1, y1],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )
    mlab.plot3d(
        [x1, x2],
        [y2, y2],
        [0, 0],
        color=(0.5, 0.5, 0.5),
        tube_radius=0.1,
        line_width=1,
        figure=fig,
    )'''

    # mlab.orientation_axes()
    '''mlab.view(
        azimuth=180,
        elevation=70,
        focalpoint=[12.0909996, -1.04700089, -2.03249991],
        distance=62.0,
        figure=fig,
    )'''
    mlab.view(azimuth=180, elevation=70, focalpoint=[19.0, 0.0, -2.0], distance=32.0, figure=fig)
    return fig


def draw_gt_boxes3d(
    gt_boxes3d,
    fig,
    color=(1, 1, 1),
    line_width=1,
    draw_text=True,
    text_scale=(1, 1, 1),
    color_list=None,
    label=""
):
    """ Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    """
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text:
            mlab.text3d(
                b[4, 0],
                b[4, 1],
                b[4, 2],
                label,
                scale=text_scale,
                color=color,
                figure=fig,
            )
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )
    # mlab.show(1)
    mlab.view(azimuth=180, elevation=70, focalpoint=[19.0, 0.0, -2.0], distance=32.0, figure=fig)
    return fig


def xyzwhl2eight(xyzwhl):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            7 -------- 6
           /|         /|
          4 -------- 5 .
          | |        | |
          . 3 -------- 2
          |/         |/
          0 -------- 1
    """
    x, y, z, w, h, l = xyzwhl[:6]
    box8 = np.array(
        [
            [
                x + w / 2,
                x + w / 2,
                x - w / 2,
                x - w / 2,
                x + w / 2,
                x + w / 2,
                x - w / 2,
                x - w / 2,
            ],
            [
                y - h / 2,
                y + h / 2,
                y + h / 2,
                y - h / 2,
                y - h / 2,
                y + h / 2,
                y + h / 2,
                y - h / 2,
            ],
            [
                z - l / 2,
                z - l / 2,
                z - l / 2,
                z - l / 2,
                z + l / 2,
                z + l / 2,
                z + l / 2,
                z + l / 2,
            ],
        ]
    )
    return box8.T


def draw_xyzwhl(
    gt_boxes3d,
    fig,
    color=(1, 1, 1),
    line_width=1,
    draw_text=True,
    text_scale=(1, 1, 1),
    color_list=None,
    rot=False,
):
    """ Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    """
    num = len(gt_boxes3d)
    for n in range(num):
        print(gt_boxes3d[n])
        box6 = gt_boxes3d[n]
        b = xyzwhl2eight(box6)
        if rot:
            b = b.dot(rotz(box6[7]))
            # b = b.dot(rotx(box6[6]))
            # print(rotz(box6[6]))
            # b = b.dot( rotz(box6[6]).dot(rotz(box6[7])) )
            vec = np.array([-1, 1, 0])
            b = b.dot(rotation_matrix_numpy(vec, box6[6]))
            # b = b.dot(roty(box6[7]))

        print(b.shape, b)
        if color_list is not None:
            color = color_list[n]
        # if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0, 4):
            # http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )

            i, j = k, k + 4
            mlab.plot3d(
                [b[i, 0], b[j, 0]],
                [b[i, 1], b[j, 1]],
                [b[i, 2], b[j, 2]],
                color=color,
                tube_radius=None,
                line_width=line_width,
                figure=fig,
            )
    # mlab.show(1)
    # mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def get_lidar_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def show_lidar_with_boxes(
    pc_velo,
    objects=None,
    calib=None,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    #if "mlab" not in sys.modules:
    #    import mayavi.mlab as mlab
    #from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    if img_fov:
        pc_velo = get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        print(("FOV point num: ", pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)
    # pc_velo=pc_velo[:,0:3]

    # draw gt boxes
    if objects is not None:
        color = (0, 1, 0)
        for obj in objects:
            if obj.type == "DontCare" or obj.type == "Van" or obj.type == "Tram" or obj.type == "Truck" or obj.type == "Misc" or obj.type == "Person sitting":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = kitti_util.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)

            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color, text_scale=(0.5,0.5,0.5) ,label=obj.type)

            # Draw depth
            '''if depth is not None:
                # import pdb; pdb.set_trace()
                depth_pt3d = depth_region_pt3d(depth, obj)
                depth_UVDepth = np.zeros_like(depth_pt3d)
                depth_UVDepth[:, 0] = depth_pt3d[:, 1]
                depth_UVDepth[:, 1] = depth_pt3d[:, 0]
                depth_UVDepth[:, 2] = depth_pt3d[:, 2]
                print("depth_pt3d:", depth_UVDepth)
                dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
                print("dep_pc_velo:", dep_pc_velo)

                draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))'''

            # Draw heading arrow
            '''_, ori3d_pts_3d = kitti_util.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )'''
    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = kitti_util.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            '''_, ori3d_pts_3d = kitti_util.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )'''
    mlab.show(1)



def key_func(x):
        return os.path.split(x)[-1]

if __name__ == "__main__":
    #pc = np.loadtxt("kitti_sample_scan.txt")

    '''lidar_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/velodyne/*.bin'), key=key_func)
    calib_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/calib/*.txt'), key=key_func)
    label_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/label_2/*.txt'), key=key_func)
    pred_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/pred/*.txt'), key=key_func)'''

    '''lidar_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/testing/velodyne/*.bin'), key=key_func)
    calib_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/testing/calib/*.txt'), key=key_func)
    pred_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/testing/pred/*.txt'), key=key_func)'''

    lidar_data = sorted(glob.glob('/home/alvari/Desktop/2011_09_26/2011_09_26_drive_0009_sync/velodyne_points/data/*.bin'), key=key_func)
    calib_data = sorted(glob.glob('/home/alvari/dataset_creator/Partial_Point_Clouds_Generation/kitti/training/calib/*.txt'), key=key_func)
    pred_data = sorted(glob.glob('/home/alvari/Desktop/2011_09_26/2011_09_26_drive_0009_sync/pred/*.txt'), key=key_func)

    Scan = LaserScan(project=True, flip_sign=False, H=64, W=2048, fov_up=3.0, fov_down=-25.0)
    for i in range(len(pred_data)):
        #i = 60
        Scan.open_scan(lidar_data[i])
        xyz_list = Scan.points
        #fig = draw_lidar(xyz_list)

        #dataset = kitti_object(root_dir, split=args.split, args=args)
        #pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]
        #calib = dataset.get_calibration(data_idx)

        pc_velo = xyz_list
        calibration = kitti_util.Calibration(calib_data[i])
        labels = kitti_util.read_label(pred_data[i])
        #is_exist = os.path.exists(pred_data[i])
        #    if is_exist:
                #return utils.read_label(pred_filename)
        #pred_obj = kitti_util.read_label(pred_data[i])

        fig = show_lidar_with_boxes(pc_velo, objects=labels, calib=calibration, img_fov=True, img_width=1380, img_height=512, objects_pred=None, depth=None, cam_img=None)

        mlab.savefig("images2/pc_view{}.jpg".format(i), figure=fig)
        mlab.close(all=True)
        #raw_input()
