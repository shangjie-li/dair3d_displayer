import os
import argparse
import matplotlib.pyplot as plt  # for WARNING: QApplication was not created in the main() thread.

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from dair_dataset import DAIRDataset
import opencv_vis_utils
import open3d_vis_utils


CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='DAIR3D Displayer')
    parser.add_argument('--image_dir', type=str, default='dair_i/training/image',
                        help='Root directory path to images.')
    parser.add_argument('--velodyne_dir', type=str, default='dair_i/training/velodyne',
                        help='Root directory path to velodyne point clouds.')
    parser.add_argument('--calib_dir', type=str, default='dair_i/training/calib',
                        help='Root directory path to calibration files.')
    parser.add_argument('--label_dir', type=str, default='dair_i/training/label',
                        help='Root directory path to label files. (Use None for the testing set.)')
    parser.add_argument('--split_file', type=str, default='dair_i/ImageSets/val.txt',
                        help='Path to the split file. (E.g. train.txt, val.txt, trainval.txt, test.txt.)')
    parser.add_argument('--show_gt_boxes', action='store_true', default=False,
                        help='Whether to show ground truth boxes.')
    parser.add_argument('--onto_image', action='store_true', default=False,
                        help='Whether to project results onto the image.')
    parser.add_argument('--frame_id',  type=str, default=None,
                        help='Frame ID for displaying. (E.g. 000008.)')

    global args
    args = parser.parse_args(argv)


def show_single_frame(data_dict, args, frame_id='dair3d'):
    if args.show_gt_boxes and data_dict.get('gt_boxes', None) is not None:
        boxes = data_dict['gt_boxes'][:, :7]
        names = data_dict['gt_names']
    else:
        boxes = None
        names = None

    if args.onto_image:
        img = data_dict['image'][:, :, ::-1]  # to BGR
        img = opencv_vis_utils.normalize_img(img)
        calib = data_dict['calib']
        opencv_vis_utils.draw_scene(
            img=img,
            calib=calib,
            boxes3d=boxes,
            names=names,
            window_name=frame_id,
        )

    else:
        points = data_dict['points_in_fov'][:, :3]
        point_colors = None
        open3d_vis_utils.draw_scene(
            points=points,
            boxes3d=boxes,
            names=names,
            point_colors=point_colors,
            window_name=frame_id,
        )


def main():
    parse_args()
    assert os.path.isfile(args.split_file), '%s is not a file.' % args.split_file
    assert os.path.isdir(args.image_dir), '%s is not a directory.' % args.image_dir
    assert os.path.isdir(args.velodyne_dir), '%s is not a directory.' % args.velodyne_dir
    assert os.path.isdir(args.calib_dir), '%s is not a directory.' % args.calib_dir
    if args.label_dir is not None:
        assert os.path.isdir(args.label_dir), '%s is not a directory.' % args.label_dir

    demo_dataset = DAIRDataset(
        CLASS_NAMES, args.split_file,
        image_dir=args.image_dir, velodyne_dir=args.velodyne_dir, calib_dir=args.calib_dir, label_dir=args.label_dir,
    )

    if args.frame_id is not None:
        assert args.frame_id in demo_dataset.frame_id_list, 'Invalid frame id: %s' % args.frame_id
        idx = demo_dataset.frame_id_list.index(args.frame_id)
        data_dict = demo_dataset[idx]
        frame_id = data_dict['frame_id']
        show_single_frame(data_dict, args, frame_id=frame_id)
    else:
        for data_dict in demo_dataset:
            frame_id = data_dict['frame_id']
            show_single_frame(data_dict, args, frame_id=frame_id)


if __name__ == '__main__':
    main()
