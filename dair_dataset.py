import os
from skimage import io
import torch
import numpy as np
import open3d

import object3d_dair
import calibration_dair


class DAIRDataset(torch.utils.data.Dataset):
    def __init__(self, class_names, split_file, image_dir, velodyne_dir, calib_dir, label_dir=None):
        self.class_names = class_names
        assert os.path.isfile(split_file), 'File not found: %s' % split_file
        self.frame_id_list = [x.strip() for x in open(split_file).readlines()]
        self.image_dir = image_dir
        self.velodyne_dir = velodyne_dir
        self.calib_dir = calib_dir
        self.label_dir = label_dir

    def get_points(self, idx):
        """
        Args:
            idx: int, sample index
        Returns:
            points: ndarray of float32, [N, 3], points of (x, y, z)
        """
        frame_id = self.frame_id_list[idx]
        pts_file = os.path.join(self.velodyne_dir, '%s.pcd' % frame_id)
        assert os.path.isfile(pts_file), 'File not found: %s' % pts_file
        return np.asarray(open3d.io.read_point_cloud(pts_file).points, dtype=np.float32).reshape(-1, 3)

    def get_image(self, idx):
        """
        Args:
            idx: int, sample index
        Returns:
            image: ndarray of uint8, [H, W, 3], RGB image
        """
        frame_id = self.frame_id_list[idx]
        img_file = os.path.join(self.image_dir, '%s.jpg' % frame_id)
        assert os.path.isfile(img_file), 'File not found: %s' % img_file
        return io.imread(img_file)

    def get_image_shape(self, idx):
        """
        Args:
            idx: int, sample index
        Returns:
            image_shape: ndarray of int, [2], H and W
        """
        img = self.get_image(idx)
        return np.array(img.shape[:2], dtype=np.int32)

    def get_calib(self, idx):
        """
        Args:
            idx: int, sample index
        Returns:
            calib: calibration_dair.Calibration
        """
        frame_id = self.frame_id_list[idx]
        intrinsic_file = os.path.join(self.calib_dir, 'camera_intrinsic', '%s.json' % frame_id)
        v2c_file = os.path.join(self.calib_dir, 'virtuallidar_to_camera', '%s.json' % frame_id)
        assert os.path.isfile(intrinsic_file), 'File not found: %s' % intrinsic_file
        assert os.path.isfile(v2c_file), 'File not found: %s' % v2c_file
        return calibration_dair.Calibration(intrinsic_file, v2c_file)

    def get_label(self, idx):
        """
        Args:
            idx: int, sample index
        Returns:
            objects: list of object3d_dair.Object3d
        """
        frame_id = self.frame_id_list[idx]
        label_file = os.path.join(self.label_dir, 'virtuallidar', '%s.json' % frame_id)
        assert os.path.isfile(label_file), 'File not found: %s' % label_file
        return object3d_dair.get_objects_from_label(label_file)

    def get_annotations(self, idx):
        """
        Args:
            idx: int, sample index
        Returns:
            annotations: dict
        """
        obj_list = self.get_label(idx)
        mask = np.array([True if obj.cls_type in self.class_names else False for obj in obj_list])
        annos = {
            'name': np.zeros(0), 'truncated': np.zeros(0),
            'occluded': np.zeros(0), 'alpha': np.zeros(0),
            'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]),
            'location': np.zeros([0, 3]), 'rotation': np.zeros(0),
            'gt_boxes_lidar': np.zeros([0, 7])
        }

        if mask.sum() > 0:
            annos['name'] = np.array([obj.cls_type for obj in obj_list])[mask]
            annos['truncated'] = np.array([obj.truncation for obj in obj_list])[mask]
            annos['occluded'] = np.array([obj.occlusion for obj in obj_list])[mask]
            annos['alpha'] = np.array([obj.alpha for obj in obj_list])[mask]
            annos['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)[mask]
            annos['dimensions'] = np.array([[obj.l, obj.w, obj.h] for obj in obj_list])[mask]
            annos['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)[mask]
            annos['rotation'] = np.array([obj.rotation for obj in obj_list])[mask]

            loc = annos['location']
            dims = annos['dimensions']
            rot = annos['rotation']
            gt_boxes_lidar = np.concatenate([loc, dims, rot[..., np.newaxis]], axis=1)
            annos['gt_boxes_lidar'] = gt_boxes_lidar  # [M, 7], (x, y, z, l, w, h, heading) in lidar coordinates

        return annos

    @staticmethod
    def get_fov_flag(points, img_shape, calib):
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        return pts_valid_flag

    def get_points_in_fov(self, idx):
        """
        Args:
            idx: int, sample index
        Returns:
            points: ndarray of float32, [N, 3], points of (x, y, z) in fov
        """
        points = self.get_points(idx)
        calib = self.get_calib(idx)
        fov_flag = self.get_fov_flag(points, self.get_image_shape(idx), calib)
        return points[fov_flag]

    def __len__(self):
        return len(self.frame_id_list)

    def __getitem__(self, idx):
        """
        Args:
            idx: int, sample index
        Returns:
            data_dict:
                frame_id: str
                points: ndarray of float32, [N, 3], points of (x, y, z)
                points_in_fov: ndarray of float32, [N, 3], points of (x, y, z) in fov
                image: ndarray of uint8, [H, W, 3], RGB image
                image_shape: ndarray of int, [2], H and W
                calib: calibration_dair.Calibration
                gt_boxes: ndarray of float, [M, 8], (x, y, z, l, w, h, heading, class_id) in lidar coordinates
                gt_names: ndarray of str, [M]
        """
        data_dict = {
            'frame_id': self.frame_id_list[idx],
            'points': self.get_points(idx),
            'points_in_fov': self.get_points_in_fov(idx),
            'image': self.get_image(idx),
            'image_shape': self.get_image_shape(idx),
            'calib': self.get_calib(idx),
        }

        if self.label_dir is not None:
            annotations = self.get_annotations(idx)
            data_dict['gt_boxes'] = annotations['gt_boxes_lidar']
            data_dict['gt_names'] = annotations['name']

        return data_dict
