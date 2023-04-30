import os
import argparse
import json
import glob
import tqdm
from skimage import io
import numpy as np


CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Labels Convertor')
    parser.add_argument('--image_dir', type=str, default='dair_i/training/image',
                        help='Root directory path to images.')
    parser.add_argument('--src_label_dir', type=str, default='dair_i/training/label/virtuallidar',
                        help='Root directory path to label files.')
    parser.add_argument('--out_label_dir', type=str, default='annotations',
                        help='Root directory path to label files.')

    global args
    args = parser.parse_args(argv)


if __name__ == '__main__':
    parse_args()
    assert os.path.isdir(args.image_dir), '%s is not a directory.' % args.image_dir
    assert os.path.isdir(args.src_label_dir), '%s is not a directory.' % args.src_label_dir
    if not os.path.exists(args.out_label_dir):
        os.makedirs(args.out_label_dir, exist_ok=True)

    src_files = glob.glob(os.path.join(args.src_label_dir, '*.json'))
    src_files.sort()
    progress_bar = tqdm.tqdm(
        total=len(src_files), dynamic_ncols=True, leave=True, desc='samples'
    )

    for f in src_files:
        with open(f, 'r') as fp:
            objs = json.load(fp)

        frame_id = os.path.basename(f).split('.')[0]
        img_file = os.path.join(args.image_dir, '%s.jpg' % frame_id)
        assert os.path.isfile(img_file), 'File not found: %s' % img_file
        img = io.imread(img_file)
        img_h, img_w = img.shape[0], img.shape[1]

        types = []
        boxes = []
        for obj in objs:
            t = obj['2d_box']
            xmin, ymin, xmax, ymax = float(t['xmin']), float(t['ymin']), float(t['xmax']), float(t['ymax'])
            box = [xmin, ymin, xmax, ymax]
            types.append(obj['type'])
            boxes.append(box)
        mask = np.array([True if t in CLASS_NAMES else False for t in types])
        types = np.asarray(types).reshape(-1)[mask]
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)[mask]

        # Convert to the YOLO format, which is [center x, center y, width, height] and all values are normalized.
        x = ((boxes[:, 0:1] + boxes[:, 2:3]) / 2) / img_w
        y = ((boxes[:, 1:2] + boxes[:, 3:4]) / 2) / img_h
        w = (boxes[:, 2:3] - boxes[:, 0:1]) / img_w
        h = (boxes[:, 3:4] - boxes[:, 1:2]) / img_h
        boxes_yolo = np.concatenate([x, y, w, h], axis=-1)

        out_file = os.path.join(args.out_label_dir, frame_id + '.txt')
        with open(out_file, 'w') as fp:
            valid_num = int(mask.sum())
            for j in range(valid_num):
                cls_id = CLASS_NAMES.index(types[j])
                box = boxes_yolo[j]
                data = cls_id, *box
                line = ('%g ' * len(data)).rstrip() % data + '\n'
                fp.write(line)

        progress_bar.update()
    progress_bar.close()
