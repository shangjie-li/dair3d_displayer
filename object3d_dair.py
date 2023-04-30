import numpy as np
import json


def get_objects_from_label(label_file):
    with open(label_file, 'r') as fp:
        objs = json.load(fp)
    objects = [Object3d(obj) for obj in objs]
    return objects


class Object3d(object):
    def __init__(self, obj):
        t = obj['type']  # Car, Truck, Van, Bus, Pedestrian, Cyclist, Tricyclist, Motorcyclist, Barrowlist, Trafficcone
        self.cls_type = t

        self.truncation = float(obj['truncated_state'])  # 0: none, 1: horizontal, 2: vertical
        self.occlusion = float(obj['occluded_state'])  # 0: none, 1: 0%-50%, 2: 50%-100%
        self.alpha = float(obj['alpha'])

        t = obj['2d_box']
        self.box2d = np.array((float(t['xmin']), float(t['ymin']), float(t['xmax']), float(t['ymax'])), dtype=np.float32)

        t = obj['3d_dimensions']
        self.h = float(t['h'])
        self.w = float(t['w'])
        self.l = float(t['l'])

        t = obj['3d_location']
        self.loc = np.array((float(t['x']), float(t['y']), float(t['z'])), dtype=np.float32)

        self.rotation = float(obj['rotation'])
