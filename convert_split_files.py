import os
import argparse
import json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Split Files Convertor')
    parser.add_argument('--split_file', type=str, default='/home/lishangjie/DAIR-V2X/data/split_datas/single-infrastructure-split-data.json',
                        help='Path to the original split file.')
    parser.add_argument('--use_yolo_format', action='store_true', default=False,
                        help='Whether to use the yolo format.')

    global args
    args = parser.parse_args(argv)


if __name__ == '__main__':
    parse_args()
    assert os.path.isfile(args.split_file), '%s is not a file.' % args.split_file

    with open(args.split_file, 'r') as fp:
        data = json.load(fp)

    train_id_list = data['train']
    val_id_list = data['val']
    trainval_id_list = train_id_list + val_id_list
    test_id_list = data['test']

    train_split_file = 'train.txt'
    val_split_file = 'val.txt'
    trainval_split_file = 'trainval.txt'
    test_split_file = 'test.txt'

    with open(train_split_file, 'w') as f:
        for i in train_id_list:
            if args.use_yolo_format:
                f.write('../../images/' + i + '.jpg' + '\n')
            else:
                f.write(i + '\n')

    with open(val_split_file, 'w') as f:
        for i in val_id_list:
            if args.use_yolo_format:
                f.write('../../images/' + i + '.jpg' + '\n')
            else:
                f.write(i + '\n')

    with open(trainval_split_file, 'w') as f:
        for i in trainval_id_list:
            if args.use_yolo_format:
                f.write('../../images/' + i + '.jpg' + '\n')
            else:
                f.write(i + '\n')

    with open(test_split_file, 'w') as f:
        for i in test_id_list:
            if args.use_yolo_format:
                f.write('../../images/' + i + '.jpg' + '\n')
            else:
                f.write(i + '\n')
