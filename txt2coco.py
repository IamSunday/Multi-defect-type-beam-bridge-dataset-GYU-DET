import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

# Example usage:
# python yolo2coco.py --root_dir VisDrone2019-DET-train --save_path train.json
# python yolo2coco.py --root_dir VisDrone2019-DET-val --save_path val.json
# python yolo2coco.py --root_dir VisDrone2019-DET-test-dev --save_path test.json

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='./dataset/valid', type=str,
                    help="Root directory containing 'images', 'labels', and 'classes.txt'")
parser.add_argument('--save_path', type=str, default='./valid.json',
                    help="Path to save output COCO format JSON (if not using split)")
parser.add_argument('--random_split', action='store_true', help="Enable random dataset splitting with default ratio 8:1:1")
parser.add_argument('--split_by_file', action='store_true',
                    help="Split dataset using predefined text files: 'train.txt', 'val.txt', and 'test.txt'")

arg = parser.parse_args()


def train_test_val_split_random(img_paths, ratio_train=0.8, ratio_test=0.1, ratio_val=0.1):
    # Randomly split the dataset by specified ratios
    assert int(ratio_train + ratio_test + ratio_val) == 1
    train_img, middle_img = train_test_split(img_paths, test_size=1 - ratio_train, random_state=233)
    ratio = ratio_val / (1 - ratio_train)
    val_img, test_img = train_test_split(middle_img, test_size=ratio, random_state=233)
    print("Split count - train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
    return train_img, val_img, test_img


def train_test_val_split_by_files(img_paths, root_dir):
    # Split dataset based on text files listing image filenames
    phases = ['train', 'val', 'test']
    img_split = []
    for p in phases:
        define_path = os.path.join(root_dir, f'{p}.txt')
        print(f'Reading {p} dataset definition from {define_path}')
        assert os.path.exists(define_path)
        with open(define_path, 'r') as f:
            img_paths = f.readlines()
            img_split.append(img_paths)
    return img_split[0], img_split[1], img_split[2]


def yolo2coco(arg):
    root_path = arg.root_dir
    print("Loading data from ", root_path)

    assert os.path.exists(root_path)
    originLabelsDir = os.path.join(root_path, 'labels')
    originImagesDir = os.path.join(root_path, 'images')
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()

    indexes = os.listdir(originImagesDir)

    if arg.random_split or arg.split_by_file:
        train_dataset = {'categories': [], 'annotations': [], 'images': []}
        val_dataset = {'categories': [], 'annotations': [], 'images': []}
        test_dataset = {'categories': [], 'annotations': [], 'images': []}

        for i, cls in enumerate(classes, 0):
            for ds in (train_dataset, val_dataset, test_dataset):
                ds['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

        if arg.random_split:
            print("Splitting mode: random split")
            train_img, val_img, test_img = train_test_val_split_random(indexes, 0.8, 0.1, 0.1)
        elif arg.split_by_file:
            print("Splitting mode: split by files")
            train_img, val_img, test_img = train_test_val_split_by_files(indexes, root_path)
    else:
        dataset = {'categories': [], 'annotations': [], 'images': []}
        for i, cls in enumerate(classes, 0):
            dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # Support png and jpg image formats
        txtFile = index.replace('images', 'txt').replace('.jpg', '.txt').replace('.png', '.txt')
        im = cv2.imread(os.path.join(root_path, 'images/') + index)
        height, width, _ = im.shape

        if arg.random_split or arg.split_by_file:
            if index in train_img:
                dataset = train_dataset
            elif index in val_img:
                dataset = val_dataset
            elif index in test_img:
                dataset = test_dataset

        dataset['images'].append({'file_name': index,
                                  'id': k,
                                  'width': width,
                                  'height': height})
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            continue

        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H

                cls_id = int(label[0])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': k,
                    'iscrowd': 0,
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)
    if arg.random_split or arg.split_by_file:
        for phase in ['train', 'val', 'test']:
            json_name = os.path.join(root_path, f'annotations/{phase}.json')
            with open(json_name, 'w') as f:
                if phase == 'train':
                    json.dump(train_dataset, f)
                elif phase == 'val':
                    json.dump(val_dataset, f)
                elif phase == 'test':
                    json.dump(test_dataset, f)
            print(f'Saved annotations to {json_name}')
    else:
        json_name = os.path.join(root_path, 'annotations/{}'.format(arg.save_path))
        with open(json_name, 'w') as f:
            json.dump(dataset, f)
            print(f'Saved annotations to {json_name}')


if __name__ == "__main__":
    yolo2coco(arg)
