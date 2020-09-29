import argparse
import logging
import sys
import os
import time
import glob
import pandas as pd
from tqdm import tqdm

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def make_dataset(image_folder, model, resize, resize_out_ratio, dataset_path):
    images = []
    exercises = os.listdir(image_folder)
    for exercise in exercises:
        for pose in os.listdir(image_folder+exercise):
            for img in os.listdir(f'{image_folder}{exercise}/{pose}'):
                images.append([f'{image_folder}{exercise}/{pose}/{img}',pose])

    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    body_points = []
    total = tqdm(range(len(images)))
    for img, _ in zip(images, total):
        centers = []
        image = common.read_imgfile(img[0], None, None)
        
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        # neck, left_shoulder, left_hip, left_knee for squat
        BodyPoints = [1, 5, 11, 12]
        
        
        for i in BodyPoints:
            if i not in humans[0].body_parts.keys():
                continue

            body_part = humans[0].body_parts[i]
            center_x, center_y = int(body_part.x * 432 + 0.5), int(body_part.y * 368 + 0.5)
            centers.append(center_x)
            centers.append(center_y)
        body_points.append(centers)
        body_points[-1].append(img[1])

    df = pd.DataFrame(body_points,
                      columns=['neck_x',
                               'neck_y',
                               'Lshoulder_x',
                               'Lshoulder_y',
                               'Lhip_x',
                               'Lhip_y',
                               'Lknee_x',
                               'Lknee_y',
                               'label'])
    
    df.to_csv(f'{args.dataset_path}/train.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize_out_ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--dataset_path', type=str)

    args = parser.parse_args()

    make_dataset(args.image_folder, args.model, args.resize, args.resize_out_ratio, args.dataset_path)