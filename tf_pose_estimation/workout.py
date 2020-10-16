import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tensorflow.keras.models import load_model


logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    pose_model = load_model("../test.h5")
    count = 0
    position = "stand"

    if cap.isOpened() is False:
        print("Error opening video stream or file")

    while cap.isOpened():
        ret_val, image = cap.read()

        humans = e.inference(image,  resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        # neck, left_shoulder, left_hip, left_knee for squat
        BodyPoints = [1, 5, 11, 12]
        
        centers = []
        if humans:
            for i in BodyPoints:
                if i not in humans[0].body_parts.keys():
                    continue
                body_part = humans[0].body_parts[i]
                center_x, center_y = int(body_part.x * 432 + 0.5), int(body_part.y * 368 + 0.5)
                centers.append(center_x)
                centers.append(center_y)
        
        centers = np.array([centers])
        pred = 0
        if centers.shape == (1,8):
            pred = pose_model.predict(centers)
            print(pred, position)
            if pred <= 1e-3 and position == "stand":
                position = "sit"
                count += 1
                
        if pred <= 0.3:
            cv2.putText(image, "sit", (220, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif pred >= 0.9:
            position = "stand"
            cv2.putText(image, "stand", (220, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        cv2.putText(image, f"count : {count}", (432, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
logger.debug('finished+')
