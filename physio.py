import argparse
import base64
import logging

from typing import List

import cv2
import numpy as np
import openpifpaf
import torch

from common import SKELETON_CONNECTIONS, write_on_image
from exercises import do_left_heel_slides, do_seated_right_knee_extension, LHS_TOTAL, SRKE_TOTAL
from processor import Processor


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    openpifpaf.decoder.cli(parser, force_complete_pose=True,
                           instance_threshold=0.1, seed_threshold=0.5)
    openpifpaf.network.nets.cli(parser)
    parser.add_argument('--resolution', default=0.4, type=float,
                        help=('Resolution prescale factor from 640x480. '
                              'Will be rounded to multiples of 16.'))
    parser.add_argument('--resize', default=None, type=str,
                        help=('Force input image resize. '
                              'Example WIDTHxHEIGHT.'))
    parser.add_argument('--video', default=None, type=str,
                        help='Path to the video file.')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='debug messages and autoreload')
    parser.add_argument('--exercise', default='seated_right_knee_extension', type=str,
                        help='Exercise ID to perform.')

    vis_args = parser.add_argument_group('Visualisation')
    vis_args.add_argument('--joints', default=False, action='store_true',
                          help='Draw joint\'s keypoints on the output video.')
    vis_args.add_argument('--skeleton', default=False, action='store_true',
                          help='Draw skeleton on the output video.')

    args = parser.parse_args()

    # Log
    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # Add args.device
    args.device = torch.device('cpu')

    return args


def visualise(img: np.ndarray, keypoint_sets: List, width: int, height: int, vis_keypoints: bool = False,
              vis_skeleton: bool = False) -> np.ndarray:
    """Draw keypoints/skeleton on the output video frame."""
    if vis_keypoints or vis_skeleton:
        for keypoints in keypoint_sets:
            coords = keypoints['coordinates']

            if vis_skeleton:
                for p1i, p2i, color in SKELETON_CONNECTIONS:
                    p1 = (int(coords[p1i][0] * width), int(coords[p1i][1] * height))
                    p2 = (int(coords[p2i][0] * width), int(coords[p2i][1] * height))

                    if p1 == (0, 0) or p2 == (0, 0):
                        continue

                    cv2.line(img=img, pt1=p1, pt2=p2, color=color, thickness=3)

            if vis_keypoints:
                for i, kps in enumerate(coords):
                    # Scale up joint coordinate
                    p = (int(kps[0] * width), int(kps[1] * height))

                    # Joint wasn't detected
                    if p == (0, 0):
                        continue

                    cv2.circle(img=img, center=p, radius=5, color=(255, 255, 255), thickness=-1)

    return img


def main():
    exercises = {
        'seated_right_knee_extension': {
            'func': do_seated_right_knee_extension,
            'steps': SRKE_TOTAL
        },
        'left_heel_slides': {
            'func': do_left_heel_slides,
            'steps': LHS_TOTAL
        }
    }
    args = cli()

    # Choose video source
    if args.video is None:
        logging.debug('Video source: webcam')
        cam = cv2.VideoCapture(0)
    else:
        logging.debug(f'Video source: {args.video}')
        cam = cv2.VideoCapture(args.video)

    ret_val, img = cam.read()

    # Resize the video
    if args.resize is None:
        height, width = img.shape[:2]
    else:
        width, height = [int(dim) for dim in args.resize.split('x')]
        
    if args.exercise not in exercises:
        logging.error(f'Exercise {args.exercise} not supported!')
        return 

    exercise_func = exercises[args.exercise]['func']
    cur_step = 0
    total_steps = exercises[args.exercise]['steps']

    width_height = (int(width * args.resolution // 16) * 16 + 1,
                    int(height * args.resolution // 16) * 16 + 1)
    logging.debug(f'Target width and height = {width_height}')
    processor_singleton = Processor(width_height, args)

    task_finished = False

    while not task_finished:
        ret_val, img = cam.read()

        if not ret_val:
            task_finished = True
            continue

        if cv2.waitKey(1) == 27:
            task_finished = True
            continue

        img = cv2.resize(img, (width, height))

        ###
        # IMP: Having force_complete_pose=False results in separate annotations
        ###

        keypoint_sets, scores, width_height = processor_singleton.single_image(
            b64image=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('UTF-8')
        )
        keypoint_sets = [{
            'coordinates': keypoints.tolist(),
            'detection_id': i,
            'score': score,
            'width_height': width_height,
        } for i, (keypoints, score) in enumerate(zip(keypoint_sets, scores))]

        img = visualise(img=img, keypoint_sets=keypoint_sets, width=width, height=height, vis_keypoints=args.joints,
                        vis_skeleton=args.skeleton)

        temp_step, mess = exercise_func(keypoint_sets, cur_step)

        # No need to change cur_step when prerequisites are not met
        if temp_step != -1:
            cur_step = temp_step

        img = write_on_image(img=img, text=mess, color=[0, 0, 0])

        cv2.imshow('You', img)

        if cur_step == total_steps:
            task_finished = True
            continue


if __name__ == '__main__':
    main()
