import argparse
import base64
import logging

import cv2
import openpifpaf
import torch

from common import write_on_image
from poses import do_seated_right_knee_extension, SRKE_TOTAL
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
    parser.add_argument('--pose', default='seated_right_knee_extension', type=str,
                        help='Pose ID to perform.')

    args = parser.parse_args()

    # log
    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG)

    # add args.device
    args.device = torch.device('cpu')

    return args


def main():
    pose_methods = {
        'seated_right_knee_extension': {
            'func': do_seated_right_knee_extension,
            'steps': SRKE_TOTAL
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

    pose_func = pose_methods[args.pose]['func']
    cur_step = 0
    pose_steps = pose_methods[args.pose]['steps']

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

        keypoint_sets, scores, width_height = processor_singleton.single_image(b64image=base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('UTF-8'))
        keypoint_sets = [{
            'coordinates': keypoints.tolist(),
            'detection_id': i,
            'score': score,
            'width_height': width_height,
        } for i, (keypoints, score) in enumerate(zip(keypoint_sets, scores))]

        # Using only the first person for visualisation
        keypoints = keypoint_sets[0]

        # For visualisation
        for i, kps in enumerate(keypoints['coordinates']):
            x = int(kps[0] * width)
            y = int(kps[1] * height)

            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        cur_step, mess = pose_func(keypoint_sets, cur_step)

        img = write_on_image(img=img, text=mess, color=[0, 0, 0])

        cv2.imshow('You', img)

        if cur_step == pose_steps:
            task_finished = True
            continue


if __name__ == '__main__':
    main()
