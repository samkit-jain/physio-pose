import argparse
import base64
import csv
import logging

from typing import List

import cv2
import numpy as np
import openpifpaf
import torch

from common import CocoPart, SKELETON_CONNECTIONS, write_on_image
from exercises import do_left_heel_slides, do_seated_right_knee_extension, do_side_lying_left_leg_lift, LHS_TOTAL, \
    SRKE_TOTAL, SLLLL_TOTAL
from processor import Processor


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # TODO: Verify the args since they were changed in v0.10.0
    openpifpaf.decoder.cli(parser, force_complete_pose=True,
                           instance_threshold=0.2, seed_threshold=0.5)
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
    vis_args.add_argument('--save-output', default=False, action='store_true',
                          help='Save the result in a video file.')
    vis_args.add_argument('--fps', default=20, type=int,
                          help='FPS for the output video.')
    vis_args.add_argument('--out-path', default='result.avi', type=str,
                          help='Save the output video at the path specified. .avi file format.')
    vis_args.add_argument('--csv-path', default='keypoints.csv', type=str,
                          help='Save the pose coordinates into a CSV file at the path specified.')

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


def write_to_csv(frame_number: int, humans: List, width: int, height: int, csv_fp: str):
    """Save keypoint coordinates of the *first* human pose identified to a CSV file.

    Coordinates are scaled to refer the resized image.

    Columns are in order frame_no, nose.(x|y|p), (l|r)eye.(x|y|p), (l|r)ear.(x|y|p), (l|r)shoulder.(x|y|p),
    (l|r)elbow.(x|y|p), (l|r)wrist.(x|y|p), (l|r)hip.(x|y|p), (l|r)knee.(x|y|p), (l|r)ankle.(x|y|p)

    l - Left side of the identified joint
    r - Right side of the identified joint
    x - X coordinate of the identified joint
    y - Y coordinate of the identified joint
    p - Probability of the identified joint

    :param frame_number: Frame number for the video file
    :param humans: List of human poses identified
    :param width: Width of the image
    :param height: Height of the image
    :param csv_fp: Path to the CSV file
    """
    # Use only the first human identified using pose estimation
    coordinates = humans[0]['coordinates'] if len(humans) > 0 else [None for _ in range(17)]

    # Final row that will be written to the CSV file
    row = [frame_number] + ['' for _ in range(51)]  # Number of coco points * 3 -> 17 * 3 -> 51

    # Update the items in the row for every joint
    # TODO: Value for joints not identified? Currently stored as 0
    for part in CocoPart:
        if coordinates[part] is not None:
            index = 1 + 3 * part.value  # Index at which the values for this joint would start in the final row
            row[index] = coordinates[part][0] * width
            row[index + 1] = coordinates[part][1] * height
            row[index + 2] = coordinates[part][2]

    with open(csv_fp, mode='a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(row)


def main():
    exercises = {
        'left_heel_slides': {
            'func': do_left_heel_slides,
            'steps': LHS_TOTAL
        },
        'seated_right_knee_extension': {
            'func': do_seated_right_knee_extension,
            'steps': SRKE_TOTAL
        },
        'side_lying_left_leg_lift': {
            'func': do_side_lying_left_leg_lift,
            'steps': SLLLL_TOTAL
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

    # Setup CSV file
    with open(args.csv_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['frame_no',
                             'nose.x', 'nose.y', 'nose.prob',
                             'l.eye.x', 'l.eye.y', 'l.eye.prob',
                             'r.eye.x', 'r.eye.y', 'r.eye.prob',
                             'l.ear.x', 'l.ear.y', 'l.ear.prob',
                             'r.ear.x', 'r.ear.y', 'r.ear.prob',
                             'l.shoulder.x', 'l.shoulder.y', 'l.shoulder.prob',
                             'r.shoulder.x', 'r.shoulder.y', 'r.shoulder.prob',
                             'l.elbow.x', 'l.elbow.y', 'l.elbow.prob',
                             'r.elbow.x', 'r.elbow.y', 'r.elbow.prob',
                             'l.wrist.x', 'l.wrist.y', 'l.wrist.prob',
                             'r.wrist.x', 'r.wrist.y', 'r.wrist.prob',
                             'l.hip.x', 'l.hip.y', 'l.hip.prob',
                             'r.hip.x', 'r.hip.y', 'r.hip.prob',
                             'l.knee.x', 'l.knee.y', 'l.knee.prob',
                             'r.knee.x', 'r.knee.y', 'r.knee.prob',
                             'l.ankle.x', 'l.ankle.y', 'l.ankle.prob',
                             'r.ankle.x', 'r.ankle.y', 'r.ankle.prob',
                             ])

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

    width_height = (int(width * args.resolution // 16) * 16 + 1, int(height * args.resolution // 16) * 16 + 1)
    logging.debug(f'Target width and height = {width_height}')
    processor_singleton = Processor(width_height, args)

    task_finished = False
    output_video = None

    frame = 0

    while not task_finished:
        frame += 1

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

        write_to_csv(frame_number=frame, humans=keypoint_sets, width=width, height=height, csv_fp=args.csv_path)

        temp_step, mess = exercise_func(keypoint_sets, cur_step)

        # No need to change cur_step when prerequisites are not met
        if temp_step != -1:
            cur_step = temp_step

        img = write_on_image(img=img, text=mess, color=[0, 0, 0])

        if output_video is None:
            if args.save_output:
                output_video = cv2.VideoWriter(filename=args.out_path, fourcc=cv2.VideoWriter_fourcc(*'MP42'),
                                               fps=args.fps, frameSize=img.shape[:2][::-1])
                logging.debug(f'Saving the output video at {args.out_path} with {args.fps} frames per seconds')
            else:
                output_video = None
                logging.debug(f'Not saving the output video')

        if output_video is not None:
            output_video.write(img)

        cv2.imshow('You', img)

        if cur_step == total_steps:
            task_finished = True
            continue


if __name__ == '__main__':
    main()
