"""Inspired from https://arxiv.org/pdf/1906.12171.pdf

@article{schneider2019gesture,
  title={Gesture Recognition in RGB Videos Using Human Body Keypoints and Dynamic Time Warping},
  author={Schneider, Pascal and Memmesheimer, Raphael and Kramer, Ivanna and Paulus, Dietrich},
  journal={arXiv preprint arXiv:1906.12171},
  year={2019}
}
"""
import csv
import math
import sys

from typing import List

import fastdtw

from scipy.spatial.distance import euclidean

from common import CocoPart


def load_csv(csv_fp: str) -> List:
    """Load keypoint coordinates stored in a CSV file.

    Columns are in order frame_no, nose.(x|y|p), (l|r)eye.(x|y|p), (l|r)ear.(x|y|p), (l|r)shoulder.(x|y|p),
    (l|r)elbow.(x|y|p), (l|r)wrist.(x|y|p), (l|r)hip.(x|y|p), (l|r)knee.(x|y|p), (l|r)ankle.(x|y|p)

    l - Left side of the identified joint
    r - Right side of the identified joint
    x - X coordinate of the identified joint
    y - Y coordinate of the identified joint
    p - Probability of the identified joint

    Coordinate list for a frame = [ [x, y], [x, y], [x, y], ... ]
    *coordinates in order specified in the CSV header

    Returns a list of coordinates for each frame = [ [...], [...], ... ]

    :param csv_fp: Path to the CSV file
    """
    pose_coordinates = []

    with open(csv_fp, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)

        for row in reader:
            # TODO: Figure out a way to handle missing joints
            coordinates = []  # List to store XY coordinates of every joint

            # Fill up the coordinate list for a single frame
            for index in range(1, len(row), 3):  # Starting from 1 to skip frame column, 3 because joint.x|y|prob
                coordinates.append([float(row[index]), float(row[index + 1])])

            # Fill up the final list with coordinate lists of all the frames
            pose_coordinates.append(coordinates)

    return pose_coordinates


def normalise(all_coordinates: List) -> List:
    """The normalization is a simple coordinate transformation done in two steps:

    1. Translation: All the key points are translated such that the nose key point becomes the origin of the coordinate
        system. This is achieved by subtracting the nose key points coordinates from all other key points.

    2. Scaling: The key points are scaled such that the distance between the left shoulder and right shoulder key point
        becomes 1. This is done by dividing all key points coordinates by the distance between the left and right
        shoulder key point.
    """
    norm_coords = []  # Hold the normalised coordinates for every frame

    # Iterate over every frame
    for coordinates in all_coordinates:
        # Step 1: Translate
        coordinates = [
            [coordinate[0] - coordinates[CocoPart.Nose.value][0], coordinate[1] - coordinates[CocoPart.Nose.value][1]]
            for coordinate in coordinates
        ]

        # Step 2: Scale
        dist = math.hypot(coordinates[CocoPart.LShoulder.value][0] - coordinates[CocoPart.RShoulder.value][0],
                          coordinates[CocoPart.LShoulder.value][1] - coordinates[CocoPart.RShoulder.value][1])
        coordinates = [[coordinate[0] / dist, coordinate[1] / dist] for coordinate in coordinates]

        norm_coords.append(coordinates)

    return norm_coords


def main():
    # CSV file paths
    pose_csv1 = sys.argv[1]
    pose_csv2 = sys.argv[2]

    # Load the CSVs
    pose1 = load_csv(csv_fp=pose_csv1)
    pose2 = load_csv(csv_fp=pose_csv2)

    # Normalization
    pose1 = normalise(pose1)
    pose2 = normalise(pose2)

    # Dynamic Time Warping
    from itertools import chain
    pose1 = [list(chain(*p)) for p in pose1]
    pose2 = [list(chain(*p)) for p in pose2]
    distance, _ = fastdtw.fastdtw(pose1, pose2, radius=30, dist=euclidean)
    print(distance)


if __name__ == '__main__':
    main()
