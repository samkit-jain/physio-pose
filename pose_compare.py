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

from itertools import chain
from typing import List

import fastdtw
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt
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


def dimension_selection(frames: List) -> List:
    """Remove indices that don't vary a lot during the pose.

    Key points that do not move significantly in the sequence will cause the signals of the respective coordinates to be
    roughly constant with only little variance. All signals whose variance is below a threshold will be filtered out and
    are assumed to be uninformative.

    :returns: A set of indices of dimensions that should be kept
    """

    def keep_sequence(seq: List) -> bool:
        """Check whether the points in the sequence vary a lot or not."""
        # Use a median filter for smoothing
        seq = medfilt(seq, kernel_size=3)

        # Filter the dimension based on variance
        return np.var(seq) > 0.10

    # Reorder the coordinates such that they are per joint and not per frame
    frames = [list(chain(*frame)) for frame in frames]  # Flatten the nested lists inside
    sequences = list(map(list, zip(*frames)))  # Transpose the list

    # Drop low variance columns
    dimensions = [i for i, sequence in enumerate(sequences) if keep_sequence(sequence)]

    return dimensions


def calculate_score(seq1: List, seq2: List, dimensions: List) -> float:
    """Calculate how similar the two pose sequences are."""

    def process_signal(signal: List) -> List:
        """Final processing before dynamic time warping."""
        # Apply Gaussian filter for further processing
        signal = gaussian_filter(signal, sigma=1)

        # Make the sequence/signal zero-mean by subtracting the mean from it
        mean = np.mean(signal)
        return [x - mean for x in signal]

    distance = 0.0

    for dim in dimensions:
        sig1 = process_signal(signal=seq1[dim])
        sig2 = process_signal(signal=seq2[dim])

        temp_distance, _ = fastdtw.fastdtw(sig1, sig2, radius=30, dist=euclidean)
        distance += temp_distance

    # Normalise DTW score
    distance /= len(dimensions)

    return distance


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

    # Dimension Selection
    pose1_dimensions = dimension_selection(pose1.copy())
    pose2_dimensions = dimension_selection(pose2.copy())

    dimensions = sorted(set(pose1_dimensions + pose2_dimensions))  # Take a union to get final list of dimensions

    score = calculate_score(pose1, pose2, dimensions)
    print(f'Score = {score:.6f}')


if __name__ == '__main__':
    main()
