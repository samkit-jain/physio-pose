from enum import IntEnum, unique
from typing import List, Tuple

import cv2
import numpy as np


@unique
class CocoPart(IntEnum):
    """Body part locations in the 'coordinates' list."""
    Nose = 0
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16


SKELETON_CONNECTIONS = [(0, 1, (210, 182, 247)), (0, 2, (127, 127, 127)), (1, 2, (194, 119, 227)),
                        (1, 3, (199, 199, 199)), (2, 4, (34, 189, 188)), (3, 5, (141, 219, 219)),
                        (4, 6, (207, 190, 23)), (5, 6, (150, 152, 255)), (5, 7, (189, 103, 148)),
                        (5, 11, (138, 223, 152)), (6, 8, (213, 176, 197)), (6, 12, (40, 39, 214)),
                        (7, 9, (75, 86, 140)), (8, 10, (148, 156, 196)), (11, 12, (44, 160, 44)),
                        (11, 13, (232, 199, 174)), (12, 14, (120, 187, 255)), (13, 15, (180, 119, 31)),
                        (14, 16, (14, 127, 255))]


def get_angle(p0: List, p1: List, p2: List) -> float:
    """Compute angle (in degrees) for p0p1p2 corner."""
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    return np.degrees(np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1)))


def get_intersection_point(line1: List, line2: List) -> Tuple:
    """Return the point of intersection between two lines.

    Source: https://stackoverflow.com/a/42727584/7760998
    """
    # Make float
    line1 = [[float(line1[0][0]), float(line1[0][1])], [float(line1[1][0]), float(line1[1][1])]]
    line2 = [[float(line2[0][0]), float(line2[0][1])], [float(line2[1][0]), float(line2[1][1])]]

    s = np.vstack([line1[0], line1[1], line2[0], line2[1]])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection

    if z == 0:  # lines are parallel
        raise ValueError('lines do not intersect')

    return x / z, y / z


def make_360(angle: float) -> float:
    return angle if angle >= 0 else 360.0 + angle


def write_on_image(img: np.ndarray, text: str, color: List) -> np.ndarray:
    """Write text at the top of the image."""
    # Add a white border to top of image for writing text
    img = cv2.copyMakeBorder(src=img,
                             top=int(0.25 * img.shape[0]),
                             bottom=0,
                             left=0,
                             right=0,
                             borderType=cv2.BORDER_CONSTANT,
                             dst=None,
                             value=[255, 255, 255])
    for i, line in enumerate(text.split('\n')):
        y = 30 + i * 30
        cv2.putText(img=img,
                    text=line,
                    org=(20, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=color,
                    thickness=3)

    return img
