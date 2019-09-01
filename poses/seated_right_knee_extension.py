from typing import Dict, List, Tuple

from common import CocoPart, get_angle, make_360

TOTAL_STEPS = 3  # Number of steps in the exercise


def do_seated_right_knee_extension(humans: List, current_step: int) -> Tuple[int, str]:
    """Perform seated knee flexion and extension for the right leg.
    Link: https://youtu.be/OpFov55bKZo
    """
    satisfies, err_mess = satisfies_prerequisites(humans=humans)

    if not satisfies:
        return -1, err_mess

    new_step, mess = perform_step(human=humans[0], cur_step=current_step)

    return new_step, mess


def perform_step(human: Dict, cur_step: int) -> Tuple[int, str]:
    """
    Steps:
    1. Right leg should be in a seated position, i.e., making an inner angle in range (120, 150).
    2. Extend the right leg such that the inner angle is more than 180.
    3. Bring the right leg back to the starting position.
    """
    right_ankle = human['coordinates'][CocoPart.RAnkle.value][:2]
    right_knee = human['coordinates'][CocoPart.RKnee.value][:2]
    right_hip = human['coordinates'][CocoPart.RHip.value][:2]

    angle = get_angle(p0=[right_ankle[0], 1 - right_ankle[1]],
                      p1=[right_knee[0], 1 - right_knee[1]],
                      p2=[right_hip[0], 1 - right_hip[1]])

    if cur_step == 0:
        if 120 < angle < 150:
            return 1, 'Initial position set\nSlowly extend right leg'

        return 0, 'Move right leg to seating position'

    elif cur_step == 1:
        if make_360(angle=angle) >= 180:
            return 2, 'Extension limit reached\nSlowly lower right leg'

        return 1, 'Slowly extend right leg'

    elif cur_step == 2:
        if make_360(angle=angle) >= 180:
            return 2, 'Extension limit reached\nSlowly lower right leg'

        if 0 < angle < 150:
            return 3, 'Right leg back in starting position'

        return 2, 'Slowly lower right leg'

    return cur_step, 'Nothing'


def satisfies_prerequisites(humans: List) -> Tuple[bool, str]:
    """Check whether the prerequisites for the exercise are met.

    Y axis increases downwards  (hence the `1-`)
    X axis increases rightwards

    Coordinates are 0 to 1 scaled hence the `+0.5` in X axis

    Prerequisites:
    1. Only 1 human in frame.
    2. Both legs fully visible.
    3. Left leg in seated position.
    """
    if len(humans) == 0:
        return False, 'No human in sight'

    if len(humans) > 1:
        return False, 'More than 1 human in sight'

    right_ankle = humans[0]['coordinates'][CocoPart.RAnkle.value][:2]
    right_knee = humans[0]['coordinates'][CocoPart.RKnee.value][:2]
    right_hip = humans[0]['coordinates'][CocoPart.RHip.value][:2]
    left_ankle = humans[0]['coordinates'][CocoPart.LAnkle.value][:2]
    left_knee = humans[0]['coordinates'][CocoPart.LKnee.value][:2]
    left_hip = humans[0]['coordinates'][CocoPart.LHip.value][:2]

    if any(joint == [0, 0] for joint in [left_ankle, left_knee, left_hip, right_ankle, right_knee, right_hip]):
        return False, 'Full legs not visible'

    angle = get_angle(p0=[left_ankle[0], 1-left_ankle[1]],
                      p1=[left_knee[0], 1-left_knee[1]],
                      p2=[left_hip[0], 1-left_hip[1]])

    if not (90 <= int(angle) <= 150):
        return False, 'Left leg not seated'

    return True, 'Satisfies'
