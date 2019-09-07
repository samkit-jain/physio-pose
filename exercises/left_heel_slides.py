from typing import Dict, List, Tuple

from common import CocoPart, get_angle

TOTAL_STEPS = 3  # Number of steps in the exercise


def do_left_heel_slides(humans: List, current_step: int) -> Tuple[int, str]:
    """Perform heel slides with the left leg.

    Description: Slide the heel towards the buttocks as far as possible. Hold it for 5 seconds and relax.

    Link: https://youtu.be/Bz0wSFRjH2c
    """
    satisfies, err_mess = satisfies_prerequisites(humans=humans)

    if not satisfies:
        return -1, err_mess

    new_step, mess = perform_step(human=humans[0], cur_step=current_step)

    return new_step, mess


def perform_step(human: Dict, cur_step: int) -> Tuple[int, str]:
    """
    Steps:
    1. Left leg should be in a lying position, i.e., making an inner angle in range [165, 180].
    2. Slide the heel backwards such that the inner angle is less than 90.
    3. Slide the heel forward to the starting position.
    """
    left_ankle = human['coordinates'][CocoPart.LAnkle.value][:2]
    left_knee = human['coordinates'][CocoPart.LKnee.value][:2]
    left_hip = human['coordinates'][CocoPart.LHip.value][:2]

    angle = get_angle(p0=[left_ankle[0], 1-left_ankle[1]],
                      p1=[left_knee[0], 1-left_knee[1]],
                      p2=[left_hip[0], 1-left_hip[1]])

    if cur_step == 0:
        if 165 <= angle <= 180:
            return cur_step + 1, 'Initial position set\nSlowly slide left leg backwards'

        return cur_step, 'Move left leg to lying position'

    elif cur_step == 1:
        if 165 <= angle <= 180:
            return cur_step, 'Initial position set\nSlowly slide left leg backwards'

        if angle <= 40:
            return cur_step + 1, 'Limit reached\nSlide left leg forwards'

        return cur_step, 'Continue sliding left leg backwards'

    elif cur_step == 2:
        if angle <= 40:
            return cur_step, 'Limit reached\nSlide left leg forwards'

        if 165 <= angle <= 180:
            return cur_step + 1, 'Left leg back in starting position'

        return cur_step, 'Continue sliding left leg forwards'

    return cur_step, ''


def satisfies_prerequisites(humans: List) -> Tuple[bool, str]:
    """Check whether the prerequisites for the exercise are met.

    Y axis increases downwards  (hence the `1-`)
    X axis increases rightwards

    Coordinates are 0 to 1 scaled hence the `+0.5` in X axis

    Prerequisites:
    1. Only 1 human in frame.
    2. Left leg fully visible.
    3. Left leg in lying position.
    """
    if len(humans) == 0:
        return False, 'No human in sight'

    if len(humans) > 1:
        return False, 'More than 1 human in sight'

    left_ankle = humans[0]['coordinates'][CocoPart.LAnkle.value][:2]
    left_knee = humans[0]['coordinates'][CocoPart.LKnee.value][:2]
    left_hip = humans[0]['coordinates'][CocoPart.LHip.value][:2]

    if any(joint == [0, 0] for joint in [left_ankle, left_knee, left_hip]):
        return False, 'Left leg not visible'

    if not (left_ankle[0] < left_knee[0] < left_hip[0]):
        return False, 'Left leg not in lying position'

    return True, 'Satisfies'
