from typing import Dict, List, Tuple

from common import CocoPart, get_angle, get_intersection_point, make_360

TOTAL_STEPS = 3  # Number of steps in the exercise


def do_side_lying_left_leg_lift(humans: List, current_step: int) -> Tuple[int, str]:
    """Perform side lying leg lift with the left leg.

    Description: If you feel unsteady, bend the bottom leg for support. Toes should face forward.

    Link: https://youtu.be/jgh6sGwtTwk
    """
    satisfies, err_mess = satisfies_prerequisites(humans=humans)

    if not satisfies:
        return -1, err_mess

    new_step, mess = perform_step(human=humans[0], cur_step=current_step)

    return new_step, mess


def perform_step(human: Dict, cur_step: int) -> Tuple[int, str]:
    """
    Steps:
    1. Left leg should be straight and in a lying position, i.e., making an inner angle in range [170, 180]
        and hip coord less than knee and knee coord less than ankle.
    2. Extend the left leg upwards keeping it straight such that the inner angle between thighs is more than 45.
    3. Bring the left leg back to the starting position.
    """
    if cur_step == 0:
        left_ankle = human['coordinates'][CocoPart.LAnkle.value][:2]
        left_knee = human['coordinates'][CocoPart.LKnee.value][:2]
        left_hip = human['coordinates'][CocoPart.LHip.value][:2]

        # X coordinate of hip should be before knee which should be before ankle
        # and the max difference between Y coordinates should be of 0.1
        if not (
            left_hip[0] < left_knee[0] < left_ankle[0]
            and max(left_ankle[1], left_knee[1], left_hip[1]) - min(left_ankle[1], left_knee[1], left_hip[1]) <= 0.1
        ):
            return cur_step + 1, 'Initial position set\nSlowly move the left leg up'

        return cur_step, 'Move left leg to lying position'

    elif cur_step == 1:
        left_knee = human['coordinates'][CocoPart.LKnee.value][:2]
        left_hip = human['coordinates'][CocoPart.LHip.value][:2]
        right_knee = human['coordinates'][CocoPart.RKnee.value][:2]
        right_hip = human['coordinates'][CocoPart.RHip.value][:2]

        try:
            inter = get_intersection_point(line1=[left_hip, left_knee], line2=[right_hip, right_knee])
        except:
            return cur_step, 'Legs are parallel'

        angle = get_angle(p0=[left_knee[0], 1 - left_knee[1]],
                          p1=[inter[0], 1 - inter[1]],
                          p2=[right_knee[0], 1 - right_knee[1]])

        if -45 < angle < -40:
            return cur_step + 1, 'Extension limit reached\nLower the left leg'

        return cur_step, 'Continue moving the left leg up'

    elif cur_step == 2:
        left_ankle = human['coordinates'][CocoPart.LAnkle.value][:2]
        left_knee = human['coordinates'][CocoPart.LKnee.value][:2]
        left_hip = human['coordinates'][CocoPart.LHip.value][:2]

        # X coordinate of hip should be before knee which should be before ankle
        # and the max difference between Y coordinates should be of 0.1
        if (
            left_hip[0] < left_knee[0] < left_ankle[0]
            and max(left_ankle[1], left_knee[1], left_hip[1]) - min(left_ankle[1], left_knee[1], left_hip[1]) <= 0.1
        ):
            return cur_step + 1, 'Back in starting position'

        return cur_step, 'Continue lowering the left leg'

    return cur_step, ''


def satisfies_prerequisites(humans: List) -> Tuple[bool, str]:
    """Check whether the prerequisites for the exercise are met.

    Y axis increases downwards  (hence the `1-`)
    X axis increases rightwards

    Coordinates are 0 to 1 scaled hence the `+0.5` in X axis

    Prerequisites:
    1. Only 1 human in frame.
    2. Both legs fully visible.
    3. Right leg in lying position.
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

    # X coordinate of hip should be before knee which should be before ankle
    # and the max difference between Y coordinates should be of 0.1
    if not (
        right_hip[0] < right_knee[0] < right_ankle[0]
        and max(right_ankle[1], right_knee[1], right_hip[1]) - min(right_ankle[1], right_knee[1], right_hip[1]) <= 0.1
    ):
        return False, 'Right leg not in lying position'

    angle = get_angle(p0=[right_hip[0], 1 - right_hip[1]],
                      p1=[right_knee[0], 1 - right_knee[1]],
                      p2=[right_ankle[0], 1 - right_ankle[1]])

    if not (165 <= int(abs(angle)) <= 180):
        return False, 'Right leg not straight'

    angle = get_angle(p0=[left_hip[0], 1 - left_hip[1]],
                      p1=[left_knee[0], 1 - left_knee[1]],
                      p2=[left_ankle[0], 1 - left_ankle[1]])

    if not (165 <= int(abs(angle)) <= 180):
        return False, 'Left leg not straight'

    return True, 'Satisfies'
