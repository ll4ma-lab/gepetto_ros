"""`PlayPlan` goal <-> `robot_plan.SolvePlan`, and nothing else.

Two functions, both trivial, and that is the point. The ENCODING -- how a plan
flattens to rectangular arrays, what order the block is in, what a ragged one does
-- lives in ``gepetto_solvers.core.robot_plan.wire``, where it is pure numpy and
where ``tests/core/test_plan_wire.py`` proves it round-trips without a built
colcon workspace anywhere in sight. What is left here is the part that genuinely
is ROS: turning 4x4 matrices into ``geometry_msgs/Pose`` and back.

Writing the encoding here instead would mean writing it TWICE, once to send and
once to receive, with the two copies as the only check on each other. That is the
shape of bug that plays a plan correctly in every test and drives the wrong joints
on the robot.
"""

import numpy as np

from gepetto_solvers.core import robot_plan


def plan_to_goal(plan, goal, matrix_to_pose):
    """Fill ``goal``'s plan fields from ``plan``. Returns ``goal``.

    ``matrix_to_pose`` is passed in rather than imported so this module needs no
    opinion about which pose type is being filled -- the caller has a
    ``geometry_msgs/Pose`` to hand and knows how it wants it built.
    """
    from geometry_msgs.msg import Pose

    fields = robot_plan.flatten_plan(plan)
    goal.wrist_poses = [matrix_to_pose(T, Pose()) for T in fields["wrist_poses"]]
    goal.digit_names = list(fields["digit_names"])
    goal.dof_per_digit = int(fields["dof_per_digit"])
    goal.command_kind = str(fields["command_kind"])
    goal.digit_cmd = [float(v) for v in fields["digit_cmd"]]
    goal.notes = list(fields["notes"])
    goal.corner_viz = [float(v) for v in fields["corner_viz"]]
    return goal


def goal_to_plan(goal, pose_to_matrix):
    """A ``SolvePlan`` from ``goal``'s plan fields. The inverse of the above.

    ``command_kind`` may be empty on a goal from an older client; it falls back
    to the tendon units, which is what such a client can only have meant. A
    MISMATCHED kind is not handled here -- the executor refuses that goal
    outright, before the arm moves, rather than letting it reach a driver.
    """
    return robot_plan.unflatten_plan({
        "wrist_poses": np.array([pose_to_matrix(p) for p in goal.wrist_poses],
                                float).reshape(-1, 4, 4),
        "digit_names": list(goal.digit_names),
        "dof_per_digit": int(goal.dof_per_digit),
        "command_kind": (goal.command_kind
                         or robot_plan.TENDON_DISPLACEMENT_M),
        "digit_cmd": np.asarray(goal.digit_cmd, float),
        "notes": list(goal.notes),
        "corner_viz": np.asarray(goal.corner_viz, float),
    })
