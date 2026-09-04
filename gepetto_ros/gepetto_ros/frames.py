"""Registering the visualizer's world against the robot's, from TF.

Used by both sides of the split -- `robot_bridge` (the visualizer's client, for
"Get robot state") and `trajectory_executor` (the control loop, for every tick)
-- so there is one definition of the convention rather than two that can drift.

REGISTRATION. The visualizer's world frame is a local scene frame; the robot works
in `lbr_link_0`. The link between them is a physical feature both already
describe: the workspace table.

  * URDF (`workspace_table_description.xacro`): `lbr_workspace_table_link` is a
    0.4 x 0.4 m slab whose frame origin sits on the corner NEAREST THE ROBOT, on
    the TOP face, with the slab extending along its own +x/+y.
  * Viser (`scene.table_corner`): the minimum corner of a `TABLE_SPAN` = 0.4 m
    square, on the top face, with the square extending +X/+Y.

Same corner, same square, same convention -- so the viser world frame IS the URDF
table frame, translated by where the corner sits in viser coordinates:

    T_base<-viz = T_base<-workspace_table @ Translate(-corner_viz)

Nothing here has to be measured or configured. Note the URDF joint carries a +90
degree yaw (table +x is robot +y), which is why every conversion goes through the
TF transform rather than assuming the two worlds share an axis convention.

THE SAME TABLE FOR BOTH ROBOTS. `med14_allegro.xacro` includes the very same
`med14_epfl/workspace_table_description.xacro`, so `lbr_workspace_table_link`
exists and means the same thing on either description and this registration is
hand-independent. What IS hand-dependent is the WRIST frame -- the SRDF chain tip,
`lbr_epfl_wrist_link` on one and `palm_link` on the other -- which is why a
:class:`FrameLookup` is constructed with the profile's frames rather than with
constants of its own.

The registration is recomputed from `corner_viz` on every call, never cached: the
viser support plane is seated UNDER the object, so changing the object moves the
scene relative to a bench that is physically bolted down.
"""

import numpy as np
import rclpy
from tf2_ros import Buffer, TransformListener

#: The bench both descriptions carry. Not a profile field, because a second table
#: would be a second cell rather than a second robot.
TABLE_FRAME = "lbr_workspace_table_link"


def transform_to_matrix(transform):
    """geometry_msgs/TransformStamped -> 4x4. Quaternion by hand (w, x, y, z ->
    rotation matrix) so this module needs no scipy."""
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def matrix_to_pose(T, pose):
    """Fill a geometry_msgs/Pose from a 4x4. Quaternion by hand, same reason.

    Shepperd's method rather than the naive `w = sqrt(1 + trace)/2`: that form
    loses all precision when the trace approaches -1, which is an ordinary
    180-degree wrist and not an edge case worth getting wrong on the wire.
    """
    T = np.asarray(T, float)
    R = T[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w, x, y, z = (0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = ((R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s)

    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in T[:3, 3])
    pose.orientation.w = float(w)
    pose.orientation.x = float(x)
    pose.orientation.y = float(y)
    pose.orientation.z = float(z)
    return pose


def pose_to_matrix(pose):
    """geometry_msgs/Pose -> 4x4. The inverse of :func:`matrix_to_pose`."""
    q = pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm > 0.0:
        # A pose that crossed the wire is not guaranteed unit; an un-normalized
        # quaternion silently scales the rotation matrix, which reads downstream
        # as a wrist that is subtly the wrong size rather than as an error.
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return T


class FrameLookup:
    """One node's TF buffer, with the two lookups this stack ever makes.

    ``planning_frame`` and ``wrist_frame`` come from the robot profile. The wrist
    one is the SRDF chain tip, which is the frame MoveIt Servo's Jacobian is
    taken at AND the frame the solver's wrist variable is -- the two coincide by
    construction on both robots, and that coincidence is what lets a solved wrist
    pose be commanded with no mount conversion.
    """

    def __init__(self, node, planning_frame, wrist_frame,
                 table_frame=TABLE_FRAME):
        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, node)
        self._clock = node.get_clock()
        self.planning_frame = planning_frame
        self.wrist_frame = wrist_frame
        self.table_frame = table_frame

    def registration(self, corner_viz):
        """``T_base<-viz`` for a viser scene whose table corner is at ``corner_viz``."""
        try:
            transform = self._buffer.lookup_transform(
                self.planning_frame, self.table_frame, rclpy.time.Time())
        except Exception as exc:
            raise RuntimeError(
                f"no TF {self.planning_frame} <- {self.table_frame} ({exc}). The "
                f"robot description has to be up -- start the arm half of the "
                f"cell (gepetto_bringup.launch.py brings it up for you).") from exc
        shift = np.eye(4)
        shift[:3, 3] = -np.asarray(corner_viz, float).reshape(3)
        return transform_to_matrix(transform) @ shift

    def wrist_pose_base(self):
        """The hand's wrist frame in the planning frame, from TF."""
        return self.wrist_pose_base_stamped()[0]

    def wrist_pose_base_stamped(self):
        """``(T, age_seconds)``.

        The age matters in the control loop and nowhere else: `lookup_transform`
        with a zero time returns the LATEST transform available, which is a
        perfectly good answer right up until the publisher dies, at which point
        it keeps returning the same stale pose forever. Tracking error against a
        frozen measurement reads as an arm that is following perfectly.
        """
        try:
            transform = self._buffer.lookup_transform(
                self.planning_frame, self.wrist_frame, rclpy.time.Time())
        except Exception as exc:
            raise RuntimeError(
                f"no TF {self.planning_frame} <- {self.wrist_frame} ({exc}).") from exc
        stamp = rclpy.time.Time.from_msg(transform.header.stamp)
        age = (self._clock.now() - stamp).nanoseconds * 1e-9
        return transform_to_matrix(transform), age
