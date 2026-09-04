"""Tick-driven wrappers around the ARM's two servo command topics.

These are `lbr_motion`'s `twist_servo_pub` / `joint_servo_pub` with their fixed
demo payloads replaced by a `tick()` / `command()` the caller drives. Nothing here
owns a timer or a thread: the caller runs the control loop and calls once per
period, which is what lets one loop drive the arm and the hand in lockstep instead
of two timers drifting against each other.

THE HAND IS NOT HERE. It used to be, alongside these, when there was one hand;
now the hand end is a driver behind :mod:`gepetto_ros.hand_drivers`, because the
EPFL hand integrates delta jogs in metres of tendon while the Allegro takes
absolute joint positions in radians, and no single class was going to be both.
The arm, by contrast, is genuinely identical for the two robots -- same servo
node, same scales, same topics -- so it is one file with no profile in it.

UNITS:

  ArmTwistServo   `/lbr/servo_node/delta_twist_cmds`, TwistStamped in the
                  planning frame. MoveIt Servo runs `command_in_type: unitless`,
                  so a component of 1.0 means `scale.linear` (0.4 m/s) or
                  `scale.rotational` (0.8 rad/s). This class takes REAL m/s and
                  rad/s and does the division, so callers never hold the yaml
                  file in their head.
  ArmJointServo   `/lbr/servo_node/delta_joint_cmds`, JointJog in rad, likewise
                  divided by `scale.joint` (0.5).

Both stop the same way: stop publishing. The arm halts on MoveIt Servo's
`incoming_command_timeout` (0.1 s). `halt()` makes that immediate and explicit
rather than waiting out the timeout.
"""

import numpy as np
from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Header

# lbr_bringup/config/moveit_servo.yaml, and moveit_servo_allegro.yaml, which
# agree on all three. Duplicated here rather than read from the parameter server
# because servo_node owns them as its own private parameters and a wrong guess is
# a silently mis-scaled command; they are exposed as constructor arguments so a
# retune is a one-line change at the call site.
SERVO_SCALE_LINEAR = 0.4        # m/s at a twist component of 1.0
SERVO_SCALE_ROTATIONAL = 0.8    # rad/s
SERVO_SCALE_JOINT = 0.5         # rad

#: The arm's own base. Both descriptions plan in it, so unlike the wrist frame it
#: is not a per-robot fact -- but it is still passed in from the profile, so that
#: a cell with a differently-named base needs no edit here.
DEFAULT_PLANNING_FRAME = "lbr_link_0"


def _clamp_norm(vector, limit):
    """Scale ``vector`` down to ``limit`` if it is longer, preserving DIRECTION.

    Clamping each component independently would bend the commanded velocity away
    from the straight line to the target -- the arm would visibly cut a corner
    whenever one axis saturated."""
    norm = float(np.linalg.norm(vector))
    if norm <= limit or norm < 1e-12:
        return np.asarray(vector, float)
    return np.asarray(vector, float) * (limit / norm)


class ArmTwistServo:
    """The Cartesian twist channel: clamp a commanded twist and publish it.

    The frame being servoed is the SRDF chain tip of group "arm" -- which on BOTH
    robots is the solver's own wrist frame: `lbr_epfl_wrist_link` on `med14_epfl`
    (which the URDF places at exactly `MountConfig.flange_from_wrist_*`), and
    `palm_link` on `med14_allegro` (which `AllegroHand.mount_pose` was measured
    against with `tf2_echo lbr_link_ee palm_link`). So a wrist pose out of the
    solver is commandable here with no mount conversion, on either robot, and
    that equality is the whole reason this integration is simple.

    `frame_id` is only the frame the twist COMPONENTS are expressed in; it does
    not change which point moves.
    """

    def __init__(self, node, topic="/lbr/servo_node/delta_twist_cmds",
                 frame_id=DEFAULT_PLANNING_FRAME, max_linear=0.2, max_angular=0.4,
                 scale_linear=SERVO_SCALE_LINEAR,
                 scale_rotational=SERVO_SCALE_ROTATIONAL):
        self._node = node
        self._frame_id = frame_id
        self._pub = node.create_publisher(TwistStamped, topic, 10)
        self.max_linear = max_linear
        self.max_angular = max_angular
        self._scale_linear = scale_linear
        self._scale_rotational = scale_rotational
        #: Whether the last `command` hit the direction-preserving clamp, and
        #: whether `_publish` then had to clip a component against the servo's
        #: own unitless scale. Diagnostics only; nothing reads these to decide.
        self.clamped_linear = False
        self.clamped_angular = False
        self.clipped_component = False

    def command(self, v, w):
        """Clamp one already-resolved twist to the speed ceilings and publish it.

        ``v`` (m/s) and ``w`` (rad/s) are the linear velocity of the WRIST ORIGIN
        and the angular velocity of the wrist, both with components expressed in
        `frame_id`.

        THE CONTROL LAW IS NOT HERE. It lives in `trajectory_executor`, which is
        the module that owns the loop and which already has the se(3) helpers
        `robot_plan` provides. Keeping it there rather than duplicating that
        algebra in this file is deliberate: two independently maintained copies of
        a matrix logarithm is exactly the kind of drift that shows up as an arm
        moving slightly wrongly rather than as an error.

        `max_linear` / `max_angular` MUST BE SET ABOVE THE PATH'S OWN SPEED, or
        the correction the caller computed is decorative: a path timed to travel
        at `max_linear` arrives here already at that norm, and `_clamp_norm` then
        scales the sum straight back down, leaving the correction able only to
        tilt a saturated command by a few degrees. It can never close a gap, so
        the pipeline's inherent lag accumulates until a watchdog fires -- in
        proportion to the commanded speed, which is why slowing down used to look
        like a fix. `trajectory_executor` sets these to `velocity_headroom` times
        the path speed, capped at the scales below: above them `_publish` clips
        per component, which would bend the direction the clamp preserved.
        """
        v = np.asarray(v, float)
        w = np.asarray(w, float)
        v_out = _clamp_norm(v, self.max_linear)
        w_out = _clamp_norm(w, self.max_angular)
        # Reported so the caller can say how much of a run was spent against the
        # ceiling. A command that is clamped most of the time is not a tracking
        # problem to be tuned -- it is the arm being forbidden from moving as
        # fast as the correction is asking for, which no gain can fix.
        self.clamped_linear = bool(np.linalg.norm(v) > self.max_linear + 1e-12)
        self.clamped_angular = bool(np.linalg.norm(w) > self.max_angular + 1e-12)
        self._publish(v_out, w_out)

    def halt(self):
        """Command zero. Publishing stops after this; the 0.1 s
        `incoming_command_timeout` would have done the same thing a beat later."""
        self._publish(np.zeros(3), np.zeros(3))

    def _publish(self, v, w):
        msg = TwistStamped()
        header = Header()
        header.stamp = self._node.get_clock().now().to_msg()
        header.frame_id = self._frame_id
        msg.header = header
        # "unitless": each component is a FRACTION of the configured scale, so
        # real units are divided out here and clipped to the [-1, 1] the servo
        # accepts. Clipping per component is correct at this stage -- the
        # direction-preserving clamp already happened, in real units.
        unit_v = np.asarray(v, float) / self._scale_linear
        unit_w = np.asarray(w, float) / self._scale_rotational
        # A component outside [-1, 1] here is the servo's own scale refusing the
        # command, AFTER the direction-preserving clamp has already run -- so it
        # bends the commanded direction. Worth knowing about; it means the clamp
        # limits above are set higher than the servo can express.
        self.clipped_component = bool(np.any(np.abs(unit_v) > 1.0)
                                      or np.any(np.abs(unit_w) > 1.0))
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = \
            np.clip(unit_v, -1.0, 1.0)
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = \
            np.clip(unit_w, -1.0, 1.0)
        self._pub.publish(msg)


class ArmJointServo:
    """Proportional joint-space servoing, for moves that must not think about IK.

    Home is a joint configuration, not a pose: it is defined by where the arm is
    folded, it is reachable by construction, and driving to it in joint space
    cannot stall in a singularity or pick a different elbow than the one it
    started with. That is why `home_node` uses this rather than the twist servo.
    """

    def __init__(self, node, joint_names, topic="/lbr/servo_node/delta_joint_cmds",
                 frame_id=DEFAULT_PLANNING_FRAME, max_rate=0.4, kp=1.5,
                 scale_joint=SERVO_SCALE_JOINT):
        self._node = node
        self._frame_id = frame_id
        self.joint_names = list(joint_names)
        self._pub = node.create_publisher(JointJog, topic, 10)
        self.max_rate = max_rate
        self.kp = kp
        self._scale_joint = scale_joint

    def tick(self, target, current):
        """One command toward ``target`` (rad, in ``joint_names`` order).
        Returns the largest absolute joint error."""
        error = np.asarray(target, float) - np.asarray(current, float)
        rate = np.clip(self.kp * error, -self.max_rate, self.max_rate)
        self._publish(rate)
        return float(np.max(np.abs(error))) if error.size else 0.0

    def halt(self):
        self._publish(np.zeros(len(self.joint_names)))

    def _publish(self, rate):
        msg = JointJog()
        header = Header()
        header.stamp = self._node.get_clock().now().to_msg()
        header.frame_id = self._frame_id
        msg.header = header
        msg.joint_names = list(self.joint_names)
        # Velocities, not displacements: MoveIt Servo treats delta_joint_cmds
        # velocities as fractions of scale.joint, and a rate is what a
        # proportional controller naturally produces.
        msg.velocities = [float(v) for v in np.clip(
            np.asarray(rate, float) / self._scale_joint, -1.0, 1.0)]
        msg.duration = 0.0
        self._pub.publish(msg)
