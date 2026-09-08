"""A fake arm + hand for exercising executor_node with no hardware and no MoveIt.

    ros2 run gepetto_ros fake_cell -- --hand epfl
    ros2 run gepetto_ros fake_cell -- --hand allegro --tau 0.15 --gain 0.2

The arm is the part that matters: it models the thing that caused the bug, a
first-order lag between the commanded twist and the actual motion (MoveIt Servo's
smoother, the fieldbus, the joint controller), plus a steady-state GAIN deficit.
With `--tau 0.15` a controller that does not correct cannot track at all.

`--freeze-at` pins the arm in place after N seconds, to exercise the watchdog.

EITHER HAND, and the hand half is modelled the way that hand's real node behaves:

* `epfl` -- an INTEGRATING, rate-limited jog node, like `finger_servo_node`:
  deltas accumulate into a setpoint, which then slews toward the goal;
* `allegro` -- an ABSOLUTE first-order chase, like `allegro_mock_node.py`: the
  reported position lags the commanded one and there is nothing to integrate.

That difference is deliberate. The two are not interchangeable, and a fake cell
that modelled only one of them would let a driver bug through -- an integrating
driver pointed at an absolute node commands the DIFFERENCE and barely moves; an
absolute driver pointed at an integrating one commands the TOTAL every tick and
runs away.

It publishes the same TF the real description does (`lbr_link_0` ->
the profile's wrist frame, and the bench), so the registration is exercised rather
than being an identity.
"""

import argparse
import math
import threading
import time

import numpy as np
import rclpy
from control_msgs.msg import JointJog
from geometry_msgs.msg import TransformStamped, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray
from tf2_ros import TransformBroadcaster

from gepetto_solvers.core.hands import get_hand

from .arm_servo import SERVO_SCALE_LINEAR, SERVO_SCALE_ROTATIONAL
from .frames import TABLE_FRAME
from .hand_drivers import allegro as allegro_driver
from .hand_drivers import tendon as tendon_driver
from .robot_profile import get_profile

COMMAND_TIMEOUT = 0.1       # MoveIt Servo's incoming_command_timeout


def rot_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s)
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                (R[1, 2] + R[2, 1]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s, 0.25 * s)


def expm_so3(w):
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


class _FakeTendonHand:
    """`finger_servo_node`, in miniature: integrate deltas, slew toward them."""

    def __init__(self, node, hand, publish_hz=5.0, step_hz=20.0):
        self._names = list(tendon_driver.actuator_names(hand).values())
        self._rate = float(hand.max_digit_speed)
        self._step_hz = step_hz
        self._at = {name: 0.0 for name in self._names}
        self._goal = {name: 0.0 for name in self._names}
        node.create_subscription(JointJog, tendon_driver.DEFAULT_COMMAND_TOPIC,
                                 self._on_cmd, 10)
        self._pub = node.create_publisher(
            JointState, tendon_driver.DEFAULT_STATE_TOPIC, 10)
        self._node = node
        node.create_timer(1.0 / step_hz, self.step)
        node.create_timer(1.0 / publish_hz, self.publish)

    def _on_cmd(self, msg):
        for name, delta in zip(msg.joint_names, msg.displacements):
            if name in self._goal:
                self._goal[name] += float(delta)

    def step(self):
        step = self._rate / self._step_hz
        for name in self._names:
            error = self._goal[name] - self._at[name]
            self._at[name] += max(-step, min(step, error))

    def publish(self):
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(self._names)
        msg.position = [self._at[n] for n in self._names]
        self._pub.publish(msg)


class _FakeAllegroHand:
    """`allegro_mock_node.py`, in miniature: absolute positions through a lag.

    Reads `position` POSITIONALLY and rejects anything under sixteen entries,
    exactly as the real driver does -- which is the half of the contract a mock
    most needs to reproduce, since getting it wrong on the real hand moves the
    wrong joints rather than failing.
    """

    #: Counts per metre of commanded overshoot past the object. Scaled so the
    #: default 300-count target lands a few millimetres in, which is about what a
    #: fingertip deflects on a real one.
    _COUNTS_PER_M = 4e4

    #: The real driver's validity clamp, verbatim from `AllegroHandDrv.cpp`:
    #: anything outside this reads as ZERO, and only then is filtered. Reproduced
    #: rather than approximated, because "a crushed fingertip reports no contact"
    #: is the single sharpest edge in the tactile close and a fake cell that
    #: smoothed it over would let the bug it guards against through.
    _SENSOR_VALID = (0, 5000)
    _SENSOR_ALPHA = 0.25

    def __init__(self, node, hand, publish_hz=100.0, tau=0.1,
                 contact_at=None, tactile_gain=1.0):
        from gepetto_solvers.core.hands.allegro import spec

        self._order = list(spec.DRIVER_JOINT_ORDER)
        self._digits = list(spec.DIGIT_NAMES)
        self._n = len(self._order)
        self._desired = np.zeros(self._n)
        self._current = np.zeros(self._n)
        self._tau = tau
        self._dt = 1.0 / publish_hz
        self._node = node

        #: How far each fingertip may TRAVEL from where it started before it runs
        #: into the virtual object, in metres; None for an empty hand.
        #:
        #: A DISTANCE, not a joint angle, and the first version got this wrong in
        #: a way worth recording. Blocking joints at a fixed angle looks
        #: equivalent and is not: the servo's step is minimum-norm, so it puts
        #: essentially all the motion into whichever single joint has the best
        #: lever arm -- j1 for a finger, j0 for the thumb -- and a threshold on
        #: "the flexion joints" therefore models an object the thumb can never
        #: touch. Fingertip distance is both what an operator means by "the
        #: object is 30 mm away" and the one quantity every digit shares.
        self._contact_at = contact_at
        self._tactile_gain = tactile_gain
        self._tactile = np.zeros(len(self._digits))
        #: Where each fingertip is blocked, as the joint vector it had when it
        #: first reached the object. The hand stays there while the COMMAND runs
        #: on past it, and that gap is what makes the reading -- which is exactly
        #: how a position-controlled hand generates force.
        self._blocked = [None] * len(self._digits)
        self._fk = hand.digit_fk() if contact_at is not None else None
        self._tip0 = (None if self._fk is None else
                      [self._fk.tip_position(name, np.zeros(4))
                       for name in self._digits])

        node.create_subscription(JointState,
                                 allegro_driver.DEFAULT_COMMAND_TOPIC,
                                 self._on_cmd, 1)
        # Published on the topic `lbr_allegro.launch.py` remaps the real driver's
        # own joint_states onto, so what the executor subscribes to is what it
        # would subscribe to in the assembled cell.
        self._pub = node.create_publisher(
            JointState, allegro_driver.DEFAULT_STATE_TOPIC, 3)
        self._tactile_pub = node.create_publisher(
            Int32MultiArray, allegro_driver.DEFAULT_TACTILE_TOPIC, 3)
        node.create_timer(self._dt, self.step)

    def _on_cmd(self, msg):
        if len(msg.position) < self._n:
            self._node.get_logger().warn(
                f"ignoring joint_cmd with {len(msg.position)} positions; the "
                f"driver reads {self._n} positionally.",
                throttle_duration_sec=5.0)
            return
        self._desired = np.asarray(msg.position[:self._n], float)

    def step(self):
        alpha = 1.0 if self._tau <= 0.0 else 1.0 - math.exp(-self._dt / self._tau)
        # Where the object lets the finger get to, which is not where it was
        # told to go. `_desired` is untouched: the gap between the two IS the
        # contact force, and collapsing them would make the fake hand a thing
        # that can never be pressed against.
        reachable, raw = self._contact()
        self._current = self._current + alpha * (reachable - self._current)

        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(self._order)
        msg.position = [float(v) for v in self._current]
        self._pub.publish(msg)
        self._publish_tactile(raw)

    def _contact(self):
        """``(reachable joint vector, raw counts per digit)``.

        The object is a sphere of travel about each fingertip's starting point:
        a finger may move ``contact_at`` metres and no further, and everything
        the command asks for past that turns into a reading instead of motion.
        """
        reachable = self._desired.copy()
        raw = np.zeros(len(self._digits))
        if self._contact_at is None:
            return reachable, raw

        for d, name in enumerate(self._digits):
            block = slice(4 * d, 4 * d + 4)
            want = self._desired[block]
            travel = float(np.linalg.norm(
                self._fk.tip_position(name, want) - self._tip0[d]))
            if travel <= self._contact_at:
                self._blocked[d] = None
                continue
            if self._blocked[d] is None:
                # First tick past the surface: the finger stops HERE. Held from
                # now on, so the reading grows with the command rather than with
                # a posture that is no longer changing.
                self._blocked[d] = self._current[block].copy()
            reachable[block] = self._blocked[d]
            raw[d] = (self._tactile_gain * self._COUNTS_PER_M
                      * (travel - self._contact_at))
        return reachable, raw

    def _publish_tactile(self, raw):
        lo, hi = self._SENSOR_VALID
        for i, value in enumerate(raw):
            # THE DRIVER'S OWN ORDER OF OPERATIONS: clamp to zero first, filter
            # second. Out of range therefore DECAYS toward "no contact" over
            # about a dozen samples rather than pinning high, which is the
            # behaviour `core.solvers.grasp_servo`'s collapse detector exists for
            # and the only way to exercise it without a real fingertip.
            value = 0.0 if (value < lo or value > hi) else value
            self._tactile[i] = (self._SENSOR_ALPHA * value
                                + (1.0 - self._SENSOR_ALPHA) * self._tactile[i])

        out = Int32MultiArray()
        out.data = [int(v) for v in self._tactile]
        self._tactile_pub.publish(out)


HANDS = {"epfl": _FakeTendonHand, "allegro": _FakeAllegroHand}


class FakeCell(Node):

    def __init__(self, profile, hand, tau, freeze_at, gain=1.0,
                 hand_kwargs=None):
        super().__init__("fake_cell")
        self._hand_kwargs = dict(hand_kwargs or {})
        self._profile = profile
        self._tau = tau
        self._gain = gain
        self._freeze_at = freeze_at
        self._started = time.monotonic()

        # Wrist starts a little above the table corner, arbitrary but reachable.
        self._p = np.array([0.20, 0.10, 0.30])
        self._R = np.eye(3)
        self._v = np.zeros(3)       # actual (lagged)
        self._w = np.zeros(3)
        self._v_cmd = np.zeros(3)   # commanded (instant)
        self._w_cmd = np.zeros(3)
        self._last_cmd = 0.0
        self._lock = threading.Lock()

        self.create_subscription(TwistStamped,
                                 "/lbr/servo_node/delta_twist_cmds",
                                 self._on_twist, 10)
        # Only the Allegro models a fingertip sensor, so only it takes the
        # tactile knobs. Passed by keyword and filtered by the hand's own
        # signature rather than branched on here, so a second sensing hand
        # needs no edit at this line.
        import inspect
        accepted = inspect.signature(HANDS[profile.name]).parameters
        self._hand = HANDS[profile.name](
            self, hand,
            **{k: v for k, v in self._hand_kwargs.items() if k in accepted})
        self._tf = TransformBroadcaster(self)

        self._dt = 1.0 / 200.0
        self.create_timer(self._dt, self._step)
        self.get_logger().info(
            f"fake cell ({profile.name}): pipeline tau={tau:.2f} s, "
            f"gain={gain:.2f}, wrist frame {profile.wrist_frame}"
            + (f", freezing at t={freeze_at:.1f} s" if freeze_at else ""))

    def _on_twist(self, msg):
        with self._lock:
            self._v_cmd = np.array([msg.twist.linear.x, msg.twist.linear.y,
                                    msg.twist.linear.z]) * SERVO_SCALE_LINEAR
            self._w_cmd = np.array([msg.twist.angular.x, msg.twist.angular.y,
                                    msg.twist.angular.z]) * SERVO_SCALE_ROTATIONAL
            self._last_cmd = time.monotonic()

    def _step(self):
        now = time.monotonic()
        with self._lock:
            # Just like MoveIt Servo: a publisher that stops is an arm that stops.
            if now - self._last_cmd > COMMAND_TIMEOUT:
                self._v_cmd = np.zeros(3)
                self._w_cmd = np.zeros(3)
            frozen = (self._freeze_at
                      and now - self._started > self._freeze_at)
            alpha = self._dt / max(self._tau, 1e-6)
            # `gain` is a STEADY-STATE deficit, and it is a different fault from
            # `tau`. A lag makes the arm late and it still arrives; a gain below
            # 1.0 means the arm never travels as fast as it is told, however long
            # it is given, so a path timed at the full ceiling can never be
            # followed at any controller gain -- the command is already clamped
            # at the servo's own scale. That is the shape of the real pipeline
            # (Servo re-basing + FRI smoothing + a publish-rate mismatch), and it
            # is invisible to a lag-only model, which is why it is modelled here.
            self._v += (self._v_cmd * self._gain - self._v) * min(alpha, 1.0)
            self._w += (self._w_cmd * self._gain - self._w) * min(alpha, 1.0)
            if frozen:
                self._v[:] = 0.0
                self._w[:] = 0.0
            self._p = self._p + self._v * self._dt
            self._R = expm_so3(self._w * self._dt) @ self._R
            p, R = self._p.copy(), self._R.copy()

        stamp = self.get_clock().now().to_msg()
        self._send_tf(stamp, self._profile.planning_frame,
                      self._profile.wrist_frame, p, R)
        # The bench, bolted down. Yawed +90 deg like the real URDF joint, so the
        # registration is exercised rather than being an identity.
        table_R = expm_so3(np.array([0.0, 0.0, math.pi / 2.0]))
        self._send_tf(stamp, self._profile.planning_frame, TABLE_FRAME,
                      np.array([0.35, -0.20, 0.0]), table_R)

    def _send_tf(self, stamp, parent, child, p, R):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(p[0])
        t.transform.translation.y = float(p[1])
        t.transform.translation.z = float(p[2])
        w, x, y, z = rot_to_quat(R)
        t.transform.rotation.w = float(w)
        t.transform.rotation.x = float(x)
        t.transform.rotation.y = float(y)
        t.transform.rotation.z = float(z)
        self._tf.sendTransform(t)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="allegro",
                        help="robot profile to fake. Decides the wrist frame and "
                             "which hand node is modelled.")
    parser.add_argument("--gain", type=float, default=1.0,
                        help="steady-state fraction of commanded Cartesian speed "
                             "the arm actually realises. 1.0 is a perfect "
                             "actuator behind a lag; the real pipeline is nearer "
                             "0.2 and that is what makes a path finish short.")
    parser.add_argument("--tau", type=float, default=0.15,
                        help="servo pipeline lag, seconds")
    parser.add_argument("--freeze-at", type=float, default=0.0,
                        help="pin the arm after N seconds (stall test)")
    parser.add_argument("--tactile-contact-at", type=float, default=None,
                        metavar="METRES",
                        help="Allegro only. How far each fingertip may travel "
                             "before it runs into the virtual object: past it "
                             "the finger stops while the command keeps "
                             "advancing, which is how a position-controlled "
                             "hand makes force. Omit for an EMPTY hand, which "
                             "is the no-contact timeout test; 0.02 is an object "
                             "well within reach of a 50 mm budget.")
    parser.add_argument("--tactile-gain", type=float, default=1.0,
                        help="Stiffness multiplier. 1.0 puts a 300-count "
                             "reading a few millimetres into the object. Raise "
                             "it to trip the over-force abort sooner; raise it "
                             "far enough that the TRUE reading passes 5000 and "
                             "the sensor goes out of range, which this cell "
                             "reproduces faithfully -- the driver reports zero "
                             "for that, not a maximum.")
    args, ros_args = parser.parse_known_args(argv)

    profile = get_profile(args.hand)
    hand = get_hand(profile.solver_hand)

    rclpy.init(args=ros_args)
    node = FakeCell(profile, hand, args.tau, args.freeze_at, args.gain,
                    hand_kwargs=dict(contact_at=args.tactile_contact_at,
                                     tactile_gain=args.tactile_gain))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
