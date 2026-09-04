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

    def __init__(self, node, hand, publish_hz=100.0, tau=0.1):
        from gepetto_solvers.core.hands.allegro import spec

        self._order = list(spec.DRIVER_JOINT_ORDER)
        self._n = len(self._order)
        self._desired = np.zeros(self._n)
        self._current = np.zeros(self._n)
        self._tau = tau
        self._dt = 1.0 / publish_hz
        self._node = node
        node.create_subscription(JointState,
                                 allegro_driver.DEFAULT_COMMAND_TOPIC,
                                 self._on_cmd, 1)
        # Published on the topic `lbr_allegro.launch.py` remaps the real driver's
        # own joint_states onto, so what the executor subscribes to is what it
        # would subscribe to in the assembled cell.
        self._pub = node.create_publisher(
            JointState, allegro_driver.DEFAULT_STATE_TOPIC, 3)
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
        self._current = self._current + alpha * (self._desired - self._current)
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(self._order)
        msg.position = [float(v) for v in self._current]
        self._pub.publish(msg)


HANDS = {"epfl": _FakeTendonHand, "allegro": _FakeAllegroHand}


class FakeCell(Node):

    def __init__(self, profile, hand, tau, freeze_at, gain=1.0):
        super().__init__("fake_cell")
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
        self._hand = HANDS[profile.name](self, hand)
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
    args, ros_args = parser.parse_known_args(argv)

    profile = get_profile(args.hand)
    hand = get_hand(profile.solver_hand)

    rclpy.init(args=ros_args)
    node = FakeCell(profile, hand, args.tau, args.freeze_at, args.gain)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
