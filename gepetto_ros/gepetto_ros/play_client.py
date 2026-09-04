"""Send one PlayPlan goal to executor_node and report what happened.

    ros2 run gepetto_ros play_client -- --hand epfl
    ros2 run gepetto_ros play_client -- --hand allegro --speed 0.3
    ros2 run gepetto_ros play_client -- --hand-only     # bring-up: no arm involved

Builds a plan BY HAND -- no solver, no viser, no browser -- that sweeps the wrist
across the table and curls the digits, which is the shape a real solve produces.
That is the point of the tool: it exercises the whole command path (action,
executor, control loop, drivers, servo, hand node) with nothing upstream of it, so
a failure here is unambiguously in the ROS half.

WHICH HAND IT BUILDS FOR is `--hand`, and it matters: the plan carries the digit
names, the width and the `command_kind` of the profile's hand, and the executor
REFUSES a goal whose kind does not match the hand it was started with. Sending a
tendon plan to an Allegro executor is caught before the arm moves.

The digit command it ramps is a fraction of the hand's own travel, so it is 8 mm
of tendon on one hand and a gentle curl in radians on the other, without this file
holding either number.
"""

import argparse
import math
import threading
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool

from gepetto_ros_interfaces.action import PlayPlan
from gepetto_solvers.core import robot_plan
from gepetto_solvers.core.hands import get_hand

from .arm_servo import SERVO_SCALE_LINEAR, SERVO_SCALE_ROTATIONAL
from .frames import matrix_to_pose
from .robot_profile import get_profile
from .topics import ESTOP_QOS, ESTOP_TOPIC, PLAY_PLAN_ACTION
from .wire import plan_to_goal

#: How far into its travel each digit is driven, as a fraction. Well short of the
#: stop on either hand -- this is a path shape, not a grasp.
CURL_FRACTION = 0.45


def expm_so3(w):
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _target(hand):
    """``{digit: (K,) vector}`` the sweep ramps toward, from the hand's travel.

    Read off ``robot_plan.travel_limits`` rather than typed in, so the sweep is
    inside what the executor will clamp to on either hand and so this file states
    no per-robot number. Falls back to a small absolute curl if the limits are
    unavailable -- the tool must still run on a machine that cannot see them.
    """
    width = len(hand.actuation.drive_indices)
    limits = robot_plan.travel_limits(hand)
    out = {}
    for name in hand.digit_names:
        bounds = None if limits is None else limits.get(name)
        if bounds is None:
            out[name] = np.full(width, 0.008 if width == 1 else 0.3)
            continue
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
        # From the ZERO command toward the upper stop. Zero is the tendon hand's
        # rest, so on it the ramp starts where a parked hand already is; on a
        # joint-space hand it is the URDF neutral rather than where `home_node`
        # parks (see its docstring), so the first sample uncurls before the sweep
        # -- harmless in a bring-up tool that is deliberately commanding a
        # straight line through the whole range.
        out[name] = np.clip(hi * CURL_FRACTION, lo, hi)
    return out


def build_plan(hand, waypoints=12, rotation_deg=40.0):
    """A sweep in the VISER frame: 25 cm of travel, some rotation, digits curling.
    Shaped like an AL iterate history -- dense, monotonic.

    `rotation_deg` decides WHICH CHANNEL SETS THE PACE, and that is the whole
    difference between reproducing the reported bug and not. With 40 deg the
    segments are angular-limited, so the linear command has slack even when its
    ceiling equals the path speed. With 0 they are linear-limited: the
    feed-forward alone saturates the clamp, which is the condition under which a
    proportional term with no headroom cannot correct anything.
    """
    target = _target(hand)
    waypoints_out = []
    for i in range(waypoints):
        s = i / (waypoints - 1)
        T = np.eye(4)
        T[:3, :3] = expm_so3(np.array([0.0, math.radians(rotation_deg) * s, 0.0]))
        T[:3, 3] = np.array([0.05 + 0.25 * s, 0.10,
                             0.15 + 0.05 * math.sin(math.pi * s)])
        waypoints_out.append(robot_plan.Waypoint(
            wrist_pose=T,
            digit_cmd={name: value * s for name, value in target.items()},
            note=f"iterate {i}"))
    return robot_plan.SolvePlan(
        waypoints=waypoints_out,
        corner_viz=np.zeros(3),
        digit_names=list(hand.digit_names),
        open_lengths={},
        command_kind=robot_plan.command_kind(hand))


def build_goal(hand, speed, enable_arm=True, enable_hand=True, rotation_deg=40.0):
    goal = plan_to_goal(build_plan(hand, rotation_deg=rotation_deg),
                        PlayPlan.Goal(), matrix_to_pose)
    goal.enable_arm = bool(enable_arm)
    goal.enable_hand = bool(enable_hand)
    # One fraction across all three ceilings, which is what the visualizer's
    # single playback-speed slider sends. Scaling them apart cannot desynchronize
    # the channels -- they share a per-segment duration -- it only changes which
    # one is the slowest, so there was never a second number to set.
    goal.max_linear = speed * SERVO_SCALE_LINEAR
    goal.max_angular = speed * SERVO_SCALE_ROTATIONAL
    goal.max_digit = speed * hand.max_digit_speed
    return goal


class PlayClient(Node):

    def __init__(self):
        super().__init__("play_client")
        self.client = ActionClient(self, PlayPlan, PLAY_PLAN_ACTION)
        self.estop = self.create_publisher(Bool, ESTOP_TOPIC, ESTOP_QOS)
        self.worst_error = 0.0
        self.worst_rotation = 0.0
        self.hold_ticks = 0
        self.path_total = 0.0
        self.feedback_count = 0
        self.last_feedback = 0.0
        self.gaps = []

    def on_feedback(self, message):
        fb = message.feedback
        now = time.monotonic()
        if self.last_feedback:
            self.gaps.append(now - self.last_feedback)
        self.last_feedback = now
        self.feedback_count += 1
        self.worst_error = max(self.worst_error, fb.position_error)
        self.worst_rotation = max(self.worst_rotation, fb.rotation_error)
        self.path_total = fb.path_total
        if fb.settling:
            self.hold_ticks += 1
        tag = "holding" if fb.settling else f"wp {fb.waypoint}/{fb.num_waypoints}"
        print(f"    t={fb.path_time:5.2f}/{fb.path_total:5.2f}  {tag:>18}  "
              f"err {fb.position_error * 1e3:6.1f} mm / "
              f"{math.degrees(fb.rotation_error):5.1f}deg")


def run(node, goal, estop_after=0.0, label=""):
    print(f"\n--- {label} ---")
    if not node.client.wait_for_server(timeout_sec=10.0):
        raise SystemExit(f"no action server on {PLAY_PLAN_ACTION}")

    node.worst_error = 0.0
    node.worst_rotation = 0.0
    node.hold_ticks = 0
    node.path_total = 0.0
    node.feedback_count = 0
    node.last_feedback = 0.0
    node.gaps = []

    send = node.client.send_goal_async(goal, feedback_callback=node.on_feedback)
    rclpy.spin_until_future_complete(node, send)
    handle = send.result()
    if not handle.accepted:
        print("  goal REJECTED -- check the executor's log; it says why. The "
              "usual cause is a hand mismatch: this client and the executor "
              "were started with different `--hand` / `-p hand:=`.")
        return None

    if estop_after:
        def trip():
            time.sleep(estop_after)
            print(f"  >>> E-STOP at t+{estop_after:.1f}s (topic + cancel)")
            msg = Bool()
            msg.data = True
            node.estop.publish(msg)
            handle.cancel_goal_async()
        threading.Thread(target=trip, daemon=True).start()

    started = time.monotonic()
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future, timeout_sec=120.0)
    elapsed = time.monotonic() - started
    result = result_future.result().result

    print(f"  verdict            : {result.verdict}")
    print(f"  success            : {result.success}")
    print(f"  ticks / elapsed    : {result.ticks} / {result.elapsed:.1f}s "
          f"(wall {elapsed:.1f}s)")
    print(f"  final error        : {result.final_position_error * 1e3:.2f} mm / "
          f"{math.degrees(result.final_rotation_error):.2f} deg")
    print(f"  resumable          : {result.resumable} at waypoint "
          f"{result.resume_waypoint} (t={result.resume_path_time:.2f})")
    print(f"  worst error seen   : {node.worst_error * 1e3:.1f} mm / "
          f"{math.degrees(node.worst_rotation):.1f} deg")
    # THE headline check. The clock is free-running, so a completed run must take
    # the path's own duration plus the fixed terminal hold and nothing else --
    # whatever the plant is doing. An overrun here means the clock is being
    # steered by tracking again, which is the bug this design removed.
    if result.success and node.path_total:
        print(f"  duration vs plan   : wall {elapsed:.2f}s against a "
              f"{node.path_total:.2f}s path + hold "
              f"(overrun {elapsed - node.path_total:+.2f}s -- should be just "
              f"the terminal hold)")
    if node.gaps:
        print(f"  feedback jitter    : mean {np.mean(node.gaps) * 1e3:.0f} ms, "
              f"max {np.max(node.gaps) * 1e3:.0f} ms")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="allegro",
                        help="robot profile to build the plan for. MUST match "
                             "the executor's own `-p hand:=`, which refuses a "
                             "plan in the wrong units.")
    parser.add_argument("--speed", type=float, default=0.5,
                        help="playback speed as a fraction of every channel's own "
                             "maximum -- the same single number the visualizer's "
                             "slider sets")
    # Per-channel playback lives HERE and not in the visualizer, whose Robot
    # folder has one arming checkbox and always drives both. Half a coordinated
    # trajectory is a bring-up tool, not an operating mode: pair --hand-only with
    # a hand node in dry-run or mock mode to exercise the whole command path with
    # no arm involved.
    channels = parser.add_mutually_exclusive_group()
    channels.add_argument("--arm-only", action="store_true",
                          help="send wrist waypoints, leave the hand alone")
    channels.add_argument("--hand-only", action="store_true",
                          help="send digit waypoints, leave the arm alone")
    parser.add_argument("--estop-after", type=float, default=0.0)
    parser.add_argument("--label", default="play")
    parser.add_argument("--clear-estop", action="store_true")
    parser.add_argument("--rotation-deg", type=float, default=40.0)
    args, ros_args = parser.parse_known_args(argv)

    profile = get_profile(args.hand)
    hand = get_hand(profile.solver_hand)
    print(f"building a {robot_plan.command_kind(hand)} plan for "
          f"{hand.name}: {len(hand.digit_names)} digits x "
          f"{len(hand.actuation.drive_indices)}")

    rclpy.init(args=ros_args)
    node = PlayClient()
    try:
        if args.clear_estop:
            msg = Bool()
            msg.data = False
            node.estop.publish(msg)
            time.sleep(0.5)
        run(node, build_goal(hand, args.speed,
                             enable_arm=not args.hand_only,
                             enable_hand=not args.arm_only,
                             rotation_deg=args.rotation_deg),
            estop_after=args.estop_after, label=args.label)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
