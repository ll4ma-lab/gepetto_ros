"""Send one tactile close to `executor_node`, with no visualizer in the loop.

    ros2 run gepetto_ros grasp_client -- --force 300
    ros2 run gepetto_ros grasp_client -- --release

The bring-up tool for the close, and the counterpart of `play_client`: an ordinary
console script that pokes a LIVE graph, because what it exercises is ROS plumbing
and there is nothing in it worth a unit test that the pure-numpy suite in
`gepetto_solvers/tests/core/test_grasp_servo.py` does not already cover better.

WHAT IT IS FOR is checking the half that suite cannot: that the action encodes and
decodes, that the executor's refusals fire on the right conditions, that the
tactile subscription reaches the hand, and that each stop reason can actually be
provoked end to end. Against `fake_cell` every one of them can be:

    # a normal close onto an object 20 mm away; each digit stops on `contact`
    ros2 run gepetto_ros fake_cell -- --hand allegro --tactile-contact-at 0.02
    ros2 run gepetto_ros grasp_client -- --force 300

    # nothing in the hand: every digit must give up at the travel budget
    ros2 run gepetto_ros fake_cell -- --hand allegro
    ros2 run gepetto_ros grasp_client -- --force 300

    # an object stiff enough to overshoot the target inside one tick, which is
    # what it takes to reach the ceiling -- a compliant one just stops on target
    ros2 run gepetto_ros fake_cell -- --hand allegro --tactile-contact-at 0.02 \
        --tactile-gain 150
    ros2 run gepetto_ros grasp_client -- --force 1200

    # past the sensor's range, where the driver reports ZERO rather than a
    # maximum -- the failure the collapse detector exists for
    ros2 run gepetto_ros fake_cell -- --hand allegro --tactile-contact-at 0.02 \
        --tactile-gain 3000
    ros2 run gepetto_ros grasp_client -- --force 1000

    # the release, and the refusal when nothing has been closed yet
    ros2 run gepetto_ros grasp_client -- --release

    # the stop, mid-close: the fingers must freeze where they are
    ros2 run gepetto_ros grasp_client -- --force 300 --estop-after 1.5

THE DIRECTIONS ARE SYNTHETIC HERE. A real close takes them from the solve's
h_grasp,E normals; this one drives every finger along +X in the wrist frame,
which is the direction Allegro's fingers FLEX in (they extend along +z at the
identity wrist -- see `AllegroHand.DEFAULT_WRIST_RPY`'s note). It is not a check
of the normals, which is what the workbench is for, so do not read a close that
works here as a grasp that will work there.

Do not "fix" it to -Z. That points along an outstretched finger's own axis, where
the fingertip Jacobian is singular and the damped least squares correctly
delivers nothing -- which is a real thing to be able to provoke (the servo stops
such a digit on `cannot move`) but is not a close.
"""

import argparse
import threading

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Bool

from gepetto_ros_interfaces.action import GraspClose
from gepetto_solvers.core.hands import get_hand
from gepetto_solvers.core.solvers import pack_vectors

from .robot_profile import get_profile
from .topics import ESTOP_QOS, ESTOP_TOPIC, GRASP_CLOSE_ACTION


class GraspClient(Node):

    def __init__(self):
        super().__init__("gepetto_grasp_client")
        self._client = ActionClient(self, GraspClose, GRASP_CLOSE_ACTION)
        self._estop_pub = self.create_publisher(Bool, ESTOP_TOPIC, ESTOP_QOS)
        self._done = threading.Event()
        self.result = None

    def clear_estop(self):
        msg = Bool()
        msg.data = False
        self._estop_pub.publish(msg)

    def trip_estop(self):
        msg = Bool()
        msg.data = True
        self._estop_pub.publish(msg)
        self.get_logger().warn("E-STOP published")

    def send(self, goal):
        if not self._client.wait_for_server(timeout_sec=5.0):
            raise SystemExit(
                f"no grasp server on {GRASP_CLOSE_ACTION}. Start it with "
                f"`ros2 run gepetto_ros executor_node --ros-args -p "
                f"hand:=allegro`.")
        future = self._client.send_goal_async(
            goal, feedback_callback=self._on_feedback)
        future.add_done_callback(self._on_sent)

    def _on_feedback(self, message):
        fb = message.feedback
        self.get_logger().info(
            f"  {fb.digits_stopped}/{fb.num_digits} stopped  "
            + "  ".join(f"{t * 1e3:5.1f}mm {c:5.0f}c"
                        for t, c in zip(fb.travel, fb.contact_counts))
            + f"   {fb.note}")

    def _on_sent(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error(
                "goal REFUSED -- check the executor's log, it says which "
                "condition rejected it")
            self._done.set()
            return
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future):
        self.result = future.result().result
        self._done.set()

    def wait(self, timeout=120.0):
        return self._done.wait(timeout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand", default="allegro",
                        help="must match the executor's `-p hand:=`")
    parser.add_argument("--digits", nargs="*", default=None,
                        help="which digits to close. Defaults to this hand's "
                             "own default contact set (the precision tripod).")
    parser.add_argument("--force", type=float, default=300.0,
                        help="per-digit target, in RAW tactile counts above "
                             "each fingertip's resting reading. Not newtons: "
                             "nothing in this stack calibrates these.")
    parser.add_argument("--speed", type=float, default=0.01,
                        help="fingertip travel along the direction, m/s")
    parser.add_argument("--max-travel", type=float, default=0.05,
                        help="give up on a digit past this, in metres")
    parser.add_argument("--abort-counts", type=float, default=1500.0,
                        help="raw ceiling; a digit at or past it stops at once")
    parser.add_argument("--direction", nargs=3, type=float,
                        default=(1.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                        help="close direction in the WRIST frame, same for "
                             "every digit. The default is the way this hand's "
                             "fingers flex; `0 0 -1` points along an "
                             "outstretched finger's own axis and provokes the "
                             "singular `cannot move` stop instead.")
    parser.add_argument("--release", action="store_true",
                        help="back out to the posture the last close started "
                             "from, instead of closing")
    parser.add_argument("--estop-after", type=float, default=0.0,
                        metavar="SECONDS",
                        help="trip the cell e-stop mid-close, to check the "
                             "close stops within a tick and the fingers hold")
    parser.add_argument("--clear-estop", action="store_true",
                        help="publish a clear before sending, for a cell left "
                             "latched by an earlier run")
    args, ros_args = parser.parse_known_args(argv)

    profile = get_profile(args.hand)
    hand = get_hand(profile.solver_hand)
    names = list(args.digits if args.digits else hand.default_contact_digits)

    goal = GraspClose.Goal()
    goal.digit_names = names
    goal.release = bool(args.release)
    if not args.release:
        # +X in the wrist frame: the direction this hand's fingers flex.
        # Synthetic; see the module docstring, including why not -Z.
        direction = np.array([float(v) for v in args.direction])
        goal.directions = pack_vectors(names, {n: direction for n in names})
        goal.force_targets = [float(args.force)] * len(names)
    goal.speed = float(args.speed)
    goal.max_travel = float(args.max_travel)
    goal.abort_counts = float(args.abort_counts)

    rclpy.init(args=ros_args)
    node = GraspClient()
    if args.clear_estop:
        node.clear_estop()

    if args.estop_after > 0.0:
        threading.Timer(args.estop_after, node.trip_estop).start()

    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        node.send(goal)
        if not node.wait():
            node.get_logger().error("timed out waiting for the result")
        elif node.result is not None:
            result = node.result
            # The EXECUTOR'S names, not the ones sent. A release names no digits
            # -- the posture to back out to is the executor's -- so labelling
            # these columns with what this client happened to ask for would
            # mislabel them the moment the two disagree.
            names = list(result.digit_names) or names
            print(f"\nsuccess: {result.success}   ({result.elapsed:.1f} s)")
            for i, name in enumerate(names):
                print(f"  {name:<8} {result.travel[i] * 1e3:6.1f} mm  "
                      f"{result.contact_counts[i]:6.0f} counts  "
                      f"(peak {result.peak_counts[i]:.0f})  "
                      f"{result.stop_reasons[i]}"
                      + (f" -- {result.stop_notes[i]}"
                         if result.stop_notes[i] else ""))
            # A digit that ends LOW with a high peak did not let go, it lost its
            # sensor. Printed here rather than left to be spotted in the columns.
            for i, name in enumerate(names):
                if (result.peak_counts[i] > 2.0 * max(result.contact_counts[i], 1.0)
                        and result.peak_counts[i] > 100.0):
                    print(f"  ! {name} finished well below its own peak -- "
                          f"check whether its sensor went out of range")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
