"""The node that actually drives the robot. No visualizer in this process.

    ros2 run gepetto_ros executor_node --ros-args -p hand:=epfl
    ros2 run gepetto_ros executor_node --ros-args -p hand:=allegro

ONE NODE, EITHER ROBOT. `hand` picks a :class:`~gepetto_ros.robot_profile.
RobotProfile`, which decides three things and nothing else: which
``gepetto_solvers`` hand builds the plan, which TF frame is the wrist, and which
:mod:`~gepetto_ros.hand_drivers` class turns a per-digit command into messages.
The control loop below is the same code for both.

WHY THIS IS ITS OWN PROCESS. The playback loop used to live inside `viz_node`,
alongside viser's asyncio websocket server, its 32-thread callback pool, and a
scene layer that rebuilds trimesh cylinders every frame. One process, one GIL. A
scene update long enough to outlast MoveIt Servo's `incoming_command_timeout`
(0.1 s) stopped the arm dead while the trajectory clock kept advancing, and the
tracking error that produced tripped the watchdog. Control work does not belong
behind a renderer, so it is not behind one any more: `viz_node` builds plans and
draws pictures, this executes them, and nothing in here can be delayed by
anything a browser does.

This is the ONLY process in the stack that publishes to
`/lbr/servo_node/delta_twist_cmds` and to the hand's command topic during
playback. `home_node` owns them for its own move; the two are mutually exclusive
by the same cell-wide e-stop they both watch.

TWO ACTIONS, ONE HAND. `PlayPlan` plays a decided path; `GraspClose` closes onto
an object against the fingertip tactile sensors, discovering its trajectory a
tick at a time because the planner has no dynamics and cannot say how far a
finger must travel to press with a given force. They live in the SAME node
deliberately: both drive the hand's command topic, so they must never overlap,
and one node holding both makes that impossible -- `_on_goal` on either refuses
while the other is running. A separate grasp node would need a lock the two
processes could disagree about.

E-STOP. Two independent paths reach this node and either one is sufficient:

  * `/gepetto/estop`, latched and cell-wide. The subscription is on its OWN
    callback group so a `MultiThreadedExecutor` services it on a free thread
    while the control loop is busy -- if they shared a group the stop would queue
    behind the very loop it is meant to interrupt. It halts the publishers from
    inside the callback, without waiting for the loop to come round.
  * A cancel on the running `PlayPlan` goal, which the loop notices within a
    tick.

The visualizer's E-STOP button fires both. Note that the halt executes HERE
rather than in the viser process, which means a stop pressed during a heavy scene
update no longer waits behind the scene update -- the button got faster by
leaving.
"""

import math

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Int8

from gepetto_ros_interfaces.action import GraspClose, PlayPlan
from gepetto_solvers.core import robot_plan
from gepetto_solvers.core.hands import get_hand
from gepetto_solvers.core.solvers import pack_vectors, unpack_vectors

from .frames import FrameLookup, pose_to_matrix
from .grasp_servo import GraspLimits, GraspServo
from .hand_state import HandStateReader
from .robot_profile import get_profile, registered_profiles
from .tactile import TactileReader
from .topics import ESTOP_QOS, ESTOP_TOPIC, GRASP_CLOSE_ACTION, PLAY_PLAN_ACTION
from .trajectory_executor import ExecutorLimits, TrajectoryExecutor
from .wire import goal_to_plan


class ExecutorNode(Node):

    def __init__(self):
        super().__init__("gepetto_executor")

        self.declare_parameter("hand", "allegro")
        self.declare_parameter("control_hz", 100.0)
        # Empty means "whatever this hand's driver says", which is the answer in
        # every assembled cell. It is a parameter at all so a bring-up against a
        # hand node in a namespace can point at it without a code change.
        self.declare_parameter("hand_state_topic", "")
        # MoveIt Servo says WHY it is scaling a command down -- approaching a
        # singularity, near a collision, against a joint bound. Without it, all
        # of those look identical from here: a commanded twist that the arm does
        # not follow. `moveit_servo.yaml` sets `status_topic: ~/status`.
        self.declare_parameter("servo_status_topic", "/lbr/servo_node/status")
        # --- the tactile close; see grasp_servo.GraspLimits for every default ---
        # Empty means "whatever this hand's driver says", as above. A hand whose
        # driver names no tactile topic simply cannot be closed this way, and the
        # GraspClose goal is refused with that as the reason.
        self.declare_parameter("tactile_topic", "")
        # DECLARED BY TYPE, not by an empty-list default: rclpy infers
        # BYTE_ARRAY from `[]` and every `-p tactile_digit_order:=[...]` override
        # would then be rejected as the wrong type. Same trap `home_node`
        # documents for `hand_home_flat`. Empty here means "the driver's own
        # order", which is verified against this hardware.
        self.declare_parameter("tactile_digit_order", Parameter.Type.STRING_ARRAY)
        self.declare_parameter("tactile_max_age", 0.2)        # s; a HARD fault
        self.declare_parameter("grasp_baseline_samples", 20)
        self.declare_parameter("grasp_contact_counts", 50.0)
        self.declare_parameter("grasp_collapse_fraction", 0.5)
        self.declare_parameter("grasp_stall_fraction", 0.25)
        self.declare_parameter("kp_linear", 2.0)
        self.declare_parameter("kp_angular", 2.0)
        self.declare_parameter("velocity_headroom", 2.0)
        self.declare_parameter("pipeline_gain", 0.07)         # see ExecutorLimits
        self.declare_parameter("position_watchdog", 0.15)     # m
        self.declare_parameter("rotation_watchdog", 45.0)     # deg
        self.declare_parameter("terminal_hold", 0.5)          # s -- see ExecutorLimits
        self.declare_parameter("tf_max_age", 0.2)             # s
        self.declare_parameter("tf_stale_grace", 1.0)         # s -- see ExecutorLimits
        self.declare_parameter("trace_hz", 1.0)               # 0 disables

        name = self.get_parameter("hand").value
        try:
            self._profile = get_profile(name)
        except KeyError as exc:
            raise RuntimeError(
                f"{exc} Set it with `-p hand:=<name>`.") from None
        self._hand = get_hand(self._profile.solver_hand)
        self.get_logger().info(f"robot: {self._profile.description}")

        # Callback groups, and the split is load-bearing -- see the module
        # docstring. The action server may run a goal for minutes; the e-stop
        # must never be behind it in a queue.
        self._action_group = ReentrantCallbackGroup()
        self._estop_group = MutuallyExclusiveCallbackGroup()

        self._frames = FrameLookup(self, self._profile.planning_frame,
                                   self._profile.wrist_frame,
                                   self._profile.table_frame)

        self._state = HandStateReader(
            self, self._hand, self._profile.hand_driver,
            topic=self.get_parameter("hand_state_topic").value or None,
            callback_group=self._estop_group)

        self._servo_status = None
        self.create_subscription(
            Int8, self.get_parameter("servo_status_topic").value,
            self._on_servo_status, 10, callback_group=self._estop_group)

        travel = robot_plan.travel_limits(self._hand)
        if travel is None:
            # Not fatal -- the plan is already clamped client-side by
            # `robot_plan.clamp_to_travel`, and a hand node may enforce its own
            # copy -- but this node's own clamp is gone, and a missing safety
            # clamp must never be silent.
            self.get_logger().warn(
                f"NO TRAVEL LIMITS for {self._hand.name}: this node will not "
                f"clamp what it commands. On a tendon hand that means "
                f"epfl_hand_core is not importable here; on a joint hand it "
                f"means the URDF's limits could not be read. The plan is still "
                f"clamped client-side, but fix the environment.")
            travel = {}
        else:
            kind = robot_plan.command_kind(self._hand)
            suffix, scale = robot_plan.COMMAND_UNITS[kind]
            self.get_logger().info(
                f"travel limits ({suffix}): "
                + ", ".join(
                    f"{n} [{lo.min() * scale:.1f}, {hi.max() * scale:.1f}]"
                    for n, (lo, hi) in sorted(travel.items())))

        self._executor = TrajectoryExecutor(
            self, self._profile, self._hand, self._frames,
            limits=self._limits(),
            hand_state=self._state.read,
            servo_status=self.servo_status,
            travel_limits=travel)

        # On the e-stop group with the other feedback readers: a reading that
        # queued behind a running close would go stale by definition, and the
        # close aborts on stale readings.
        # `get_parameter_or`, not `get_parameter`: declaring by TYPE leaves the
        # parameter uninitialized until someone sets it, and the plain getter
        # raises on that. Same reader `home_node` uses for `hand_home_flat`,
        # which is declared the same way and for the same reason.
        order = self.get_parameter_or("tactile_digit_order").value
        self._tactile = TactileReader(
            self, self._hand, self._profile.hand_driver,
            topic=self.get_parameter("tactile_topic").value or None,
            order=list(order) if order else None,
            callback_group=self._estop_group)
        self._grasp = GraspServo(
            self, self._profile, self._hand, self._tactile,
            limits=self._grasp_limits(),
            hand_state=self._state.read,
            travel_limits=travel)

        self._estop_reason = ""
        self._estop_pub = self.create_publisher(Bool, ESTOP_TOPIC, ESTOP_QOS)
        self.create_subscription(Bool, ESTOP_TOPIC, self._on_estop_msg,
                                 ESTOP_QOS, callback_group=self._estop_group)

        self._server = ActionServer(
            self, PlayPlan, PLAY_PLAN_ACTION,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._action_group)

        # Same group as the playback action, and the mutual exclusion between
        # them is enforced in the two `_on_goal`s rather than by the group: they
        # are both long-running, and a REENTRANT group is what lets a cancel on
        # one be serviced while the other is mid-run.
        self._grasp_server = ActionServer(
            self, GraspClose, GRASP_CLOSE_ACTION,
            execute_callback=self._execute_grasp,
            goal_callback=self._on_grasp_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._action_group)

        self.get_logger().info(
            f"trajectory executor up on {PLAY_PLAN_ACTION} at "
            f"{self.get_parameter('control_hz').value:.0f} Hz "
            f"(hand state on {self._state.topic}); "
            f"profiles: {', '.join(registered_profiles())}")
        if self._tactile.available:
            self.get_logger().info(
                f"tactile close up on {GRASP_CLOSE_ACTION} "
                f"(fingertips on {self._tactile.topic})")
        else:
            # Said once, at startup, rather than only when a close is refused:
            # an operator looking at a greyed-out button should be able to find
            # out why from the log the node already printed.
            self.get_logger().info(
                f"no tactile close: the {self._hand.name} hand's driver declares "
                f"no fingertip sensor. {GRASP_CLOSE_ACTION} will refuse goals.")

    def _limits(self):
        def get(name):
            return self.get_parameter(name).value
        return ExecutorLimits(
            hz=float(get("control_hz")),
            kp_linear=float(get("kp_linear")),
            kp_angular=float(get("kp_angular")),
            velocity_headroom=float(get("velocity_headroom")),
            pipeline_gain=float(get("pipeline_gain")),
            position_watchdog=float(get("position_watchdog")),
            rotation_watchdog=math.radians(float(get("rotation_watchdog"))),
            terminal_hold=float(get("terminal_hold")),
            tf_max_age=float(get("tf_max_age")),
            tf_stale_grace=float(get("tf_stale_grace")),
            trace_hz=float(get("trace_hz")))

    def _grasp_limits(self):
        def get(name):
            return self.get_parameter(name).value
        return GraspLimits(
            hz=float(get("control_hz")),
            tactile_max_age=float(get("tactile_max_age")),
            baseline_samples=int(get("grasp_baseline_samples")),
            contact_counts=float(get("grasp_contact_counts")),
            collapse_fraction=float(get("grasp_collapse_fraction")),
            stall_fraction=float(get("grasp_stall_fraction")),
            trace_hz=float(get("trace_hz")))

    # --------------------------------------------------------------- feedback

    def _on_servo_status(self, msg):
        self._servo_status = int(msg.data)

    def servo_status(self):
        """MoveIt Servo's own status code, or None if it has never published.

        None is meaningfully different from 0: 0 is "Servo is running and has no
        complaint", None is "nothing on the status topic at all", which usually
        means servo_node is not up or the topic name is wrong -- and in that case
        every twist this node publishes is going nowhere.
        """
        return self._servo_status

    # ------------------------------------------------------------------ estop

    def _on_estop_msg(self, msg):
        """A trip from anywhere in the cell. Halts from THIS callback rather than
        leaving it to the loop: the loop polls between ticks, and one tick of a
        saturated twist is exactly what a stop button exists to prevent."""
        if msg.data:
            if not self._executor.estop.is_set():
                self.get_logger().error("E-STOP engaged -- halting")
            # BOTH loops, unconditionally. Only one of them can be running, but
            # this callback must not have to know which -- halting the wrong one
            # is free, and halting neither because the check picked wrong is not.
            self._executor.estop.set()
            self._executor.halt()
            self._grasp.estop.set()
            self._grasp.halt()
        else:
            if self._executor.estop.is_set():
                self.get_logger().info("E-STOP cleared")
            self._executor.estop.clear()
            self._grasp.estop.clear()

    def _trip(self, reason):
        """Latch the cell-wide e-stop from here (a stall the loop detected).

        Published rather than merely set locally, because the visualizer greys
        out on the topic and an operator needs to be told WHY the run stopped,
        not just that it did."""
        self._estop_reason = reason
        self._executor.estop.set()
        self._executor.halt()
        msg = Bool()
        msg.data = True
        self._estop_pub.publish(msg)

    # ----------------------------------------------------------------- action

    def _on_goal(self, goal):
        if self._executor.estop.is_set():
            self.get_logger().warn("rejecting goal: E-STOP engaged")
            return GoalResponse.REJECT
        if self._executor.playing:
            self.get_logger().warn("rejecting goal: already playing")
            return GoalResponse.REJECT
        if self._grasp.running:
            # The other half of the mutual exclusion. A playback started during
            # a close would put two loops on the hand's command topic, each
            # slew-limiting toward a different setpoint -- with fingers already
            # loaded against an object.
            self.get_logger().warn("rejecting goal: a tactile close is running")
            return GoalResponse.REJECT
        if not (goal.enable_arm or goal.enable_hand):
            self.get_logger().warn("rejecting goal: no channel enabled")
            return GoalResponse.REJECT
        if not goal.wrist_poses:
            self.get_logger().warn("rejecting goal: no waypoints")
            return GoalResponse.REJECT
        # A plan built for ANOTHER HAND is refused here rather than played. It
        # would otherwise reach the driver as a width mismatch mid-run, with the
        # arm already moving -- and on the Allegro a width-1 plan would be a
        # sixteen-wide message the driver rejects outright, i.e. an arm that
        # travels the path with a hand that never moves.
        expected = robot_plan.command_kind(self._hand)
        if goal.command_kind and goal.command_kind != expected:
            self.get_logger().warn(
                f"rejecting goal: it carries {goal.command_kind!r} but this "
                f"executor is driving {self._hand.name}, which is commanded in "
                f"{expected!r}. Is the visualizer on the same `hand`?")
            return GoalResponse.REJECT
        width = len(self._hand.actuation.drive_indices)
        if goal.enable_hand and int(goal.dof_per_digit) != width:
            self.get_logger().warn(
                f"rejecting goal: {goal.dof_per_digit} values per digit, but "
                f"{self._hand.name} drives {width} actuators per digit")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle):
        """Always accepted. A cancel is how the E-STOP button reaches a running
        goal, and a stop request that can be refused is not a stop request."""
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        goal = goal_handle.request
        result = PlayPlan.Result()

        def on_progress(progress):
            feedback = PlayPlan.Feedback()
            feedback.tick = int(progress.tick)
            feedback.path_time = float(progress.path_time)
            feedback.path_total = float(progress.path_total)
            feedback.waypoint = int(progress.waypoint)
            feedback.num_waypoints = int(progress.num_waypoints)
            feedback.note = progress.note
            feedback.position_error = float(progress.position_error)
            feedback.rotation_error = float(progress.rotation_error)
            feedback.time_scale = float(progress.time_scale)
            # The wire field is still called `settling`; what it now reports is
            # the fixed terminal hold. Renaming it would need an interface
            # rebuild for no behavioural gain -- see the .action file's comment.
            feedback.settling = bool(progress.holding)
            goal_handle.publish_feedback(feedback)

        try:
            outcome = self._executor.play(
                goal_to_plan(goal, pose_to_matrix),
                enable_arm=goal.enable_arm,
                enable_hand=goal.enable_hand,
                speeds=dict(max_linear=goal.max_linear,
                            max_angular=goal.max_angular,
                            max_digit=goal.max_digit),
                on_progress=on_progress,
                should_stop=lambda: goal_handle.is_cancel_requested)
        except Exception as exc:
            self.get_logger().error(f"playback failed: {exc}")
            result.success = False
            result.verdict = f"**playback failed:** `{exc}`"
            goal_handle.abort()
            return result

        if outcome.trip_reason:
            # The loop's watchdog fired. Latching here rather than inside the
            # loop keeps the loop free of ROS publishing.
            self.get_logger().error(f"playback watchdog: {outcome.trip_reason}")
            self._trip(outcome.trip_reason)

        result.success = outcome.success
        result.verdict = outcome.verdict
        result.ticks = int(outcome.ticks)
        result.elapsed = float(outcome.elapsed)
        result.final_position_error = float(outcome.position_error)
        result.final_rotation_error = float(outcome.rotation_error)
        result.resumable = bool(outcome.resumable)
        result.resume_waypoint = int(outcome.resume_waypoint)
        result.resume_path_time = float(outcome.resume_path_time)

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif outcome.success:
            goal_handle.succeed()
        else:
            # Ran but did not arrive, or was stopped by the e-stop. Not an
            # `abort()` -- the result is meaningful and carries the resume point.
            goal_handle.succeed()
        return result

    # ------------------------------------------------------- the tactile close

    def _on_grasp_goal(self, goal):
        if self._executor.estop.is_set() or self._grasp.estop.is_set():
            self.get_logger().warn("rejecting close: E-STOP engaged")
            return GoalResponse.REJECT
        if self._executor.playing:
            self.get_logger().warn("rejecting close: a playback is running")
            return GoalResponse.REJECT
        if self._grasp.running:
            self.get_logger().warn("rejecting close: already closing")
            return GoalResponse.REJECT

        if goal.release:
            if not self._grasp.can_release:
                self.get_logger().warn(
                    "rejecting release: no close has run in this executor, so "
                    "there is no recorded posture to back out to")
                return GoalResponse.REJECT
            return GoalResponse.ACCEPT

        if not self._tactile.available:
            self.get_logger().warn(
                f"rejecting close: {self._hand.name}'s driver declares no "
                f"fingertip sensor, so there would be nothing to stop it")
            return GoalResponse.REJECT
        _counts, age = self._tactile.read()
        if age is None:
            # REFUSED, not started hopefully: a close whose sensor never
            # publishes runs its whole travel budget into whatever is in front
            # of it and reports "no contact" about it.
            self.get_logger().warn(
                f"rejecting close: nothing has ever arrived on "
                f"`{self._tactile.topic}`. Is the hand node publishing?")
            return GoalResponse.REJECT
        if not goal.digit_names:
            self.get_logger().warn("rejecting close: no digits named")
            return GoalResponse.REJECT
        unknown = [n for n in goal.digit_names
                   if n not in self._hand.digit_names]
        if unknown:
            self.get_logger().warn(
                f"rejecting close: {', '.join(unknown)} are not digits of "
                f"{self._hand.name}. Is the visualizer on the same `hand`?")
            return GoalResponse.REJECT
        if len(goal.directions) != 3 * len(goal.digit_names):
            self.get_logger().warn(
                f"rejecting close: {len(goal.directions)} direction values for "
                f"{len(goal.digit_names)} digits, expected "
                f"{3 * len(goal.digit_names)}")
            return GoalResponse.REJECT
        if len(goal.force_targets) != len(goal.digit_names):
            self.get_logger().warn(
                f"rejecting close: {len(goal.force_targets)} targets for "
                f"{len(goal.digit_names)} digits")
            return GoalResponse.REJECT
        if goal.speed <= 0.0 or goal.max_travel <= 0.0:
            self.get_logger().warn(
                "rejecting close: speed and max_travel must both be positive")
            return GoalResponse.REJECT
        if any(t >= goal.abort_counts for t in goal.force_targets):
            # A target at or above the ceiling can never be reached: the abort
            # fires first, every time, and every digit ends on over-force.
            self.get_logger().warn(
                f"rejecting close: a force target is at or above the "
                f"{goal.abort_counts:.0f}-count abort ceiling, so it could "
                f"never be reached")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _execute_grasp(self, goal_handle):
        goal = goal_handle.request
        result = GraspClose.Result()
        # A RELEASE NAMES NO DIGITS -- the posture to back out to is this node's,
        # not the caller's -- so they are resolved here, before the feedback
        # closure below captures them. Left until after the run, the feedback
        # would arrive as empty arrays for the whole release.
        names = (self._grasp.release_digits() if goal.release
                 else list(goal.digit_names))

        def on_progress(progress):
            feedback = GraspClose.Feedback()
            feedback.travel = [float(progress.travel.get(n, 0.0)) for n in names]
            feedback.contact_counts = [float(progress.counts.get(n, 0.0))
                                       for n in names]
            feedback.digits_stopped = int(progress.stopped)
            feedback.num_digits = int(progress.total)
            feedback.note = progress.note
            goal_handle.publish_feedback(feedback)

        try:
            if goal.release:
                outcome = self._grasp.release(
                    on_progress=on_progress,
                    should_stop=lambda: goal_handle.is_cancel_requested)
            else:
                outcome = self._grasp.close(
                    names,
                    unpack_vectors(names, goal.directions),
                    dict(zip(names, (float(t) for t in goal.force_targets))),
                    speed=float(goal.speed),
                    max_travel=float(goal.max_travel),
                    abort_counts=float(goal.abort_counts),
                    on_progress=on_progress,
                    should_stop=lambda: goal_handle.is_cancel_requested)
        except Exception as exc:
            self.get_logger().error(f"grasp close failed: {exc}")
            result.success = False
            result.verdict = f"**grasp close failed:** `{exc}`"
            goal_handle.abort()
            return result

        result.success = outcome.success
        result.verdict = outcome.verdict
        result.digit_names = list(names)
        result.stop_reasons = [outcome.stop_reasons.get(n, "") for n in names]
        result.stop_notes = [outcome.stop_notes.get(n, "") for n in names]
        result.travel = [float(outcome.travel.get(n, 0.0)) for n in names]
        result.contact_counts = [float(outcome.counts.get(n, 0.0))
                                 for n in names]
        result.peak_counts = [float(outcome.peak_counts.get(n, 0.0))
                              for n in names]
        result.q_final = pack_vectors(
            names, outcome.q_final,
            width=len(self._hand.actuation.drive_indices))
        result.elapsed = float(outcome.elapsed)

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        else:
            # `succeed()` even for a close where digits gave up, the playback's
            # rule: the result is meaningful -- it says which digit stopped why,
            # which is the whole readout -- and an `abort()` would throw it away
            # and report only that something went wrong.
            goal_handle.succeed()
        return result

    def destroy_node(self):
        # Halt before the publishers go away, so a node killed mid-playback does
        # not leave the arm coasting on its last twist for the full
        # incoming_command_timeout. Only while the context is still up: on Ctrl-C
        # rclpy's SIGINT handler has already invalidated it and publishing merely
        # raises. The arm still stops -- an interrupted process publishes
        # nothing, which is what the servo's command timeout is for.
        if rclpy.ok():
            try:
                self._executor.estop.set()
                self._executor.halt()
                self._grasp.estop.set()
                self._grasp.halt()
            except Exception as exc:
                self.get_logger().warn(f"error halting on shutdown: {exc}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExecutorNode()
    # Multi-threaded so the e-stop subscription and the TF listener are serviced
    # on threads the control loop is not holding. See the module docstring.
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
