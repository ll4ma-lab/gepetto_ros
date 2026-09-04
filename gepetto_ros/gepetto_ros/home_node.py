"""Send the robot home: a predefined arm configuration, and the hand at rest.

    ros2 action send_goal /gepetto/move_home gepetto_ros_interfaces/action/MoveHome "{}"
    ros2 service call /home_node/go_home std_srvs/srv/Trigger      # same move
    ros2 run gepetto_ros home_node --ros-args -p hand:=allegro     # standalone

A RESIDENT NODE, NOT A ONE-SHOT. It comes up with the cell in
`gepetto_bringup.launch.py` and stays up; every trip home after that is a
`MoveHome` goal or a `go_home` call on the process that is already running.
Homing used to mean launching this node, watching it move and killing it again,
which is a strange way to press a button.

WHICH MEANS IT DOES NOT MOVE ON START-UP THERE. `on_start` still defaults to true,
because a bare `ros2 run` of this node is a request to go home and it would be
obtuse to make that take a second command -- but the launch file passes
`home_on_start:=false`, so bringing the cell up does not move the arm. Anything
that homes the instant it is launched cannot be launched with the cell.

TWO WAYS IN, ONE MOVE. The action carries feedback, cancellation and a per-goal
choice of channels; the service is one line to type and has none of that. Both
funnel into `go_home`, so there is no second control loop to keep in step.

WHY JOINT SPACE FOR THE ARM. Home is where the arm is folded, not a pose it
happens to reach: a joint target is reachable by construction, cannot stall in a
singularity, and cannot pick a different elbow than the one it left.

WHAT "HOME" IS FOR THE HAND follows from what its command MEANS, which is not the
same thing on the two hands:

* the tendon hand's command is DISPLACEMENT from the hand-open pose, so its rest
  is the ZERO vector -- exactly as it was when this node knew only that hand;
* the Allegro's command is joint POSITION, where zero is the straight, uncurled
  hand. That is inside every joint's limit and is still not a posture anyone
  wants to park in, so its rest is the hand's OWN default posture --
  `AllegroHand.default_pose()`, the slightly-curled pre-grasp the viser
  workbench opens on and returns to on *Reset defaults*. Reading it off the hand
  rather than copying the numbers is what keeps the two from drifting: retune
  the workbench's opening posture and this follows.

Neither is a table here, and neither is per-cell. Override `hand_home_flat` if a
particular cell wants to park somewhere else; it is a flat vector in the driver's
own digit and actuator order.

WHY SERVO AND NOT MoveIt. `move_group` (and `lbr_motion`'s `move_home_server`,
which is the same idea with planning) drives `joint_trajectory_controller`, while
MoveIt Servo drives `forward_position_controller`. They are mutually exclusive, so
in a cell brought up for servoing, going home has to go through the servo topics
too. That is what this node is: a proportional controller with a stopping rule
over the same publishers `executor_node` uses.

It refuses to move while `/gepetto/estop` is true, and aborts immediately if the
stop is pressed on the way, which is the same latch the visualizer's E-STOP button
raises. It is MUTUALLY EXCLUSIVE with a running playback for that reason: both
publish to the same command topics, and nothing but the operator sequences them.
Being resident does not change that -- an idle `home_node` publishes nothing at
all, so it costs a running playback nothing to have one sitting there.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from gepetto_ros_interfaces.action import MoveHome
from gepetto_solvers.core import robot_plan
from gepetto_solvers.core.hands import get_hand

from .arm_servo import ArmJointServo
from .hand_state import HandStateReader
from .robot_profile import get_profile
from .topics import ESTOP_QOS, ESTOP_TOPIC, MOVE_HOME_ACTION

ARM_JOINT_NAMES = ["lbr_A1", "lbr_A2", "lbr_A3", "lbr_A4",
                   "lbr_A5", "lbr_A6", "lbr_A7"]

# WHERE THE ARM HOME LIVES: `RobotProfile.arm_home`, one folded configuration per
# cell, in this same order. It is per robot because the fold has to keep the HAND
# clear -- the pose that tucks the tendon hand in drives the longer Allegro
# through the table -- so a single constant here would be wrong for one of them.


@dataclass
class HomeOutcome:
    """What one trip home did, in the units the action reports.

    A record rather than the old ``(ok, message)`` pair because the action has a
    result to fill and the service does not: both come out of the same loop, and
    a loop that returned less to one caller than the other is how the service and
    the action would come to disagree about what happened.
    """

    success: bool
    message: str
    joint_error: float = 0.0        # rad
    hand_error: float = 0.0         # command units (metres or radians)
    elapsed: float = 0.0            # wall seconds


class HomeNode(Node):

    def __init__(self):
        super().__init__("home_node")

        self.declare_parameter("hand", "allegro")
        name = self.get_parameter("hand").value
        try:
            self._profile = get_profile(name)
        except KeyError as exc:
            raise RuntimeError(f"{exc} Set it with `-p hand:=<name>`.") from None

        # Declared AFTER the hand is known, because its default is the profile's:
        # each cell folds up somewhere else. Not move_home_server's all-zeros --
        # that is the arm stretched straight up, which is a fine mathematical
        # origin and a poor place to park -- so override the parameter if you
        # want the zero configuration instead.
        self.declare_parameter("home_joint_positions", list(self._profile.arm_home))
        self.declare_parameter("joint_names", ARM_JOINT_NAMES)
        self.declare_parameter("joint_tolerance", 0.01)        # rad
        # In the hand's own command units: metres of tendon, or radians of joint.
        self.declare_parameter("hand_tolerance", 0.0005)
        self.declare_parameter("control_hz", 100.0)
        self.declare_parameter("max_joint_rate", 0.4)          # rad/s
        # Empty means the hand's own rest posture -- see the module docstring on
        # what that is for each hand and why. A non-empty value overrides it, and
        # is flat, in digit-then-actuator order.
        #
        # DECLARED BY TYPE, not by an empty default: rclpy infers a parameter's
        # type from the default it is given, and `[]` infers BYTE_ARRAY -- so
        # every `-p hand_home_flat:=[...]` the docstring advertises was rejected
        # as a DOUBLE_ARRAY against a BYTE_ARRAY before it ever reached
        # _resolve_hand_home. There is no non-empty default that could carry the
        # type instead (having none is the whole point of the parameter), so it
        # is stated. Left UNINITIALIZED when nobody passes one, which is why it
        # is read with `get_parameter_or` below -- plain `get_parameter` raises
        # on a declared-but-unset parameter.
        self.declare_parameter("hand_home_flat", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("timeout", 120.0)                # s
        # The defaults for a move that does not say otherwise. A `MoveHome` goal
        # says otherwise per goal (`skip_arm` / `skip_hand`), so these are what
        # the service and the start-up move use.
        self.declare_parameter("move_arm", True)
        self.declare_parameter("open_hand", True)
        # True for a standalone `ros2 run`, false from the launch file -- see the
        # module docstring on why a resident node must not move on start-up.
        self.declare_parameter("on_start", True)

        self._hand_model = get_hand(self._profile.solver_hand)
        self._width = len(self._hand_model.actuation.drive_indices)

        self._joint_names = list(self.get_parameter("joint_names").value)
        self._home = np.asarray(
            self.get_parameter("home_joint_positions").value, float)
        if self._home.size != len(self._joint_names):
            raise ValueError(
                f"home_joint_positions has {self._home.size} entries but there "
                f"are {len(self._joint_names)} joint_names.")

        self._hz = float(self.get_parameter("control_hz").value)
        self._timeout = float(self.get_parameter("timeout").value)
        self._joint_tolerance = float(self.get_parameter("joint_tolerance").value)
        self._hand_tolerance = float(self.get_parameter("hand_tolerance").value)
        self._hand_home = self._resolve_hand_home()

        self._arm = ArmJointServo(
            self, self._joint_names,
            frame_id=self._profile.planning_frame,
            max_rate=float(self.get_parameter("max_joint_rate").value))
        # Clamped like a playback is: home is a commanded posture too, and a
        # zero outside a joint's limits (which no current hand has, but a future
        # one might) must be caught here rather than sent.
        self._hand = self._profile.hand_driver(
            self, self._hand_model,
            limits=robot_plan.travel_limits(self._hand_model) or {},
            hz=self._hz)

        self._joint_state = None
        self._lock = threading.Lock()
        self._busy = threading.Lock()

        # Callback groups, and the split is load-bearing exactly as it is in
        # `executor_node`. A goal runs the control loop for as long as the move
        # takes, inside the action's own callback; the e-stop and the feedback
        # the loop READS must be serviced on threads that goal is not holding.
        # Sharing one group would deadlock the loop against its own inputs --
        # it would poll for a joint state that could not be delivered until it
        # returned.
        self._action_group = ReentrantCallbackGroup()
        self._feedback_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(JointState, "/lbr/joint_states",
                                 self._on_joint_state, 10,
                                 callback_group=self._feedback_group)
        self._hand_state = HandStateReader(self, self._hand_model,
                                           self._profile.hand_driver,
                                           callback_group=self._feedback_group)

        self._estop = threading.Event()
        self.create_subscription(Bool, ESTOP_TOPIC, self._on_estop, ESTOP_QOS,
                                 callback_group=self._feedback_group)

        self.create_service(Trigger, "~/go_home", self._on_go_home,
                            callback_group=self._action_group)
        self._server = ActionServer(
            self, MoveHome, MOVE_HOME_ACTION,
            execute_callback=self._execute,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._action_group)

        if bool(self.get_parameter("on_start").value):
            # On a thread: the run loop needs subscriptions to be delivering, and
            # nothing is delivered until the executor is spinning, which it is
            # not yet inside __init__.
            threading.Thread(target=self._run_when_ready, daemon=True).start()
        suffix, _scale = robot_plan.COMMAND_UNITS[
            robot_plan.command_kind(self._hand_model)]
        self.get_logger().info(
            f"home_node ready ({self._profile.name}) on {MOVE_HOME_ACTION} "
            f"and ~/go_home. Arm home: {np.round(self._home, 3).tolist()} rad"
            + (f", hand to its rest posture ({suffix})"
               if self.get_parameter("open_hand").value else ""))

    def _resolve_hand_home(self):
        """``{digit: (K,) vector}`` for the hand's rest posture.

        The hand's own, unless `hand_home_flat` says otherwise. Which one that is
        is decided by what the command MEANS -- see the module docstring:

        * a DISPLACEMENT hand measures its command from the open hand, so rest is
          the zero vector and there is nothing to look up;
        * a POSITION hand's zero is merely the URDF's neutral pose, so rest is
          `default_pose()` -- the posture the hand itself nominates, which is the
          one the viser workbench opens on.

        Keyed off the command kind rather than off the profile name so a third
        hand of either kind gets the right answer with no entry here.
        """
        digits = list(self._hand_model.digit_names)
        flat = list(self.get_parameter_or("hand_home_flat").value or [])
        if flat:
            expected = len(digits) * self._width
            if len(flat) != expected:
                raise ValueError(
                    f"hand_home_flat has {len(flat)} values; {len(digits)} "
                    f"digits x {self._width} actuators needs {expected}")
            block = np.asarray(flat, float).reshape(len(digits), self._width)
            return {name: block[i] for i, name in enumerate(digits)}

        kind = robot_plan.command_kind(self._hand_model)
        if kind == robot_plan.TENDON_DISPLACEMENT_M:
            return {name: np.zeros(self._width) for name in digits}
        _wrist, means = self._hand_model.default_pose()
        home = {}
        for i, name in enumerate(digits):
            q = np.asarray(means[i], float).ravel()
            if q.size != self._width:
                # A hand whose default posture does not describe every driven
                # actuator: refuse rather than pad, which would silently home
                # some joints and leave the rest wherever they were.
                raise ValueError(
                    f"{self._hand_model.name}.default_pose() gives {q.size} "
                    f"values for digit {name!r}, but the driver commands "
                    f"{self._width}")
            home[name] = q
        return home

    # ------------------------------------------------------------- feedback

    def _on_joint_state(self, msg):
        index = {name: i for i, name in enumerate(msg.name)}
        if not all(name in index for name in self._joint_names):
            return          # a partial /joint_states (the hand's, say) -- ignore
        with self._lock:
            self._joint_state = np.array(
                [msg.position[index[name]] for name in self._joint_names], float)

    def _on_estop(self, msg):
        if msg.data:
            self._estop.set()
            self._arm.halt()
            self.get_logger().warn(f"E-STOP on {ESTOP_TOPIC}: holding position.")
        else:
            self._estop.clear()

    # -------------------------------------------------------------- commands

    def _on_go_home(self, _request, response):
        """The Trigger service, kept alongside the action.

        Not superseded by it: a Trigger is one line of `ros2 service call` with
        no goal to fill in, which is what a bring-up session actually types. The
        action is for a client that wants feedback, cancellation, or a channel
        skipped -- and both go through `go_home`, so neither can park the robot
        somewhere the other would not."""
        outcome = self.go_home()
        response.success = outcome.success
        response.message = outcome.message
        return response

    # ----------------------------------------------------------------- action

    def _on_goal(self, goal):
        """Refuse a goal that could only end badly, rather than starting it.

        The same three gates `executor_node` puts on a play, for the same
        reasons: a stopped cell does not move, one mover at a time, and a goal
        that asks for nothing is a mistake at the client rather than a no-op to
        be honoured silently."""
        if self._estop.is_set():
            self.get_logger().warn("rejecting home goal: E-STOP engaged")
            return GoalResponse.REJECT
        if self._busy.locked():
            self.get_logger().warn("rejecting home goal: already going home")
            return GoalResponse.REJECT
        if goal.skip_arm and goal.skip_hand:
            self.get_logger().warn(
                "rejecting home goal: it skips both the arm and the hand")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle):
        """Always accepted -- see the action file. The loop notices within a
        tick and halts both channels on its way out."""
        return CancelResponse.ACCEPT

    def _execute(self, goal_handle):
        goal = goal_handle.request
        suffix, scale = robot_plan.COMMAND_UNITS[
            robot_plan.command_kind(self._hand_model)]
        move_arm = not goal.skip_arm
        open_hand = not goal.skip_hand

        def on_progress(progress):
            feedback = MoveHome.Feedback()
            feedback.joint_error = float(progress.joint_error)
            feedback.hand_error = float(progress.hand_error * scale)
            feedback.elapsed = float(progress.elapsed)
            # Reported per channel, and a SKIPPED channel is at home by
            # definition -- its error is fixed at zero, and calling that "not
            # there yet" would be a progress bar that never fills.
            feedback.arm_at_home = (
                not move_arm or progress.joint_error <= self._joint_tolerance)
            feedback.hand_at_home = (
                not open_hand or progress.hand_error <= self._hand_tolerance)
            goal_handle.publish_feedback(feedback)

        outcome = self.go_home(
            move_arm=move_arm, open_hand=open_hand,
            timeout=goal.timeout or None,
            on_progress=on_progress,
            should_stop=lambda: goal_handle.is_cancel_requested)

        result = MoveHome.Result()
        result.success = bool(outcome.success)
        result.message = outcome.message
        result.joint_error = float(outcome.joint_error)
        result.hand_error = float(outcome.hand_error * scale)
        result.hand_units = suffix
        result.elapsed = float(outcome.elapsed)

        self.get_logger().info(f"move_home: {outcome.message}")
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
        elif outcome.success:
            goal_handle.succeed()
        else:
            # ABORTED, unlike PlayPlan's did-not-arrive case: a home that did not
            # reach home has no partial result worth acting on -- the robot is
            # somewhere between where it was and where it was asked to be, and a
            # caller must not read that as parked.
            goal_handle.abort()
        return result

    # -------------------------------------------------------------- start-up

    def _run_when_ready(self):
        """Wait for the first feedback, then go home once."""
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and rclpy.ok():
            with self._lock:
                have_arm = self._joint_state is not None
            _state, age = self._hand_state.read()
            if (have_arm or not self.get_parameter("move_arm").value) and \
                    (age is not None or not self.get_parameter("open_hand").value):
                break
            time.sleep(0.1)
        outcome = self.go_home()
        self.get_logger().info(f"go_home: {outcome.message}")
        if not outcome.success:
            self.get_logger().error("go_home did not complete.")

    # ------------------------------------------------------------- the move

    def go_home(self, move_arm=None, open_hand=None, timeout=None,
                on_progress=None, should_stop=None):
        """Drive to the home configuration. Returns a :class:`HomeOutcome`.

        ``move_arm`` / ``open_hand`` / ``timeout`` default to the node's
        parameters; the action passes its goal's values instead, which is what
        lets one resident node serve an arm-only home to one caller and a whole
        one to the next without a reconfigure.

        ``on_progress`` is called each tick with the running
        :class:`HomeOutcome`, and ``should_stop`` is polled each tick -- the two
        hooks the action needs for feedback and for cancellation. Both are
        optional, and with neither this behaves exactly as the service always
        has.
        """
        if not self._busy.acquire(blocking=False):
            return HomeOutcome(False, "already going home")
        try:
            return self._go_home(move_arm, open_hand, timeout,
                                 on_progress, should_stop)
        finally:
            self._busy.release()

    def _go_home(self, move_arm=None, open_hand=None, timeout=None,
                 on_progress=None, should_stop=None):
        if self._estop.is_set():
            return HomeOutcome(
                False, f"refused: E-STOP is engaged ({ESTOP_TOPIC} is true)")

        if move_arm is None:
            move_arm = bool(self.get_parameter("move_arm").value)
        if open_hand is None:
            open_hand = bool(self.get_parameter("open_hand").value)
        timeout = float(timeout or self._timeout)
        if not (move_arm or open_hand):
            return HomeOutcome(
                False, "nothing to do: the arm and the hand are both skipped")

        with self._lock:
            joints = self._joint_state
        measured, age = self._hand_state.read()
        if move_arm and joints is None:
            return HomeOutcome(False, "no /lbr/joint_states yet -- is the robot up?")
        if open_hand and age is None:
            return HomeOutcome(
                False, f"nothing on `{self._hand_state.topic}` yet -- is the "
                       f"hand node running?")
        if open_hand:
            # Seed from the measured hand, so the first command is measured from
            # where the digits actually are rather than from an assumed rest.
            self._hand.seed(measured)

        period = 1.0 / self._hz
        started = time.monotonic()
        deadline = started + timeout
        joint_error = hand_error = float("inf")
        suffix, scale = robot_plan.COMMAND_UNITS[
            robot_plan.command_kind(self._hand_model)]

        def outcome(success, message):
            return HomeOutcome(
                success, message,
                joint_error=0.0 if joint_error == float("inf") else joint_error,
                hand_error=0.0 if hand_error == float("inf") else hand_error,
                elapsed=time.monotonic() - started)

        try:
            while rclpy.ok():
                if self._estop.is_set():
                    return outcome(False, "aborted: E-STOP engaged mid-move")
                if should_stop is not None and should_stop():
                    # The operator stopped it part way. Not a failure of the
                    # move, so it says so plainly rather than reporting an error
                    # the caller would have to interpret.
                    return outcome(False, "cancelled part way home")
                if time.monotonic() > deadline:
                    return outcome(
                        False, f"timed out after {timeout:.0f} s (joint error "
                               f"{joint_error:.3f} rad, hand error "
                               f"{hand_error * scale:.2f} {suffix})")

                with self._lock:
                    joints = self._joint_state
                measured, _age = self._hand_state.read()

                joint_error = 0.0
                if move_arm and joints is not None:
                    joint_error = self._arm.tick(self._home, joints)

                hand_error = 0.0
                if open_hand:
                    self._hand.tick(self._hand_home)
                    if measured:
                        # Against the MEASURED state, not against what has been
                        # commanded: a driver's own setpoint reaches its target
                        # immediately and would report success before the hand
                        # had moved at all.
                        hand_error = max(
                            (float(np.max(np.abs(
                                np.ravel(value) - self._hand_home[name])))
                             for name, value in measured.items()
                             if name in self._hand_home),
                            default=0.0)

                if on_progress is not None:
                    on_progress(outcome(False, ""))

                if joint_error <= self._joint_tolerance and \
                        hand_error <= self._hand_tolerance:
                    return outcome(True, f"home reached (joint error "
                                         f"{joint_error * 1e3:.1f} mrad, hand "
                                         f"error {hand_error * scale:.2f} "
                                         f"{suffix})")
                time.sleep(period)
        finally:
            self._arm.halt()
            self._hand.halt()
        return outcome(False, "interrupted")


def main(args=None):
    rclpy.init(args=args)
    node = HomeNode()
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
