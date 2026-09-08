"""The ROS half of the visualizer integration: frames, feedback, and dispatch.

`viz_interactive.HandVizApp` in ROS mode talks to a duck-typed bridge -- `status`,
`read_state`, `play`, `set_estop` -- and this is the implementation. The split is
deliberate and load-bearing: the visualizer knows about solves and viser and
nothing about ROS, this knows about ROS and frames and nothing about solving, and
`gepetto_solvers`' `robot_plan` -- which neither imports from the other -- is the
pure conversion in between. That is what keeps `gepetto_solvers` free of an rclpy
dependency and lets the plan export be tested with no robot.

HAND-AGNOSTIC BY CONSTRUCTION. Nothing here reads a tendon, a joint or a finger
name. The plan arrives already carrying whatever this hand is commanded in, the
readback is decoded by the hand's own driver class, and what crosses the wire is
`robot_plan.flatten_plan`'s output field for field. The only per-robot object this
file holds is the profile, and it uses it for two things: which TF frame is the
wrist, and which driver class knows how to decode a state message.

THIS PROCESS DOES NOT DRIVE THE ROBOT. It used to: the playback loop ran here,
inside `viz_node`, behind viser's GIL, and a scene update long enough to outlast
MoveIt Servo's command timeout turned into tracking error and a tripped watchdog.
The loop now lives in `executor_node`, in its own process, and `play()` is an
action client that ships a plan there and relays feedback back. Nothing in this
file publishes a motion command -- the only publisher left is the e-stop.

E-STOP, FROM HERE. `set_estop(True)` fires both paths, in this order:

  1. Latch `/gepetto/estop`. Cell-wide, transient-local, and the one that must
     not be lost -- `home_node` and `executor_node` both watch it, and the
     executor halts its publishers from inside the subscription callback.
  2. Cancel the running `PlayPlan` goal, without waiting on the future (this runs
     on a viser button-click thread and must not block).

Either alone is sufficient; both are cheap. The halt itself executes in the
executor's process rather than this one, which is why the button got FASTER by
moving the control loop out: a stop pressed during a heavy scene update no longer
queues behind the scene update.

REGISTRATION -- the table convention that links the viser world to `lbr_link_0`
-- lives in `frames.py`, since the executor needs the same conversion.
"""

import threading
from dataclasses import dataclass, field

import numpy as np
from rclpy.action import ActionClient
from std_msgs.msg import Bool

from gepetto_ros_interfaces.action import GraspClose, MoveHome, PlayPlan
from gepetto_solvers.core import robot_plan
from gepetto_solvers.core.solvers import pack_vectors

from .frames import FrameLookup, matrix_to_pose
from .hand_state import HandStateReader
from .tactile import TactileReader
from .topics import (
    ESTOP_QOS,
    ESTOP_TOPIC,
    GRASP_CLOSE_ACTION,
    MOVE_HOME_ACTION,
    PLAY_PLAN_ACTION,
)
from .wire import plan_to_goal

__all__ = ["ESTOP_QOS", "ESTOP_TOPIC", "RobotState", "RobotBridge"]


@dataclass
class RobotState:
    """What the robot says it is doing, in the visualizer's terms."""
    wrist_pose: np.ndarray                      # 4x4, VISER world frame
    #: Solver digit name -> the digit's measured command vector, in the units the
    #: plan for this hand is written in: metres of tendon displacement (width 1)
    #: or radians of joint position (width 4). Same shape as a
    #: ``robot_plan.Waypoint``'s, deliberately -- the visualizer plots the two
    #: against each other.
    digit_cmd: dict[str, np.ndarray] = field(default_factory=dict)
    age: float | None = None                 # seconds since the state message
    #: Where the state was read from, for a status line that has to say what is
    #: missing when nothing arrives.
    source: str = ""


@dataclass
class _ResumePoint:
    """Where an interrupted run stopped, and which plan it stopped in.

    The signature exists to answer one question: is the plan being played now the
    same one that was interrupted? Resuming into a DIFFERENT trajectory -- the
    operator re-solved, moved the scrubber, or switched objects while the cell was
    stopped -- would start the arm from the middle of a path it has never been on.
    Worth the few lines to make impossible.
    """
    signature: tuple
    waypoint: int


def _plan_signature(plan):
    """Cheap identity for a plan: how many waypoints, where it starts and ends,
    and which scene it was built in. Rounded, because a plan that round-tripped
    through float64 poses on the wire is the same plan."""
    first = np.round(plan.waypoints[0].wrist_pose, 6).tobytes()
    last = np.round(plan.waypoints[-1].wrist_pose, 6).tobytes()
    corner = np.round(np.asarray(plan.corner_viz, float), 6).tobytes()
    return (len(plan.waypoints), first, last, corner)


class RobotBridge:
    """Frames, feedback, and dispatch to the executor, for one visualizer."""

    def __init__(self, node, profile, hand, hand_state_topic=None,
                 result_timeout=600.0):
        self._node = node
        self._log = node.get_logger()
        self._profile = profile
        self._hand = hand
        self._frames = FrameLookup(node, profile.planning_frame,
                                   profile.wrist_frame, profile.table_frame)
        self._result_timeout = float(result_timeout)

        # Decoded by the HAND's driver class, so this file never learns what a
        # `*_flex` name or a `joint_7_0` is.
        self._state = HandStateReader(node, hand, profile.hand_driver,
                                      topic=hand_state_topic)

        self._estop = threading.Event()
        self._estop_reason = ""
        self._latch = None              # the visualizer's EStop, once bound
        self._estop_pub = node.create_publisher(Bool, ESTOP_TOPIC, ESTOP_QOS)
        node.create_subscription(Bool, ESTOP_TOPIC, self._on_estop_msg, ESTOP_QOS)

        # Read for the status line only. The close itself is run entirely in the
        # executor's process -- this side never sees a tactile count while one is
        # in flight, it sees the feedback the action sends back -- but "is the
        # sensor even publishing" has to be answerable BEFORE the operator arms
        # anything, and the Robot folder is where they would look.
        self._tactile = TactileReader(node, hand, profile.hand_driver)

        self._client = ActionClient(node, PlayPlan, PLAY_PLAN_ACTION)
        self._grasp_client = ActionClient(node, GraspClose, GRASP_CLOSE_ACTION)
        # Served by `home_node`, a DIFFERENT process from the executor above --
        # resident, with its own control loop over the same servo and hand
        # publishers. Which is exactly why it shares the goal slot below: the two
        # processes drive one robot and must never overlap.
        self._home_client = ActionClient(node, MoveHome, MOVE_HOME_ACTION)
        #: ONE slot for all three actions. They are mutually exclusive on the
        #: hardware -- the executor refuses a play while a close runs and vice
        #: versa, and `home_node` refuses a home while the cell is stopped -- so
        #: there can never be two live handles, and one slot means the e-stop's
        #: cancel path does not have to know which kind of run it is interrupting.
        self._goal_handle = None
        self._goal_lock = threading.Lock()
        self._resume = None
        self._last_note = ""

    # ---------------------------------------------------------------- e-stop

    def bind_estop(self, latch):
        """Adopt the visualizer's :class:`EStop` as the authority.

        Two directions, and both are needed: the visualizer's latch reaches this
        bridge (it registers `set_estop` as a listener), and a trip arriving over
        ROS reaches the visualizer through this reference, so the browser greys
        out for a stop pressed elsewhere in the cell."""
        self._latch = latch

    def set_estop(self, tripped, reason=""):
        """The latch moved. Mirror it onto ROS and interrupt any running goal.

        Order matters: the latched topic goes first because it is the one that
        must not be lost -- it is what actually halts the publishers, in the
        executor's process, from its own subscription callback. The cancel is a
        second, independent path to the same stop; it is issued without waiting
        on its future, because this runs on the viser thread that serviced the
        button click and blocking it is how a stop button stops working.
        """
        if tripped:
            self._estop.set()
            self._estop_reason = reason
        else:
            self._estop.clear()
            self._estop_reason = ""
        self._publish_estop(tripped)
        if tripped:
            self._cancel_goal()

    def _cancel_goal(self):
        with self._goal_lock:
            handle = self._goal_handle
        if handle is None:
            return
        try:
            handle.cancel_goal_async()      # deliberately not waited on
        except Exception as exc:
            # Never fatal: the latched topic above has already stopped the cell,
            # and this path exists only to end the goal a beat sooner.
            self._log.warn(f"could not cancel playback goal: {exc}")

    def _publish_estop(self, tripped):
        msg = Bool()
        msg.data = bool(tripped)
        self._estop_pub.publish(msg)

    def _on_estop_msg(self, msg):
        """A trip from elsewhere in the cell. Ignored unless it is news, so the
        state we publish ourselves does not loop back in as a fresh trip."""
        if not msg.data or self._estop.is_set():
            return
        self._estop.set()
        self._estop_reason = f"trip received on {ESTOP_TOPIC}"
        self._cancel_goal()
        if self._latch is not None:
            # Runs the visualizer's own trip path, so the GUI greys out and the
            # solve loop stops too -- not just the playback.
            self._latch.trip(self._estop_reason)

    def stop(self, reason="stopped"):
        """Halt everything now. Kept as a name of its own because it reads at the
        call site as what it is."""
        self.set_estop(True, reason)

    # ------------------------------------------------------------ registration

    def registration(self, corner_viz):
        """``T_base<-viz`` for a viser scene whose table corner is at ``corner_viz``."""
        return self._frames.registration(corner_viz)

    def wrist_pose_base(self):
        """The hand's wrist frame in the planning frame, from TF."""
        return self._frames.wrist_pose_base()

    # --------------------------------------------------------------- feedback

    def hand_state(self):
        """``({digit: vector}, age_seconds)``; ``({}, None)`` if nothing arrived."""
        return self._state.read()

    def read_state(self, corner_viz):
        """The robot's current state, expressed for the visualizer."""
        T_base_viz = self.registration(corner_viz)
        wrist_viz = np.linalg.inv(T_base_viz) @ self.wrist_pose_base()
        digit_cmd, age = self.hand_state()
        return RobotState(wrist_pose=wrist_viz, digit_cmd=digit_cmd, age=age,
                          source=self._state.topic)

    def status(self):
        """One markdown line for the Robot folder: what is actually reachable."""
        parts = [f"**{self._profile.name}**"]
        try:
            self.registration(np.zeros(3))
            parts.append("table registered")
        except RuntimeError:
            parts.append("**no table TF**")
        try:
            self.wrist_pose_base()
            parts.append(f"wrist TF ok (`{self._profile.wrist_frame}`)")
        except RuntimeError:
            parts.append(f"**no wrist TF** (`{self._profile.wrist_frame}`)")

        digit_cmd, age = self.hand_state()
        if age is None:
            parts.append(f"**no hand state** on `{self._state.topic}`")
        else:
            suffix, scale = robot_plan.COMMAND_UNITS[
                robot_plan.command_kind(self._hand)]
            # The digit's LARGEST commanded value, not all of them: sixteen
            # numbers do not fit a status line, and what the line is being read
            # for is "is anything moving, and roughly where is it".
            parts.append(
                f"hand {age:.1f}s ago ("
                + ", ".join(f"{k} {float(np.max(v)) * scale:.1f}"
                            for k, v in sorted(digit_cmd.items()))
                + f" {suffix} peak)")
        # The control loop is a different process, so "is it there" is a real
        # question with a real answer, and a missing executor should say so
        # rather than looking like a play that quietly did nothing.
        if not self._client.server_is_ready():
            parts.append(f"**no executor on {PLAY_PLAN_ACTION}** "
                         f"(`ros2 run gepetto_ros executor_node "
                         f"--ros-args -p hand:={self._profile.name}`)")
        # Only on a hand that HAS fingertips. A line saying "no tactile" about
        # the tendon hand would be reporting the absence of a sensor it was never
        # built with, which reads as a fault rather than as a fact.
        if self._tactile.available:
            _counts, tactile_age = self._tactile.read()
            if tactile_age is None:
                parts.append(f"**no tactile** on `{self._tactile.topic}`")
            elif tactile_age > 1.0:
                parts.append(f"**tactile {tactile_age:.1f}s old**")
            else:
                parts.append("tactile ok")
            if not self._grasp_client.server_is_ready():
                parts.append(f"**no grasp server on {GRASP_CLOSE_ACTION}**")
        # Resident, so its absence is a bring-up fault rather than a state --
        # and *Go home* is the one button on the panel that needs it.
        if not self._home_client.server_is_ready():
            parts.append(f"**no home server on {MOVE_HOME_ACTION}** "
                         f"(`ros2 run gepetto_ros home_node "
                         f"--ros-args -p hand:={self._profile.name} "
                         f"-p on_start:=false`)")
        if self._resume is not None:
            parts.append(f"paused at waypoint {self._resume.waypoint}")
        if self._estop.is_set():
            parts.append(f"**E-STOP: {self._estop_reason}**")
        return " &nbsp;|&nbsp; ".join(parts)

    # --------------------------------------------------------------- playback

    def play(self, plan, enable_arm, enable_hand, speeds, on_progress=None,
             should_stop=None, on_sample=None):
        """Send ``plan`` to the executor and block until it finishes.

        The same blocking contract the in-process loop had, so the visualizer's
        worker thread behaves identically -- it is just that the ticking now
        happens somewhere a browser cannot reach.
        """
        if self._estop.is_set():
            raise RuntimeError(f"E-STOP engaged: {self._estop_reason}")
        if not self._client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError(
                f"no executor on {PLAY_PLAN_ACTION}. Start it with "
                f"`ros2 run gepetto_ros executor_node --ros-args -p "
                f"hand:={self._profile.name}`, or use "
                f"`ros2 launch gepetto_ros_launch gepetto_bringup.launch.py`, "
                f"which does.")

        # Signed BEFORE slicing: the client sends the whole plan every time, so
        # the identity a resume point is checked against has to be the whole
        # plan's, and the waypoint it names has to be an index into it.
        signature = _plan_signature(plan)
        plan, offset, resume_note = self._apply_resume(plan, signature)

        goal = self._to_goal(plan, enable_arm, enable_hand, speeds)

        def on_feedback(message):
            if on_sample is not None:
                try:
                    on_sample(message.feedback)
                except Exception:
                    # A sampler is a diagnostic. It must never be able to take
                    # down the feedback path that carries the status line, and
                    # it must never abort a run that is physically in progress.
                    self._log.warn("playback sampler raised; continuing",
                                   throttle_duration_sec=5.0)
            if on_progress is None:
                return
            line = self._format_feedback(message.feedback)
            # The resume note rides along on EVERY feedback line rather than
            # being announced once. Posted once it survived about 100 ms before
            # the first feedback overwrote it, which left a resumed run --
            # legitimately playing only the tail it was interrupted in -- looking
            # exactly like a playback that had silently dropped most of its
            # waypoints. Those are different faults and must not read the same.
            on_progress(f"{resume_note}  \n{line}" if resume_note else line)

        send_future = self._client.send_goal_async(goal, feedback_callback=on_feedback)
        handle = self._wait(send_future, "sending the playback goal")
        if handle is None or not handle.accepted:
            raise RuntimeError(
                "the executor refused the goal -- it is already playing, the "
                "e-stop is engaged, no channel was enabled, or it is running a "
                "different hand than this visualizer is. Check its log; it says "
                "which.")

        with self._goal_lock:
            self._goal_handle = handle
        try:
            # The E-STOP path may cancel this handle from another thread at any
            # point below; that is the whole reason it is stored first.
            if should_stop is not None and should_stop():
                handle.cancel_goal_async()
            result = self._wait(handle.get_result_async(), "the playback result",
                                timeout=self._result_timeout)
        finally:
            with self._goal_lock:
                self._goal_handle = None

        if result is None:
            raise RuntimeError("the executor never returned a result")
        return self._absorb(signature, offset, result.result, resume_note)

    # ------------------------------------------------------------ grasp close

    def grasp_close(self, digit_names, directions, targets, speed, max_travel,
                    abort_counts, release=False, on_progress=None,
                    should_stop=None):
        """Close the named digits onto the object, or back out of a close.

        ``directions`` is ``{digit: (3,)}`` and must already be in the WRIST
        frame -- see the action file on why the rotation happens on the client.
        ``targets`` is ``{digit: counts above rest}``. Blocks until the run
        finishes, the same contract :meth:`play` has, so the visualizer's worker
        thread behaves identically for both.

        Returns the verdict as markdown.
        """
        if self._estop.is_set():
            raise RuntimeError(f"E-STOP engaged: {self._estop_reason}")
        if not self._grasp_client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError(
                f"no grasp server on {GRASP_CLOSE_ACTION}. It is served by the "
                f"same node as playback, so if that is up this is an executor "
                f"too old to have it -- rebuild `gepetto_ros` and "
                f"`gepetto_ros_interfaces`.")

        names = list(digit_names)
        goal = GraspClose.Goal()
        goal.digit_names = names
        goal.directions = ([] if release
                           else pack_vectors(names, directions))
        goal.force_targets = ([] if release
                              else [float(targets[n]) for n in names])
        goal.speed = float(speed)
        goal.max_travel = float(max_travel)
        goal.abort_counts = float(abort_counts)
        goal.release = bool(release)

        def on_feedback(message):
            if on_progress is None:
                return
            on_progress(self._format_grasp_feedback(names, message.feedback))

        send_future = self._grasp_client.send_goal_async(
            goal, feedback_callback=on_feedback)
        handle = self._wait(send_future, "sending the grasp goal")
        if handle is None or not handle.accepted:
            raise RuntimeError(
                "the executor refused the close -- it is already playing or "
                "closing, the e-stop is engaged, the fingertip sensor is not "
                "publishing, or (for a release) no close has run yet. Check its "
                "log; it says which.")

        with self._goal_lock:
            self._goal_handle = handle
        try:
            if should_stop is not None and should_stop():
                handle.cancel_goal_async()
            result = self._wait(handle.get_result_async(), "the grasp result",
                                timeout=self._result_timeout)
        finally:
            with self._goal_lock:
                self._goal_handle = None

        if result is None:
            raise RuntimeError("the executor never returned a grasp result")
        self._last_note = result.result.verdict
        return result.result.verdict

    def grasp_available(self):
        """``(available, why_not)`` -- whether a tactile close could run now.

        Asked by the visualizer to decide whether to build the panel at all, so
        it answers with a REASON rather than a bool: a greyed-out button that
        cannot say why is worse than no button.
        """
        if not self._tactile.available:
            return False, (f"the {self._hand.name} hand's driver declares no "
                           f"fingertip sensor")
        if not self._grasp_client.server_is_ready():
            return False, f"no grasp server on `{GRASP_CLOSE_ACTION}`"
        _counts, age = self._tactile.read()
        if age is None:
            return False, f"nothing has arrived on `{self._tactile.topic}`"
        return True, ""

    # --------------------------------------------------------------- go home

    def move_home(self, skip_arm=False, skip_hand=False, timeout=0.0,
                  on_progress=None, should_stop=None):
        """Park the cell: the arm folded, the hand at its rest posture.

        NOT a one-waypoint :meth:`play`, and the difference is the point. Home is
        a property of the CELL rather than of the caller -- the arm's folded
        configuration is ``RobotProfile.arm_home`` and the hand's rest is the
        hand's own -- so the target is deliberately absent from the goal, and the
        arm is driven in JOINT space, where the move is reachable by construction
        and cannot stall in a singularity or come back on a different elbow.
        Served by ``home_node``, which is resident: it comes up with the cell and
        every trip home is a goal on it.

        ``timeout`` of 0 means the node's own ``timeout`` parameter, which is the
        answer unless the caller has a reason of its own. Blocks until the run
        finishes -- the same contract :meth:`play` and :meth:`grasp_close` have,
        so the visualizer's worker thread behaves identically for all three.

        Returns the outcome as markdown.
        """
        if self._estop.is_set():
            raise RuntimeError(f"E-STOP engaged: {self._estop_reason}")
        if not self._home_client.wait_for_server(timeout_sec=2.0):
            raise RuntimeError(
                f"no home server on {MOVE_HOME_ACTION}. `home_node` is resident "
                f"-- it is meant to be up with the cell -- so this is a cell "
                f"brought up without it, or a `gepetto_ros` too old to have it. "
                f"Start one with `ros2 run gepetto_ros home_node --ros-args "
                f"-p hand:={self._profile.name} -p on_start:=false`.")

        goal = MoveHome.Goal()
        goal.skip_arm = bool(skip_arm)
        goal.skip_hand = bool(skip_hand)
        goal.timeout = float(timeout)

        def on_feedback(message):
            if on_progress is None:
                return
            on_progress(self._format_home_feedback(message.feedback))

        send_future = self._home_client.send_goal_async(
            goal, feedback_callback=on_feedback)
        handle = self._wait(send_future, "sending the home goal")
        if handle is None or not handle.accepted:
            raise RuntimeError(
                "`home_node` refused the goal -- it is already going home, the "
                "e-stop is engaged, or the goal skipped both channels. Check "
                "its log; it says which.")

        with self._goal_lock:
            self._goal_handle = handle
        try:
            if should_stop is not None and should_stop():
                handle.cancel_goal_async()
            result = self._wait(handle.get_result_async(), "the home result",
                                timeout=self._result_timeout)
        finally:
            with self._goal_lock:
                self._goal_handle = None

        if result is None:
            raise RuntimeError("`home_node` never returned a result")
        self._last_note = result.result.message
        return result.result.message

    def _format_home_feedback(self, feedback):
        # `hand_error` arrives ALREADY in the hand's display units -- home_node
        # scales it before publishing -- so this reads the suffix off the hand
        # and must not scale it a second time.
        suffix, _scale = robot_plan.COMMAND_UNITS[
            robot_plan.command_kind(self._hand)]
        marks = ("arm " + ("at home" if feedback.arm_at_home else "moving"),
                 "hand " + ("at home" if feedback.hand_at_home else "moving"))
        return (f"**going home** &nbsp; {feedback.elapsed:.1f}s"
                f"  \n`arm ` {feedback.joint_error * 1e3:6.1f} mrad &nbsp; "
                f"`hand` {feedback.hand_error:6.2f} {suffix}"
                f"  \n_{' &nbsp; '.join(marks)}_")

    @staticmethod
    def _format_grasp_feedback(names, feedback):
        # Positional labels where a name is missing. A RELEASE goal names no
        # digits -- the executor supplies them from the close it remembers -- so
        # this side genuinely does not know them until the result comes back, and
        # a numbered row is better than a blank block.
        labels = [names[i] if i < len(names) else f"digit {i}"
                  for i in range(len(feedback.travel))]
        cells = "  \n".join(
            f"`{label:<7}` {travel * 1e3:5.1f} mm &nbsp; {counts:.0f} counts"
            for label, travel, counts in zip(labels, feedback.travel,
                                             feedback.contact_counts))
        return (f"**closing** &nbsp; {feedback.digits_stopped}"
                f"/{feedback.num_digits} stopped &nbsp; _{feedback.note}_"
                f"  \n{cells}")

    def _wait(self, future, what, timeout=30.0):
        """Block a worker thread on a future.

        `spin_until_future_complete` is NOT used: this runs on a viser worker
        thread while the node is already being spun by the node's own executor,
        and spinning it from a second thread is how rclpy deadlocks. Waiting on
        the future's own event is what an action client is meant to do from a
        thread that is not the spinner.
        """
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            raise RuntimeError(f"timed out after {timeout:.0f} s waiting for {what}")
        return future.result()

    def _apply_resume(self, plan, signature):
        """Slice a stored resume point onto ``plan``, if it is the same plan.

        Returns ``(plan, offset, note)``; ``offset`` is how far into the original
        the sliced plan starts, so a second interruption still reports an index
        the original would recognise.

        Done here rather than in the executor so the action stays a plain "play
        these waypoints" and knows nothing about continuations.
        """
        resume = self._resume
        self._resume = None
        if resume is None:
            return plan, 0, None
        if signature != resume.signature:
            # The operator re-solved, moved the scrubber, or switched objects
            # while the cell was stopped. Resuming into a different trajectory
            # would start the arm from the middle of a path it has never been on.
            return plan, 0, ("_the paused run was for a different solve; playing "
                             "this one from the start._")
        if resume.waypoint <= 0 or resume.waypoint >= len(plan.waypoints):
            return plan, 0, None

        from dataclasses import replace
        sliced = replace(plan, waypoints=list(plan.waypoints[resume.waypoint:]))
        return sliced, resume.waypoint, (
            f"**resuming** from waypoint {resume.waypoint}"
            f"/{len(plan.waypoints) - 1} -- the executor re-approaches it from "
            f"wherever the arm actually is.")

    def _to_goal(self, plan, enable_arm, enable_hand, speeds):
        """The plan's own flat encoding, plus this run's switches.

        The plan half is `wire.plan_to_goal`, which is `robot_plan.flatten_plan`
        with poses built -- so there is no second copy of the layout here for the
        executor's decoder to disagree with.
        """
        goal = plan_to_goal(plan, PlayPlan.Goal(), matrix_to_pose)
        goal.enable_arm = bool(enable_arm)
        goal.enable_hand = bool(enable_hand)
        goal.max_linear = float(speeds["max_linear"])
        goal.max_angular = float(speeds["max_angular"])
        goal.max_digit = float(speeds["max_digit"])
        return goal

    def _format_feedback(self, feedback):
        if feedback.settling:
            # The brief fixed hold at the end, not a convergence wait -- so this
            # reports where the arm IS, not how far it has left to go.
            head = (f"**holding** &nbsp; {feedback.position_error * 1e3:.1f} mm / "
                    f"{np.degrees(feedback.rotation_error):.1f}° from the goal")
        else:
            head = (f"**playing** &nbsp; t={feedback.path_time:.1f}"
                    f"/{feedback.path_total:.1f}s &nbsp; "
                    f"waypoint {feedback.waypoint}/{feedback.num_waypoints}"
                    f" &nbsp; _{feedback.position_error * 1e3:.0f} mm behind_")
        return f"{head}  \n{feedback.note}"

    def _absorb(self, signature, offset, result, resume_note):
        """Turn a result into the status line, and remember where it stopped."""
        if result.resumable:
            # `offset` puts the executor's index -- which counts from the start
            # of what it was SENT -- back into the original plan's numbering.
            self._resume = _ResumePoint(
                signature=signature,
                waypoint=offset + int(result.resume_waypoint))
        note = result.verdict
        if result.ticks:
            note += f" &nbsp; {result.ticks} ticks, {result.elapsed:.1f} s elapsed"
        if result.final_position_error or result.final_rotation_error:
            note += (f"  \nfinal error "
                     f"{result.final_position_error * 1e3:.1f} mm / "
                     f"{np.degrees(result.final_rotation_error):.1f}°")
        if resume_note:
            note = f"{resume_note}  \n{note}"
        self._last_note = note
        return note

    @property
    def playing(self):
        with self._goal_lock:
            return self._goal_handle is not None
