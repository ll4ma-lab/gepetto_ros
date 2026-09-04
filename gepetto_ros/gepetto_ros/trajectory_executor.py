"""The control loop: a time-scaled path, played on one clock.

This runs in `executor_node`, which has no visualizer in it. That separation is
the point -- see the node's docstring for why -- and this module is where the
consequences live.

ONE LOOP, EITHER HAND. Nothing in this file knows which robot it is driving. The
plan arrives carrying per-digit command VECTORS and a `command_kind` saying what
they mean, and the hand end of the tick is a `HandDriver` handed in by the node --
so a tendon hand's metres of displacement and an Allegro's radians of joint
position are the same code path from here down. The arm half is identical for both
robots outright: same servo node, same scales, same chain-tip-is-the-wrist
coincidence (see `ArmTwistServo`).

WHAT COORDINATES THE ARM AND THE HAND. One number: the path parameter `t_path`.
`robot_plan.plan_schedule` gives every segment a single duration -- the longest
its wrist translation, its wrist rotation and its digit travel each need at their
own ceilings -- and both channels are then sampled from the same `t_path` by
`robot_plan.sample_at`. Neither channel has a clock of its own, so neither can run
ahead of the other. Simultaneous arrival is a property of the schedule, not
something the loop has to steer toward.

WHY THE CLOCK IS FREE-RUNNING. `t_path` advances by exactly one period every
tick, always. It is never stretched, never held, and never waits for the arm.
This is the whole point and it is worth being blunt about the consequence: the
DURATION OF A RUN IS A FUNCTION OF THE PLAN ALONE. It does not depend on how well
the arm tracked, which is what makes "the fingers and the wrist arrive together"
a guarantee rather than a hope.

An earlier version paced `t_path` against the arm's tracking error, on the
reasoning that a clock which cannot outrun the arm can never skip a waypoint.
That is true, and it is also precisely what broke the coordination: the hand has
no error term in that ramp, so the clock stretched for the arm alone while the
digits -- which track their far slower ceiling closely -- sat exactly where the
stretched path put them. The two channels were synchronized in COMMAND and
divergent in REALITY. Tracking error is now corrected where it belongs, in the
control law below, instead of being fed back into time.

THE CONTROL LAW. Feed-forward plus proportional correction, on the manifold:

    T_d      = T_k @ se3_exp(V_cmd * elapsed)        # the reference, from the plan
    V_err    = se3_log(T_act^-1 @ T_d)               # body-frame error twist
    V_u      = Ad(T_act^-1 @ T_d) @ V_cmd + Kp @ V_err

The Adjoint is not decoration. It maps the feed-forward twist -- which is defined
relative to the REFERENCE pose -- onto the frame the arm is ACTUALLY at. It drops
to the identity only when `T_act == T_d`, and the standing lag described below is
exactly the condition that keeps them apart, so dropping it would leave the
feed-forward subtly wrong in precisely the regime it matters.

WHY THE COMMAND CLAMP IS NOT THE PATH SPEED. `segment_durations` times the path
so it travels at `max_linear`; if the command is then also clamped to
`max_linear`, the feed-forward alone saturates it and the proportional term has
no room to do anything but rotate the command a few degrees. It can never close
a gap -- error only ever grows. The clamp gets `velocity_headroom` times the path
speed, capped at MoveIt Servo's own scale (above which `ArmTwistServo._publish`
would clip per component and bend the direction).

WHY THE CORRECTION TERM CANNOT BE DROPPED. The pipeline below this loop does not
move the arm at the speed it is told to. MoveIt Servo re-bases on the reported
joint state every cycle and adds one `publish_period` of travel, so its output
never leads the state by more than that step; the FRI position command is then
exponentially smoothed (`joint_position_tau`, 0.04 s), and with `open_loop: true`
it is that SMOOTHED COMMAND which comes back as the joint state. The step that
actually lands is therefore `1 - exp(-period / tau)` of the step asked for -- and
`moveit_servo.yaml` sizes its step for 200 Hz while `controller_manager` runs at
100 Hz, halving it again. Measured end to end, the wrist moves at about a TENTH
of the commanded Cartesian speed. A purely open-loop stream of feed-forward
twists against that plant undershoots every segment and compounds the shortfall
along the path; the proportional term is what makes the reference land.

THE TERMINAL HOLD IS NOT A SETTLE. When `t_path` reaches the end, the loop holds
the final pose with zero feed-forward for a FIXED `terminal_hold` seconds and
then stops, unconditionally. It never tests an error threshold and its length
never depends on convergence -- it exists only so the smoother's in-flight
command lands instead of being cut dead mid-lag. What it replaced was a real
settle that ran up to twenty seconds against a 2 mm tolerance, which is what put
seconds of arm-only creep after the fingers had visibly stopped.

WHAT THE WATCHDOG IS FOR. Safety, and nothing else. It is not a transition
criterion: being behind never changes what the loop does next, it only decides
whether continuing is safe at all. A servo that never came up, a singularity it
cannot leave or a collision stop all show up the same way against a free-running
clock -- error that grows without bound -- so the plain ceiling catches them, and
the separate stall detector the paced clock needed is gone with the pacing.
"""

import threading
import time
from dataclasses import dataclass

import numpy as np

from gepetto_solvers.core import robot_plan

from .arm_servo import SERVO_SCALE_LINEAR, SERVO_SCALE_ROTATIONAL, ArmTwistServo

#: MoveIt Servo's `StatusCode`, which it publishes on `~/status`. The names are
#: the whole reason to subscribe: every one of these makes the arm move slower
#: than commanded, and from inside this loop they are otherwise indistinguishable
#: from each other and from a plant that is simply sluggish. Collision and
#: singularity deceleration are ON by default in `moveit_servo.yaml`
#: (`check_collisions: true`, `lower_singularity_threshold: 17.0`), and a hand
#: closing around an object on a table is exactly the situation that trips the
#: proximity thresholds.
SERVO_STATUS = {
    0: "ok",
    1: "DECELERATING near a singularity",
    2: "DECELERATING near a collision",
    3: "HALTED at a singularity",
    4: "HALTED at a collision",
    5: "HALTED at a joint bound",
}

# `robot_plan` is imported at module scope, plainly. It used to be fetched through
# a lazy shim because it lived in a tree that was not an installed package and had
# to be put on `sys.path` by the node at startup; `gepetto_solvers` is a real
# installed package now, so there is nothing left to defer.
#
# This module leans on it for more than the plan types: the se(3) exponential,
# logarithm and Adjoint the control law is written in all live there, as the
# single copy. Duplicating them here would put a second matrix logarithm in the
# tree, which drifts silently.


@dataclass
class ExecutorLimits:
    """Everything the loop is tuned by. One object so the node can declare them
    as parameters and hand them over without a fourteen-argument call."""
    hz: float = 100.0
    #: Proportional gains on the body-frame error twist, as the two diagonal
    #: blocks of Kp: linear on the first three components, angular on the last.
    kp_linear: float = 2.0
    kp_angular: float = 2.0
    #: The fraction of commanded Cartesian velocity the pipeline below this loop
    #: actually realises, measured. EVERY COMMAND IS DIVIDED BY IT.
    #:
    #: This is gain compensation, and it is the right tool because the loss is a
    #: known STATIC GAIN rather than a disturbance. MoveIt Servo re-bases on the
    #: measured joint state each cycle (`q_cmd = q_meas + v * publish_period`)
    #: and `lbr_fri_ros2` low-passes the position command, so in steady state the
    #: arm advances `alpha * publish_period / controller_period` of what is
    #: asked. Measured on this cell: 0.004 m/s achieved for 0.060 commanded, and
    #: 0.009 for 0.114 -- about 7% both times, with Servo reporting `ok 100%`.
    #:
    #: Fighting that with `kp` alone does not work and the numbers say why: at
    #: kp = 2.0 a 25 mm error asks for 0.05 m/s, so the loop never pushed past a
    #: seventh of the ceiling it had been given and left 0% of ticks saturated.
    #: Dividing instead makes the feed-forward land the path speed EXACTLY --
    #: `g * (v/g) == v` -- with zero steady-state error, and leaves `kp` to do
    #: what a proportional term should: reject what the model got wrong. It also
    #: restores the meaning of `kp`, whose effective gain becomes `kp` again
    #: rather than `kp * 0.07`.
    #:
    #: RAISE IT if the pipeline is fixed (see the `publish_period` note below);
    #: the `implied gain` line in the diagnostic block reports the measurement to
    #: set it from. Too high overshoots and the correction pulls it back; too low
    #: undershoots exactly as it did before this existed.
    pipeline_gain: float = 0.07
    #: How much faster than the path the command may be, so the correction term
    #: has room to close a gap. `play` caps the PATH speed at
    #: `servo scale * pipeline_gain / headroom`, so this one number sets both how
    #: much authority the correction has AND how fast a path may be timed.
    #:
    #: Back to 2.0 now that `pipeline_gain` carries the plant's loss: this is
    #: once again just "let the correction ask for twice the path speed", in the
    #: units the arm actually moves in.
    #:
    #: WORTH FIXING UPSTREAM rather than compensating for forever:
    #: `moveit_servo.yaml` sets `publish_period: 0.005` against a
    #: `controller_manager` `update_rate: 100`, i.e. half a step per cycle, from
    #: a line whose own comment says it should be 1/(controller rate). Correcting
    #: it to 0.01 should roughly double `pipeline_gain` and halve every commanded
    #: twist for the same motion.
    velocity_headroom: float = 2.0
    #: Safety ceilings, not transition criteria. Nothing about the loop's
    #: sequencing depends on the tracking error; these only decide whether
    #: continuing to command the arm is safe at all.
    position_watchdog: float = 0.15     # m
    rotation_watchdog: float = np.radians(45.0)
    #: How long the final pose is held with zero feed-forward before the run
    #: ends. A fixed duration, never a convergence test: the loop counts off this
    #: many ticks and stops, whatever the error is doing.
    #:
    #: HOW LONG IT NEEDS TO BE DEPENDS ON HOW FAST THE PATH ENDS. When the
    #: reference stops while the arm is still travelling, the plant coasts about
    #: `v * tau` past it and the error goes UP before the correction brings it
    #: back -- so a hold sized for a path that ends at speed has to outlast that
    #: bump, which on the harness means seconds.
    #:
    #: A REAL SOLVE DOES NOT END AT SPEED. An AL history converges, so its tail
    #: is micro-moves and then nothing. Measured on a 32-iterate solve of the
    #: default scene: every segment pinned to `min_duration`, peak wrist travel
    #: 0.022 mm per segment, and the last eight segments -- 0.4 s of the 1.55 s
    #: replay -- exactly zero on both channels. The arm is already stationary
    #: when the clock runs out, so there is no coast to absorb and the converged
    #: tail has acted as its own settle without being one. Half a second is
    #: margin on top of that, not a convergence budget.
    #:
    #: The synthetic path `ros2 run gepetto_ros play_client` sends is the opposite
    #: case -- it ends mid-sweep at full commanded speed -- so expect tens of
    #: millimetres of final error there and do not tune this against it. If a REAL
    #: playback finishes visibly short, measure first: the client's per-tick error
    #: print shows whether the tail is actually slow, and raising
    #: `kp_linear`/`kp_angular` treats the lag rather than waiting it out. What
    #: must NOT happen is this growing back into a convergence-gated settle.
    terminal_hold: float = 0.5          # s
    #: How old the wrist TF may be and still be servoed on. NOT a liveness
    #: check -- see `tf_stale_grace` for that.
    #:
    #: The floor is set by `robot_state_publisher`, which republishes TF at its
    #: `publish_frequency` (20 Hz by default, measured at 18 Hz in this cell), so
    #: a perfectly healthy cell sawtooths this figure between roughly 0 and 60 ms.
    #: Under load -- a solve running in `viz_node`, Servo's collision check going
    #: long -- that floor rises: `run_duration: 0.69 (0.005)` out of
    #: `moveit_servo` and a TF age of 200 ms are the same event seen twice.
    tf_max_age: float = 0.2             # s
    #: How long the loop will coast on the LAST GOOD pose before it gives up.
    #:
    #: This is the liveness check, and it is separate from `tf_max_age` because
    #: the two questions are different: "is this pose fresh enough to correct
    #: against" is asked every tick and is routinely no on a busy desktop, while
    #: "has the publisher stopped" needs to be answered once. Aborting a playback
    #: on the first stale sample conflated them, and the symptom was a run that
    #: died on tick two with `wrist TF is 202 ms old` because a GTSAM solve was
    #: running in another process.
    #:
    #: COASTING IS SAFE HERE BECAUSE THE PATH CLOCK IS FREE-RUNNING. The
    #: reference keeps advancing while the measurement does not, so the tracking
    #: error GROWS -- a genuinely frozen publisher walks straight into
    #: `position_watchdog` within a few hundred milliseconds and aborts there,
    #: with a message about tracking rather than about TF. That is the backstop
    #: that makes a grace window a bounded risk rather than an open one. The
    #: other backstop is outside this process: Servo's `incoming_command_timeout`
    #: is 0.1 s, so an executor that stops publishing halts the arm regardless.
    tf_stale_grace: float = 1.0         # s
    #: Rate of the per-tick trace line in the executor's own log. 0 disables it.
    #: On by default at 1 Hz because a run that goes wrong is usually reported by
    #: pasting this log, and a time series is worth far more than an end summary
    #: when the question is WHEN the arm fell behind.
    trace_hz: float = 1.0


@dataclass
class Progress:
    """What the loop hands back each feedback interval. Deliberately plain data:
    the loop must not format markdown or touch a GUI."""
    tick: int = 0
    path_time: float = 0.0
    path_total: float = 0.0
    waypoint: int = 0
    num_waypoints: int = 0
    note: str = ""
    position_error: float = 0.0
    rotation_error: float = 0.0
    #: Pinned at 1.0. The clock is free-running, so the path never plays at
    #: anything but its commanded speed; kept because it is a wire field and
    #: a reader that has one is better served by a constant than by a gap.
    time_scale: float = 1.0
    #: True during the fixed terminal hold, not during a convergence wait.
    holding: bool = False


@dataclass
class Outcome:
    """How a run ended."""
    success: bool = False
    verdict: str = ""
    ticks: int = 0
    elapsed: float = 0.0
    position_error: float = 0.0
    rotation_error: float = 0.0
    #: Worst tracking error seen at any tick. With a free-running clock this is
    #: the quality number for a run: the final error only says where the arm
    #: finished, this says how well it rode the path to get there.
    max_position_error: float = 0.0
    max_rotation_error: float = 0.0
    resumable: bool = False
    resume_waypoint: int = 0
    resume_path_time: float = 0.0
    #: Set when the run must trip the cell-wide e-stop (a fault, not a cancel).
    trip_reason: str = ""


class TrajectoryExecutor:
    """Plays one plan at a time. Not reentrant; the node's action server is what
    enforces that (one goal at a time)."""

    def __init__(self, node, profile, hand, frames, limits=None,
                 hand_state=None, travel_limits=None, servo_status=None):
        self._node = node
        self._log = node.get_logger()
        self._profile = profile
        self._hand_model = hand
        self._frames = frames
        self.limits = limits or ExecutorLimits()
        #: Callable returning ``({digit: (K,) vector}, age)`` in SOLVER digit
        #: names and the plan's own units. See `hand_state.HandStateReader`.
        self._hand_state = hand_state or (lambda: ({}, None))
        #: Callable returning MoveIt Servo's status code, or None. See
        #: `SERVO_STATUS` for why this is worth a subscription.
        self._servo_status = servo_status or (lambda: None)
        #: ``{digit: (lo, hi)}`` arrays, or {} when the hand could not supply
        #: them -- which the NODE reports loudly; a missing clamp is never silent.
        self._travel = travel_limits or {}

        self._arm = ArmTwistServo(node, frame_id=profile.planning_frame)
        # Built per run, not here: it holds the running state of one playback
        # (an integrating total on one hand, a slew-limited setpoint on the
        # other) and starting a second run with the first one's is exactly the
        # lurch `seed` exists to prevent.
        self._hand = None
        self._estop = threading.Event()
        self._playing = False
        #: Last wrist pose that arrived inside `tf_max_age`, and when the current
        #: run of stale reads began. Reset per run -- see `_measure`.
        self._tf_good = None
        self._tf_stale_since = None

    # ------------------------------------------------------------------ estop

    @property
    def estop(self):
        """The Event the loop polls. The node's e-stop subscription sets it."""
        return self._estop

    @property
    def playing(self):
        return self._playing

    def halt(self):
        """Zero the arm and stop commanding the hand, right now.

        Called from the e-stop SUBSCRIPTION CALLBACK, not from the loop -- the
        loop polls the Event and will exit within a tick, but a tick is 10 ms of
        saturated twist and a stop button exists to prevent exactly that. Safe to
        call concurrently with the loop: the worst case is a zero twist landing
        between two commands, and the loop is about to exit anyway.
        """
        self._arm.halt()
        if self._hand is not None:
            self._hand.halt()

    # ---------------------------------------------------------------- helpers
    #
    # THE HARDWARE'S NAMES ARE NOT THIS FILE'S PROBLEM any more. Everything here
    # speaks SOLVER digit names -- they are what the plan carries and what the
    # visualizer shows -- and the translation to `index_flex`, or to a positional
    # slot in a sixteen-wide JointState, happens inside the hand driver at the
    # single point where a message actually leaves. That is what lets one loop
    # drive two hands whose naming schemes have nothing in common.

    def _servo_arm(self, T_target, twist_ff_body):
        """One feed-forward-plus-correction command toward ``T_target``.

        Returns ``(position_error_m, rotation_error_rad)``.

        ``T_target`` is the reference pose in the ROBOT BASE frame; ``twist_ff_body``
        is the segment's constant body twist, which needs no frame conversion on
        the way in because it is expressed in the wrist's own frame either way.
        See the module docstring for the law and why the Adjoint stays.

        The errors REPORTED are the plain Euclidean distance and rotation angle,
        not the components of the se(3) log. The log's translational part carries
        an inverse-Jacobian factor and is not a distance; the watchdog thresholds
        and every status line downstream are quoted in millimetres and degrees, so
        they get the quantity those units actually name.
        """
        T_act = self._measure()
        T_err = np.linalg.inv(T_act) @ T_target
        V_err = robot_plan.se3_log(T_err)

        gains = np.concatenate([np.full(3, self.limits.kp_linear),
                                np.full(3, self.limits.kp_angular)])
        # Ad pulls the feed-forward -- defined at the REFERENCE -- onto the frame
        # the arm is actually at. Identity only when the two coincide.
        # DIVIDED BY THE PIPELINE'S GAIN, both terms. The feed-forward then
        # lands the path speed exactly rather than a fourteenth of it, and `kp`
        # recovers its nominal meaning because `plant_gain * (kp / plant_gain)`
        # is just `kp`. See `ExecutorLimits.pipeline_gain`.
        V_body = (robot_plan.se3_adjoint(T_err) @ np.asarray(twist_ff_body, float)
                  + gains * V_err) / max(self.limits.pipeline_gain, 1e-6)

        # Body twist to what MoveIt Servo wants: the velocity of the wrist ORIGIN
        # and the angular velocity, both with components in the base frame. That
        # is a ROTATION of each half, not the full spatial Adjoint -- the latter
        # adds a `p x w` term that grows with distance from the base origin and
        # would command a point that is not the wrist.
        R_act = T_act[:3, :3]
        self._arm.command(R_act @ V_body[:3], R_act @ V_body[3:])

        # --- diagnostics only; nothing below changes what is commanded ------
        d = self._diag
        d["ticks"] += 1
        # What the control law ASKED for, before the ceiling took it away, and
        # what actually went out after. The gap between these two is the arm
        # being forbidden to move, which is a different fault from the arm being
        # unable to move and has a different fix.
        want_v = float(np.linalg.norm(V_body[:3]))
        want_w = float(np.linalg.norm(V_body[3:]))
        sent_v = min(want_v, self._arm.max_linear)
        sent_w = min(want_w, self._arm.max_angular)
        d["want_v_sum"] += want_v
        d["want_v_max"] = max(d["want_v_max"], want_v)
        d["sent_v_sum"] += sent_v
        d["sent_v_max"] = max(d["sent_v_max"], sent_v)
        d["want_w_max"] = max(d["want_w_max"], want_w)
        d["sent_w_max"] = max(d["sent_w_max"], sent_w)
        if self._arm.clamped_linear:
            d["clamped_v"] += 1
        if self._arm.clamped_angular:
            d["clamped_w"] += 1
        if self._arm.clipped_component:
            d["clipped"] += 1
        code = self._servo_status()
        d["servo"][code] = d["servo"].get(code, 0) + 1

        if self._last_actual is not None:
            step = float(np.linalg.norm(T_act[:3, 3] - self._last_actual))
            d["arm_arc"] += step
            # ACHIEVED SPEED IS MEASURED OVER A WINDOW, NOT PER TICK. The loop
            # runs at 100 Hz; the wrist TF does not necessarily update that
            # fast, so per-tick differencing alternates between zero and a
            # double step, and its "peak" is the TF's update rate showing
            # through rather than anything the arm did. A tenth-second window is
            # longer than any plausible TF period and still short enough to show
            # a genuine burst.
            d["win_arc"] += step
            d["win_ticks"] += 1
            if d["win_ticks"] >= max(1, int(self.limits.hz / 10)):
                got = d["win_arc"] / (d["win_ticks"] / self.limits.hz)
                d["got_v_sum"] += got
                d["got_v_n"] += 1
                d["got_v_max"] = max(d["got_v_max"], got)
                d["win_arc"] = 0.0
                d["win_ticks"] = 0
            d["path_arc"] += float(np.linalg.norm(
                T_target[:3, 3] - self._last_reference))
        else:
            d["arm_start"] = np.array(T_act[:3, 3])
            d["path_start"] = np.array(T_target[:3, 3])
        d["arm_end"] = np.array(T_act[:3, 3])
        d["path_end"] = np.array(T_target[:3, 3])
        self._last_reference = np.array(T_target[:3, 3])
        self._last_actual = np.array(T_act[:3, 3])

        position_error = float(np.linalg.norm(T_target[:3, 3] - T_act[:3, 3]))
        rotation_error = float(np.linalg.norm(V_err[3:]))
        return position_error, rotation_error

    @staticmethod
    def _fresh_diag():
        """Zeroed per-run counters. Diagnostics only -- see `_diagnostic_block`."""
        return {"ticks": 0, "clamped_v": 0, "clamped_w": 0, "clipped": 0,
                "want_v_sum": 0.0, "want_v_max": 0.0,
                "sent_v_sum": 0.0, "sent_v_max": 0.0,
                "got_v_sum": 0.0, "got_v_max": 0.0, "got_v_n": 0,
                "win_arc": 0.0, "win_ticks": 0,
                "want_w_max": 0.0, "sent_w_max": 0.0,
                "arm_arc": 0.0, "path_arc": 0.0,
                "arm_start": None, "arm_end": None,
                "path_start": None, "path_end": None,
                "tf_stale": 0, "tf_age_max": 0.0,
                "servo": {}}

    def _diagnostic_block(self, speeds, schedule):
        """Everything needed to tell four different faults apart, in one block.

        The four look identical from outside -- the arm finishes short -- and
        have different fixes, so guessing between them is what wastes time:

          1. The PATH is timed for a channel you are not watching. A run paced
             entirely by rotation does not go quicker when the linear ceiling is
             raised. `pacing` answers this.
          2. The COMMAND is being clamped. `clamped` high means the control law
             is asking for more speed than `velocity_headroom * path speed`
             allows, and no gain fixes a ceiling. Note that ceiling SHRINKS as
             the speed slider goes down, so slowing a path down also weakens its
             ability to catch up -- the reason turning the speed down does not
             improve the realised fraction.
          3. The ARM cannot deliver. `achieved peak` well below `sent peak` with
             `clamped` LOW is the servo pipeline attenuating (see the module
             docstring); that is a plant property, not something to tune here.
          4. Nothing is wrong and the metric lied. An optimizer history wanders,
             so the reference's ARC is much longer than its NET displacement and
             a smoothed arm legitimately travels less arc while still arriving.
             Both are reported for exactly this reason -- judge arrival by net
             and by the final error, never by arc alone.
        """
        d = self._diag
        n = max(d["ticks"], 1)
        if d["arm_start"] is None:
            return ""

        def net(a, b):
            return float(np.linalg.norm(d[b] - d[a]))

        arm_net, path_net = net("arm_start", "arm_end"), net("path_start", "path_end")
        lines = [
            f"  \n`ceilings ` path {speeds['max_linear']:.3f} m/s / "
            f"{speeds['max_angular']:.3f} rad/s, command "
            f"{self._arm.max_linear:.3f} m/s / {self._arm.max_angular:.3f} rad/s "
            f"({self.limits.velocity_headroom:.1f}x), servo scale "
            f"{SERVO_SCALE_LINEAR:.2f} m/s / {SERVO_SCALE_ROTATIONAL:.2f} rad/s",

            f"  \n`linear   ` wanted {d['want_v_sum'] / n:.3f} avg / "
            f"{d['want_v_max']:.3f} peak -> sent {d['sent_v_sum'] / n:.3f} / "
            f"{d['sent_v_max']:.3f} -> ACHIEVED "
            f"{d['got_v_sum'] / max(d['got_v_n'], 1):.3f} / "
            f"{d['got_v_max']:.3f} m/s",

            f"  \n`angular  ` wanted {d['want_w_max']:.3f} peak -> sent "
            f"{d['sent_w_max']:.3f} rad/s",

            f"  \n`saturated` linear {100.0 * d['clamped_v'] / n:.0f}% of ticks, "
            f"angular {100.0 * d['clamped_w'] / n:.0f}%, servo-scale clip "
            f"{100.0 * d['clipped'] / n:.0f}%",

            f"  \n`travel   ` arm {arm_net * 1e3:.0f} mm net "
            f"({d['arm_arc'] * 1e3:.0f} mm arc), reference {path_net * 1e3:.0f} mm "
            f"net ({d['path_arc'] * 1e3:.0f} mm arc)",

            "  \n`servo    ` " + (", ".join(
                (SERVO_STATUS.get(code, f"code {code}") if code is not None
                 else "**SILENT** -- no status published; is servo_node up and "
                      "is `servo_status_topic` right? If so, every twist this "
                      "node sends is going nowhere")
                + f" {100.0 * hits / n:.0f}%"
                for code, hits in sorted(d["servo"].items(),
                                         key=lambda kv: -kv[1])) or "no samples"),
        ]

        # ONLY WHEN IT HAPPENED. A run that never coasted should not carry a line
        # saying so -- the point of the line is that a degraded run cannot look
        # like a clean one, and a permanent zero teaches the reader to skip it.
        if d["tf_stale"]:
            lines.append(
                f"  \n`wrist TF ` stale on {100.0 * d['tf_stale'] / n:.0f}% of "
                f"ticks, worst {d['tf_age_max'] * 1e3:.0f} ms against a "
                f"{self.limits.tf_max_age * 1e3:.0f} ms budget -- the correction "
                f"ran on an old pose that often. This is the MACHINE, not the "
                f"arm: robot_state_publisher republishes TF at 20 Hz and a solve "
                f"or a long Servo collision check pushes it past the budget. "
                f"Raise `tf_max_age` if it is routine.")

        # One plain-language read of the above, so the block is actionable
        # without holding all four hypotheses in your head.
        #
        # THE NUMBER `pipeline_gain` SHOULD BE SET TO, measured on this run. It
        # is the whole loop's calibration in one figure: achieved over commanded,
        # which is exactly what the compensation divides by. Reported always,
        # because a run where it has drifted looks identical to a run where the
        # path was simply too fast.
        sent_avg = d["sent_v_sum"] / n
        got_avg = d["got_v_sum"] / max(d["got_v_n"], 1)
        if sent_avg > 1e-6:
            implied = got_avg / sent_avg
            lines.append(
                f"  \n`gain     ` implied {implied:.3f} against a configured "
                f"`pipeline_gain` of {self.limits.pipeline_gain:.3f}"
                + ("" if 0.6 < implied / max(self.limits.pipeline_gain, 1e-9) < 1.6
                   else f" -- **set it to about {implied:.3f}**"))

        if d["clamped_v"] > 0.25 * n:
            lines.append(
                f"  \n**the command was clamped {100.0 * d['clamped_v'] / n:.0f}% "
                f"of the time** -- the control law wanted up to "
                f"{d['want_v_max']:.3f} m/s but the ceiling is "
                f"{self._arm.max_linear:.3f}. That ceiling is "
                f"{self.limits.velocity_headroom:.1f}x the PATH speed, so raising "
                f"the playback speed raises it too; lowering the speed lowers it.")
        elif got_avg < 0.6 * speeds["max_linear"]:
            # Compared against the PATH speed, not against the commanded twist.
            # Commanded-vs-achieved is supposed to differ by `pipeline_gain` now;
            # the question that matters is whether the arm kept up with the path.
            lines.append(
                f"  \n**the arm averaged {got_avg:.3f} m/s against a path timed "
                f"at {speeds['max_linear']:.3f} m/s** without the command "
                f"clamping -- the compensation is under-shooting, so "
                f"`pipeline_gain` is set higher than the pipeline is really "
                f"delivering. Use the implied figure above.")
        return "".join(lines)

    def _measure(self):
        """Current wrist pose in the base frame, coasting on the last good one.

        A stale read is not, on its own, a fault. `robot_state_publisher`
        republishes TF at 20 Hz and anything that keeps the machine busy for a
        few of those periods -- a solve in `viz_node`, a long Servo collision
        check -- pushes the age past `tf_max_age` for a moment. What that
        deserves is a warning and the previous pose, not a dead playback.

        What it must NOT become is silent, so the first stale read of a run says
        so and `_diag` counts the rest. And it must not become unbounded, so
        after `tf_stale_grace` of unbroken staleness this raises. See both
        fields' docstrings for why the grace is a bounded risk.
        """
        T, age = self._frames.wrist_pose_base_stamped()
        now = time.monotonic()
        if age <= self.limits.tf_max_age:
            self._tf_good = T
            self._tf_stale_since = None
            return T

        self._diag["tf_stale"] += 1
        self._diag["tf_age_max"] = max(self._diag["tf_age_max"], age)
        if self._tf_good is None:
            # Nothing to coast on. Only reachable before the first good read,
            # and `play` waits for one, so this is a cell that never had TF.
            raise RuntimeError(
                f"wrist TF is {age * 1e3:.0f} ms old and nothing fresher has "
                f"ever arrived (limit {self.limits.tf_max_age * 1e3:.0f} ms) -- "
                f"is robot_state_publisher running?")
        if self._tf_stale_since is None:
            self._tf_stale_since = now
            self._log.warn(
                f"wrist TF went stale at {age * 1e3:.0f} ms (limit "
                f"{self.limits.tf_max_age * 1e3:.0f} ms); coasting on the last "
                f"good pose for up to {self.limits.tf_stale_grace:.1f} s. This "
                f"is the machine being busy, not the arm -- the correction is "
                f"running on an old measurement until it clears.",
                throttle_duration_sec=2.0)
        coasted = now - self._tf_stale_since
        if coasted > self.limits.tf_stale_grace:
            raise RuntimeError(
                f"wrist TF has been stale for {coasted:.1f} s (now "
                f"{age * 1e3:.0f} ms old, limit "
                f"{self.limits.tf_max_age * 1e3:.0f} ms) -- is robot_state_"
                f"publisher still running? Servoing against a frozen "
                f"measurement would read as perfect tracking.")
        return self._tf_good

    def _wait_for_tf(self, timeout):
        """Block until the wrist TF is fresh. Returns its age; raises on timeout.

        Called ONCE, before anything moves, and it waits rather than refusing
        because pressing *Play* straight after a solve is the normal way to use
        this -- the machine is at its busiest exactly then, and a run that will
        be fine three TF periods later should not be turned away. Nothing is
        moving while this waits, which is what makes waiting the right answer
        here and coasting the right answer inside the loop.
        """
        deadline = time.monotonic() + timeout
        while True:
            T, age = self._frames.wrist_pose_base_stamped()
            if age <= self.limits.tf_max_age:
                self._tf_good = T
                self._tf_stale_since = None
                return age
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"wrist TF is {age * 1e3:.0f} ms old (limit "
                    f"{self.limits.tf_max_age * 1e3:.0f} ms) and did not go "
                    f"fresh in {timeout:.1f} s -- is robot_state_publisher "
                    f"running, and is this machine keeping up? Servoing against "
                    f"a frozen measurement would read as perfect tracking.")
            time.sleep(min(0.02, 1.0 / max(self.limits.hz, 1.0)))

    # ------------------------------------------------------------------- play

    def play(self, plan, enable_arm, enable_hand, speeds, on_progress=None,
             should_stop=None):
        """Execute ``plan``. Blocks until it finishes, stops or aborts.

        ``should_stop`` is polled once per tick alongside the e-stop Event; it is
        how the action server's cancel request reaches the loop.
        """
        if self._playing:
            raise RuntimeError("already playing")
        if self._estop.is_set():
            raise RuntimeError("E-STOP engaged")

        limits = self.limits
        T_base_viz = self._frames.registration(plan.corner_viz)

        # BEFORE the first measurement, not with the rest of the per-run state
        # further down: `_measure` counts stale TF reads into `_diag`, and the
        # approach pose below is a `_measure` call.
        self._diag = self._fresh_diag()
        self._tf_good = None
        self._tf_stale_since = None

        # Wait for a fresh wrist TF rather than refusing on a stale one. Pressing
        # *Play* right after a solve is the normal way to use this, and that is
        # exactly when the machine is least able to keep TF current.
        tf_age = self._wait_for_tf(limits.tf_stale_grace)
        if tf_age > limits.tf_max_age * 0.5:
            # Not a fault -- but a cell that starts at half the budget will spend
            # the run coasting, and that is worth knowing before it does.
            self._log.info(
                f"wrist TF was {tf_age * 1e3:.0f} ms old at the start "
                f"(budget {limits.tf_max_age * 1e3:.0f} ms); expect coasting if "
                f"this machine gets busier")

        # Start from where the robot actually is, so the first segment is an
        # ordinary approach at the configured speeds instead of a step change.
        # This is also what turns a resumed run into a normal re-approach.
        # See robot_plan.prepend_current.
        current_wrist_viz = np.linalg.inv(T_base_viz) @ self._measure()
        measured, age = self._hand_state()
        if enable_hand and age is None:
            # The approach segment would otherwise be measured from the plan's
            # own first waypoint -- i.e. from wherever the SOLVE started, not
            # from where the hand is -- and on an absolute-position hand that is
            # a sweep to the plan's opening posture at full slew. Refusing is the
            # only safe answer: there is no conservative default posture, since
            # "open" is a different vector on every hand.
            raise RuntimeError(
                f"nothing has arrived on the hand's state topic, so there is no "
                f"measured posture to start from. Is the hand node up and "
                f"publishing? (expected on `{self._profile.hand_driver.state_topic()}`)")
        plan = robot_plan.prepend_current(plan, current_wrist_viz, measured)
        # Everything below indexes the PREPENDED plan; the client asked about the
        # original, so a resume index has to have this offset taken back off.
        prepended = 1

        # CAP THE PATH BELOW THE SERVO SCALE, so the correction term always has
        # somewhere to go. The command ceiling is `path * velocity_headroom` and
        # cannot usefully exceed the servo's own scale -- `ArmTwistServo._publish`
        # clips per component above it, which would bend the direction the
        # direction-preserving clamp just protected. So if the path itself is
        # allowed up to the servo scale, the ceiling lands exactly ON the path
        # speed, `_clamp_norm` eats the whole correction, and the loop degenerates
        # to open-loop feed-forward against a lagging plant. Capping the PATH
        # instead makes the headroom exist by construction at every speed setting.
        # The command is `path / pipeline_gain`, so the PATH that keeps the
        # command inside the servo's scale with `velocity_headroom` left over for
        # the correction is that much smaller. Leaving the gain out here is what
        # would put the feed-forward alone at the ceiling with nothing spare --
        # the exact failure `velocity_headroom` exists to prevent.
        gain = max(limits.pipeline_gain, 1e-6)
        usable_linear = SERVO_SCALE_LINEAR * gain / limits.velocity_headroom
        usable_angular = SERVO_SCALE_ROTATIONAL * gain / limits.velocity_headroom
        speeds = dict(speeds)
        requested_linear = speeds["max_linear"]
        speeds["max_linear"] = min(speeds["max_linear"], usable_linear)
        speeds["max_angular"] = min(speeds["max_angular"], usable_angular)
        if speeds["max_linear"] < requested_linear - 1e-9:
            # Not a warning: this is the design working. But a slider asking for
            # 0.40 m/s and a robot moving at 0.20 m/s needs to explain itself.
            self._log.info(
                f"path speed capped at {speeds['max_linear']:.3f} m/s (asked "
                f"{requested_linear:.3f}): the pipeline realises "
                f"{gain * 100:.0f}% of what it is told, so commanding that "
                f"needs {speeds['max_linear'] / gain:.2f} m/s, and the "
                f"{limits.velocity_headroom:.1f}x correction headroom has to fit "
                f"under the {SERVO_SCALE_LINEAR:.2f} m/s servo scale too")

        schedule = robot_plan.plan_schedule(plan, hz=limits.hz, **speeds)

        self._arm.max_linear = (speeds["max_linear"] / gain
                                * limits.velocity_headroom)
        self._arm.max_angular = (speeds["max_angular"] / gain
                                 * limits.velocity_headroom)

        # Per-run, so one playback's shortfall never colours the next one's
        # report. `_diag` itself was reset earlier -- see the note up there.
        self._last_reference = None
        self._last_actual = None

        # The driver for THIS robot, fresh for this run. Which class it is comes
        # from the profile and is the only per-robot choice the loop makes.
        self._hand = self._profile.hand_driver(
            self._node, self._hand_model, limits=self._travel, hz=limits.hz)
        self._hand.seed(measured)

        self._log.info(
            f"playback: {len(plan.waypoints)} waypoints, {schedule.total:.1f} s "
            f"plus a {limits.terminal_hold:.1f} s hold -- a fixed duration, "
            f"arm={enable_arm} hand={enable_hand}, command ceiling "
            f"{self._arm.max_linear:.2f} m/s "
            f"({limits.velocity_headroom:.1f}x the path)")
        # WHICH CHANNEL IS ACTUALLY SETTING THE PACE. Without this, a path timed
        # entirely by its rotation or pinned at `min_duration` is indistinguish-
        # able from one timed by its wrist travel, and only one of the three
        # responds to the linear speed ceiling.
        self._log.info("plan: " + robot_plan.describe_pacing(
            plan, speeds["max_linear"], speeds["max_angular"],
            speeds["max_digit"]))

        self._playing = True
        started = time.monotonic()
        try:
            outcome = self._run(plan, schedule, T_base_viz,
                                enable_arm, enable_hand, on_progress,
                                should_stop, prepended, speeds)
        finally:
            self._playing = False
            self._arm.halt()
            self._hand.halt()

        outcome.elapsed = time.monotonic() - started
        self._log.info(f"playback: {outcome.verdict}")
        return outcome

    def _run(self, plan, schedule, T_base_viz, enable_arm,
             enable_hand, on_progress, should_stop, prepended, speeds):
        limits = self.limits
        period = 1.0 / limits.hz
        feedback_every = max(1, int(round(limits.hz / 10.0)))   # ~10 Hz
        trace_every = (max(1, int(round(limits.hz / limits.trace_hz)))
                       if limits.trace_hz > 0 else 0)
        hold_ticks = max(0, int(round(limits.terminal_hold * limits.hz)))
        self._trace_tick = 0
        self._trace_got = 0.0
        self._trace_clamp = 0

        outcome = Outcome()
        t_path = 0.0
        tick = 0
        position_error = rotation_error = 0.0
        # The hold is counted in TICKS, not measured against a deadline and not
        # against an error: a fixed number of periods is what makes the run's
        # duration a property of the plan rather than of how the arm behaved.
        held = 0

        next_wake = time.monotonic()
        while True:
            # FIRST, before anything is commanded. One tick of a saturated twist
            # is what a stop button exists to prevent.
            if self._estop.is_set():
                outcome.verdict = "**E-STOP**"
                break
            if should_stop is not None and should_stop():
                outcome.verdict = "**stopped**"
                break

            holding = t_path >= schedule.total
            sample = robot_plan.sample_at(plan, schedule, t_path)

            if enable_arm:
                # Past the end `sample.body_twist` is already zero, so the hold
                # needs no special case here -- it is the same command with no
                # feed-forward left in it.
                position_error, rotation_error = self._servo_arm(
                    T_base_viz @ sample.wrist_pose, sample.body_twist)
                outcome.max_position_error = max(outcome.max_position_error,
                                                 position_error)
                outcome.max_rotation_error = max(outcome.max_rotation_error,
                                                 rotation_error)

                # Safety, not sequencing. Nothing else in this loop looks at the
                # tracking error to decide what to do next.
                if (position_error > limits.position_watchdog
                        or rotation_error > limits.rotation_watchdog):
                    outcome.trip_reason = (
                        f"tracking error {position_error * 1e3:.0f} mm / "
                        f"{np.degrees(rotation_error):.0f}° is past the hard "
                        f"limit -- the arm is not following at all. Is servo up "
                        f"and out of a singularity?")
                    outcome.verdict = f"**aborted**: {outcome.trip_reason}"
                    break

            if enable_hand:
                # The driver clamps and re-keys; what leaves here is the plan's
                # own vocabulary. A clamped digit is reported rather than
                # silently bounded -- see `hand_drivers.base.clamp_target`.
                for name in self._hand.tick(sample.digit_cmd):
                    self._log.warn(f"{name} clamped to its travel limit",
                                   throttle_duration_sec=5.0)

            # --- live trace, once a second ---------------------------------
            # The executor's log is what gets pasted when something looks wrong,
            # so the time series lives here rather than only in the GUI feedback.
            # Cheap: one line per second, and it reads what is already computed.
            if enable_arm and trace_every and tick % trace_every == 0:
                d = self._diag
                since = max(tick - self._trace_tick, 1)
                self._log.info(
                    f"  t={t_path:5.2f}/{schedule.total:5.2f} "
                    f"wp {max(0, sample.waypoint - prepended):3d} "
                    f"err {position_error * 1e3:6.1f} mm/{np.degrees(rotation_error):5.1f}deg "
                    f"cmd {np.linalg.norm(sample.body_twist[:3]):.3f}ff "
                    f"want {d['want_v_sum'] / max(d['ticks'], 1):.3f} "
                    f"got {d['got_v_sum'] / max(d['got_v_n'], 1):.3f} m/s "
                    f"clamp {100.0 * (d['clamped_v'] - self._trace_clamp) / since:3.0f}%")
                self._trace_tick, self._trace_got = tick, d["got_v_sum"]
                self._trace_clamp = d["clamped_v"]

            # --- feedback, and nothing else --------------------------------
            if on_progress is not None and tick % feedback_every == 0:
                on_progress(Progress(
                    tick=tick, path_time=t_path, path_total=schedule.total,
                    # Reported against the plan the CLIENT sent, which does not
                    # have the prepended current-state waypoint on the front.
                    waypoint=max(0, sample.waypoint - prepended),
                    num_waypoints=len(plan.waypoints) - prepended,
                    note=plan.waypoints[min(sample.waypoint,
                                            len(plan.waypoints) - 1)].note,
                    position_error=position_error,
                    rotation_error=rotation_error,
                    time_scale=1.0, holding=holding))

            tick += 1

            # --- the clock ------------------------------------------------
            # Unconditional. It is never scaled, never held for the arm, and
            # never waits on an error threshold -- see the module docstring.
            if holding:
                held += 1
                if held >= hold_ticks:
                    outcome.success = True
                    outcome.verdict = "finished"
                    break
            else:
                t_path = min(t_path + period, schedule.total)

            # Wall-clock pacing, keeping the loop at a steady rate. The path
            # clock above advances one period per tick, so this is what ties the
            # two together: a tick that runs long is real time the path has
            # already been credited with.
            next_wake += period
            slack = next_wake - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            elif slack < -period:
                # Reset rather than sprint: catching up by running flat out is
                # how a stalled loop turns into a burst of commands.
                next_wake = time.monotonic()
                self._log.warn(
                    f"control tick overran by {-slack * 1e3:.0f} ms against a "
                    f"{period * 1e3:.0f} ms budget",
                    throttle_duration_sec=5.0)

        outcome.ticks = tick
        outcome.position_error = position_error
        outcome.rotation_error = rotation_error
        # Gated on the DIAGNOSTIC tick count, not `outcome.ticks`: an abort on
        # the very first tick breaks before the loop's counter increments, and
        # that is precisely the run whose numbers are wanted.
        if enable_arm and self._diag["ticks"]:
            # On EVERY outcome, not just a clean finish. A run that aborted or
            # was stopped is exactly when the numbers are wanted, and printing
            # them only on success meant the interesting cases said least.
            outcome.verdict += (
                f" &nbsp; final {position_error * 1e3:.1f} mm / "
                f"{np.degrees(rotation_error):.1f}°, worst "
                f"{outcome.max_position_error * 1e3:.1f} mm / "
                f"{np.degrees(outcome.max_rotation_error):.1f}° along the path")
            try:
                outcome.verdict += self._diagnostic_block(speeds, schedule)
            except Exception as exc:
                # A READOUT MUST NEVER FAIL A RUN THAT ALREADY HAPPENED. This
                # block is assembled after the arm has finished moving, so a
                # raise here propagates out of `play` and reports "playback
                # failed" for a playback that in fact completed -- losing the
                # verdict, the resume point, and the operator's trust in the
                # word "failed". Measured, once, as a KeyError on a counter that
                # a half-applied edit never created.
                self._log.warn(f"diagnostics unavailable: {exc!r}")
                outcome.verdict += f"  \n_diagnostics unavailable: `{exc!r}`_"
        if not outcome.success and t_path < schedule.total:
            outcome.resumable = True
            outcome.resume_path_time = t_path
            # The waypoint being travelled TOWARD, shifted back off the prepended
            # one. Resuming from it replays the segment that was interrupted
            # rather than skipping it.
            outcome.resume_waypoint = max(
                0, robot_plan.sample_at(plan, schedule, t_path).waypoint - prepended)
            outcome.verdict += (
                f" &nbsp; paused at waypoint {outcome.resume_waypoint}"
                f"/{len(plan.waypoints) - prepended}")
        return outcome
