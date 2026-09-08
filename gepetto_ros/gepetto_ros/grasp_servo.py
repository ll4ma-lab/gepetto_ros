"""The tactile close: a loop that discovers its own trajectory, one tick at a time.

Runs in `executor_node`, alongside the playback loop and never at the same time as
it -- both drive the hand's command topic, and one node holding both actions is
what makes overlapping them impossible without a lock two processes could
disagree about.

WHY THIS IS NOT A PLAYBACK. `trajectory_executor` plays a path that was decided
before anything moved: its clock is free-running precisely so that the duration of
a run is a property of the plan alone. Nothing here is decided in advance. The
planner is purely kinematic, so it can say which DIRECTION each fingertip should
press -- the ellipsoid grasp alignment constraint's inward normals -- and cannot
say how far, because that depends on where the object really is and how it really
gives. So this loop has no schedule, no path parameter and no total: it advances
each fingertip along its normal until the sensor at the end of that finger says to
stop, and the run is over when every digit has stopped for some reason.

THE DIVISION OF LABOUR, and it is the same one the rest of this package keeps:
this file does I/O and timing, and `gepetto_solvers.core.solvers.grasp_servo` --
which imports nothing from ROS -- makes every decision. Which is what lets the
stop conditions be tested against synthetic sensor traces in a suite that needs no
workspace and no robot, and "does it stop before it breaks the fingertip" is not a
question to answer by trying it.

THE ARM IS NEVER COMMANDED FROM HERE. Not "is held", not "is servoed to a constant
pose" -- not commanded at all. MoveIt Servo's `incoming_command_timeout` is 0.1 s,
so it halts and holds by itself the moment this loop starts, which is both what we
want and what makes the wrist-frame formulation exact: with the palm stationary,
directions expressed in it stay correct for the whole close, and no TF lookup, no
table registration and no staleness handling comes anywhere near the fingers.

WHAT STOPS A RUN THAT WILL NOT STOP ITSELF. Every digit has a travel budget, so in
the ordinary case the loop terminates on its own. `timeout` covers the case it
cannot: a fingertip whose Jacobian has collapsed achieves no travel per tick, so
it never spends its budget and would sit there commanding forever. A wall clock is
the only thing that catches that.
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from gepetto_solvers.core.solvers import grasp_servo as policy


@dataclass
class GraspLimits:
    """Everything the loop is tuned by that is not part of a goal.

    The goal carries what the OPERATOR chose for this grasp -- speed, travel,
    per-digit targets, the abort ceiling. This carries what the CELL is like,
    which is a node parameter and not something to re-decide per close.
    """

    hz: float = 100.0

    #: How old a tactile reading may be and still be closed against.
    #:
    #: A hard fault, unlike the executor's `tf_max_age`, which coasts on the last
    #: good pose for a grace period. The asymmetry is deliberate and it is about
    #: what the measurement is FOR: a stale wrist pose degrades a correction term
    #: on a path that is still going where it was going, while a stale fingertip
    #: reading removes the only thing that will stop a finger closing on
    #: something solid. There is nothing safe to coast on here.
    tactile_max_age: float = 0.2

    #: Messages averaged into each digit's resting reading before a close starts.
    baseline_samples: int = 20

    #: Contact-detection floor, in counts above rest. Arms the collapsed-sensor
    #: detector; below it a reading is treated as noise on an untouched finger.
    contact_counts: float = 50.0

    #: Fraction of its own peak a touching digit may fall to before its sensor is
    #: called out of range. See `core.solvers.grasp_servo` for why this is not
    #: simply "is the reading near zero".
    collapse_fraction: float = 0.5

    #: Below this fraction of the step it asked for, a digit clamped against a
    #: joint stop is called stuck rather than merely slowed.
    stall_fraction: float = 0.25

    #: Ceiling on one joint's motion, rad/s. None takes the hand's own
    #: `max_digit_speed`, so the close and the playback share a ceiling by
    #: construction rather than by two constants agreeing.
    joint_speed: float | None = None

    #: Multiple of the theoretical worst-case duration
    #: (`max_travel / speed`) after which the run is aborted regardless. See the
    #: module docstring on what this is actually for.
    timeout_factor: float = 3.0
    timeout_floor: float = 5.0

    #: Rate of the per-tick trace line in the node's log. 0 disables it. On for
    #: the same reason the executor's is: a close that went wrong is reported by
    #: pasting this log, and when each digit stopped is the whole story.
    trace_hz: float = 1.0


@dataclass
class GraspProgress:
    """What the loop hands back each feedback interval. Plain data: the loop must
    not format markdown or touch an action message."""

    travel: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    stopped: int = 0
    total: int = 0
    note: str = ""


@dataclass
class GraspOutcome:
    """How a close ended."""

    success: bool = False
    verdict: str = ""
    digits: list = field(default_factory=list)
    stop_reasons: dict = field(default_factory=dict)
    stop_notes: dict = field(default_factory=dict)
    travel: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    peak_counts: dict = field(default_factory=dict)
    q_final: dict = field(default_factory=dict)
    elapsed: float = 0.0
    #: The posture the close STARTED from, for a later release to return to.
    #: Empty on a run that never got as far as moving.
    q_start: dict = field(default_factory=dict)


class GraspServo:
    """Runs one tactile close at a time. Not reentrant; the node's action server
    is what enforces that."""

    def __init__(self, node, profile, hand, tactile, limits=None,
                 hand_state=None, travel_limits=None):
        self._node = node
        self._log = node.get_logger()
        self._profile = profile
        self._hand_model = hand
        self._tactile = tactile
        self.limits = limits or GraspLimits()
        self._hand_state = hand_state or (lambda: ({}, None))
        self._travel = travel_limits or {}

        self._estop = threading.Event()
        self._running = False
        self._hand = None
        #: Where the last completed close started, so a release has somewhere to
        #: go. Kept HERE rather than in the goal, following `MoveHome`: a target
        #: a client could name would let a release run to a posture this hand was
        #: never at.
        self._release_to = None

    # ------------------------------------------------------------------ estop

    @property
    def estop(self):
        return self._estop

    @property
    def running(self):
        return self._running

    @property
    def can_release(self):
        return self._release_to is not None

    def release_digits(self):
        """Which digits a release would move, or ``[]`` if there is nothing to
        back out of.

        Asked BEFORE the run, so the action's feedback arrays can be sized and
        labelled: a release goal names no digits -- the posture to return to is
        the executor's, not the caller's -- and without this the feedback would
        arrive as empty lists until the result landed."""
        return [] if self._release_to is None else list(self._release_to)

    def halt(self):
        """Stop commanding, right now. Called from the e-stop SUBSCRIPTION
        CALLBACK, not from the loop: the loop polls the Event and will exit
        within a tick, and a tick is another millimetre into the object."""
        if self._hand is not None:
            self._hand.halt()

    # ------------------------------------------------------------------- run

    def close(self, digit_names, directions, targets, speed, max_travel,
              abort_counts, on_progress=None, should_stop=None):
        """Close ``digit_names`` along ``directions`` until each reads its target.

        ``directions`` is ``{digit: (3,)}`` in the WRIST frame. ``targets`` is
        ``{digit: counts above that digit's resting reading}``.
        """
        return self._run(digit_names, directions, targets, speed, max_travel,
                         abort_counts, False, on_progress, should_stop)

    def release(self, on_progress=None, should_stop=None):
        """Back out to the posture the last close started from."""
        if self._release_to is None:
            raise RuntimeError(
                "nothing to release from -- no close has run in this executor "
                "since it started. Close first, or move the hand with the "
                "joint sliders and a playback.")
        digits = list(self._release_to)
        return self._run(digits, {n: np.zeros(3) for n in digits}, {}, 0.0,
                         0.0, np.inf, True, on_progress, should_stop)

    def _run(self, digit_names, directions, targets, speed, max_travel,
             abort_counts, release, on_progress, should_stop):
        if self._running:
            raise RuntimeError("already closing")
        if self._estop.is_set():
            raise RuntimeError("E-STOP engaged")

        digit_names = list(digit_names)
        # WHERE A RELEASE IS GOING, captured before anything below overwrites it.
        # The RECORDED pre-close posture, never the current one: walking toward
        # where the hand already is is a release that finishes instantly having
        # moved nothing, which is exactly what the first version did.
        release_target = ({n: np.asarray(q, float).copy()
                           for n, q in self._release_to.items()}
                          if release else None)
        fk = self._digit_fk()
        limits = self.limits
        period = 1.0 / limits.hz

        # Start from where the fingers ARE. On an absolute-position hand an
        # unseeded first command is a sweep to wherever this loop's arithmetic
        # started, which is exactly what `seed` exists to prevent -- and here the
        # integration itself begins at the measured posture, so a missing
        # readback is not a degraded start but no start at all.
        measured, age = self._hand_state()
        if age is None:
            raise RuntimeError(
                f"nothing has arrived on the hand's state topic, so there is no "
                f"measured posture to close from. Is the hand node up? "
                f"(expected on `{self._profile.hand_driver.state_topic()}`)")
        missing = [n for n in digit_names if n not in measured]
        if missing:
            raise RuntimeError(
                f"no measured joint state for {', '.join(missing)} -- half a "
                f"digit is not a posture to start integrating from")

        baseline = {n: 0.0 for n in digit_names}
        if not release:
            # BEFORE anything moves, and per close rather than once at startup:
            # a finger already resting on the object carries its own preload, and
            # measuring the target above THAT is what makes the slider mean the
            # same thing on every run.
            baseline = self._tactile.baseline(
                digit_names, samples=limits.baseline_samples, period=period)
            self._log.info("tactile baseline: " + ", ".join(
                f"{n} {baseline[n]:.0f}" for n in digit_names) + " counts")

        q_start = {n: np.asarray(measured[n], float).copy() for n in digit_names}
        state = policy.servo_start(q_start, baseline, digits=digit_names)
        servo_limits = policy.GraspServoLimits(
            speed=float(speed),
            max_travel=float(max_travel),
            abort_counts=float(abort_counts),
            contact_counts=limits.contact_counts,
            collapse_fraction=limits.collapse_fraction,
            stall_fraction=limits.stall_fraction,
            max_joint_speed=float(
                limits.joint_speed if limits.joint_speed is not None
                else self._hand_model.max_digit_speed))

        self._hand = self._profile.hand_driver(
            self._node, self._hand_model, limits=self._travel, hz=limits.hz)
        self._hand.seed(measured)

        what = "release" if release else "close"
        self._log.info(
            f"grasp {what}: {', '.join(digit_names)}"
            + ("" if release else
               f" at {speed * 1e3:.0f} mm/s, up to {max_travel * 1e3:.0f} mm, "
               f"targets " + ", ".join(f"{n} {targets.get(n, 0.0):.0f}"
                                       for n in digit_names)
               + f", abort at {abort_counts:.0f} counts"))

        self._running = True
        started = time.monotonic()
        try:
            outcome = self._loop(fk, state, directions, targets, servo_limits,
                                 release, on_progress, should_stop,
                                 release_target)
        finally:
            self._running = False
            # One last publish of where we are, then silence. On this hand that
            # IS "stay here": the driver holds its last received position.
            self._hand.halt()

        outcome.elapsed = time.monotonic() - started
        outcome.q_start = q_start
        if not release and outcome.q_start:
            # Only a CLOSE gives a release somewhere to go, and only once it has
            # actually run: overwriting this from a release would leave the next
            # one trying to back out to a posture that is already backed out.
            self._release_to = q_start
        self._log.info(f"grasp {what}: {outcome.verdict}")
        return outcome

    def _loop(self, fk, state, directions, targets, servo_limits, release,
              on_progress, should_stop, release_target):
        limits = self.limits
        period = 1.0 / limits.hz
        feedback_every = max(1, int(round(limits.hz / 10.0)))
        trace_every = (max(1, int(round(limits.hz / limits.trace_hz)))
                       if limits.trace_hz > 0 else 0)
        deadline = time.monotonic() + max(
            limits.timeout_floor,
            limits.timeout_factor * servo_limits.max_travel
            / max(servo_limits.speed, 1e-6))

        outcome = GraspOutcome(digits=list(state.digits))
        tick = 0
        counts = {}
        aborted = ""

        next_wake = time.monotonic()
        while state.running:
            # FIRST, before anything is commanded, for the executor's reason: the
            # loop polls between ticks and one tick is another millimetre in.
            if self._estop.is_set():
                aborted = "**E-STOP**"
                break
            if should_stop is not None and should_stop():
                aborted = "**stopped**"
                break
            if time.monotonic() > deadline:
                aborted = (
                    "**timed out** -- no digit reached a stopping condition "
                    "within three times the travel budget. A fingertip that "
                    "achieves no motion per tick never spends its budget; check "
                    "the log's trace for a digit whose travel stopped growing.")
                break

            if release:
                state = policy.release_step(fk, state, release_target,
                                            servo_limits, period)
            else:
                counts, age = self._tactile.read()
                if age is None or age > limits.tactile_max_age:
                    # A HARD FAULT, never a coast. See `GraspLimits.tactile_max_age`:
                    # there is nothing safe to keep closing against.
                    aborted = (
                        "**aborted**: the tactile reading is "
                        + ("absent" if age is None else f"{age * 1e3:.0f} ms old")
                        + f" (limit {limits.tactile_max_age * 1e3:.0f} ms) on "
                          f"`{self._tactile.topic}`. Closing against a frozen "
                          f"sensor would read as a finger that never touches "
                          f"anything.")
                    break
                state = policy.servo_step(fk, state, directions, counts, targets,
                                          servo_limits, period)

            for name in self._hand.tick(state.commands()):
                self._log.warn(f"{name} clamped to its travel limit",
                               throttle_duration_sec=5.0)

            if trace_every and tick % trace_every == 0:
                self._log.info("  " + "  ".join(
                    f"{n} {state.servos[n].travel * 1e3:5.1f}mm "
                    f"{state.servos[n].counts:5.0f}c"
                    f"{'/' + state.servos[n].stopped if state.servos[n].stopped else ''}"
                    for n in state.digits))

            if on_progress is not None and tick % feedback_every == 0:
                on_progress(GraspProgress(
                    travel={n: state.servos[n].travel for n in state.digits},
                    counts={n: state.servos[n].counts for n in state.digits},
                    stopped=state.num_stopped, total=len(state.digits),
                    note=self._note(state)))

            tick += 1
            next_wake += period
            slack = next_wake - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            elif slack < -period:
                # Reset rather than sprint, the executor's rule: catching up flat
                # out is how a stalled loop becomes a burst of commands.
                next_wake = time.monotonic()
                self._log.warn(
                    f"grasp tick overran by {-slack * 1e3:.0f} ms against a "
                    f"{period * 1e3:.0f} ms budget",
                    throttle_duration_sec=5.0)

        for name in state.digits:
            servo = state.servos[name]
            outcome.stop_reasons[name] = servo.stopped or "still running"
            outcome.stop_notes[name] = servo.note
            outcome.travel[name] = servo.travel
            outcome.counts[name] = servo.counts
            outcome.peak_counts[name] = servo.peak_counts
            outcome.q_final[name] = servo.q.copy()

        good = all(r in policy.GOOD_STOPS
                   for r in outcome.stop_reasons.values())
        outcome.success = bool(good and not aborted)
        outcome.verdict = aborted or ("finished" if good else "**finished with "
                                      "digits that did not reach contact**")
        outcome.verdict += "  \n" + self._summary(state)
        return outcome

    def _note(self, state):
        return ", ".join(
            f"{n} {state.servos[n].stopped}" for n in state.digits
            if state.servos[n].stopped) or "closing"

    def _summary(self, state):
        """One line per digit: what it did and why it stopped.

        On EVERY outcome, not just a clean one. A close that gave up is exactly
        when the numbers are wanted, and the peak is reported beside the final
        reading because a digit stopped on a collapsed sensor ends LOW with a
        high peak -- the two together are the only thing that says so.
        """
        lines = []
        for name in state.digits:
            servo = state.servos[name]
            lines.append(
                f"`{name:<7}` {servo.travel * 1e3:5.1f} mm &nbsp; "
                f"{servo.counts:.0f} counts (peak {servo.peak_counts:.0f}) "
                f"&nbsp; **{servo.stopped or 'still running'}**"
                + (f" -- {servo.note}" if servo.note else ""))
        return "  \n".join(lines)

    def _digit_fk(self):
        fk = getattr(self._hand_model, "digit_fk", None)
        if fk is None:
            raise RuntimeError(
                f"the {self._hand_model.name} hand does not supply per-digit "
                f"fingertip kinematics, so there is no way to walk a fingertip "
                f"along a grasp normal. A tactile close needs a hand that "
                f"declares `digit_fk()`.")
        return fk()
