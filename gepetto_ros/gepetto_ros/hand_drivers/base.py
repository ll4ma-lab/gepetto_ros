"""What a hand driver has to be, for the control loop to drive it.

The executor ticks ONE loop that moves the arm and the hand together. It knows
how to command an arm (a twist, through MoveIt Servo) and it knows how to read a
plan, but what it does with a per-digit command vector is entirely the hand's
business:

* the EPFL tendon hand is driven by `finger_servo_node`, which INTEGRATES delta
  jogs in metres of tendon, so a driver for it has to hold a running total and
  publish differences;
* the Allegro is driven by Wonik's `allegro_node_grasp`, which takes ABSOLUTE
  joint positions in radians, sixteen of them, read positionally out of a
  `JointState`, so a driver for it slew-limits and publishes the whole vector.

Those two have nothing in common below this interface -- different message types,
different units, different sign conventions, one stateful and one not -- and that
is precisely why the interface is this small. Everything above it (pacing,
scheduling, the control law, the e-stop, the watchdogs, the action) is one path
for both robots.

THE CONTRACT, and every method matters to the loop:

``digit_names``   the solver digit names this driver will accept, in solver order.
``seed(measured)``  adopt the hand's MEASURED state as the starting point, before
                  the first ``tick``. Without it the first command is measured
                  from an assumed zero and the hand lurches by however far it
                  actually was from there. A driver with no internal state may
                  ignore it, but must still accept the call.
``tick(target)``  publish one command toward the absolute ``{digit: (K,) vector}``
                  target. Called once per control period, at `hz`. Returns the
                  list of digits it had to clamp this tick.
``hold()``        publish an explicit "stay where you are", as a keepalive while
                  another channel is still moving.
``halt()``        stop commanding, NOW. Whether that means publishing a zero or
                  publishing nothing at all is the driver's business -- what
                  matters is that the hand stops moving and holds.
``commanded``     ``{digit: (K,) vector}`` of what has been commanded so far, for
                  diagnostics.

SIGN AND UNITS ARE THE DRIVER'S, NOT THE LOOP'S. A target reaching ``tick`` is in
the plan's ``command_kind`` units and the solver's digit names; turning that into
the hardware's names, order, sign and message type is what a driver IS.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class HandDriver(Protocol):
    """See the module docstring. Structural, so a driver need not inherit."""

    #: Solver digit names this driver accepts, in solver order.
    digit_names: list[str]

    def seed(self, measured) -> None:
        """Adopt ``{digit: (K,) vector}`` of MEASURED state as the start point."""
        ...

    def tick(self, target) -> list[str]:
        """Publish one command toward ``{digit: (K,) vector}``. Returns the
        digits clamped this tick."""
        ...

    def hold(self) -> None:
        """Publish an explicit "stay", as a keepalive."""
        ...

    def halt(self) -> None:
        """Stop commanding now; the hand holds where it is."""
        ...

    @property
    def commanded(self) -> dict:
        """``{digit: (K,) vector}`` commanded so far."""
        ...


def clamp_target(target, limits, width):
    """``(clamped, clamped_names)`` for one tick's target.

    Shared because both drivers need exactly this and getting it wrong is a
    safety bug rather than a cosmetic one: a NaN is DROPPED (not clamped to
    zero), because a single NaN poisons an integrating node's setpoint
    permanently and moves an absolute one to a position no one asked for.

    ``limits`` is ``{digit: (lo, hi)}`` of arrays, as
    ``robot_plan.travel_limits`` produces; a digit with no entry is passed
    through unclamped. Empty limits clamp nothing, which the caller must have
    already complained about -- see ``executor_node``.
    """
    out, clamped = {}, []
    for name, value in target.items():
        value = np.ravel(np.asarray(value, float))
        if value.size != width:
            raise ValueError(
                f"{name}: got {value.size} commanded values, this hand drives "
                f"{width} actuators per digit")
        if not np.all(np.isfinite(value)):
            continue
        bounds = limits.get(name)
        if bounds is not None:
            bounded = np.clip(value, *bounds)
            if not np.array_equal(bounded, value):
                clamped.append(name)
            value = bounded
        out[name] = value
    return out, clamped
