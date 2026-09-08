"""Reading the hand's fingertip contact sensors, whatever hand it is.

The sibling of `hand_state.py` and built the same way for the same reasons: one
subscription, one lock, one monotonic age, and the DECODING -- which topic, which
message type, which element is which digit -- delegated to the hand driver class
so this file holds no per-robot knowledge. A driver that declares no
`tactile_topic` has no such sensor, and :class:`TactileReader` says so by
existing in a disabled state rather than by being absent, which is what lets the
executor give the operator a real reason instead of a missing feature.

WHY THE AGE MATTERS MORE HERE THAN ANYWHERE ELSE IN THIS STACK. Every other
reader in this package is feeding a display or a seed. This one is feeding the
only thing that will stop a finger closing on an object, and a reader that keeps
handing back the last value it saw after the publisher died would let the servo
run its whole travel budget into something solid while reading a comfortable
300 counts. That is the same failure the executor's `_measure` warns about for a
frozen TF -- "servoing against a frozen measurement would read as perfect
tracking" -- with a fingertip on the other end of it. The grasp loop treats a
stale reading as a fault and stops.

MONOTONIC, NOT THE MESSAGE STAMP. `Int32MultiArray` has no header to stamp, so
there is not even the option: what is being asked is "how long since anything
arrived", which is a fact about this process's clock.
"""

import threading
import time

import numpy as np


class TactileReader:
    """Per-digit fingertip readings, decoded into solver digit names.

    ``available`` is False when this hand's driver declares no tactile topic. The
    reader is still constructed in that case, and still answers ``read()`` with
    an empty dict and a None age, so a caller checks one flag instead of guarding
    every call site with a ``hasattr``.
    """

    def __init__(self, node, hand, driver_class, topic=None, order=None,
                 depth=10, callback_group=None):
        self._hand = hand
        self._driver_class = driver_class
        self._order = list(order) if order else None
        self._state = None      # (monotonic stamp, {digit: counts})
        self._lock = threading.Lock()

        resolve = getattr(driver_class, "tactile_topic", None)
        self.topic = topic or (resolve() if resolve is not None else "")
        self.available = bool(self.topic) and hasattr(
            driver_class, "decode_tactile")
        if not self.available:
            return

        msg_type = getattr(driver_class, "tactile_msg", None)
        if msg_type is None:
            from std_msgs.msg import Int32MultiArray
            msg_type = Int32MultiArray
        node.create_subscription(msg_type, self.topic, self._on_tactile, depth,
                                 callback_group=callback_group)

    def _on_tactile(self, msg):
        try:
            counts = self._driver_class.decode_tactile(
                self._hand, msg, order=self._order)
        except Exception:
            # A malformed message is dropped rather than raised out of a
            # subscription callback, exactly as `HandStateReader` does. It is not
            # silent: nothing lands, so `read()`'s age goes stale and the servo
            # refuses to start (or stops) with a message naming this topic --
            # which is the report that is actually useful.
            return
        if not counts:
            return
        with self._lock:
            self._state = (time.monotonic(), counts)

    def read(self):
        """``(counts, age_seconds)``; ``({}, None)`` if nothing has arrived."""
        with self._lock:
            if self._state is None:
                return {}, None
            stamp, counts = self._state
        return dict(counts), time.monotonic() - stamp

    def baseline(self, digits, samples=20, period=0.01, timeout=2.0):
        """Each digit's RESTING reading, averaged over ``samples`` messages.

        Every contact target is measured above this. It has to be, and not
        because of drift: the fingertips simply do not read zero untouched, they
        read different offsets from each other, and one that idles at 400 counts
        would satisfy a 300-count target the instant a close began, having
        touched nothing at all.

        Captured at the START of each close rather than once at startup, since it
        is a property of how the hand is currently sitting -- a finger already
        resting against the object reads its own preload, and measuring the
        target above THAT is what makes "300 counts more than now" mean the same
        thing on every run.

        Averaged over real messages rather than sampled once: the driver's own
        filter is an EMA with alpha 0.25, so a single reading carries three
        quarters of whatever came before it.
        """
        if not self.available:
            raise RuntimeError(
                "this hand's driver declares no tactile sensor, so there is "
                "nothing to take a baseline from")
        digits = list(digits)
        collected = {name: [] for name in digits}
        seen = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                state = self._state
            if state is not None and state[0] != seen:
                seen = state[0]
                for name in digits:
                    if name in state[1]:
                        collected[name].append(state[1][name])
                if min(len(v) for v in collected.values()) >= samples:
                    break
            time.sleep(period)

        missing = [name for name, v in collected.items() if not v]
        if missing:
            raise RuntimeError(
                f"no tactile reading arrived for {', '.join(missing)} on "
                f"`{self.topic}` within {timeout:.1f} s -- is the hand node "
                f"publishing?")
        return {name: float(np.mean(v)) for name, v in collected.items()}
