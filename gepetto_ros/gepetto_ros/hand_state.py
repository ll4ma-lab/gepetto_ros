"""Reading the hand's measured state, whatever hand it is.

One subscription, one lock, one age. The DECODING -- which topic, which message
fields, which names map onto which solver digit -- belongs to the hand driver and
is asked of it by class (see the note in `hand_drivers/tendon.py` on why those are
class methods), so this file holds no per-robot knowledge at all.

WHY THE AGE IS KEPT. `lookup`-style readers hand back the latest thing they have,
which is a perfectly good answer right up until the publisher dies -- after which
they keep returning the same stale reading forever. Two things depend on knowing
that has happened: the control loop, which would otherwise seed a playback from a
posture the hand left minutes ago, and the visualizer's Robot folder, which says
so in its status line.

MONOTONIC, NOT THE MESSAGE STAMP. What is being asked is "how long since anything
arrived", which is a fact about this process's clock and not about the publisher's
-- and the tendon node stamps its state messages with a clock the executor has no
reason to trust to be synchronised.
"""

import threading
import time


class HandStateReader:
    """The hand's measured state, decoded into solver digits and vectors."""

    def __init__(self, node, hand, driver_class, topic=None, depth=10,
                 callback_group=None):
        self._hand = hand
        self._driver_class = driver_class
        self.topic = topic or driver_class.state_topic()
        self._state = None      # (monotonic stamp, {digit: (K,) vector})
        self._lock = threading.Lock()
        # The message type is the driver's business too, but every hand's state
        # so far is a JointState and subscribing needs the type up front. A hand
        # that reports something else declares `state_msg` and this follows it.
        from sensor_msgs.msg import JointState
        msg_type = getattr(driver_class, "state_msg", JointState)
        node.create_subscription(msg_type, self.topic, self._on_state, depth,
                                 callback_group=callback_group)

    def _on_state(self, msg):
        try:
            state = self._driver_class.decode_state(self._hand, msg)
        except Exception:
            # A malformed or unexpected message on a SHARED topic is not an
            # error here: the Allegro reads `/lbr/joint_states`, which also
            # carries the arm's own seven joints from a different publisher.
            # Dropping what does not decode is the correct response to that, and
            # `read()`'s age is what reports a topic carrying nothing usable.
            return
        if not state:
            return
        with self._lock:
            self._state = (time.monotonic(), state)

    def read(self):
        """``(state, age_seconds)``; ``({}, None)`` if nothing has arrived."""
        with self._lock:
            if self._state is None:
                return {}, None
            stamp, state = self._state
        return {name: value.copy() for name, value in state.items()}, \
            time.monotonic() - stamp
