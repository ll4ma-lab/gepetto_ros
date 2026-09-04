"""The EPFL tendon hand, driven through `epfl_hand_hardware/finger_servo_node`.

Absolute tendon-displacement targets, published as the DELTAS that node wants.

`finger_servo_node` INTEGRATES what it receives, so it cannot be told "be at
5 mm" -- only "move 0.1 mm further". This driver holds the running total it has
commanded and publishes the difference, which turns an absolute target back into
the integrating interface without the caller having to track anything. Because the
total is kept in float metres and only the difference is sent, displacements far
below a motor count still accumulate rather than being lost.

Seed it from the MEASURED state before the first tick, or the first delta is
measured from an assumed zero and the hand lurches by however far it actually was
from open.

SIGN: positive = pulled in = flexing, matching the node's `~/delta_tendon_cmds`
and its state topics. (`~/delta_motor_cmds` is the opposite; this driver never
touches it.)

WIDTH 1, ALWAYS. This hand drives one actuator per digit -- the flexor -- so every
vector here is one long, and the driver says so rather than quietly reading
``[0]`` of whatever it is handed: a width-4 target arriving here is a plan built
for another hand, and it must not be half-applied.
"""

import numpy as np
from control_msgs.msg import JointJog
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from .base import clamp_target

#: What the hardware calls each solver digit. The tendon hand's own
#: ``HardwareMap.actuator_names`` says the same thing, and is preferred when a
#: hand object is available; this is the fallback for a node that has none.
DEFAULT_ACTUATOR_SUFFIX = "_flex"

DEFAULT_COMMAND_TOPIC = "/finger_servo_node/delta_tendon_cmds"
DEFAULT_STATE_TOPIC = "/finger_servo_node/measured_state"


def actuator_names(hand):
    """``{solver digit: hardware finger}``, from the hand.

    The spelling lives in the hand's own ``HardwareMap.actuator_names``, so it is
    stated once. The suffix fallback exists only for a tendon-like hand that
    supplies no map at all, and it is the convention that map already follows.
    """
    actuators = getattr(getattr(hand, "hardware", None), "actuator_names", None)
    if actuators:
        return dict(actuators)
    return {name: f"{name}{DEFAULT_ACTUATOR_SUFFIX}" for name in hand.digit_names}


class TendonHandDriver:
    """Absolute tendon displacements onto an integrating jog node."""

    #: Driven actuators per digit.
    width = 1

    def __init__(self, node, hand, limits=None, topic=DEFAULT_COMMAND_TOPIC,
                 frame_id="", hz=100.0):
        self._node = node
        self.hand = hand
        self.digit_names = list(hand.digit_names)
        self._frame_id = frame_id
        self._pub = node.create_publisher(JointJog, topic, 10)
        self._period = 1.0 / float(hz)
        self.limits = dict(limits or {})
        self._to_hardware = actuator_names(hand)
        # What we have told the node to be at, cumulatively. Not what the motors
        # have reached -- the node rate-limits, so it lags this by design.
        self._commanded = {name: np.zeros(self.width) for name in self.digit_names}

    # -- readback -----------------------------------------------------------
    #
    # CLASS METHODS, taking the hand rather than an instance: reading the hand's
    # state has to work before -- and after -- any commanding does. The executor
    # subscribes at startup and only builds a driver when a goal arrives, and the
    # bridge never builds one at all, so a decode that needed a live publisher
    # would be a decode neither of them could use.

    @staticmethod
    def state_topic():
        return DEFAULT_STATE_TOPIC

    @classmethod
    def decode_state(cls, hand, msg: JointState):
        """`measured_state` -> ``{solver digit: (1,) metres}``.

        Keyed back to SOLVER digit names on the way in, so nothing downstream has
        to know the hardware's `*_flex` spelling.
        """
        to_solver = {v: k for k, v in actuator_names(hand).items()}
        return {to_solver[name]: np.array([float(position)])
                for name, position in zip(msg.name, msg.position)
                if name in to_solver}

    # -- commanding ---------------------------------------------------------

    def seed(self, measured):
        """Adopt the hand's measured displacement as the running total, so the
        first tick commands the difference from REALITY rather than from zero."""
        for name in self.digit_names:
            value = measured.get(name)
            if value is not None:
                self._commanded[name] = np.ravel(np.asarray(value, float))[:1].copy()

    def tick(self, target):
        """Publish one message moving each digit toward its absolute ``target``
        (metres of displacement from open). Returns the digits clamped."""
        bounded, clamped = clamp_target(target, self.limits, self.width)
        names, deltas = [], []
        for name in self.digit_names:
            value = bounded.get(name)
            if value is None:
                continue
            names.append(self._to_hardware[name])
            deltas.append(float(value[0] - self._commanded[name][0]))
            self._commanded[name] = value.copy()
        if names:
            self._publish(names, deltas)
        return clamped

    def hold(self):
        """Publish a zero delta for every digit: an explicit "stay". Useful as a
        keepalive while another channel is still moving."""
        names = [self._to_hardware[n] for n in self.digit_names]
        self._publish(names, [0.0] * len(names))

    def halt(self):
        """Stop commanding. Deliberately publishes NOTHING: the node holds its
        integrated setpoint, so silence is the hold, and a zero-delta message
        would be indistinguishable from the keepalive above."""
        return

    @property
    def commanded(self):
        return {name: value.copy() for name, value in self._commanded.items()}

    def _publish(self, names, deltas):
        msg = JointJog()
        header = Header()
        header.stamp = self._node.get_clock().now().to_msg()
        # finger_servo_node ignores the header entirely -- no staleness gate, no
        # extrapolation. Stamped so bags and `ros2 topic echo` read sensibly.
        header.frame_id = self._frame_id
        msg.header = header
        msg.joint_names = list(names)
        msg.displacements = [float(d) for d in deltas]
        # Left empty: the node prefers displacements when both are filled, and
        # these ARE displacements -- a position difference, not a rate.
        msg.velocities = []
        msg.duration = self._period
        self._pub.publish(msg)
