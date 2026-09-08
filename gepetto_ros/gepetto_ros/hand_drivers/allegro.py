"""The Allegro Hand V5, driven through Wonik's `allegro_node_grasp`.

Absolute joint positions, published as the sixteen-wide `sensor_msgs/JointState`
that node takes:

    /allegroHand/joint_cmd      sensor_msgs/JointState, 16 positions, radians

THE MESSAGE IS READ POSITIONALLY. `AllegroNodeGrasp::setJointCallback` does
``desired_position[i] = msg->position[i]`` and IGNORES ``msg.name`` entirely, so
the order of `position` is the whole contract. Get it wrong and nothing fails --
the wrong joints move. That order is
:data:`gepetto_solvers.core.hands.allegro.spec.DRIVER_JOINT_ORDER`
(``joint_0_0`` .. ``joint_15_0``: index 0-3, middle 4-7, ring 8-11, thumb 12-15),
it is derived from the same table the solver builds its digits from, and
``gepetto_solvers/tests/core/test_plan_wire.py`` pins it against the mock node's
own list. ``name`` is filled in anyway, for `ros2 topic echo` and for bags.

A SHORT MESSAGE IS REJECTED, not half-applied: the driver drops anything under
sixteen entries. So every tick publishes all sixteen, including digits the plan
did not name -- those hold whatever was last commanded for them, which is what
:meth:`AllegroJointDriver.seed` exists to make a real measurement rather than a
guess of zero.

NO INTEGRATION, unlike the tendon hand's node: this one takes absolute positions,
so there is no running total to keep in step and no delta to compute. What IS kept
is a slew limit -- the driver runs its own ~333 Hz joint-PD loop and will chase
any setpoint it is given as fast as it can, so a step change in the commanded
position is a step change at the fingers. The reference is therefore rate-limited
here, per joint, to `max_speed` radians per second at the control period.

WIDTH 4, ALWAYS. Four joints per digit; a width-1 target arriving here is a plan
built for the tendon hand and is refused rather than broadcast to all four.
"""

import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray

from .base import clamp_target

DEFAULT_COMMAND_TOPIC = "/allegroHand/joint_cmd"
#: `lbr_allegro.launch.py` remaps the driver's own `allegroHand/joint_states`
#: onto `/lbr/joint_states`, so that one robot_state_publisher draws the arm and
#: the fingers from one topic. That remapped name is what to subscribe to in the
#: assembled cell; the driver's own name still works when the hand runs alone.
DEFAULT_STATE_TOPIC = "/lbr/joint_states"

#: Wonik's own fingertip tactile topic, four `int32` on
#: `std_msgs/Int32MultiArray` -- see `allegro_node.cpp::publishData`. NOT remapped
#: by `lbr_allegro.launch.py`, unlike the joint states, so this is the name in the
#: assembled cell as well as on a hand running alone.
DEFAULT_TACTILE_TOPIC = "/allegroHand/tactile_sensors"

#: Which element of that array is which solver digit.
#:
#: Verified on the hardware, and consistent with the driver's own code: the CAN
#: decode writes index pairs (0, 1) and (2, 3) from `ID_CMD_FINGERTIP_1` and
#: `_3`, and `allegro_node.cpp`'s grasp heuristic sums elements 0, 1 and 3 --
#: index, middle and thumb, the tripod. It is written out rather than derived
#: from `DIGIT_NAMES` because it is a fact about the FIRMWARE's ordering that
#: happens to agree with ours, and a hand whose firmware disagreed would need
#: this line changed and nothing else.
DEFAULT_TACTILE_ORDER = ["index", "middle", "ring", "thumb"]

#: What the driver reports for a fingertip whose reading fell outside
#: `[0, 5000]`: ZERO. See `AllegroHandDrv.cpp`, which clamps out-of-range to zero
#: and only THEN low-passes it. Consumers must not read a low number here as
#: "this fingertip is touching nothing" -- it is equally "this fingertip is being
#: crushed" -- which is why `core.solvers.grasp_servo` watches for a reading that
#: collapses from its own peak rather than for one that is simply high.
TACTILE_OUT_OF_RANGE = 0


class AllegroJointDriver:
    """Absolute joint positions onto Wonik's positional command topic."""

    #: Driven actuators per digit.
    width = 4

    def __init__(self, node, hand, limits=None, topic=DEFAULT_COMMAND_TOPIC,
                 hz=100.0, max_speed=None):
        self._node = node
        self.hand = hand
        self.digit_names = list(hand.digit_names)
        self._pub = node.create_publisher(JointState, topic, 10)
        self._period = 1.0 / float(hz)
        self.limits = dict(limits or {})
        # Per-tick slew, in radians. From the hand unless overridden, so the
        # driver's ceiling and the PATH's ceiling (robot_plan's `max_digit`,
        # which is the same number) cannot be set apart by accident.
        speed = hand.max_digit_speed if max_speed is None else max_speed
        self._max_step = float(speed) * self._period

        # {digit: [joint names]} and the flat driver order, both from the hand --
        # see the module docstring on why this table is never re-derived here.
        self._joints = hand.driver_joint_names()
        self._order = [j for name in self.digit_names for j in self._joints[name]]
        self._index = {joint: i for i, joint in enumerate(self._order)}
        if len(self._order) != len(set(self._order)):
            raise ValueError("the hand's driver_joint_names repeat a joint name")

        # The last published position vector, flat in driver order. Starts at the
        # neutral configuration; `seed` replaces it with a measurement before the
        # first tick, and until then nothing is published at all.
        self._commanded = np.zeros(len(self._order))

    # -- readback -----------------------------------------------------------
    #
    # CLASS METHODS, taking the hand rather than an instance -- see the same note
    # in `tendon.py`: the executor subscribes to the state topic at startup and
    # only builds a driver when a goal arrives, and the bridge never builds one.

    @staticmethod
    def state_topic():
        return DEFAULT_STATE_TOPIC

    @classmethod
    def decode_state(cls, hand, msg: JointState):
        """`joint_states` -> ``{solver digit: (4,) radians}``.

        BY NAME, not positionally, and that asymmetry with the command path is
        deliberate: the state topic this reads is shared with the ARM's
        `joint_state_broadcaster` (they are remapped onto one `/lbr/joint_states`
        so a single robot_state_publisher draws both), so a message here may carry
        the arm's seven joints, the hand's sixteen, or some other mix depending on
        which publisher it came from. Only the command topic is positional.

        A digit whose joints are not all present is omitted entirely rather than
        partly filled -- half a digit is not a measurement of that digit.
        """
        seen = {name: float(position)
                for name, position in zip(msg.name, msg.position)}
        out = {}
        for digit, joints in hand.driver_joint_names().items():
            if all(joint in seen for joint in joints):
                out[digit] = np.array([seen[joint] for joint in joints])
        return out

    # -- tactile ------------------------------------------------------------
    #
    # DECLARING THESE IS WHAT MAKES A HAND CLOSEABLE. `TactileReader` asks the
    # driver class for them and the grasp action refuses outright on a driver
    # that has neither, rather than degrading into a close with no stopping
    # condition. The tendon hand's driver declares neither, and so cannot be
    # asked to do a tactile close -- which is correct, it has no such sensor.

    #: The message type, so the reader can subscribe without knowing the hand.
    tactile_msg = Int32MultiArray

    @staticmethod
    def tactile_topic():
        return DEFAULT_TACTILE_TOPIC

    @classmethod
    def decode_tactile(cls, hand, msg, order=None):
        """`tactile_sensors` -> ``{solver digit: counts}``.

        POSITIONAL, like the command path and unlike the joint-state readback:
        this topic carries one publisher's fixed-length array with no names in
        it at all, so the order IS the contract. See `DEFAULT_TACTILE_ORDER`.

        A message shorter than the order is dropped entirely rather than partly
        decoded -- the same rule `decode_state` applies to a half-present digit,
        and for a stronger reason here: a short array means the elements are not
        where this thinks they are, so every value in it is suspect.
        """
        order = list(order or DEFAULT_TACTILE_ORDER)
        data = list(msg.data)
        if len(data) < len(order):
            raise ValueError(
                f"tactile message has {len(data)} entries, expected at least "
                f"{len(order)} for {order}")
        known = set(hand.digit_names)
        return {name: float(data[i]) for i, name in enumerate(order)
                if name in known}

    # -- commanding ---------------------------------------------------------

    def seed(self, measured):
        """Adopt the hand's measured posture, so the first tick steps from where
        the fingers ARE rather than from the neutral configuration.

        On this hand that matters more than on an integrating one: the very first
        message is an absolute position, so an unseeded driver commands the hand
        to wherever the plan starts in one tick's worth of slew and then keeps
        going -- the slew limit turns a lurch into a fast sweep, which is safer
        but is still a move nobody asked for.
        """
        for digit, joints in self._joints.items():
            values = measured.get(digit)
            if values is None:
                continue
            values = np.ravel(np.asarray(values, float))
            for j, joint in enumerate(joints):
                if j < values.size and np.isfinite(values[j]):
                    self._commanded[self._index[joint]] = float(values[j])

    def tick(self, target):
        """Publish one absolute command stepped toward ``target``
        (radians per joint). Returns the digits clamped."""
        bounded, clamped = clamp_target(target, self.limits, self.width)

        want = self._commanded.copy()
        for digit, values in bounded.items():
            joints = self._joints.get(digit)
            if joints is None:
                continue
            for j, joint in enumerate(joints):
                want[self._index[joint]] = values[j]

        # Rate limit the STEP, not the setpoint, and per joint: one big jump
        # ramps at max_speed instead of snapping, and a joint that is already
        # there is not held back by a neighbour that is not.
        step = np.clip(want - self._commanded, -self._max_step, self._max_step)
        self._commanded = self._commanded + step
        self._publish()
        return clamped

    def hold(self):
        """Re-publish the current setpoint. On an absolute interface that IS a
        keepalive: the driver holds its last command anyway, and repeating it
        keeps the topic live while another channel is still moving."""
        self._publish()

    def halt(self):
        """Stop commanding. Publishes the CURRENT setpoint once more and then
        nothing.

        Unlike the tendon node, this driver has no integrated state to preserve --
        it simply holds its last received position -- so one final publish of
        where we are is an explicit "stay here" rather than a move, and silence
        afterwards leaves the hand exactly there.
        """
        self._publish()

    @property
    def commanded(self):
        return {digit: np.array([self._commanded[self._index[j]] for j in joints])
                for digit, joints in self._joints.items()}

    def _publish(self):
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        # Filled for readability only -- the driver ignores it. Filling it wrongly
        # would therefore not break anything, which is exactly why it is built
        # from the same list that orders `position` rather than typed out.
        msg.name = list(self._order)
        msg.position = [float(v) for v in self._commanded]
        self._pub.publish(msg)
