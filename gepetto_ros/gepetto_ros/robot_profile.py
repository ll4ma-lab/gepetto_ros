"""The whole per-robot difference, in one table.

Every node in this package takes a ``hand`` parameter, looks the profile up here,
and is otherwise identical for the two cells. Adding a third robot is another
entry below plus a hand driver; it is not a branch anywhere else.

WHAT ACTUALLY DIFFERS, and why each field is here rather than hardcoded:

``solver_hand``    which ``gepetto_solvers`` hand poses the workbench and builds
                   the plan. It decides the plan's ``command_kind``, its digit
                   count and its per-digit width -- everything downstream follows
                   from it.
``wrist_frame``    the SRDF chain tip, which is where MoveIt Servo takes its
                   Jacobian AND what the solver's wrist variable is. Those two
                   coincide on both robots -- `lbr_epfl_wrist_link` sits exactly
                   at `MountConfig.flange_from_wrist_*`, and `palm_link` is the
                   frame `AllegroHand.mount_pose` was measured against with
                   `tf2_echo lbr_link_ee palm_link` -- which is what lets a solved
                   wrist pose be commanded with no mount conversion.
``hand_driver``    how the hand is commanded. The one genuinely different piece of
                   code; see :mod:`gepetto_ros.hand_drivers`.
``lbr_model``      the description to bring up, for the launch file.
``servo_cfg``      the Servo config, if this model needs its own. The Allegro's
                   differs from the shared one (different `ee_frame_name`, and
                   proximity thresholds sized for a hand), so it names it.
``arm_home``       where `home_node` parks the ARM. Per robot because the hand is
                   what the fold has to keep clear: the Allegro is a wider, longer
                   end effector than the tendon hand and the pose that tucks one
                   in puts the other through the table.

WHAT DOES NOT DIFFER, and is therefore NOT here: the planning frame
(`lbr_link_0`), the workspace table frame (`med14_allegro.xacro` includes the very
same table xacro), the Servo scales (0.4 m/s / 0.8 rad/s / 0.5 rad in both yamls),
the e-stop topic and the play action. Putting any of those in this table would
invite them to drift apart for no reason.
"""

from collections.abc import Callable
from dataclasses import dataclass

from .arm_servo import DEFAULT_PLANNING_FRAME
from .frames import TABLE_FRAME
from .hand_drivers import AllegroJointDriver, TendonHandDriver


@dataclass(frozen=True)
class RobotProfile:
    """One arm-plus-hand cell."""

    #: What ``hand:=`` selects this profile by.
    name: str
    #: ``gepetto_solvers.core.hands`` registry key.
    solver_hand: str
    #: `lbr_description` model, for the arm bringup.
    lbr_model: str
    #: SRDF chain tip -- the frame Servo moves and the solver's wrist. See above.
    wrist_frame: str
    #: The class that turns a per-digit command into messages this hand takes.
    hand_driver: Callable
    #: Human-readable, for status lines and log headers.
    description: str
    planning_frame: str = DEFAULT_PLANNING_FRAME
    table_frame: str = TABLE_FRAME
    #: ``(package, path)`` of a Servo config this model needs instead of the
    #: shared one, or None for the shared one.
    servo_cfg: tuple | None = None
    #: The folded configuration `home_node` drives to, in JOINT-NUMBER order
    #: (`lbr_A1` .. `lbr_A7`), radians. Recorded from `/lbr/joint_states` with
    #: the arm parked where the cell wants it -- so it is measured, not designed,
    #: and re-recording it is how it changes. Note that `/lbr/joint_states`
    #: publishes the joints in a different order than they are numbered, so a
    #: fresh recording has to be re-ordered into this one.
    arm_home: tuple[float, ...] = ()


PROFILES = {
    "epfl": RobotProfile(
        name="epfl",
        solver_hand="tendon_5f",
        lbr_model="med14_epfl",
        wrist_frame="lbr_epfl_wrist_link",
        hand_driver=TendonHandDriver,
        description="KUKA med14 + the EPFL five-digit tendon hand, driven "
                    "through finger_servo_node on the Dynamixel bus",
        arm_home=(-0.6134081775731002,    # lbr_A1
                  -0.7799179570643047,    # lbr_A2
                  0.11252438167854789,    # lbr_A3
                  1.5145508839169586,     # lbr_A4
                  0.09815118544143298,    # lbr_A5
                  0.5074425823260066,     # lbr_A6
                  1.4141795847763836),    # lbr_A7
    ),
    "allegro": RobotProfile(
        name="allegro",
        solver_hand="allegro",
        lbr_model="med14_allegro",
        wrist_frame="palm_link",
        hand_driver=AllegroJointDriver,
        description="KUKA med14 + the Allegro Hand V5 (right, type B), driven "
                    "through allegro_node_grasp on CAN",
        # The hand changes what proximity thresholds make sense, so this model
        # carries its own Servo config rather than the one every other model
        # shares -- and its ee_frame_name is palm_link, matching the chain tip.
        servo_cfg=("lbr_bringup", "config/moveit_servo_allegro.yaml"),
        arm_home=(-0.4950334567142735,    # lbr_A1
                  -0.7388789376769671,    # lbr_A2
                  0.1070175704189514,     # lbr_A3
                  1.6837356623575184,     # lbr_A4
                  0.4088504612220396,     # lbr_A5
                  0.9105970614177137,     # lbr_A6
                  2.7603514496454),       # lbr_A7
    ),
}


def get_profile(name):
    """The profile ``name`` selects. Raises with the available names.

    Mirrors ``gepetto_solvers.core.hands.get_hand``'s contract on purpose: the
    two are looked up together (a profile names a hand) and a caller that has
    learned one lookup's failure mode has learned both.
    """
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(
            f"no robot profile {name!r}. Available: "
            f"{', '.join(sorted(PROFILES))}.") from None


def registered_profiles():
    return sorted(PROFILES)
