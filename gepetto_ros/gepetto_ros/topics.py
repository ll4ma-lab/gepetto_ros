"""Names and QoS the whole cell agrees on.

Its own module so that `home_node`, `executor_node` and `robot_bridge` can share
the e-stop contract without any of them having to import another's implementation
-- a node that only wants to know the stop topic should not be pulling in the
visualizer's action client to find out what it is called.

NOT NAMESPACED PER ROBOT, deliberately. These are CELL-WIDE names: one arm, one
hand on it, one stop button. Namespacing them per hand would let an EPFL executor
and an Allegro executor come up side by side, each deaf to the other's e-stop,
which is the one failure this topic exists to prevent. Only one robot profile runs
at a time; the profile decides which hand driver is built, not which stop topic is
watched.
"""

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

#: Latched, workspace-wide software e-stop. Anything may publish True to trip it;
#: rearming is deliberately NOT done over the topic -- it is a decision someone
#: makes at the visualizer, looking at the robot.
ESTOP_TOPIC = "/gepetto/estop"

ESTOP_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    # Latched: a node that starts after the stop was pressed must come up knowing
    # the cell is stopped, not knowing nothing.
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

#: Where `executor_node` serves `gepetto_ros_interfaces/action/PlayPlan`, and
#: where `robot_bridge` sends solves to be played.
PLAY_PLAN_ACTION = "/gepetto/play_plan"

#: Where `home_node` serves `gepetto_ros_interfaces/action/MoveHome`. Cell-wide
#: for the same reason the e-stop is: it drives the SAME arm and hand as the play
#: action, and the two are mutually exclusive -- two of these under different
#: namespaces would be two things able to park one robot at once.
MOVE_HOME_ACTION = "/gepetto/move_home"

#: Where `executor_node` serves `gepetto_ros_interfaces/action/GraspClose`: the
#: tactile close, which the planner cannot express because it has no dynamics.
#:
#: Served by the SAME node as `PLAY_PLAN_ACTION`, and that is the point. Both
#: drive the hand's command topic, so they must never overlap, and one node
#: holding both is the cheapest way to make that impossible -- a second process
#: would need a lock the two of them could disagree about.
GRASP_CLOSE_ACTION = "/gepetto/grasp_close"
