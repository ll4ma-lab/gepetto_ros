"""Run the interactive hand visualizer as a ROS node, wired to the real robot.

    ros2 run gepetto_ros viz_node --ros-args -p hand:=epfl
    ros2 run gepetto_ros viz_node --ros-args -p hand:=allegro

This is `gepetto_solvers`' `viz_interactive` -- the same app, the same solvers,
the same E-STOP -- hosted inside a node and handed a bridge, so its Robot folder
can send a solve to be played on the arm and the hand and read their state back.
With the node not running, the app still runs standalone exactly as before
(`python scripts/viz_interactive.py --hand allegro`); ROS mode is additive.

ONE PARAMETER PICKS THE ROBOT. `hand` names a
:class:`~gepetto_ros.robot_profile.RobotProfile`, which says which
`gepetto_solvers` hand to pose, which TF frame is the wrist, and which driver
class the bridge decodes state with. The visualizer is then built on that hand and
sizes every panel off it -- the digit checkboxes, the tension or joint sliders,
the trajectory plots -- with nothing here doing the sizing.

The usual bringup starts this, the executor and the arm together:

    ros2 launch gepetto_ros_launch gepetto_bringup.launch.py hand:=epfl mode:=mock

then open the printed http://localhost:8080.

THIS PROCESS DOES NOT DRIVE THE ROBOT. It draws, it solves, and it dispatches.
The control loop lives in `executor_node`, in its own process, because it used to
live here and that was a bug: viser's asyncio server, its callback pool and a
scene layer that rebuilds meshes every frame share this GIL, and a stall long
enough to outlast MoveIt Servo's command timeout became tracking error and a
tripped watchdog. The bringup starts both; the Robot folder says so when the
executor is missing. See `robot_bridge` for the dispatch and `executor_node` for
the loop.

ONE PROCESS PER HAND BUS. Neither this node nor the executor touches the hand's
own bus -- they publish to whatever driver owns it (`finger_servo_node` on the
Dynamixel port, `allegro_node_grasp` on CAN). On the EPFL cell in particular, do
not run `hand_node` or `finger_slider_node` alongside: all three claim
`/dev/ttyUSB*`.
"""

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from gepetto_solvers.core.hands import get_hand

from .robot_profile import get_profile, registered_profiles


class VizNode(Node):

    def __init__(self):
        super().__init__("gepetto_viz")

        self.declare_parameter("hand", "allegro")
        self.declare_parameter("port", 8080)
        self.declare_parameter("playback_speed", 0.50)
        # There is deliberately no `enable_arm` / `enable_hand` here. The Robot
        # folder's single checkbox is a per-press safety interlock that clears
        # itself after every run, and a launch file that could pre-tick it would
        # be handing back exactly the "armed and forgotten" state the interlock
        # exists to prevent. Arming is a decision made at the browser, looking at
        # the robot -- the same rule the e-stop rearm already follows.
        #
        # Empty means "whatever this hand's driver says", which is the answer in
        # every assembled cell.
        self.declare_parameter("hand_state_topic", "")

        name = self.get_parameter("hand").value
        try:
            self._profile = get_profile(name)
        except KeyError as exc:
            raise RuntimeError(
                f"{exc} Set it with `-p hand:=<name>`.") from None
        hand = get_hand(self._profile.solver_hand)

        import viser

        from gepetto_solvers.projects.viz.viz_interactive import (
            HandVizApp,
            binding_path,
        )

        from .robot_bridge import RobotBridge

        port = int(self.get_parameter("port").value)
        self._server = viser.ViserServer(port=port)
        # No control parameters here -- the rate, the gains, the lag thresholds
        # and the watchdogs all belong to `executor_node`, which is the process
        # that actually ticks. Declaring them here as well would let the two
        # disagree about a number only one of them uses.
        self._bridge = RobotBridge(
            self, self._profile, hand,
            hand_state_topic=self.get_parameter("hand_state_topic").value or None,
        )
        self._app = HandVizApp(self._server, ros_mode=True, bridge=self._bridge,
                               hand=hand)

        # The inbound half of the e-stop: a trip published by anything else in the
        # cell now reaches the visualizer's latch, so the GUI greys out and a
        # running solve stops too. (The outbound half is the app registering
        # itself as an EStop listener, which HandVizApp does in ROS mode.)
        self._bridge.bind_estop(self._app.estop)

        # What speed a fresh browser opens on. Safe to set from a launch file
        # because it only decides how fast a playback runs once someone has armed
        # and started one -- unlike the arming itself, it commands nothing.
        self._app.g_speed.value = float(
            self.get_parameter("playback_speed").value)

        # Publish the (clear) e-stop state once at startup, so the latched topic
        # reflects this cell rather than whatever a previous run left on it.
        self._bridge.set_estop(False)

        self.get_logger().info(f"gepetto_solvers binding: {binding_path()}")
        self.get_logger().info(
            f"robot: {self._profile.description}  |  hand: {hand.name} "
            f"({hand.kinematics} kinematics, {len(hand.digit_names)} digits: "
            f"{', '.join(hand.digit_names)})  |  profiles: "
            f"{', '.join(registered_profiles())}")
        missing = [k for k, v in self._app.caps.items() if not v]
        if missing:
            self.get_logger().warn(
                f"capabilities MISSING from this build: {', '.join(missing)}")
        self.get_logger().info(
            f"viser hand visualizer on http://localhost:{port} (ROS mode)")

    def destroy_node(self):
        # Halt before the publishers go away, so a node killed mid-playback does
        # not leave the arm coasting on its last twist for the full
        # incoming_command_timeout.
        #
        # Only while the context is still up: on Ctrl-C rclpy's SIGINT handler
        # has already invalidated it by the time we get here, and publishing then
        # merely raises. The arm still stops -- an interrupted process publishes
        # nothing, which is exactly what the servo's command timeout is for.
        if rclpy.ok():
            try:
                self._bridge.stop("viz node shutting down")
            except Exception as exc:
                self.get_logger().warn(f"error halting on shutdown: {exc}")
        try:
            self._server.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VizNode()
    # Multi-threaded: viser's callbacks arrive on its own threads and run solves
    # that hold the GIL for whole seconds, while the bridge's subscriptions and
    # TF listener have to keep being serviced throughout. (The solve bindings
    # release the GIL -- capabilities()["gil_release"] -- which is what makes
    # that actually work rather than merely being configured to.)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
