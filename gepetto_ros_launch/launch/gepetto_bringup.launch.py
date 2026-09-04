"""The whole cell: arm, servo, hand driver, executor and the viser workbench.

    ros2 launch gepetto_ros_launch gepetto_bringup.launch.py mode:=mock  # allegro
    ros2 launch gepetto_ros_launch gepetto_bringup.launch.py hand:=epfl mode:=mock

Then open http://localhost:8080 and use the *Robot* folder: tick *move the real
robot* (which arms one press and clears itself afterwards), press *Play solve on
robot* to execute the solve on screen, or *Get robot state* to pull the hardware's
wrist pose and digit state back in.

WHAT `hand` SELECTS, and it defaults to `allegro`. One argument picks a
`gepetto_ros.robot_profile.RobotProfile` and everything follows from it: which
`lbr_bringup` launch brings the arm and the hand driver up, which
`gepetto_solvers` hand the workbench poses, which TF frame is the wrist, and
which driver class the executor commands with. `viz_node` and
`executor_node` are the SAME nodes either way -- they take `hand` as a parameter.

THE VISER HALF LIVES HERE, not in `lbr_fri_ros2_stack`. That stack is the
hardware, the descriptions and the servo, and it stays that way: this launch
INCLUDES its bringup with the hand-side nodes switched off and adds the workbench
itself. `lbr_epfl.launch.py`'s old `epfl_nodes:=viz` did the opposite and is gone.

THE SPLIT BETWEEN PROCESSES MATTERS. `viz_node` draws and solves; `executor_node`
owns the control loop and is the only process that publishes motion commands
during playback; `home_node` parks the cell on `/gepetto/move_home` and publishes
nothing until it is asked to; the hand's own driver owns its bus. The first two
used to be one process, which meant a heavy scene update could stall the control
loop past MoveIt Servo's command timeout and trip the tracking watchdog. Play
still works with `executor_node` down -- the Robot folder says the executor is
missing and refuses, rather than looking like it did nothing.

MOCK IS THE DEFAULT and everything below is exercisable with no hardware:

    hand:=epfl     mock arm on ros2_control + finger_servo_node dry_run:=true
    hand:=allegro  mock arm on ros2_control + allegro_mock_node.py (no CAN)

WITH mode:=hardware THE ARM IS NOT STARTED HERE. The description, ros2_control and
the arm controllers come from the robot PC, which runs

    ros2 launch lbr_bringup hardware.launch.py model:=<model> \\
        ctrl:=forward_position_controller

on its own. Start it there first, then start this. That is `lbr_epfl.launch.py`'s
and `lbr_allegro.launch.py`'s own rule and this launch simply forwards `mode` to
them; see their docstrings.

THE CONDA ENV IS INJECTED PER NODE, never launch-wide. `gepetto_solvers`,
`epfl_hand_core` and the compiled solver binding live in a conda env built against
python 3.10, while `move_group`, `servo_node`, `ros2_control_node` and RViz must
keep the system python that rclpy and MoveIt were built against. A launch-wide
`SetEnvironmentVariable` leaks conda's site-packages into all of them; the
`additional_env` below reaches only the nodes that need it.

ONE PROCESS PER HAND BUS. On the EPFL cell, `finger_servo_node`,
`finger_slider_node` and `hand_node` all claim `/dev/ttyUSB*`. This launch starts
the first; do not run the others alongside, and do not launch
`lbr_epfl.launch.py epfl_nodes:=sliders|full` at the same time.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()

    # -- what robot -----------------------------------------------------------
    ld.add_action(DeclareLaunchArgument(
        "hand", default_value="allegro", choices=["epfl", "allegro"],
        description="Which robot profile to bring up. Picks the arm description, "
                    "the hand driver, the wrist frame and the gepetto_solvers "
                    "hand -- see gepetto_ros/robot_profile.py."))
    # SNAPSHOT IT UNDER A PRIVATE NAME, IMMEDIATELY, and use only that below.
    #
    # `lbr_allegro.launch.py` declares its own `hand` (which hand NODE to start),
    # and a launch configuration set by an include does not reliably stay inside
    # it. Two different failures came out of sharing this name, in both
    # directions: `hand:=allegro` reaching that launch's `choices` and killing the
    # bringup, and -- once that was passed explicitly -- its `mock` default
    # reaching OUR nodes, which are read lazily at spawn time and so saw the
    # include's value rather than the command line's. `executor_node` then died
    # with "no robot profile 'mock'".
    #
    # `SetLaunchConfiguration` resolves eagerly, when this action runs, which is
    # before any include. So `gepetto_hand` holds what the operator actually
    # typed and nothing downstream can overwrite it.
    ld.add_action(SetLaunchConfiguration("gepetto_hand", LaunchConfiguration("hand")))
    hand = LaunchConfiguration("gepetto_hand")
    is_epfl = PythonExpression(["'", hand, "' == 'epfl'"])
    is_allegro = PythonExpression(["'", hand, "' == 'allegro'"])

    ld.add_action(DeclareLaunchArgument(
        "mode", default_value="mock", choices=["mock", "hardware", "gazebo"],
        description="Arm operation mode. With 'hardware' the arm stack comes "
                    "from the robot PC and is NOT started here; see this file's "
                    "docstring."))
    mode = LaunchConfiguration("mode")

    ld.add_action(DeclareLaunchArgument(
        "rviz", default_value="true",
        description="Whether to start RViz alongside the viser workbench. They "
                    "answer different questions -- RViz shows the robot's own "
                    "description and TF, viser shows the solve."))
    rviz = LaunchConfiguration("rviz")

    ld.add_action(DeclareLaunchArgument(
        "arm", default_value="true",
        description="Whether to bring the arm half up at all. 'false' pairs this "
                    "with `ros2 run gepetto_ros fake_cell`, which publishes the "
                    "TF and hand state a cell would with no MoveIt anywhere."))
    arm = LaunchConfiguration("arm")

    # Which HAND NODE to start, in lbr_allegro.launch.py's own vocabulary. Named
    # `hand_node` rather than `hand` because `hand` already means the ROBOT here
    # and that launch means something else by it -- see the include below on why
    # sharing the name broke the whole bringup rather than merely confusing it.
    #
    # Defaulted off `mode` the way that launch defaults its own, so the common
    # cases (all mock, or all real) need no extra argument, while
    # `hand_node:=none` still covers a real arm with the hand unplugged. EPFL
    # ignores it: its bus node is started by this file, gated on `dry_run`.
    ld.add_action(DeclareLaunchArgument(
        "hand_node",
        default_value=PythonExpression(
            ["'driver' if '", mode, "' == 'hardware' else 'mock'"]),
        choices=["driver", "mock", "none"],
        description="Allegro only: which hand node lbr_allegro.launch.py starts. "
                    "'driver' is allegro_node_grasp on CAN, 'mock' is "
                    "allegro_mock_node.py (same topics, no bus), 'none' starts "
                    "nothing. Defaults to 'driver' when mode:=hardware, else "
                    "'mock'."))

    # -- the workbench --------------------------------------------------------
    ld.add_action(DeclareLaunchArgument(
        "port", default_value="8080",
        description="Port the viser web UI listens on."))
    # No enable_arm / enable_hand: the Robot folder's arming checkbox is a
    # per-press interlock that clears itself after every run, so there is nothing
    # here that could sensibly pre-tick it. Arming happens at the browser,
    # looking at the robot.
    ld.add_action(DeclareLaunchArgument(
        "playback_speed", default_value="0.5",
        description="Trajectory playback speed as a fraction of every channel's "
                    "own maximum (0.4 m/s and 0.8 rad/s at the wrist, and the "
                    "HAND's own per-actuator ceiling -- 0.065 m/s of tendon, or "
                    "1.0 rad/s of joint). One number: each segment takes as long "
                    "as its slowest channel needs, so this is a pure duration "
                    "multiplier."))
    ld.add_action(DeclareLaunchArgument(
        "viz", default_value="true",
        description="Whether to start the viser workbench. 'false' leaves the "
                    "executor up to be driven by `ros2 run gepetto_ros "
                    "play_client`."))

    # -- going home -----------------------------------------------------------
    ld.add_action(DeclareLaunchArgument(
        "home", default_value="true",
        description="Whether to start home_node, which serves "
                    "/gepetto/move_home. It is resident and idle until asked, "
                    "so 'false' is only for a cell that wants the command "
                    "topics provably untouched by anything but the executor."))
    ld.add_action(DeclareLaunchArgument(
        "home_on_start", default_value="false",
        description="Whether home_node goes home the moment it comes up. FALSE "
                    "here, unlike a bare `ros2 run` of that node: bringing the "
                    "cell up must not move the arm before anyone is looking at "
                    "it. Send a /gepetto/move_home goal, or call "
                    "/home_node/go_home, when you want the move."))
    ld.add_action(DeclareLaunchArgument(
        "home_timeout", default_value="30.0",
        description="Seconds a trip home may take before it gives up. A "
                    "MoveHome goal may override it per goal."))

    # -- executor tuning ------------------------------------------------------
    ld.add_action(DeclareLaunchArgument(
        "control_hz", default_value="100.0",
        description="Rate of the executor control loop."))
    ld.add_action(DeclareLaunchArgument(
        "pipeline_gain", default_value="0.07",
        description="Fraction of commanded Cartesian velocity the servo pipeline "
                    "actually realises; every twist is divided by it. Measured at "
                    "~0.07 on this cell (MoveIt Servo re-basing plus the FRI "
                    "position filter). The executor prints the implied value each "
                    "run -- set this from that."))
    ld.add_action(DeclareLaunchArgument(
        "velocity_headroom", default_value="2.0",
        description="How much faster than the path the arm command may be, so "
                    "the correction term has room to close a tracking gap. The "
                    "path is capped at servo_scale * pipeline_gain / headroom, so "
                    "this also bounds how fast a path may be timed."))
    ld.add_action(DeclareLaunchArgument(
        "terminal_hold", default_value="0.5",
        description="Seconds the final pose is held with zero feed-forward "
                    "before a run ends. A fixed duration, not a convergence "
                    "wait: it lets the servo pipeline's in-flight command land "
                    "rather than being cut off mid-lag."))

    # -- the EPFL hand's own bus ---------------------------------------------
    ld.add_action(DeclareLaunchArgument(
        "dry_run", default_value="false",
        description="EPFL only: finger_servo_node opens no serial port and moves "
                    "no motors, but keeps the command and feedback topics real. "
                    "The way to exercise playback with no hand attached."))
    ld.add_action(DeclareLaunchArgument(
        "measured_state_hz", default_value="5.0",
        description="EPFL only: how often the motors are read back onto "
                    "~/measured_state -- what 'Get robot state' reads and what "
                    "playback seeds from."))
    ld.add_action(DeclareLaunchArgument(
        "moving_speed", default_value="0",
        description="EPFL only: Dynamixel moving speed 1-1023; 0 keeps the "
                    "HandConfig default."))
    ld.add_action(DeclareLaunchArgument(
        "torque_limit", default_value="0",
        description="EPFL only: Dynamixel torque limit; 0 keeps the HandConfig "
                    "default (300)."))

    # -- the conda env --------------------------------------------------------
    ld.add_action(DeclareLaunchArgument(
        "conda_prefix",
        default_value=[EnvironmentVariable("HOME"), "/miniconda3/envs/gepetto_py10"],
        description="Conda env the gepetto_ros nodes run against. It must provide "
                    "gepetto_solvers (and epfl_hand_core, for the EPFL hand's "
                    "measured calibration). NOT crest_py10 -- that env is "
                    "deprecated and carries the pre-refactor packages."))
    conda_prefix = LaunchConfiguration("conda_prefix")

    # Per node, never launch-wide -- see this file's docstring.
    #
    # GEPETTO_CONDA_PREFIX is not redundant with PYTHONPATH, and dropping it
    # breaks an editable `gepetto_solvers` install: PYTHONPATH puts the env's
    # site-packages on sys.path but does not make it a SITE directory, so its
    # `.pth` files never run and the editable install's Python half is never
    # found. `gepetto_ros/__init__.py` reads this and calls `site.addsitedir`,
    # which does. See that module for the whole story.
    gepetto_env = {
        "GEPETTO_CONDA_PREFIX": conda_prefix,
        "PYTHONPATH": [
            conda_prefix, "/lib/python3.10/site-packages:",
            EnvironmentVariable("PYTHONPATH", default_value=""),
        ],
        "LD_LIBRARY_PATH": [
            conda_prefix, "/lib:",
            EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
        ],
    }

    lbr_bringup = FindPackageShare("lbr_bringup")

    # -- the arm half, from lbr_fri_ros2_stack --------------------------------
    # Included with the hand-side nodes OFF: `epfl_nodes:=none` starts no EPFL
    # node at all (this launch starts finger_servo_node itself, below), while the
    # Allegro's own driver IS its hand node and is left to that launch.
    #
    # EVERY ARGUMENT THE INCLUDED LAUNCH ALSO DECLARES IS PASSED EXPLICITLY, and
    # that is not tidiness. A launch argument is GLOBAL: one an include does not
    # receive falls through from this scope, so any name we happen to share is
    # silently handed our value. `lbr_allegro.launch.py` declares its own `hand`
    # (`driver` | `mock` | `none`, which hand NODE to start) and ours names a
    # ROBOT, so `hand:=allegro` reached its `choices` check and the whole launch
    # died on "provided value allegro is not valid". `moving_speed`,
    # `torque_limit` and `conda_prefix` collide the same way with
    # `lbr_epfl.launch.py` -- harmlessly today, because `epfl_nodes:=none` means
    # it starts nothing that reads them, but silently and only for that reason.
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([lbr_bringup, "launch", "lbr_epfl.launch.py"])]),
        launch_arguments={
            "mode": mode, "servo": "true", "rviz": rviz,
            "epfl_nodes": "none",
            "moving_speed": LaunchConfiguration("moving_speed"),
            "torque_limit": LaunchConfiguration("torque_limit"),
            "conda_prefix": conda_prefix,
        }.items(),
        condition=IfCondition(PythonExpression([is_epfl, " and '", arm, "' == 'true'"])),
    ))
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([lbr_bringup, "launch", "lbr_allegro.launch.py"])]),
        launch_arguments={
            "mode": mode, "servo": "true", "rviz": rviz,
            # OUR `hand` is the robot; THEIRS is which hand node to start. The
            # translation is `hand_node`, declared above with their own vocabulary
            # and their own default rule.
            "hand": LaunchConfiguration("hand_node"),
        }.items(),
        condition=IfCondition(PythonExpression([is_allegro, " and '", arm, "' == 'true'"])),
    ))

    # -- the EPFL hand's bus node --------------------------------------------
    # Started HERE rather than left to lbr_epfl.launch.py, because it needs the
    # conda env and that launch deliberately keeps conda away from the MoveIt
    # nodes it also starts. The Allegro has no equivalent: its driver is a C++
    # node with no python dependency of ours, so it stays in lbr_allegro's hands.
    ld.add_action(Node(
        package="epfl_hand_hardware", executable="finger_servo_node",
        name="finger_servo_node", output="screen",
        additional_env=gepetto_env,
        parameters=[{
            "measured_state_hz": ParameterValue(
                LaunchConfiguration("measured_state_hz"), value_type=float),
            "dry_run": ParameterValue(
                LaunchConfiguration("dry_run"), value_type=bool),
            "moving_speed": ParameterValue(
                LaunchConfiguration("moving_speed"), value_type=int),
            "torque_limit": ParameterValue(
                LaunchConfiguration("torque_limit"), value_type=int),
        }],
        condition=IfCondition(is_epfl),
    ))

    # -- the control loop -----------------------------------------------------
    ld.add_action(Node(
        package="gepetto_ros", executable="executor_node",
        name="gepetto_executor", output="screen",
        additional_env=gepetto_env,
        parameters=[{
            "hand": ParameterValue(hand, value_type=str),
            "control_hz": ParameterValue(
                LaunchConfiguration("control_hz"), value_type=float),
            "velocity_headroom": ParameterValue(
                LaunchConfiguration("velocity_headroom"), value_type=float),
            "pipeline_gain": ParameterValue(
                LaunchConfiguration("pipeline_gain"), value_type=float),
            "terminal_hold": ParameterValue(
                LaunchConfiguration("terminal_hold"), value_type=float),
        }],
    ))

    # -- going home -----------------------------------------------------------
    # Resident, and started AFTER the executor for no reason but reading order:
    # the two never contend at start-up because this one does not move until it
    # is asked (`home_on_start` is false above), and when it is asked they are
    # kept apart by the cell-wide e-stop and by an operator, not by launch order.
    ld.add_action(Node(
        package="gepetto_ros", executable="home_node",
        name="home_node", output="screen",
        additional_env=gepetto_env,
        parameters=[{
            "hand": ParameterValue(hand, value_type=str),
            "control_hz": ParameterValue(
                LaunchConfiguration("control_hz"), value_type=float),
            "on_start": ParameterValue(
                LaunchConfiguration("home_on_start"), value_type=bool),
            "timeout": ParameterValue(
                LaunchConfiguration("home_timeout"), value_type=float),
        }],
        condition=IfCondition(LaunchConfiguration("home")),
    ))

    # -- the workbench --------------------------------------------------------
    ld.add_action(Node(
        package="gepetto_ros", executable="viz_node",
        name="gepetto_viz", output="screen",
        additional_env=gepetto_env,
        parameters=[{
            "hand": ParameterValue(hand, value_type=str),
            "port": ParameterValue(LaunchConfiguration("port"), value_type=int),
            "playback_speed": ParameterValue(
                LaunchConfiguration("playback_speed"), value_type=float),
        }],
        condition=IfCondition(LaunchConfiguration("viz")),
    ))

    return ld
