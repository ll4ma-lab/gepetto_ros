# gepetto_ros

The bridge between the [`gepetto_solvers`](../gepetto_solvers) planner and a KUKA
LBR med14 wearing **either** hand:

| `hand:=` | arm description | hand | commanded in | driver |
|---|---|---|---|---|
| `epfl` | `med14_epfl` | EPFL five-digit tendon hand | metres of tendon displacement, 1 per digit | `finger_servo_node` on the Dynamixel bus |
| `allegro` | `med14_allegro` | Allegro Hand V5 (right, type B) | radians of joint position, 4 per digit | `allegro_node_grasp` on CAN |

One argument picks the robot, and it defaults to `allegro`. The nodes, the control
loop, the action, the e-stop and the visualizer are the same code either way.

```
gepetto_ros/
├── gepetto_ros_interfaces/   ament_cmake   — PlayPlan.action, MoveHome.action
├── gepetto_ros/              ament_python  — the bridge library and its nodes
└── gepetto_ros_launch/       ament_python  — gepetto_bringup.launch.py
```

It sits beside `gepetto_solvers`, `epfl_hand_ros` and `lbr_fri_ros2_stack` in a
colcon workspace's `src/`. The dependency direction is one-way and worth stating:
`epfl_hand_ros → gepetto_ros → gepetto_solvers`. Nothing here knows `epfl_hand_ros`
exists, and nothing in `gepetto_solvers` imports rclpy.

---

## The one idea

> **A plan carries, per waypoint, the wrist pose plus one vector of
> `len(hand.actuation.drive_indices)` numbers per digit.**

The EPFL hand drives one actuator per digit (its flexor); the Allegro drives four
(its joints). Everything downstream of the solver is that same problem at two
widths, so `gepetto_solvers.core.robot_plan` carries a `command_kind` saying which
units the vectors are in, and *nothing* between the solver and the hand driver
branches on it. Two further coincidences make the arm half identical as well:

* **the SRDF chain tip IS the solver's wrist frame on both robots** —
  `lbr_epfl_wrist_link` sits exactly at `MountConfig.flange_from_wrist_*`, and
  `palm_link` is the frame `AllegroHand.mount_pose()` was measured against with
  `tf2_echo lbr_link_ee palm_link`. So a solved wrist pose is commandable through
  MoveIt Servo with no mount conversion, either way;
* **`lbr_workspace_table_link` exists in both descriptions** (`med14_allegro.xacro`
  includes `med14_epfl`'s table xacro), so the viser↔robot registration is one
  convention rather than two.

What is left over — the wrist frame's name, the driver class, the description to
bring up — is `robot_profile.py`, and that file is the whole of the difference.

---

## Layout

### `gepetto_ros_interfaces`

`action/PlayPlan.action` — a `robot_plan.SolvePlan` flattened for the wire, field
for field as `robot_plan.flatten_plan` produces it. The goal carries
`digit_names`, `dof_per_digit`, `command_kind` and a row-major `digit_cmd` block
over `(waypoint, digit, actuator)`.

`action/MoveHome.action` — park the cell. The goal carries no target: home is a
property of the cell (`RobotProfile.arm_home` and the hand's own rest posture),
not something a caller names, so the goal only says which channels to *skip* and
how long to allow. Both channels are skipped by negation, so `"{}"` homes
everything.

### `gepetto_ros`

| module | what it is |
|---|---|
| `robot_profile.py` | **the per-robot table.** A third robot is an entry here plus a driver. |
| `topics.py` | `/gepetto/estop` (latched, transient-local), `/gepetto/play_plan` and `/gepetto/move_home`. Cell-wide, never per hand. |
| `frames.py` | the viser↔`lbr_link_0` registration, and pose↔matrix conversion |
| `arm_servo.py` | `ArmTwistServo` / `ArmJointServo` — the arm half, identical for both robots |
| `hand_drivers/` | `base.py` (the interface), `tendon.py`, `allegro.py` — **the one place the robots genuinely differ** |
| `hand_state.py` | one subscription, decoded by the hand's own driver class |
| `wire.py` | `PlayPlan` goal ↔ `SolvePlan`. Thin: the encoding lives in `gepetto_solvers`, where it is testable without ROS. |
| `trajectory_executor.py` | the control loop: a time-scaled path on a free-running clock |
| `executor_node.py` | the only process that publishes motion commands during playback |
| `viz_node.py` | the viser workbench, hosted in a node and handed a bridge |
| `robot_bridge.py` | the duck-typed bridge `HandVizApp` talks to |
| `home_node.py` | park the arm and rest the hand, through the servo topics. **Resident** — it comes up with the cell and serves `/gepetto/move_home`. |
| `play_client.py`, `fake_cell.py` | bring-up tools, as console scripts |

There is deliberately **no `test/` directory**. The logic worth testing — the plan,
the units, the flat encoding, the Allegro's positional joint order — is pure numpy
and is tested in `gepetto_solvers/tests/core/test_plan_wire.py`, where it needs no
built workspace. What is left here is ROS plumbing, exercised by `fake_cell` +
`play_client` against a live graph.

---

## Building

`gepetto_solvers` is a **pip package in a conda env**, not an ament package, so it
is deliberately not a `<depend>` — colcon would look for it in the workspace and
fail. Build with system python, conda fully deactivated:

```bash
conda deactivate          # possibly twice; `echo $CONDA_PREFIX` must be empty
which python3             # MUST print /usr/bin/python3
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to gepetto_ros_launch
source install/setup.bash
```

### The conda env, and why `PYTHONPATH` alone is not enough

The nodes run under `/usr/bin/python3` (what `rclpy`, `tf2_ros` and MoveIt were
built against) while `gepetto_solvers`, `viser` and the compiled solver binding
live in **`gepetto_py10`** — same CPython 3.10 ABI, different prefix. The launch
files inject that env per node via `additional_env`, never launch-wide, so conda's
`site-packages` never reaches `move_group`, `servo_node`, `ros2_control_node` or
RViz.

Injecting `PYTHONPATH` puts the directory on `sys.path` but does **not** make it a
*site* directory — and only a site directory has its `.pth` files executed. That
distinction is invisible for an ordinary install and fatal for an **editable** one:
`gepetto_solvers` is installed editable, so `site-packages/gepetto_solvers/` holds
nothing but the `.so` while a `.pth` points at the source tree for the Python half.
With the `.pth` unread, `import gepetto_solvers` succeeds and
`gepetto_solvers.core` does not exist.

`gepetto_ros/__init__.py` fixes that with `site.addsitedir`, driven by
`$GEPETTO_CONDA_PREFIX` (which the launch files set alongside the `PYTHONPATH`).
It is a no-op when the solvers are already importable.

> **`crest_py10` is deprecated.** It carries the pre-refactor `crest_sparse` and
> `gepetto_core`, and nothing in this package uses it. Only `epfl_hand_ros`'s
> `planner` and `state_estimator` still do — see that package's README.

---

## Running

### Everything, mock, no hardware

```bash
ros2 launch gepetto_ros_launch gepetto_bringup.launch.py mode:=mock            # allegro
ros2 launch gepetto_ros_launch gepetto_bringup.launch.py hand:=epfl mode:=mock dry_run:=true
```

Brings up the arm on mock `ros2_control` + MoveIt Servo + RViz (from
`lbr_bringup`), the hand's driver in its mock/dry-run form, `executor_node` and
the viser workbench on <http://localhost:8080>.

In the browser: solve, tick **move the real robot** (which arms *one* press and
clears itself afterwards), then **Play solve on robot**. **Get robot state** reads
the wrist pose from TF and the hand's measured state back onto the controls.

### The real cell

Start the arm on the robot PC first — this launch does not, by design:

```bash
# on the robot PC
ros2 launch lbr_bringup hardware.launch.py model:=med14_epfl ctrl:=forward_position_controller
# here
ros2 launch gepetto_ros_launch gepetto_bringup.launch.py hand:=epfl mode:=hardware
```

### Without MoveIt at all

`fake_cell` publishes the TF and hand state a cell would, with a first-order lag
and a steady-state gain deficit modelling the real servo pipeline:

```bash
ros2 run gepetto_ros fake_cell   --hand allegro --tau 0.15 --gain 0.2
ros2 run gepetto_ros executor_node --ros-args -p hand:=allegro
ros2 run gepetto_ros play_client --hand allegro --speed 0.5 --clear-estop
```

`play_client` builds a plan by hand — no solver, no browser — so a failure there is
unambiguously in the ROS half. `--hand-only` / `--arm-only` split the channels for
bring-up; the workbench never does, because a coordinated trajectory played half
way is not an operating mode.

### Going home

`home_node` is started by the bringup and stays up, so this is a request to the
running cell rather than another process to launch and kill:

```bash
# the whole cell, with feedback and a cancel you can press
ros2 action send_goal -f /gepetto/move_home gepetto_ros_interfaces/action/MoveHome "{}"
# the same move, one line, no goal to fill in
ros2 service call /home_node/go_home std_srvs/srv/Trigger
# one channel only, and a deadline of your own
ros2 action send_goal /gepetto/move_home gepetto_ros_interfaces/action/MoveHome \
    "{skip_hand: true, timeout: 15.0}"
```

It does **not** move when the launch starts it (`home_on_start:=false`) — bringing
the cell up must not move the arm. A bare `ros2 run gepetto_ros home_node
--ros-args -p hand:=allegro` still homes once on start-up, because that is a
request to go home rather than a cell coming up.

A goal is rejected while the e-stop is engaged, while another home is running, and
when it would skip both channels; a running goal aborts within a tick if the stop
is pressed, and a cancel is always accepted.

The arm goes to a folded joint configuration — a **different one per robot**
(`RobotProfile.arm_home`), because the fold has to keep that cell's hand clear.
The hand goes to its own rest posture: the **zero command vector** on a tendon
hand, where the command is displacement from the open hand, and the hand's own
`default_pose()` on a joint one, which is the posture the viser workbench opens on
and restores with *Reset defaults* (zero there is merely the URDF's straight,
uncurled hand). Mutually exclusive with a playback — both publish to the same
topics.

---

## The E-STOP

`/gepetto/estop` is latched (`TRANSIENT_LOCAL`), cell-wide, and anything may
publish `true` to trip it. `executor_node` halts its publishers from inside the
subscription callback, on its own callback group, so a stop never queues behind
the loop it is interrupting. Rearming is deliberately **not** done over the topic —
it is a decision someone makes at the visualizer, looking at the robot.

A latched trip survives the process that published it. A node coming up into a
stopped cell correctly refuses to move; `play_client --clear-estop` is how a
bring-up session clears one.

---

## Gotchas

- **`ModuleNotFoundError: No module named 'gepetto_solvers.core'`** — the editable
  install's `.pth` was not processed. Set `GEPETTO_CONDA_PREFIX`, or start the
  nodes from the launch file, which does. See above.
- **`playback failed: wrist TF has been stale for 1.0 s`** — `robot_state_publisher`
  really has stopped. Note the wording: a *single* stale reading is not this. The
  loop coasts on the last good pose for `tf_stale_grace` (1.0 s) first, because
  `robot_state_publisher` republishes TF at only 20 Hz and anything that keeps the
  machine busy for a few of those periods — a solve running in `viz_node`, a Servo
  collision check going long (`run_duration: 0.69 (0.005)`) — pushes the age past
  `tf_max_age` for a moment. Those moments show up as a throttled *coasting on the
  last good pose* warning and a `wrist TF` line in the run's diagnostic block, and
  the run continues. If that line says the loop coasted for most of a run, the
  machine is the problem, not the arm: raise `tf_max_age`, or take load off it.
- **The executor rejects a goal with "it carries … but this executor is driving …"**
  — the visualizer and the executor were started with different `hand`. That check
  exists because the alternative is an arm that travels the path with a hand that
  never moves.
- **`colcon build` fails with `ModuleNotFoundError: No module named 'catkin_pkg'`**
  — a conda env is active and CMake picked up its python. `conda deactivate` until
  `which python3` is `/usr/bin/python3`, then `rm -rf build/gepetto_ros_interfaces`
  (CMake caches `PYTHON_EXECUTABLE`) and rebuild.
- **`/dev/ttyUSB0` is busy** (EPFL only) — `finger_servo_node`, `finger_slider_node`
  and `hand_node` are mutually exclusive. `fuser -v /dev/ttyUSB0`.
- **The CAN bus is not configured by any launch** (Allegro only). Bring it up
  first: `sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up`.
