# Robot Simulation and Analysis Project

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dependencies](#dependencies)
4. [Robot Description](#robot-description)
   - [Visual Components](#visual-components)
   - [Detailed Robot Structure](#detailed-robot-structure)
   - [URDF Structure](#urdf-structure)
5. [URDF Generation](#urdf-generation)
   - [XACRO Files](#xacro-files)
   - [Generating URDF](#generating-urdf)
6. [Main Components](#main-components)
   - [Robot Simulation](#1-robot-simulation-scriptstest_robotpy)
   - [Data Visualization](#2-data-visualization-scriptsplot_robot_datapy)
7. [Control Strategy Details](#control-strategy-details)
8. [Inverse Kinematics Module](#inverse-kinematics-module)
9. [Usage Guide](#usage-guide)
10. [Customization Options](#customization-options)
11. [Common Issues and Solutions](#common-issues-and-solutions)
12. [Future Improvements](#future-improvements)
13. [Contributing](#contributing)
14. [License](#license)

## Overview

This project implements a 3-DOF robot simulation with real-time visualization, joint control, and data analysis capabilities using PyBullet. The project focuses on simulating robot dynamics, tracking end-effector positions, and analyzing joint torques.

## Project Structure

```
.
├── scripts/
│   ├── test_robot.py         # Main simulation script
│   ├── plot_robot_data.py    # Data visualization script
│   ├── inverse_kinematics.py # Inverse kinematics module
│   └── test_inverse_kinematics.py # IK testing script
├── urdf/
│   ├── robot.urdf           # Compiled URDF file
│   ├── robot.urdf.xacro    # Main robot description
│   └── robot_materials.xacro # Material definitions
├── data/
│   ├── robot_data_*.csv    # Generated data files
│   └── robot_data_*.png    # Generated plot files
└── README.md
```

## Dependencies

- Python 3.x
- PyBullet (`pip install pybullet`)
- NumPy (`pip install numpy`)
- Pandas (`pip install pandas`)
- Matplotlib (`pip install matplotlib`)
- ROS XACRO (`sudo apt install ros-*-xacro`) - For URDF generation

## Robot Description

The robot is a 3-DOF manipulator with the following features:
- Fixed base
- 4 revolute joints with full 360° rotation capability
- Gripper end-effector
- Visual materials for better appearance
- Properly defined inertial properties

### Visual Components

#### Base Structure
![Robot Base](images/base.png)
*The base structure of the robot*

#### Links and Joints Configuration
![Links and Joints](images/links_and_joints.png)
*Detailed view of the robot's links and joints arrangement*

#### Collision Model
![Collision Model](images/collision.png)
*The collision model used for physics simulation*

#### Complete Robot Assembly
![Complete Robot](images/complete_robot.png)
*The fully assembled 3-DOF robot*

### Detailed Robot Structure

#### Links Specification

| Link Name | Geometry | Dimensions | Mass |
|-----------|----------|------------|------|
| Base Link | Box | 0.2 x 0.2 x 0.02m | 2.0 kg |
| Link 1 | Cylinder | r: 0.024m, l: 0.25m | 0.8 kg |
| Link 2 | Cylinder | r: 0.020m, l: 0.15m | 0.6 kg |
| Link 3 | Cylinder | r: 0.020m, l: 0.15m | 0.4 kg |
| Link 4 | Cylinder | r: 0.015m, l: 0.10m | 0.3 kg |
| Gripper | Box | Base: 0.04 x 0.03 x 0.02m<br>Fingers: 2x (0.01 x 0.015 x 0.05m) | - |

#### Joints Configuration

| Joint Name | Type | Connection | Additional Info |
|------------|------|------------|-----------------|
| world_to_base | Fixed | world → base_link | Position: Origin (0, 0, 0) |
| base_to_link_1 | Fixed | base_link → link_1 | Position: z = 0.02m |
| link_1_to_link_2 | Revolute | link_1 → link_2 | Axis: Y, Limits: ±180° |
| link_2_to_link_3 | Revolute | link_2 → link_3 | Axis: Y, Limits: ±180° |
| link_3_to_link_4 | Revolute | link_3 → link_4 | Axis: Y, Limits: ±180° |
| link_4_to_gripper | Revolute | link_4 → gripper | Axis: Y, Limits: ±180° |

### URDF Structure
- Base link with fixed joint to world
- 4 links connected by revolute joints
- Joint limits: ±3.14 radians (±180 degrees)
- Properly defined inertial properties for dynamics simulation

## URDF Generation

### XACRO Files
The robot's URDF is generated from XACRO files for better maintainability and modularity:

1. `robot_materials.xacro`: Contains material definitions
   - Metallic silver for base and primary structure
   - Deep orange for dynamic parts
   - Sleek dark gray for secondary parts
   - Electric blue for joints
   - Metallic white for highlights

2. `robot.urdf.xacro`: Main robot description
   - Includes material definitions
   - Defines robot structure
   - Specifies joint properties and limits
   - Contains inertial properties

### Generating URDF

To generate the URDF file from XACRO files, use the following command:

```bash
cd urdf/
xacro robot.urdf.xacro > robot.urdf
```

This command:
1. Processes the XACRO files
2. Resolves all macros and includes
3. Generates a complete URDF file
4. Saves the output as `robot.urdf`

## Main Components

### 1. Robot Simulation (`scripts/test_robot.py`)

#### Key Functions:

##### `get_end_effector_state(robot, gripper_link_index=-1)`
- **Purpose**: Gets end-effector position and orientation
- **Parameters**:
  - `robot`: PyBullet robot ID
  - `gripper_link_index`: Index of end-effector link (default: last link)
- **Returns**: Dictionary with:
  - `position`: [x, y, z] coordinates
  - `euler_angles`: [roll, pitch, yaw] in radians

##### `get_joint_forces(robot, joint_indices)`
- **Purpose**: Reads joint torque sensors
- **Parameters**:
  - `robot`: PyBullet robot ID
  - `joint_indices`: List of joint indices to monitor
- **Returns**: Dictionary mapping joint names to torque values (Nm)

### Data Collection and CSV Format

#### CSV File Structure
The data is saved in CSV format with the following columns and their sources:

1. **Time and Position Data**:
   | Column | Source | Description |
   |--------|--------|-------------|
   | timestamp | `time.time()` | Unix timestamp when data point was collected |
   | target_x | `target_x` from trajectory | Target X position from generated trajectory |
   | target_z | `target_z` from trajectory | Target Z position from generated trajectory |
   | actual_x | `actual_pos[0]` from `p.getLinkState()` | Actual X position of end effector |
   | actual_z | `actual_pos[2]` from `p.getLinkState()` | Actual Z position of end effector |

2. **Joint States** (from `p.getJointState(robot_id, joint_index)`):
   | Column | Source | Description |
   |--------|--------|-------------|
   | joint1_angle | `state[0]` | Position of joint 1 (rad) |
   | joint2_angle | `state[0]` | Position of joint 2 (rad) |
   | joint3_angle | `state[0]` | Position of joint 3 (rad) |
   | joint1_velocity | `state[1]` | Velocity of joint 1 (rad/s) |
   | joint2_velocity | `state[1]` | Velocity of joint 2 (rad/s) |
   | joint3_velocity | `state[1]` | Velocity of joint 3 (rad/s) |
   | joint1_acceleration | Calculated | Filtered acceleration of joint 1 (rad/s²) |
   | joint2_acceleration | Calculated | Filtered acceleration of joint 2 (rad/s²) |
   | joint3_acceleration | Calculated | Filtered acceleration of joint 3 (rad/s²) |
   | joint1_torque | `state[3]` | Applied torque on joint 1 (N⋅m) |
   | joint2_torque | `state[3]` | Applied torque on joint 2 (N⋅m) |
   | joint3_torque | `state[3]` | Applied torque on joint 3 (N⋅m) |

3. **Control Parameters**:
   | Column | Source | Description |
   |--------|--------|-------------|
   | kp | `visualizer.get_gains()` | Position gain from slider |
   | kd | `visualizer.get_gains()` | Velocity gain from slider |
   | radius | `visualizer.get_trajectory_params()` | Trajectory radius (m) |
   | center_x | `visualizer.get_trajectory_params()` | X coordinate of trajectory center (m) |
   | center_z | `visualizer.get_trajectory_params()` | Z coordinate of trajectory center (m) |

#### Data Collection Details

1. **Joint State Data**:
   ```python
   joint_states = p.getJointState(robot_id, joint_index)
   # Returns tuple containing:
   # [0] -> joint angle (rad)
   # [1] -> joint velocity (rad/s)
   # [2] -> joint reaction forces (ignored)
   # [3] -> applied motor torque (N⋅m)
   ```

2. **Acceleration Calculation**:
   ```python
   # Moving average filter on velocities (window size = 5)
   smoothed_vel_current = np.mean(velocity_history[-2:])
   smoothed_vel_prev = np.mean(velocity_history[:-1])
   
   # Calculate acceleration with low-pass filter
   acceleration = (smoothed_vel_current - smoothed_vel_prev) / dt
   filtered_acc = 0.2 * acceleration + 0.8 * prev_acceleration
   
   # Clip to prevent extreme values
   filtered_acc = np.clip(filtered_acc, -5.0, 5.0)  # ±5 rad/s²
   ```

3. **Torque Behavior**:
   - Negative torques indicate counterclockwise effort in joint frame
   - Primary component counteracts gravity (always negative)
   - Motion control component is smaller than gravity compensation
   - Joint 2 (shoulder) shows largest negative torque due to load
   - Joint 3 (elbow) shows moderate negative torque
   - Joint 1 (base) shows smallest torque (horizontal plane)

4. **Filename Format**:
   ```
   semi_circle_trajectory_kp{kp:.2f}_kd{kd:.1f}_r{radius:.2f}_x{center_x:.2f}_z{center_z:.2f}_semicircle{number}_{timestamp}.csv
   ```
   Example: `semi_circle_trajectory_kp0.10_kd0.4_r0.10_x0.25_z0.10_semicircle3_20250724_131550.csv`

#### Data Collection Frequency
- Simulation runs at 240Hz (dt = 1/240 second)
- Data points collected every simulation step
- Visualization updates every 5 steps
- Tkinter events processed every 10 steps

## Control Strategy Details

### Control System Architecture

#### Torque Behavior Analysis
The robot exhibits predominantly negative torques during both clockwise (forward) and counterclockwise (return) motions due to several factors:

1. **Gravitational Effect**:
   - The robot operates in a vertical plane under gravity (9.81 m/s²)
   - Joints must continuously generate torque to counteract gravitational pull
   - This creates a persistent negative torque bias regardless of motion direction

2. **Joint-wise Torque Distribution**:
   | Joint | Torque Characteristics | Reason |
   |-------|----------------------|---------|
   | Joint 1 (Base) | Smallest magnitude | Rotates in horizontal plane, minimal gravity effect |
   | Joint 2 (Shoulder) | Largest negative | Supports weight of links 3, 4, and end-effector |
   | Joint 3 (Elbow) | Moderate negative | Supports weight of link 4 and end-effector |

3. **Direction Convention**:
   - Negative torque indicates counterclockwise effort in joint frame
   - Primary torque component counteracts gravitational pull
   - Motion-related torque components are smaller than gravity compensation

4. **Motion Phase Analysis**:
   ```
   Forward Motion (Clockwise):
   - Negative torque to support links against gravity
   - Additional negative torque for motion control
   
   Return Motion (Counterclockwise):
   - Negative torque still needed for gravity compensation
   - Gravity compensation dominates over motion control torques
   ```

5. **Torque Components**:
   ```
   Total Torque = Gravity Compensation + Motion Control
   where:
   - Gravity Compensation: Always negative (counterclockwise in joint frame)
   - Motion Control: Can be positive or negative but smaller magnitude
   ```

This behavior is normal and expected for a vertically-operating robotic arm, where gravity compensation is the dominant factor in torque generation.

#### PD Control Implementation

The project uses PyBullet's built-in Position Control mode through `p.setJointMotorControl2()`. This is a PD (Proportional-Derivative) controller implemented internally by PyBullet.

#### Controller Type: PD Position Control
```python
p.setJointMotorControl2(
    robot,                    # Robot ID
    joint_id,                 # Joint to control
    p.POSITION_CONTROL,       # Control mode
    targetPosition=position,  # Desired position
    force=100.0,             # Maximum force/torque
    maxVelocity=1.0          # Velocity limit
)
```

#### Control Parameters:
1. **Position Target**
   - Generated using sinusoidal functions
   - `position = 0.5 * sin(t + i * π/4)`
   - Amplitude: ±0.5 radians (≈±28.6 degrees)
   - Phase offset: `i * π/4` for each joint
   - Smooth, continuous motion

2. **Force Limit (100.0 N)**
   - Maximum force/torque the controller can apply
   - Prevents excessive motor effort
   - Safety feature to avoid damaging the robot
   - Higher than joint torque limits (50 Nm) to ensure tracking

3. **Velocity Limit (1.0 rad/s)**
   - Maximum joint velocity
   - Ensures smooth, controlled motion
   - Matches URDF specifications
   - Safety feature to prevent rapid movements

4. **Control Frequency**
   - Update rate: 100 Hz (0.01s timestep)
   - Balances between:
     - Control precision
     - Computational load
     - Real-time performance

#### How the Controller Works:

1. **Position Error Calculation**
   ```
   error = targetPosition - currentPosition
   ```

2. **PD Control Law (internal to PyBullet)**
   ```
   torque = Kp * position_error + Kd * velocity_error
   torque = clamp(torque, -force_limit, +force_limit)
   ```
   Where:
   - `Kp`: Position gain (internally tuned)
   - `Kd`: Velocity gain (internally tuned)

3. **Velocity Limiting**
   ```
   velocity = clamp(computed_velocity, -maxVelocity, +maxVelocity)
   ```

4. **Joint Limits**
   - Enforced by URDF specifications
   - Position: ±1.57 radians (±90 degrees)
   - Maximum torque: 50 Nm
   - Maximum velocity: 1.0 rad/s

#### Trajectory Generation:

```python
positions = [
    0.5 * np.sin(t + i * np.pi/4)  # Different phase for each joint
    for i in range(len(controllable_joints))
]
```

- Each joint follows a sinusoidal trajectory
- Phase differences create coordinated motion
- Continuous, differentiable path
- Easy to modify for different patterns

#### Advantages of This Control Strategy:

1. **Simplicity**
   - Easy to implement and understand
   - Built-in PyBullet implementation
   - Minimal tuning required

2. **Stability**
   - Natural motion damping
   - Smooth trajectory following
   - Built-in safety limits

3. **Performance**
   - Computationally efficient
   - Real-time capable
   - Suitable for testing and prototyping

4. **Flexibility**
   - Easy to modify trajectories
   - Adjustable limits and parameters
   - Compatible with various robot configurations

#### Limitations:

1. **No Direct Force Control**
   - Cannot directly control interaction forces
   - Limited for contact tasks

2. **Fixed Gains**
   - Internal PD gains not directly accessible
   - Cannot optimize for specific tasks

3. **Simple Trajectory**
   - Basic sinusoidal motion
   - Not task-specific
   - No obstacle avoidance

#### Alternative Control Strategies:

1. **Velocity Control**
   ```python
   p.setJointMotorControl2(
       robot, joint_id,
       p.VELOCITY_CONTROL,
       targetVelocity=velocity
   )
   ```

2. **Torque Control**
   ```python
   p.setJointMotorControl2(
       robot, joint_id,
       p.TORQUE_CONTROL,
       force=torque
   )
   ```

3. **Future Improvements**
   - Implement inverse kinematics
   - Add trajectory planning
   - Include obstacle avoidance
   - Develop task-specific controllers

#### Data Logging:
- CSV format with timestamps
- End-effector position and orientation
- Joint torques
- Real-time data flushing for safety

## Inverse Kinematics Module

The project includes a comprehensive inverse kinematics (IK) solution for the 3-DOF robotic arm. The IK module provides functions to calculate joint angles for desired end-effector positions and analyze the robot's workspace.

### Mathematical Formulation

#### Inverse Kinematics

The inverse kinematics solution for the 3-DOF planar robot arm in the X-Z plane follows these steps:

1. **End-Effector Orientation Angle (φ)**:
   ```math
   φ = arctan2(z, x)
   ```

2. **Wrist Position Calculation**:
   ```math
   x_2 = x - l_3 \cos(φ)
   z_2 = z - l_3 \sin(φ)
   ```

3. **Second Joint Angle (θ₂)**:
   Using the cosine law:
   ```math
   D = \frac{x_2^2 + z_2^2 - l_1^2 - l_2^2}{2l_1l_2}
   θ₂ = arccos(D)
   ```
   Where D must satisfy |D| ≤ 1 for the target to be reachable.

4. **First Joint Angle (θ₁)**:
   ```math
   k_1 = l_1 + l_2\cos(θ₂)
   k_2 = l_2\sin(θ₂)
   θ₁ = arctan2(z_2, x_2) - arctan2(k_2, k_1)
   ```

5. **Third Joint Angle (θ₃)**:
   ```math
   θ₃ = φ - (θ₁ + θ₂)
   ```

#### Workspace Analysis

The workspace analysis function generates reachable points by:
1. Sampling joint angles in their valid ranges:
   ```math
   θ₁, θ₂ ∈ [-π, π]
   θ₃ ∈ [-π, π]
   ```

2. Forward kinematics for each configuration:
   ```math
   x_1 = l_1\cos(θ₁)
   z_1 = l_1\sin(θ₁)
   x_2 = x_1 + l_2\cos(θ₁ + θ₂)
   z_2 = z_1 + l_2\sin(θ₁ + θ₂)
   x_3 = x_2 + l_3\cos(θ₁ + θ₂ + θ₃)
   z_3 = z_2 + l_3\sin(θ₁ + θ₂ + θ₃)
   ```

Where:
- `l₁, l₂, l₃`: Link lengths
- `(x, z)`: Target end-effector position
- `θ₁, θ₂, θ₃`: Joint angles
- `(x₁, z₁), (x₂, z₂), (x₃, z₃)`: Joint positions

### Implementation Details

1. **Geometric Approach**
   - Uses analytical solution for 3-DOF planar arm
   - Employs cosine law for angle calculations
   - Maintains end-effector vertical orientation

2. **Joint Limits**
   - Enforces ±90 degree limits per joint
   - Validates solutions before returning
   - Handles edge cases and singularities

3. **Workspace Analysis**
   - Calculates maximum reach: l1 + l2 + l3
   - Determines minimum reach based on joint limits
   - Maps reachable workspace boundaries

4. **Error Handling**
   - Validates input coordinates
   - Checks joint limit violations
   - Provides meaningful error messages

### 2. Data Visualization (`scripts/plot_robot_data.py`)

#### Plot Types:
1. **End Effector Trajectory (3D)**
   - Complete path visualization
   - Start (green) and end (red) points marked
   - 3D perspective with grid

2. **End Effector Position vs Time**
   - X, Y, Z coordinates over time
   - Shows motion timing
   - Grid for easy reading

3. **End Effector Orientation**
   - Roll, pitch, yaw angles
   - Converted to degrees for readability
   - Shows orientation changes

4. **Joint Torques**
   - Individual torque plots for each joint
   - Shows load distribution
   - Helps identify peak loads

## Why This Approach?

1. **PyBullet Over ROS2**
   - Lighter weight solution
   - Easier setup and dependencies
   - Good for prototyping and testing
   - Built-in physics engine

2. **Position Control**
   - Simple and stable control method
   - Good for trajectory following
   - Suitable for most basic applications

3. **Data Collection Strategy**
   - Real-time logging prevents data loss
   - Comprehensive data collection for analysis
   - CSV format for easy post-processing

4. **Visualization Approach**
   - Multiple plot types for complete analysis
   - Real-time and post-process visualization
   - Clear representation of robot state

## Usage Guide

1. **Setup Environment**
   ```bash
   pip install pybullet numpy pandas matplotlib
   ```

2. **Run Simulation**
   ```bash
   python3 scripts/test_robot.py
   ```
   - Watch real-time visualization
   - Press Ctrl+C to stop and save data

3. **Analyze Data**
   ```bash
   python3 scripts/plot_robot_data.py
   ```
   - Views the most recent data file
   - Creates PNG plot file in data directory

## Customization Options

1. **Robot Parameters**
   - Modify URDF files for different robot configurations
   - Adjust joint limits and properties

2. **Control Parameters**
   - Change trajectory pattern in `scripts/test_robot.py`
   - Adjust control gains and limits

3. **Visualization**
   - Modify plot styles in `scripts/plot_robot_data.py`
   - Add new analysis plots

## Common Issues and Solutions

1. **PyBullet Installation**
   - Use pip for standard install
   - Check Python version compatibility

2. **URDF Loading**
   - Ensure correct file paths
   - Check URDF syntax
   - Verify joint definitions

3. **Data Collection**
   - Monitor CSV file size
   - Adjust sampling rate if needed
   - Check disk space

## Future Improvements

1. **Additional Features**
   - Inverse kinematics implementation
   - Path planning capabilities
   - Collision detection

2. **Analysis Tools**
   - Real-time plotting
   - Advanced analytics
   - Performance metrics

3. **User Interface**
   - GUI for parameter adjustment
   - Interactive control interface
   - Real-time data visualization

## Contributing

Feel free to contribute by:
1. Forking the repository
2. Creating feature branches
3. Submitting pull requests

## License

This project is open-source and available under the MIT License. 