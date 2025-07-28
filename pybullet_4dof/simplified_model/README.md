# Simplified 4-DOF Planar Robot Arm Model

This is a simplified model of a planar robot arm with 3 moving joints (DOF) and one fixed vertical link, implemented using matplotlib for 2D visualization. The model is based on the URDF specification and provides basic forward kinematics and visualization capabilities. The robot operates in the X-Z plane, with all joints rotating around the Y-axis.

## Features

- Forward kinematics calculation for planar motion
- Inverse kinematics for target position control
- Interactive GUI for position control
- Circular trajectory generation and control
- Speed control for trajectory following
- 2D visualization using matplotlib
- Simulation of robot motion in X-Z plane
- Workspace demonstration

## Robot Specifications

- 1 fixed vertical link (Link 1)
- 3 revolute joints (pitch joints around Y-axis)
  - Joint 2: Between Link 1 and Link 2
  - Joint 3: Between Link 2 and Link 3
  - Joint 4: Between Link 3 and Link 4
- Link lengths:
  - Link 1: 0.28m (fixed vertical)
  - Link 2: 0.15m
  - Link 3: 0.15m
  - Link 4: 0.10m
- Joint limits: ±3.14 radians (±180 degrees) for all moving joints
- Planar motion: All movement confined to X-Z plane

## Files

- `robot_motion.py`: Core robot model with forward kinematics, inverse kinematics, and visualization
- `circular_trajectory.py`: GUI for generating and controlling circular end-effector trajectories

## Usage

1. Make sure you have the required dependencies:
   ```bash
   pip install numpy matplotlib scipy
   ```

2. Run the robot motion control:
   ```bash
   python robot_motion.py
   ```
   Choose from the following modes:
   1. Point-to-Point Control: Move robot between specified positions
   2. Circular Trajectory: Generate circular end-effector paths
   3. Wave Motion: Demonstrate a waving motion
   4. Workspace Demonstration: Show robot's reachable workspace

3. Run the circular trajectory controller:
   ```bash
   python circular_trajectory.py
   ```
   Features:
   - Set circle center (X,Z coordinates)
   - Adjust circle radius
   - Control motion speed
   - Start/stop trajectory execution
   - Real-time visualization

## Interactive Control Features

### Point-to-Point Control
- Enter X,Z coordinates for start and end positions
- Visualize the robot's current configuration
- Move the robot smoothly between positions
- Get feedback if positions are unreachable
- See the motion planning in real-time

### Circular Trajectory Control
- Define circle parameters (center and radius)
- Adjust motion speed during execution
- Real-time trajectory visualization
- Dynamic target point display
- Workspace boundary checking

## Visualization

- Blue lines: Robot links
- Black circle: Fixed base joint (Joint 1)
- Red circles: Moving joints (Joints 2, 3, and 4)
- Green circle: End-effector position
- Red dashed circle: Target trajectory (in circular mode)
- Red X: Current target point (in circular mode)
- Grid: Helps visualize the scale and position
- X-axis: Horizontal motion
- Z-axis: Vertical motion (height) 