# Path Planning Implementation Guide

This guide explains the implementation of path planning, trajectory tracking, and GUI control for the 3-DOF robot arm with gripper position tracking.

## Classes Overview

### 1. PathPlanner Class

The main class responsible for robot control and path planning.

#### Key Attributes
- `robot_id`: PyBullet body ID for the robot
- `joint_indices`: List of revolute joint indices [2, 3, 4, 5]
- `num_joints`: Number of actuated joints (3)
- `gripper_joint_index`: Index for gripper tracking (5)
- `trajectory_data`: Dictionary storing motion data

#### Core Methods

##### Joint Control and State
```python
def get_joint_positions(self):
    """Get positions of actuated joints (excluding gripper)"""

def set_joint_positions(self, positions, control=True):
    """Set joint positions with optional position control"""

def get_end_effector_position(self):
    """Get gripper position for tracking"""
```

##### Path Planning
```python
def linear_interpolation(self, start, end, num_points):
    """Generate linear path between configurations"""

def plan_joint_path(self, target_positions, num_points=50):
    """Plan collision-free path in joint space"""

def inverse_kinematics(self, target_position):
    """Compute IK for target gripper position"""
```

##### Data Collection and Visualization
```python
def get_joint_torques(self):
    """Get torques from actuated joints"""

def collect_data(self):
    """Record trajectory and torque data"""

def plot_data(self):
    """Plot X-Z trajectory and joint torques"""
```

### 2. RobotGUI Class

Provides graphical interface for robot control.

#### Key Attributes
- `planner`: Reference to PathPlanner instance
- `root`: Main Tkinter window
- `x_entry`, `z_entry`: Position input fields

#### Core Methods
```python
def move_robot(self):
    """Execute movement to target position"""

def plot_data(self):
    """Display trajectory plots"""

def update(self):
    """Handle GUI updates"""
```

## Implementation Details

### 1. Joint Configuration

The robot has 4 revolute joints:
- Joints 1-3 (indices 2,3,4): Actuated for motion
- Joint 4 (index 5): Gripper joint, used only for position tracking

### 2. Trajectory Tracking

Data collected during motion:
- End effector position (X,Z coordinates)
- Joint torques for actuated joints
- Timestamps for plotting

### 3. Visualization

Two synchronized plots:
- Top: End effector trajectory in X-Z plane
- Bottom: Joint torques over time

### 4. GUI Interface

Features:
- Input fields for X and Z coordinates
- "Move" button for execution
- "Plot Data" button for visualization
- Real-time updates (240Hz simulation)

## Usage Example

```python
# Initialize system
robot_id, client = spawn_robot(create_connection=True)
planner = PathPlanner(robot_id)
gui = RobotGUI(planner)

# GUI Input
# 1. Enter X,Z coordinates
# 2. Click "Move"
# 3. Click "Plot Data" to view trajectory

# Automatic Data Collection
- Position and torque data recorded during motion
- Plots update after each movement
```

## Data Structure

### Trajectory Data
```python
trajectory_data = {
    'time': [],      # Timestamps
    'x': [],         # X coordinates
    'z': [],         # Z coordinates
    'torques': []    # Joint torques
}
```

## Error Handling

1. **Input Validation**
   - Numeric input checking
   - Path planning failure detection
   - GUI closure handling

2. **Motion Safety**
   - Collision checking
   - Joint limit validation
   - Path execution monitoring

## Dependencies

- `pybullet`: Robot simulation and control
- `numpy`: Numerical computations
- `matplotlib`: Trajectory visualization
- `tkinter`: GUI implementation
- `time`: Timing and delays

## Best Practices

1. **Motion Planning**
   - Use reasonable step sizes (num_points=50)
   - Check path validity before execution
   - Monitor torques during motion

2. **GUI Usage**
   - Enter coordinates within robot workspace
   - Wait for motion completion before new commands
   - Use plot data to analyze movement quality

3. **Data Collection**
   - Clear previous data before new movements
   - Ensure consistent sampling rate
   - Monitor system resources during plotting

## Limitations

1. **Planning**
   - Linear interpolation only
   - No obstacle avoidance
   - Fixed velocity profile

2. **Control**
   - Position control only
   - No dynamic trajectory adjustment
   - Limited speed control

3. **Visualization**
   - 2D trajectory display only
   - Post-motion plotting only
   - No real-time plot updates

## Future Improvements

1. **Motion Planning**
   - Advanced algorithms (RRT, PRM)
   - Dynamic obstacle avoidance
   - Optimized trajectories

2. **Visualization**
   - Real-time plot updates
   - 3D trajectory visualization
   - Interactive waypoint setting

3. **Control**
   - Velocity control
   - Dynamic trajectory adjustment
   - Force/torque feedback control 