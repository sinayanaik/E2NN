# Path Planning

This document explains the path planning implementation for the 3-DOF robot arm. The planner provides both joint-space and task-space trajectory generation with collision checking.

## Technical Overview

### Path Planning Components

1. **Joint Space Planning**
   - Direct interpolation between joint configurations
   - Linear trajectories in joint space
   - Collision checking along the path
   - Joint limit validation

2. **Task Space Planning**
   - Inverse kinematics for end-effector positioning
   - Linear trajectories in Cartesian space
   - Multiple waypoint support
   - Automatic joint solution computation

3. **Collision Detection**
   - Self-collision checking
   - Environment obstacle detection
   - Configuration validation
   - Path safety verification

### Implementation Details

1. **Joint Control**
   ```python
   def set_joint_positions(self, positions, control=True):
       for idx, pos in zip(self.joint_indices, positions):
           p.resetJointState(self.robot_id, idx, pos)
           if control:
               p.setJointMotorControl2(self.robot_id, idx,
                                     p.POSITION_CONTROL,
                                     targetPosition=pos)
   ```
   - Direct joint state control
   - Optional position control mode
   - Synchronized joint movement

2. **Path Generation**
   ```python
   def linear_interpolation(self, start, end, num_points):
       path = []
       for i in range(num_points):
           t = i / (num_points - 1)
           point = start + t * (end - start)
           path.append(point)
       return path
   ```
   - Linear interpolation between configurations
   - Configurable resolution
   - Time-parameterized trajectories

3. **Collision Checking**
   ```python
   def check_collision(self, joint_positions):
       # Set test positions
       self.set_joint_positions(joint_positions)
       # Check for collisions
       for i in range(p.getNumJoints(self.robot_id)):
           if p.getContactPoints(self.robot_id, self.robot_id, i):
               return True
       return False
   ```
   - State validation
   - Contact point detection
   - Self-collision prevention

## Usage Examples

1. **Joint Space Movement**
   ```python
   planner = PathPlanner(robot_id)
   target_joints = np.array([0.5, -0.5, 0.5])
   path = planner.plan_joint_path(target_joints)
   planner.execute_path(path)
   ```

2. **Task Space Movement**
   ```python
   target_position = [0.2, 0, 0.4]
   target_joints = planner.inverse_kinematics(target_position)
   path = planner.plan_joint_path(target_joints)
   planner.execute_path(path)
   ```

3. **Multi-Waypoint Trajectory**
   ```python
   waypoints = [
       [0.2, 0, 0.4],
       [0.2, 0, 0.3],
       [0.1, 0, 0.3]
   ]
   for target in waypoints:
       target_joints = planner.inverse_kinematics(target)
       path = planner.plan_joint_path(target_joints)
       planner.execute_path(path)
   ```

## Mathematical Foundation

1. **Linear Interpolation**
   - Path point P(t) = P₁ + t(P₂ - P₁)
   - t ∈ [0,1] for path parameter
   - Uniform time distribution

2. **Inverse Kinematics**
   - Uses PyBullet's built-in IK solver
   - Numerical optimization approach
   - Joint limit consideration
   - Multiple solution handling

3. **Collision Detection**
   - Contact point computation
   - Distance threshold checking
   - Configuration space validation

## Limitations and Considerations

1. **Path Planning**
   - Simple linear interpolation
   - No obstacle avoidance
   - No path optimization
   - Basic collision checking

2. **Performance**
   - Fixed time step execution
   - Synchronous joint control
   - Limited speed control

3. **Future Improvements**
   - RRT/PRM implementation
   - Dynamic obstacle avoidance
   - Path optimization
   - Velocity profiling
   - Time-optimal trajectories 