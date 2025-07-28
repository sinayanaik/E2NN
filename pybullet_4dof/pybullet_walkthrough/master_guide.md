# PyBullet Robot Control System - Built-in Functions Reference

This guide documents the key PyBullet built-in functions and classes used in our robot control system.

## Table of Contents
- [Connection and Setup](#connection-and-setup)
- [Robot Loading and Configuration](#robot-loading-and-configuration)
- [Joint and Link State Management](#joint-and-link-state-management)
- [Robot Control](#robot-control)
- [Debug and Visualization](#debug-and-visualization)
- [Physics Configuration](#physics-configuration)

## Connection and Setup

### p.connect()
Connects to the physics server.
- **Parameters**:
  - `mode` (int): Connection mode
    - `p.DIRECT`: Non-graphical mode
    - `p.GUI`: Graphical mode with 3D visualization
- **Returns**: `physicsClientId` (int)
- **Usage**: `physics_client = p.connect(p.GUI)`

### p.configureDebugVisualizer()
Configures visualization options.
- **Parameters**:
  - `flag` (int): Option to configure
    - `p.COV_ENABLE_GUI`: Show/hide GUI controls
    - `p.COV_ENABLE_RENDERING`: Enable/disable rendering
  - `enable` (int): 0 or 1
- **Usage**: `p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)`

### p.setAdditionalSearchPath()
Adds search path for URDF/robot files.
- **Parameters**:
  - `path` (str): Additional search path
- **Usage**: `p.setAdditionalSearchPath(pybullet_data.getDataPath())`

## Robot Loading and Configuration

### p.loadURDF()
Loads a robot from URDF file.
- **Parameters**:
  - `fileName` (str): Path to URDF file
  - `basePosition` (list[float]): [x, y, z] starting position
  - `baseOrientation` (list[float]): [x, y, z, w] quaternion
  - `useFixedBase` (bool): If True, base won't move
  - `flags` (int): Loading options
    - `p.URDF_USE_INERTIA_FROM_FILE`
    - `p.URDF_USE_SELF_COLLISION`
- **Returns**: `bodyUniqueId` (int)
- **Usage**: 
```python
robot = p.loadURDF("robot.urdf", 
                   basePosition=[0, 0, 0],
                   useFixedBase=True)
```

### p.getBasePositionAndOrientation()
Gets robot base pose.
- **Parameters**:
  - `bodyUniqueId` (int): Robot ID
- **Returns**: Tuple with indices:
```python
BASE_POS = 0    # List[float]: Base position [x, y, z]
BASE_ORN = 1    # List[float]: Base orientation quaternion [x, y, z, w]
```
- **Usage**:
```python
pos, orn = p.getBasePositionAndOrientation(robot_id)
base_position = pos      # [x, y, z]
base_quaternion = orn    # [x, y, z, w]
```

### p.getBaseVelocity()
Gets robot base velocity.
- **Parameters**:
  - `bodyUniqueId` (int): Robot ID
- **Returns**: Tuple with indices:
```python
BASE_LINEAR_VEL = 0     # List[float]: Linear velocity [vx, vy, vz]
BASE_ANGULAR_VEL = 1    # List[float]: Angular velocity [wx, wy, wz]
```
- **Usage**:
```python
linear_vel, angular_vel = p.getBaseVelocity(robot_id)
```

## Joint and Link State Management

### p.getNumJoints()
Gets number of joints in robot.
- **Parameters**:
  - `bodyUniqueId` (int): Robot ID
- **Returns**: Number of joints (int)
- **Usage**: `num_joints = p.getNumJoints(robot)`

### p.getJointInfo()
Gets detailed information about a joint.
- **Parameters**:
  - `bodyUniqueId` (int): Robot ID
  - `jointIndex` (int): Index of joint
- **Returns**: 17-element tuple with indices:
```python
JOINT_INFO_JOINT_INDEX = 0          # Int: Index of the joint
JOINT_INFO_JOINT_NAME = 1           # Str: Name of the joint
JOINT_INFO_JOINT_TYPE = 2           # Int: Type of joint (REVOLUTE, PRISMATIC, etc.)
JOINT_INFO_Q_INDEX = 3              # Int: First position index in state array
JOINT_INFO_U_INDEX = 4              # Int: First velocity index in state array
JOINT_INFO_FLAGS = 5                # Int: Joint flags
JOINT_INFO_JOINT_DAMPING = 6        # Float: Joint damping value
JOINT_INFO_JOINT_FRICTION = 7       # Float: Joint friction value
JOINT_INFO_JOINT_LOWER_LIMIT = 8    # Float: Lower joint limit
JOINT_INFO_JOINT_UPPER_LIMIT = 9    # Float: Upper joint limit
JOINT_INFO_JOINT_MAX_FORCE = 10     # Float: Maximum force
JOINT_INFO_JOINT_MAX_VELOCITY = 11  # Float: Maximum velocity
JOINT_INFO_LINK_NAME = 12           # Str: Name of the link
JOINT_INFO_JOINT_AXIS = 13          # List[float]: Joint axis [x, y, z]
JOINT_INFO_PARENT_FRAME_POS = 14    # List[float]: Parent frame position [x, y, z]
JOINT_INFO_PARENT_FRAME_ORN = 15    # List[float]: Parent frame orientation [x, y, z, w]
JOINT_INFO_PARENT_INDEX = 16        # Int: Parent link index
```
- **Usage**:
```python
joint_info = p.getJointInfo(robot_id, joint_index)
joint_name = joint_info[JOINT_INFO_JOINT_NAME]
joint_type = joint_info[JOINT_INFO_JOINT_TYPE]
joint_limits = (joint_info[JOINT_INFO_JOINT_LOWER_LIMIT], 
               joint_info[JOINT_INFO_JOINT_UPPER_LIMIT])
```

### p.getJointState()
Gets current state of a joint.
- **Parameters**:
  - `bodyUniqueId` (int): Robot ID
  - `jointIndex` (int): Joint index
- **Returns**: 4-element tuple with indices:
```python
JOINT_STATE_POS = 0          # Float: Joint position (radians or meters)
JOINT_STATE_VEL = 1          # Float: Joint velocity
JOINT_STATE_FORCES = 2       # List[float]: Reaction forces [fx, fy, fz, mx, my, mz]
JOINT_STATE_TORQUE = 3       # Float: Applied motor torque
```
- **Usage**:
```python
joint_state = p.getJointState(robot_id, joint_index)
current_pos = joint_state[JOINT_STATE_POS]
current_vel = joint_state[JOINT_STATE_VEL]
applied_torque = joint_state[JOINT_STATE_TORQUE]
```

### p.getLinkState()
Gets state of a robot link.
- **Parameters**:
  - `bodyUniqueId` (int): Robot ID
  - `linkIndex` (int): Link index
  - `computeLinkVelocity` (bool): Include velocity info
  - `computeForwardKinematics` (bool): Use FK
- **Returns**: 8-element tuple with indices:
```python
LINK_STATE_POS = 0                  # List[float]: Center of mass position [x, y, z]
LINK_STATE_ORN = 1                  # List[float]: Center of mass orientation [x, y, z, w]
LINK_STATE_LOCAL_POS = 2            # List[float]: Local inertial frame pos
LINK_STATE_LOCAL_ORN = 3            # List[float]: Local inertial frame orn
LINK_STATE_WORLD_LINK_POS = 4       # List[float]: World link frame position
LINK_STATE_WORLD_LINK_ORN = 5       # List[float]: World link frame orientation
LINK_STATE_WORLD_LINK_LINEAR_VEL = 6  # List[float]: World linear velocity [vx, vy, vz]
LINK_STATE_WORLD_LINK_ANGULAR_VEL = 7 # List[float]: World angular velocity [wx, wy, wz]
```
- **Usage**:
```python
link_state = p.getLinkState(robot_id, link_index, computeLinkVelocity=1)
link_pos = link_state[LINK_STATE_WORLD_LINK_POS]
link_orn = link_state[LINK_STATE_WORLD_LINK_ORN]
link_vel = link_state[LINK_STATE_WORLD_LINK_LINEAR_VEL]
```

## Robot Control

### p.setJointMotorControl2()
Main function for joint control.
- **Parameters**:
  - `bodyUniqueId` (int): Robot ID
  - `jointIndex` (int): Joint to control
  - `controlMode` (int): Control mode:
    - `p.POSITION_CONTROL`
    - `p.VELOCITY_CONTROL`
    - `p.TORQUE_CONTROL`
    - `p.PD_CONTROL`
  - `targetPosition` (float): Target angle (radians)
  - `targetVelocity` (float): Target velocity
  - `force` (float): Maximum force to apply
  - `positionGain` (float): kp in PD control
  - `velocityGain` (float): kd in PD control
- **Usage**:
```python
p.setJointMotorControl2(robot,
                       jointIndex=0,
                       controlMode=p.POSITION_CONTROL,
                       targetPosition=1.57,
                       force=100)
```

### p.resetJointState()
Sets joint state directly (bypassing physics).
- **Parameters**:
  - `bodyUniqueId` (int): Robot ID
  - `jointIndex` (int): Joint index
  - `targetValue` (float): Target position
  - `targetVelocity` (float): Target velocity
- **Usage**: `p.resetJointState(robot, 0, targetValue=1.57)`

### p.getContactPoints()
Gets contact information between bodies.
- **Parameters**:
  - `bodyA` (int): First body ID
  - `bodyB` (int): Second body ID
- **Returns**: List of contact point tuples with indices:
```python
CONTACT_FLAG_BODY_A = 0          # Int: Body A unique id
CONTACT_FLAG_BODY_B = 1          # Int: Body B unique id
CONTACT_FLAG_LINK_INDEX_A = 2    # Int: Link index of body A
CONTACT_FLAG_LINK_INDEX_B = 3    # Int: Link index of body B
CONTACT_FLAG_POSITION = 4        # List[float]: Contact position [x, y, z]
CONTACT_FLAG_NORMAL = 5          # List[float]: Contact normal on B [nx, ny, nz]
CONTACT_FLAG_DISTANCE = 6        # Float: Contact distance
CONTACT_FLAG_NORMAL_FORCE = 7    # Float: Normal force applied
CONTACT_FLAG_LATERAL_FRICTION1 = 8  # Float: Lateral friction force 1
CONTACT_FLAG_LATERAL_FRICTION2 = 9  # Float: Lateral friction force 2
CONTACT_FLAG_CONTACT_FLAGS = 10  # Int: Contact flags
```
- **Usage**:
```python
contact_points = p.getContactPoints(bodyA=robot_id, bodyB=ground_id)
for contact in contact_points:
    contact_pos = contact[CONTACT_FLAG_POSITION]
    normal_force = contact[CONTACT_FLAG_NORMAL_FORCE]
```

## Debug and Visualization

### p.addUserDebugParameter()
Creates slider in GUI for parameter control.
- **Parameters**:
  - `paramName` (str): Name of parameter
  - `rangeMin` (float): Minimum value
  - `rangeMax` (float): Maximum value
  - `startValue` (float): Initial value
- **Returns**: `paramId` (int)
- **Usage**: 
```python
slider = p.addUserDebugParameter("Joint1", -3.14, 3.14, 0)
```

### p.readUserDebugParameter()
Reads current value of debug parameter.
- **Parameters**:
  - `itemUniqueId` (int): Parameter ID
- **Returns**: Current value (float)
- **Usage**: `value = p.readUserDebugParameter(slider)`

## Physics Configuration

### p.setTimeStep()
Sets simulation timestep.
- **Parameters**:
  - `timestep` (float): Time in seconds
- **Usage**: `p.setTimeStep(1/240)`

### p.setGravity()
Sets gravity vector.
- **Parameters**:
  - `gravX` (float): X component
  - `gravY` (float): Y component
  - `gravZ` (float): Z component
- **Usage**: `p.setGravity(0, 0, -9.81)`

### p.setRealTimeSimulation()
Enables/disables real-time simulation.
- **Parameters**:
  - `enableRealTimeSimulation` (int): 0 or 1
- **Usage**: `p.setRealTimeSimulation(1)`

### p.stepSimulation()
Steps simulation forward one timestep.
- **Usage**: `p.stepSimulation()`

## Kinematics and Path Planning

### Inverse Kinematics

### p.calculateInverseKinematics()
Calculates joint angles to achieve desired end-effector pose.
- **Parameters**:
  - `bodyUniqueId` (int): Robot ID
  - `endEffectorLinkIndex` (int): Link index of end effector
  - `targetPosition` (list[float]): Target [x, y, z] position
  - `targetOrientation` (list[float], optional): Target orientation as quaternion [x, y, z, w]
  - `lowerLimits` (list[float], optional): Lower joint limits
  - `upperLimits` (list[float], optional): Upper joint limits
  - `jointRanges` (list[float], optional): Range for each joint
  - `restPoses` (list[float], optional): Initial pose for optimization
  - `maxNumIterations` (int, optional): Maximum iterations for IK solver
  - `residualThreshold` (float, optional): Convergence threshold
- **Returns**: List of joint angles
- **Usage**:
```python
# Simple position-only IK
joint_poses = p.calculateInverseKinematics(
    robot,
    endEffectorLinkIndex=end_effector_idx,
    targetPosition=[0.5, 0, 0.8]
)

# Position and orientation IK with additional parameters
joint_poses = p.calculateInverseKinematics(
    robot,
    endEffectorLinkIndex=end_effector_idx,
    targetPosition=[0.5, 0, 0.8],
    targetOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    lowerLimits=lower_limits,
    upperLimits=upper_limits,
    jointRanges=ranges,
    restPoses=rest_poses,
    maxNumIterations=100,
    residualThreshold=1e-4
)
```

### p.calculateJacobian()
Calculates the Jacobian matrix for a specific link.
- **Parameters**:
  - `bodyUniqueId` (int): Robot ID
  - `linkIndex` (int): Link index
  - `localPosition` (list[float]): Point on link to calculate Jacobian for [x, y, z]
  - `objPositions` (list[float]): Joint positions
  - `objVelocities` (list[float]): Joint velocities
  - `objAccelerations` (list[float]): Joint accelerations
- **Returns**: Tuple of:
  - `linearJacobian` (list[list[float]]): 3xN matrix for linear velocity
  - `angularJacobian` (list[list[float]]): 3xN matrix for angular velocity
- **Usage**:
```python
linear_jac, angular_jac = p.calculateJacobian(
    robot,
    linkIndex=end_effector_idx,
    localPosition=[0, 0, 0],
    objPositions=joint_positions,
    objVelocities=[0.0] * num_joints,
    objAccelerations=[0.0] * num_joints
)
```

### Path Planning

### p.computeViewMatrix()
Computes view matrix for collision checking and visualization.
- **Parameters**:
  - `cameraEyePosition` (list[float]): Camera position [x, y, z]
  - `cameraTargetPosition` (list[float]): Look-at position [x, y, z]
  - `cameraUpVector` (list[float]): Up vector [x, y, z]
- **Returns**: View matrix as 16 floats
- **Usage**:
```python
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[2, 2, 2],
    cameraTargetPosition=[0, 0, 0],
    cameraUpVector=[0, 0, 1]
)
```

### p.getClosestPoints()
Computes closest points between objects for collision checking.
- **Parameters**:
  - `bodyA` (int): First body ID
  - `bodyB` (int): Second body ID
  - `distance` (float): Maximum distance to report points
  - `linkIndexA` (int, optional): Link index for first body
  - `linkIndexB` (int, optional): Link index for second body
- **Returns**: List of contact points with same structure as getContactPoints()
- **Usage**:
```python
# Check for potential collisions within 0.1 units
closest_points = p.getClosestPoints(
    bodyA=robot,
    bodyB=obstacle,
    distance=0.1
)
```

### p.rayTest()
Performs raycast collision detection.
- **Parameters**:
  - `rayFromPosition` (list[float]): Start position [x, y, z]
  - `rayToPosition` (list[float]): End position [x, y, z]
- **Returns**: List of hit objects with:
  - `objectUniqueId` (int): Hit object ID
  - `linkIndex` (int): Hit link index
  - `hitFraction` (float): Distance along ray (0 to 1)
  - `hitPosition` (list[float]): Hit position [x, y, z]
  - `hitNormal` (list[float]): Surface normal at hit point
- **Usage**:
```python
# Check if path is clear
hits = p.rayTest(
    rayFromPosition=[0, 0, 0],
    rayToPosition=[1, 1, 1]
)
```

### Important Notes for Path Planning

### Collision Detection Methods
- `getClosestPoints()`: For checking near-collisions and minimum distances
- `getContactPoints()`: For actual contact/collision detection
- `rayTest()`: For line-of-sight and trajectory checking

### IK Solver Tips
- Use `restPoses` to bias solution toward preferred configuration
- Adjust `maxNumIterations` and `residualThreshold` for accuracy vs. speed
- Consider joint limits and ranges for better solutions
- Multiple IK solutions may exist; solver returns one

### Path Planning Strategy
1. Use IK to get start and goal configurations
2. Use collision checking to validate configurations
3. Implement RRT/PRM algorithms using PyBullet's collision checking
4. Validate paths with rayTest() or getClosestPoints()
5. Smooth resulting path for better motion

Note: While PyBullet provides building blocks (IK, collision checking), it doesn't include high-level path planners like RRT or PRM. These need to be implemented using PyBullet's low-level functions.

## Important Notes

### Joint Types
- `p.REVOLUTE_JOINT`: Rotational joint
- `p.PRISMATIC_JOINT`: Sliding joint
- `p.SPHERICAL_JOINT`: Ball joint
- `p.PLANAR_JOINT`: 2D movement joint
- `p.FIXED_JOINT`: Immovable joint

### Control Modes
- `p.POSITION_CONTROL`: Position control with PD controller
- `p.VELOCITY_CONTROL`: Direct velocity control
- `p.TORQUE_CONTROL`: Direct torque control
- `p.PD_CONTROL`: PD control with custom gains

### Coordinate Systems
- Right-handed coordinate system
- Angles in radians
- Quaternions in [x, y, z, w] format

### Performance Tips
- Use `p.DIRECT` mode for headless simulation
- Disable GUI when not needed
- Adjust timestep based on needs
- Use `p.setPhysicsEngineParameter()` for fine-tuning 