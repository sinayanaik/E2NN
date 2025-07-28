# Joint Monitoring in PyBullet - Your Robot's Console!

This guide explains how to watch your robot's joint information right in your terminal/console. Think of it like having a command center for your robot!

## What is Joint Monitoring?

Joint monitoring means watching:
1. What each joint is doing
2. Where it is
3. How it's moving
4. What forces it's experiencing

It's like having X-ray vision into your robot's "muscles"!

## Why Monitor in the Terminal?

### Advantages
1. Works everywhere (no GUI needed)
2. Easy to read
3. Great for debugging
4. Can be logged to files
5. Fast updates

### When to Use It
- During development
- When debugging problems
- For logging data
- On remote computers
- In automated tests

## Understanding Joint Types

PyBullet has different kinds of joints:
```python
joint_types = {
    p.JOINT_REVOLUTE: "Revolute",     # Rotates (like an elbow)
    p.JOINT_PRISMATIC: "Prismatic",   # Slides (like a drawer)
    p.JOINT_SPHERICAL: "Spherical",   # Moves in all directions (like a shoulder)
    p.JOINT_PLANAR: "Planar",         # Moves in a plane
    p.JOINT_FIXED: "Fixed"            # Doesn't move (like welded parts)
}
```

## Getting Joint Information

### The `get_joint_info` Function
```python
def get_joint_info(robot):
    """Print detailed info about all joints"""
    num_joints = p.getNumJoints(robot)
    print(f"\nRobot has {num_joints} joints")
    
    for joint in range(num_joints):
        info = p.getJointInfo(robot, joint)
        name = info[1].decode('utf-8')
        type = joint_types.get(info[2], "Unknown")
        
        # Show the details
        print(f"\nJoint {joint}: {name}")
        print(f"  Type: {type}")
        
        # For rotating joints, show limits
        if info[2] == p.JOINT_REVOLUTE:
            lower_deg = rad2deg(info[8])    # Convert to degrees
            upper_deg = rad2deg(info[9])
            print(f"  Limits: {lower_deg:.1f}° to {upper_deg:.1f}°")
            print(f"  Max Force: {info[10]:.1f} N")
            print(f"  Max Speed: {info[11]:.1f} rad/s")
```

### What You'll See
```
Robot has 3 joints

Joint Information:
--------------------------------------------------

Joint 0: shoulder
  Type: Revolute
  Limits: -90.0° to 90.0°
  Max Force: 100.0 N
  Max Speed: 10.0 rad/s

Joint 1: elbow
  Type: Revolute
  Limits: 0.0° to 135.0°
  Max Force: 80.0 N
  Max Speed: 8.0 rad/s
...
```

## Monitoring Joint States

### The `monitor_joint_states` Function
```python
def monitor_joint_states(robot):
    """Show real-time joint information"""
    try:
        while True:
            print("\nCurrent Joint States:")
            print("-" * 50)
            
            # Check each joint
            for joint in range(p.getNumJoints(robot)):
                # Get all the information
                info = p.getJointInfo(robot, joint)
                name = info[1].decode('utf-8')
                
                if info[2] == p.JOINT_REVOLUTE:
                    # Get current state
                    state = p.getJointState(robot, joint)
                    pos = rad2deg(state[0])    # Position in degrees
                    vel = state[1]             # Velocity in rad/s
                    forces = state[2]          # Forces and torques
                    motor = state[3]           # Motor torque
                    
                    # Get position in world
                    link_pos = p.getLinkState(robot, joint)[0]
                    x, z = link_pos[0], link_pos[2]
                    
                    # Show everything nicely formatted
                    print(f"\n{name}:")
                    print(f"  Position: {pos:.1f}°")
                    print(f"  Location: X={x:.3f}, Z={z:.3f} m")
                    print(f"  Speed: {vel:.1f} rad/s")
                    print(f"  Forces: [{forces[0]:.1f}, {forces[1]:.1f}, {forces[2]:.1f}] N")
                    print(f"  Torques: [{forces[3]:.1f}, {forces[4]:.1f}, {forces[5]:.1f}] N⋅m")
                    print(f"  Motor Power: {motor:.1f} N⋅m")
            
            print("\nPress Ctrl+C to stop")
            p.stepSimulation()
            time.sleep(0.5)  # Update twice per second
            
    except KeyboardInterrupt:
        print("\nStopping monitor...")
        p.disconnect()
```

### What You'll See
```
Current Joint States:
--------------------------------------------------

shoulder:
  Position: 45.0°
  Location: X=0.123, Z=0.456 m
  Speed: 0.5 rad/s
  Forces: [0.0, -9.8, 0.0] N
  Torques: [0.0, 0.0, 5.0] N⋅m
  Motor Power: 2.5 N⋅m

elbow:
  Position: 90.0°
  Location: X=0.789, Z=0.321 m
  Speed: -0.2 rad/s
  Forces: [1.0, 0.0, -4.9] N
  Torques: [0.0, 2.0, 0.0] N⋅m
  Motor Power: 1.5 N⋅m
...
```

## Understanding the Numbers

### Position (Degrees)
- 0° = Usually straight/neutral
- Positive = Clockwise
- Negative = Counter-clockwise
- Examples:
  - 90° = Quarter turn right
  - -45° = Eighth turn left

### Speed (rad/s)
- 0 = Not moving
- Positive = Moving one way
- Negative = Moving other way
- Typical values:
  - 0.1 to 0.5 = Slow
  - 0.5 to 2.0 = Normal
  - 2.0+ = Fast

### Forces (Newtons)
- Shows pushing/pulling
- Direction matters:
  - X: Forward(+)/Back(-)
  - Y: Left(+)/Right(-)
  - Z: Up(+)/Down(-)
- Examples:
  - [0, 0, -9.8] = Just gravity
  - [5, 0, 0] = 5N forward push

### Torques (N⋅m)
- Shows twisting forces
- Around each axis:
  - X: Front-back spin
  - Y: Left-right spin
  - Z: Standing spin
- Examples:
  - [0, 0, 5] = 5N⋅m twist up
  - [2, 0, 0] = 2N⋅m twist forward

### Motor Power (N⋅m)
- How hard motor works
- Higher = More effort
- Examples:
  - 0-1 = Light work
  - 1-5 = Normal work
  - 5+ = Heavy work

## How to Use It

### Basic Usage
```python
from monitor_joints import get_joint_info, monitor_joint_states

# Get your robot
robot, _ = spawn_robot()

# See what joints exist
get_joint_info(robot)

# Start watching them
monitor_joint_states(robot)
```

### What Happens
1. Shows all joints first
2. Updates every 0.5 seconds
3. Clear formatting
4. Easy to read
5. Press Ctrl+C to stop

## Tips for Monitoring

### 1. Watch for Problems
- Sudden position changes
- Unusual forces
- High motor power
- Stuck joints

### 2. Normal Operation
- Smooth movement
- Expected forces
- Reasonable motor power
- Regular patterns

### 3. Debugging Help
- Check limits
- Watch forces
- Monitor speeds
- Track positions

## Common Questions

### Numbers Jumping Around?
Could be:
1. Fast movement
2. Unstable control
3. Collisions
4. Sensor noise

### Strange Forces?
Check for:
1. Collisions
2. Joint limits
3. Gravity effects
4. Control issues

### Motor Power High?
Might be:
1. Heavy load
2. Fighting gravity
3. Joint friction
4. Control problems

## Advanced Usage

### 1. Custom Monitoring
```python
# Example: Watch specific joints
joints_to_watch = ['shoulder', 'elbow']
for joint in joints_to_watch:
    # Show just these joints
```

### 2. Data Logging
```python
# Example: Save to file
with open('joint_log.txt', 'w') as f:
    f.write(f"Time: {time.time()}, Position: {pos}...")
```

### 3. Specific Updates
```python
# Example: Update faster/slower
time.sleep(0.1)  # 10 times per second
```

## Next Steps
After mastering monitoring:
1. Add data logging
2. Create custom displays
3. Set up alerts
4. Analyze patterns
5. Automate testing 

## PyBullet API Reference

### `getNumJoints(bodyUniqueId)`
Returns the number of joints in a robot.

```python
num_joints = p.getNumJoints(robot)  # robot is the body ID
```

**Parameters:**
- `bodyUniqueId` (int): The unique ID returned by `loadURDF`

**Returns:**
- (int): Number of joints in the robot

### `getJointInfo(bodyUniqueId, jointIndex)`
Gets detailed information about a specific joint.

```python
joint_info = p.getJointInfo(robot, joint_index)
```

**Parameters:**
- `bodyUniqueId` (int): The robot's ID
- `jointIndex` (int): Index of the joint (0 to getNumJoints-1)

**Returns:**
A tuple with 17 items:
```python
[
    0:  jointIndex,     # int: same as input parameter
    1:  jointName,      # str: name of joint (as bytes, needs .decode('utf-8'))
    2:  jointType,      # int: type of joint (see joint_types below)
    3:  qIndex,         # int: position index in state arrays (-1 for fixed)
    4:  uIndex,         # int: velocity index in state arrays (-1 for fixed)
    5:  flags,          # int: joint flags
    6:  jointDamping,   # float: damping coefficient (Ns/m for linear, Nms/rad for revolute)
    7:  jointFriction,  # float: friction coefficient (N or Nm)
    8:  jointLowerLimit,# float: lower position limit (meters or radians)
    9:  jointUpperLimit,# float: upper position limit (meters or radians)
    10: jointMaxForce,  # float: maximum force (N or Nm)
    11: jointMaxVelocity,# float: maximum velocity (m/s or rad/s)
    12: linkName,       # str: name of link (as bytes)
    13: jointAxis,      # list: joint axis [x, y, z]
    14: parentFramePos, # list: joint position in parent frame [x, y, z]
    15: parentFrameOrn, # list: joint orientation in parent frame [x, y, z, w]
    16: parentIndex     # int: parent link index (-1 for base)
]
```

Example usage:
```python
info = p.getJointInfo(robot, 0)  # Get info for first joint
name = info[1].decode('utf-8')   # Joint name
type = info[2]                   # Joint type
limits = (info[8], info[9])      # Position limits
max_force = info[10]             # Maximum force
max_speed = info[11]             # Maximum velocity
axis = info[13]                  # Rotation/movement axis
parent = info[16]                # Parent link
```

### `joint_types`
PyBullet's joint type constants and their meanings:

```python
joint_types = {
    p.JOINT_REVOLUTE: "Revolute",     # Type: 0
    p.JOINT_PRISMATIC: "Prismatic",   # Type: 1
    p.JOINT_SPHERICAL: "Spherical",   # Type: 2
    p.JOINT_PLANAR: "Planar",         # Type: 3
    p.JOINT_FIXED: "Fixed"            # Type: 4
}
```

**Details:**
1. **JOINT_REVOLUTE (0)**:
   - Rotates around an axis
   - Has angle limits (radians)
   - Example: Robot elbow, door hinge

2. **JOINT_PRISMATIC (1)**:
   - Slides along an axis
   - Has distance limits (meters)
   - Example: Hydraulic piston, drawer

3. **JOINT_SPHERICAL (2)**:
   - Rotates in all directions
   - No limits by default
   - Example: Ball joint, shoulder

4. **JOINT_PLANAR (3)**:
   - Moves in a 2D plane
   - Can translate and rotate
   - Example: Moving platform

5. **JOINT_FIXED (4)**:
   - No movement allowed
   - Rigidly connects parts
   - Example: Welded parts

### `getLinkState(bodyUniqueId, linkIndex)`
Gets the state of a robot link in world coordinates.

```python
link_state = p.getLinkState(robot, link_index)
```

**Parameters:**
- `bodyUniqueId` (int): The robot's ID
- `linkIndex` (int): Index of the link (same as joint index)

**Returns:**
A tuple with 8 items:
```python
[
    0: linkWorldPosition,     # list: [x, y, z] position in world frame
    1: linkWorldOrientation,  # list: [x, y, z, w] quaternion in world frame
    2: localInertialFramePosition,    # list: [x, y, z] inertial frame pos
    3: localInertialFrameOrientation, # list: [x, y, z, w] inertial frame orn
    4: worldLinkFramePosition,        # list: [x, y, z] COM position
    5: worldLinkFrameOrientation,     # list: [x, y, z, w] COM orientation
    6: worldLinkLinearVelocity,       # list: [vx, vy, vz] linear velocity
    7: worldLinkAngularVelocity       # list: [wx, wy, wz] angular velocity
]
```

Example usage:
```python
state = p.getLinkState(robot, 0)  # Get state of first link

# Position in world
pos = state[0]  # [x, y, z]
x, y, z = pos   # Unpack coordinates

# Orientation (quaternion)
orn = state[1]  # [x, y, z, w]

# Center of Mass position
com_pos = state[4]

# Velocities
lin_vel = state[6]  # Linear velocity [vx, vy, vz]
ang_vel = state[7]  # Angular velocity [wx, wy, wz]
```

### Common Function Combinations

#### 1. Getting Joint Names and Types
```python
def get_joint_names_and_types(robot):
    joints = {}
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        name = info[1].decode('utf-8')
        joint_type = joint_types.get(info[2], "Unknown")
        joints[name] = joint_type
    return joints
```

#### 2. Getting Joint Limits
```python
def get_joint_limits(robot):
    limits = {}
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        name = info[1].decode('utf-8')
        if info[2] == p.JOINT_REVOLUTE:
            limits[name] = {
                'lower': rad2deg(info[8]),  # Convert to degrees
                'upper': rad2deg(info[9]),
                'max_force': info[10],
                'max_velocity': info[11]
            }
    return limits
```

#### 3. Getting Complete Joint State
```python
def get_joint_state_complete(robot, joint_index):
    # Get basic joint info
    info = p.getJointInfo(robot, joint_index)
    name = info[1].decode('utf-8')
    
    # Get current state
    state = p.getJointState(robot, joint_index)
    pos, vel, forces, torque = state
    
    # Get world position
    link_state = p.getLinkState(robot, joint_index)
    world_pos = link_state[0]
    world_orn = link_state[1]
    velocities = link_state[6:8]  # Linear and angular
    
    return {
        'name': name,
        'position': rad2deg(pos),  # degrees
        'velocity': vel,           # rad/s
        'forces': forces,          # [fx, fy, fz, mx, my, mz]
        'motor_torque': torque,    # Nm
        'world_position': world_pos,
        'world_orientation': world_orn,
        'velocities': velocities
    }
```

## Tips for Using These Functions

1. **Performance**:
   - Cache joint info that doesn't change (names, types, limits)
   - Only get link state when needed (it's computationally expensive)
   - Use appropriate update rates (e.g., 60 Hz for display)

2. **Error Handling**:
   - Always check if joint indices are valid
   - Handle byte string decoding for names
   - Verify joint types before operations

3. **Units**:
   - Remember to convert radians ↔ degrees for display
   - Forces are in Newtons (N)
   - Torques are in Newton-meters (N⋅m)
   - Positions are in meters
   - Time is in seconds

4. **Coordinate Systems**:
   - World frame: Global coordinates
   - Link frame: Relative to link
   - Inertial frame: Center of mass
   - Parent frame: Relative to parent link 