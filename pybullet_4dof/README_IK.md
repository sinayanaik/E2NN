# PyBullet Inverse Kinematics Guide

## Overview
PyBullet provides built-in inverse kinematics (IK) functionality through the `calculateInverseKinematics()` function. This guide explains how to use it effectively and avoid common pitfalls.

## Basic Usage

```python
joint_positions = p.calculateInverseKinematics(
    bodyUniqueId,      # Robot/body ID
    endEffectorLinkIndex,  # Link index of end effector
    targetPosition,    # [x, y, z] target position
    targetOrientation  # [x, y, z, w] quaternion for target orientation
)
```

## Important Parameters

### Required Parameters
- `bodyUniqueId`: The unique ID of the robot returned by `loadURDF()`
- `endEffectorLinkIndex`: The link index of your end effector
- `targetPosition`: [x, y, z] list/tuple/numpy array of target position
- `targetOrientation`: (Optional) Quaternion [x, y, z, w] for target orientation

### Optional Parameters
- `lowerLimits`: List of lower joint limits
- `upperLimits`: List of upper joint limits
- `jointRanges`: List of joint ranges
- `restPoses`: Initial/rest position of joints
- `maxNumIterations`: Maximum iterations for IK solver (default: 20)
- `residualThreshold`: Convergence threshold (default: 1e-5)
- `currentPositions`: Current joint positions for better initialization
- `jointDamping`: Damping coefficients for joints

## Common Issues and Solutions

1. **No Solution Found**
   - Check if target is within workspace
   - Increase `maxNumIterations`
   - Try different initial positions using `currentPositions`
   - Add joint damping to stabilize solution

2. **Unreachable Targets**
   - Verify target coordinates are in robot's workspace
   - Check joint limits
   - Consider robot's kinematic structure

3. **Unstable Solutions**
   - Add joint damping
   - Use lower `residualThreshold`
   - Implement smooth trajectory between solutions

## Best Practices

1. **Initialization**
   ```python
   # Get current joint positions for better initialization
   current_positions = []
   for joint in range(num_joints):
       pos = p.getJointState(robot_id, joint)[0]
       current_positions.append(pos)
   ```

2. **Joint Limits**
   ```python
   # Get joint limits from URDF
   lower_limits = []
   upper_limits = []
   for joint in range(num_joints):
       info = p.getJointInfo(robot_id, joint)
       lower_limits.append(info[8])
       upper_limits.append(info[9])
   ```

3. **Multiple Attempts Strategy**
   ```python
   best_solution = None
   min_error = float('inf')
   
   for _ in range(num_attempts):
       solution = p.calculateInverseKinematics(
           robot_id,
           ee_link,
           target_pos,
           target_orn,
           currentPositions=current_positions,
           maxNumIterations=100,
           residualThreshold=1e-5
       )
       
       # Validate solution
       error = calculate_position_error(solution)
       if error < min_error:
           min_error = error
           best_solution = solution
   ```

## Error Handling

```python
try:
    joint_positions = p.calculateInverseKinematics(...)
    
    # Validate solution
    ee_pos = p.getLinkState(robot_id, ee_link)[0]
    error = np.sqrt(sum((a - b) ** 2 for a, b in zip(ee_pos, target_pos)))
    
    if error > threshold:
        raise Exception("IK solution exceeds error threshold")
        
except Exception as e:
    print(f"IK calculation failed: {str(e)}")
    # Implement fallback behavior
```

## Tips for Better Results

1. **Orientation Handling**
   - If exact orientation isn't critical, try multiple orientations
   - Use Euler angles for intuitive orientation specification
   ```python
   orientation = p.getQuaternionFromEuler([roll, pitch, yaw])
   ```

2. **Solution Refinement**
   - Use small position adjustments if exact position needed
   - Implement iterative refinement with smaller steps
   - Consider using null-space control for redundant robots

3. **Performance Optimization**
   - Cache joint limits and other robot parameters
   - Use appropriate `maxNumIterations` and `residualThreshold`
   - Implement solution caching for similar targets

## Debugging

1. **Visualize Target**
   ```python
   p.addUserDebugLine(
       [0, 0, 0],
       target_position,
       [1, 0, 0],  # Red color
       lineWidth=2,
       lifeTime=0  # Persistent
   )
   ```

2. **Monitor End Effector**
   ```python
   ee_pos = p.getLinkState(robot_id, ee_link)[0]
   p.addUserDebugText(
       f"EE: {ee_pos}",
       ee_pos,
       textSize=1.5
   )
   ```

## References

- [PyBullet User Manual](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.9i02ojf4k3ve)
- [PyBullet Quickstart Guide](https://usermanual.wiki/Document/pybullet20quickstart20guide.479068914/view) 