# Robot Spawning in PyBullet - A Beginner's Guide

This guide explains how to spawn (create and place) a robot in a PyBullet simulation environment. Think of PyBullet as a virtual world where you can place and control robots, just like in a video game!

## What is PyBullet?

PyBullet is like a physics playground for robots. It helps you:
- Create virtual robots from description files (URDF)
- Simulate real-world physics (gravity, collisions, etc.)
- Control and monitor robot movements
- Visualize everything in 3D

## The `spawn_robot()` Function Explained

### What Does It Do?
The `spawn_robot()` function does four main things:
1. Creates a virtual world (PyBullet environment)
2. Adds a floor (ground plane)
3. Places your robot in the world
4. Sets up a camera to view everything

### How to Use It
```python
from spawn_robot import spawn_robot

# Create the world and get your robot
robot, physics_client = spawn_robot()

# Now you can use the robot ID for control!
```

### Step-by-Step Breakdown

#### 1. Connecting to PyBullet
```python
physicsClient = p.connect(p.GUI)
```
- What it does: Creates a window to show your robot
- Options:
  - `p.GUI`: Shows a visual window (good for learning/testing)
  - `p.DIRECT`: No window (good for fast computations)
- Why: You need this to see what's happening!

#### 2. Setting Up the World
```python
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
```
- Search Path: Tells PyBullet where to find built-in models
- Gravity: Makes things fall realistically
  - `0, 0, -9.81`: Earth's gravity (meters/second²)
  - Can be changed for space/moon simulations!
  - Example: Moon gravity would be `0, 0, -1.62`

#### 3. Creating the Floor
```python
planeId = p.loadURDF("plane.urdf")
```
- What: Loads an infinite checkerboard ground
- Why: Gives your robot something to stand on
- Height: Always at z=0 (like sea level)
- Pattern: Helps you see distances better

#### 4. Loading the Robot
```python
robot = p.loadURDF(urdf_path, 
                  startPos=[0, 0, 0],
                  startOrientation=[0, 0, 0],
                  useFixedBase=True)
```
- URDF Path: Where to find your robot's description file
- Start Position `[x, y, z]`:
  - `[0, 0, 0]`: Center of the world
  - Can be changed like `[1, 2, 0.5]` to start elsewhere
- Orientation: How the robot is rotated
  - `[0, 0, 0]`: Standing straight
  - Units: Euler angles in radians
- Fixed Base:
  - `True`: Robot stays in place (like a factory arm)
  - `False`: Robot can move around (like a mobile robot)

#### 5. Camera Setup
```python
p.resetDebugVisualizerCamera(
    cameraDistance=1.0,  # How far from target
    cameraYaw=0,        # Left/right rotation
    cameraPitch=0,      # Up/down angle
    cameraTargetPosition=[0, 0, 0]  # What to look at
)
```
- Distance: How far the camera is (in meters)
- Yaw: Rotation around vertical axis (degrees)
  - 0°: Front view
  - 90°: Left side view
  - 180°: Back view
  - 270°: Right side view
- Pitch: Up/down angle (degrees)
  - 0°: Horizontal view
  - 90°: Looking down
  - -90°: Looking up
- Target: What point to look at `[x, y, z]`

### Interactive Camera Controls
You can move the camera with your mouse:
- 🖱️ Left Button + Drag: Rotate view
- 🖱️ Right Button + Drag: Move camera sideways
- 🖱️ Scroll Wheel: Zoom in/out
- 🖱️ Middle Button + Drag: Pan view
- 🎮 Ctrl + Left Button + Drag: Tilt camera

## Common Issues and Solutions

### 1. "No module named 'pybullet'"
```bash
pip install pybullet
```

### 2. Can't See the Robot
- Check if URDF path is correct
- Make sure robot isn't too small/large
- Try different camera positions
- Look for error messages in terminal

### 3. Robot Looks Wrong
- Verify URDF file is valid
- Check if all mesh files exist
- Confirm units in URDF (meters vs millimeters)

### 4. Performance Issues
- Try `p.DIRECT` mode instead of `p.GUI`
- Reduce simulation frequency
- Check GPU drivers
- Close other heavy applications

## Simulation Loop Example
```python
try:
    while True:
        # Step physics forward
        p.stepSimulation()
        
        # Don't go too fast!
        time.sleep(1./240.)  # 240 Hz simulation
        
except KeyboardInterrupt:
    # Clean up when you press Ctrl+C
    p.disconnect()
```
- 240 Hz: Standard for smooth simulation
- Can be slower (like 60 Hz) for simpler tasks
- Faster for high-precision needs

## Coordinate System
Think of it like a 3D graph:
- X-axis: Forward/Backward (red arrow)
- Y-axis: Left/Right (green arrow)
- Z-axis: Up/Down (blue arrow)
- Units: Everything in meters!

## Next Steps
After spawning your robot, you can:
1. Move its joints (see joint_control.md)
2. Monitor its state (see monitor_joints.md)
3. Display information (see joint_display.md)
4. Add objects to interact with
5. Create custom control programs 