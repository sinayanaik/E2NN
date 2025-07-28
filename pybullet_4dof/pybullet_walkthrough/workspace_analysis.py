import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from spawn_robot import spawn_robot
import time

def print_joint_info(robot_id):
    """Print detailed information about all joints."""
    num_joints = p.getNumJoints(robot_id)
    print(f"\nJoint Information:")
    print(f"Total joints: {num_joints}")
    print("\nDetailed joint info:")
    print("Index | Type | Name | Axis | Lower | Upper | Max Force")
    print("-" * 70)
    
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        joint_type = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"][joint_info[2]] if joint_info[2] < 5 else "UNKNOWN"
        # Get joint axis
        axis = joint_info[13] if len(joint_info) > 13 else (0, 0, 0)
        print(f"{joint_idx:5d} | {joint_type:8s} | {joint_info[1].decode('utf-8'):20s} | {axis!r:12s} | {joint_info[8]:6.2f} | {joint_info[9]:6.2f} | {joint_info[10]:9.2f}")

def get_revolute_joint_indices(robot_id):
    """Get indices of all revolute joints."""
    joint_indices = []
    num_joints = p.getNumJoints(robot_id)
    
    for joint_idx in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_idx)
        # Only include revolute joints and skip visual joints
        if joint_info[2] == p.JOINT_REVOLUTE and "visual" not in joint_info[1].decode('utf-8'):
            joint_indices.append(joint_idx)
    
    return joint_indices

def get_joint_limits(robot_id):
    """Get joint limits for revolute joints."""
    joint_limits = []
    revolute_joints = get_revolute_joint_indices(robot_id)
    
    for joint_idx in revolute_joints:
        joint_info = p.getJointInfo(robot_id, joint_idx)
        limits = {
            'joint_index': joint_idx,
            'lower': joint_info[8],
            'upper': joint_info[9],
            'name': joint_info[1].decode('utf-8'),
            'max_force': joint_info[10],
            'max_velocity': joint_info[11],
            'axis': joint_info[13]
        }
        joint_limits.append(limits)
    
    return joint_limits

def get_end_effector_position(robot_id):
    """Get the end effector position."""
    num_joints = p.getNumJoints(robot_id)
    # Get the state of the last revolute joint's child link
    end_effector_state = p.getLinkState(robot_id, num_joints - 1)
    pos = end_effector_state[0]
    # Project position onto X-Z plane
    return (pos[0], pos[2])  # Return X and Z coordinates

def reset_robot_state(robot_id):
    """Reset robot to default state."""
    joint_limits = get_joint_limits(robot_id)
    for joint in joint_limits:
        p.resetJointState(robot_id, joint['joint_index'], 0)
        # Enable motor control
        p.setJointMotorControl2(robot_id, joint['joint_index'], 
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=joint['max_force'])
    p.stepSimulation()

def sample_workspace(robot_id, samples_per_joint=20):
    """Sample the workspace by trying different joint configurations."""
    # Print detailed joint information
    print_joint_info(robot_id)
    
    # Get joint limits
    joint_limits = get_joint_limits(robot_id)
    workspace_points = []
    
    # Create evenly spaced samples for each joint
    joint_samples = []
    for joint in joint_limits:
        samples = np.linspace(joint['lower'], joint['upper'], samples_per_joint)
        joint_samples.append(samples)
    
    # Try all combinations of joint angles
    total_combinations = samples_per_joint ** len(joint_limits)
    print(f"\nFound {len(joint_limits)} revolute joints for workspace analysis:")
    for joint in joint_limits:
        print(f"  {joint['name']}:")
        print(f"    - Index: {joint['joint_index']}")
        print(f"    - Limits: {joint['lower']:.2f} to {joint['upper']:.2f} rad")
        print(f"    - Max Force: {joint['max_force']}")
        print(f"    - Axis: {joint['axis']}")
    
    print(f"\nSampling {total_combinations} configurations...")
    
    # Reset robot to starting position
    reset_robot_state(robot_id)
    
    # Sample configurations
    for i, angles in enumerate(np.ndindex(tuple([samples_per_joint] * len(joint_limits)))):
        if i % 1000 == 0:
            print(f"Progress: {i}/{total_combinations} configurations")
            
        # Set joint positions
        for idx, angle in enumerate(angles):
            joint = joint_limits[idx]
            joint_angle = joint_samples[idx][angle]
            p.resetJointState(robot_id, joint['joint_index'], joint_angle)
            p.setJointMotorControl2(robot_id, joint['joint_index'],
                                  p.POSITION_CONTROL,
                                  targetPosition=joint_angle,
                                  force=joint['max_force'])
        
        # Step simulation multiple times to ensure stability
        for _ in range(10):  # Increased simulation steps
            p.stepSimulation()
        
        # Get end effector position
        ee_pos = get_end_effector_position(robot_id)
        if any(np.isnan(ee_pos)):
            print(f"Warning: NaN position detected at configuration {angles}")
            continue
            
        # Only add point if it's not too close to an existing point
        workspace_points.append(ee_pos)
    
    return workspace_points

def plot_workspace(points, output_file=None):
    """Plot the workspace points and calculate metrics."""
    if not points:
        print("Error: No valid points to plot!")
        return
    
    x_coords, z_coords = zip(*points)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(x_coords, z_coords, alpha=0.1, s=1)
    plt.title('Robot End Effector Workspace (X-Z Plane)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Z Position (m)')
    plt.grid(True)
    plt.axis('equal')
    
    if output_file:
        plt.savefig(output_file)
        print(f"Workspace plot saved to {output_file}")
    else:
        plt.show()
    
    # Calculate and display workspace metrics
    x_range = max(x_coords) - min(x_coords)
    z_range = max(z_coords) - min(z_coords)
    print(f"\nWorkspace Metrics:")
    print(f"X Range: {x_range:.3f} m ({min(x_coords):.3f} to {max(x_coords):.3f})")
    print(f"Z Range: {z_range:.3f} m ({min(z_coords):.3f} to {max(z_coords):.3f})")
    print(f"Approximate Area: {x_range * z_range:.3f} m²")
    print(f"Number of sampled points: {len(points)}")

def main():
    # Connect to PyBullet
    physics_client = p.connect(p.DIRECT)  # Use DIRECT mode for faster sampling
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)  # Disable gravity for workspace analysis
    
    # Spawn robot and get robot ID
    robot_id, _ = spawn_robot()  # Unpack the tuple to get just the robot ID
    
    # Sample the workspace
    print("Analyzing workspace...")
    workspace_points = sample_workspace(robot_id, samples_per_joint=20)
    
    # Plot and save the results
    output_file = "workspace_analysis.png"
    plot_workspace(workspace_points, output_file)
    
    # Disconnect from PyBullet
    p.disconnect()

if __name__ == "__main__":
    main() 