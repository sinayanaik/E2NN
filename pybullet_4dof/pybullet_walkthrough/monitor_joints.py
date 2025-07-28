import pybullet as p
import numpy as np
from spawn_robot import spawn_robot
import time

def rad2deg(rad):
    return rad * 180.0 / np.pi

def get_joint_info(robot):
    """Get information about all joints"""
    num_joints = p.getNumJoints(robot)
    print(f"\nRobot has {num_joints} joints")
    print("\nJoint Information:")
    print("-" * 50)
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        
        # Get type name
        type_name = {
            p.JOINT_REVOLUTE: "Revolute",
            p.JOINT_PRISMATIC: "Prismatic",
            p.JOINT_SPHERICAL: "Spherical",
            p.JOINT_PLANAR: "Planar",
            p.JOINT_FIXED: "Fixed"
        }.get(joint_type, "Unknown")
        
        print(f"\nJoint {i}: {joint_name}")
        print(f"  Type: {type_name}")
        
        # Show limits for both revolute and prismatic joints
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            lower = joint_info[8]
            upper = joint_info[9]
            
            if joint_type == p.JOINT_REVOLUTE:
                # Convert to degrees for revolute joints
                lower_display = f"{rad2deg(lower):.1f}°"
                upper_display = f"{rad2deg(upper):.1f}°"
                current_units = "deg"
            else:
                # Keep meters for prismatic joints
                lower_display = f"{lower:.3f}m"
                upper_display = f"{upper:.3f}m"
                current_units = "m"
                
            print(f"  Limits: [{lower_display} to {upper_display}]")
            print(f"  Range: {abs(upper - lower):.1f} {current_units}")
            print(f"  Max Force: {joint_info[10]:.1f} N")
            print(f"  Max Velocity: {joint_info[11]:.1f} {'rad/s' if joint_type == p.JOINT_REVOLUTE else 'm/s'}")

def monitor_joint_states(robot):
    """Monitor joint states in real-time"""
    try:
        # Get joint info once at start (it doesn't change)
        num_joints = p.getNumJoints(robot)
        joint_data = {}
        
        # Cache joint information
        for i in range(num_joints):
            info = p.getJointInfo(robot, i)
            if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                joint_data[i] = {
                    'name': info[1].decode('utf-8'),
                    'type': info[2],
                    'lower': info[8],
                    'upper': info[9]
                }
        
        while True:
            print("\nCurrent Joint States:")
            print("-" * 50)
            
            for i, data in joint_data.items():
                # Get joint state
                state = p.getJointState(robot, i)
                current_pos = state[0]  # Current position
                vel = state[1]
                forces = state[2]
                torque = state[3]
                
                # Get link position
                link_state = p.getLinkState(robot, i)
                link_pos = link_state[0]  # World position of center of mass
                x_pos, z_pos = link_pos[0], link_pos[2]  # Extract X and Z coordinates
                
                print(f"\n{data['name']}:")
                
                if data['type'] == p.JOINT_REVOLUTE:
                    # Convert everything to degrees for revolute joints
                    current_pos_deg = rad2deg(current_pos)
                    lower_deg = rad2deg(data['lower'])
                    upper_deg = rad2deg(data['upper'])
                    
                    # Calculate position as percentage of range
                    range_deg = upper_deg - lower_deg
                    if range_deg != 0:  # Avoid division by zero
                        position_percent = ((current_pos_deg - lower_deg) / range_deg) * 100
                    else:
                        position_percent = 0
                    
                    print(f"  Position: {current_pos_deg:.1f}° [{position_percent:.1f}% of range]")
                    print(f"  Limits: [{lower_deg:.1f}° to {upper_deg:.1f}°]")
                    print(f"  Coordinates: X={x_pos:.3f}, Z={z_pos:.3f} m")
                    print(f"  Velocity: {vel:.1f} rad/s")
                else:  # JOINT_PRISMATIC
                    # Keep everything in meters for prismatic joints
                    position_percent = ((current_pos - data['lower']) / (data['upper'] - data['lower'])) * 100 if data['upper'] != data['lower'] else 0
                    
                    print(f"  Position: {current_pos:.3f}m [{position_percent:.1f}% of range]")
                    print(f"  Limits: [{data['lower']:.3f}m to {data['upper']:.3f}m]")
                    print(f"  Coordinates: X={x_pos:.3f}, Z={z_pos:.3f} m")
                    print(f"  Velocity: {vel:.1f} m/s")
                
                print(f"  Forces (xyz): [{forces[0]:.1f}, {forces[1]:.1f}, {forces[2]:.1f}] N")
                print(f"  Torques (rpy): [{forces[3]:.1f}, {forces[4]:.1f}, {forces[5]:.1f}] N⋅m")
                print(f"  Motor Torque: {torque:.1f} N⋅m")
            
            print("\nPress Ctrl+C to exit")
            p.stepSimulation()
            time.sleep(1/120)  # 120 Hz
            
    except KeyboardInterrupt:
        p.disconnect()

def main():
    # Spawn robot and get the robot ID
    robot, physics_client = spawn_robot()
    
    # Print initial joint information
    get_joint_info(robot)
    
    # Start monitoring
    monitor_joint_states(robot)

if __name__ == "__main__":
    main() 