import pybullet as p
import pybullet_data
import numpy as np
import time
from spawn_robot import spawn_robot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk

class PathPlanner:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.joint_indices = self._get_revolute_joints()
        self.num_joints = len(self.joint_indices) - 1  # Exclude gripper joint
        self.gripper_joint_index = self.joint_indices[-1]
        self.trajectory_data = {'time': [], 'x': [], 'z': [], 'torques': []}
        print(f"Initialized PathPlanner with {self.num_joints} actuated joints and gripper tracking at index {self.gripper_joint_index}")
        
    def _get_revolute_joints(self):
        """Get indices of revolute joints, excluding fixed and visual joints."""
        indices = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE and "visual" not in joint_info[1].decode('utf-8'):
                print(f"Found revolute joint: {joint_info[1].decode('utf-8')} at index {i}")
                indices.append(i)
        return indices
    
    def get_joint_positions(self):
        """Get current joint positions."""
        positions = []
        for idx in self.joint_indices[:-1]:  # Exclude gripper joint
            pos = p.getJointState(self.robot_id, idx)[0]
            positions.append(pos)
        return np.array(positions)
    
    def get_end_effector_position(self):
        """Get current end effector position."""
        gripper_state = p.getLinkState(self.robot_id, self.gripper_joint_index)
        return np.array(gripper_state[0])
    
    def get_joint_torques(self):
        """Get current joint torques."""
        torques = []
        for idx in self.joint_indices[:-1]:  # Exclude gripper joint
            torque = p.getJointState(self.robot_id, idx)[3]
            torques.append(torque)
        return np.array(torques)
    
    def collect_data(self):
        """Collect trajectory and torque data."""
        pos = self.get_end_effector_position()
        torques = self.get_joint_torques()
        self.trajectory_data['time'].append(time.time())
        self.trajectory_data['x'].append(pos[0])
        self.trajectory_data['z'].append(pos[2])
        self.trajectory_data['torques'].append(torques)
    
    def plot_data(self):
        """Plot trajectory and torque data."""
        # Normalize time to start from 0
        t0 = self.trajectory_data['time'][0]
        times = [t - t0 for t in self.trajectory_data['time']]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot trajectory
        ax1.plot(self.trajectory_data['x'], self.trajectory_data['z'], 'b-', label='End Effector Path')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Z Position (m)')
        ax1.set_title('End Effector Trajectory')
        ax1.grid(True)
        ax1.legend()
        
        # Plot torques
        torques = np.array(self.trajectory_data['torques'])
        for i in range(self.num_joints):
            ax2.plot(times, torques[:, i], label=f'Joint {i+1}')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Torque (N⋅m)')
        ax2.set_title('Joint Torques')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def set_joint_positions(self, positions, control=True):
        """Set joint positions and optionally use position control."""
        for idx, pos in zip(self.joint_indices[:-1], positions):  # Exclude gripper joint
            p.resetJointState(self.robot_id, idx, pos)
            if control:
                p.setJointMotorControl2(self.robot_id, idx,
                                      p.POSITION_CONTROL,
                                      targetPosition=pos)
    
    def linear_interpolation(self, start, end, num_points):
        """Create a linear path between start and end positions."""
        path = []
        for i in range(num_points):
            t = i / (num_points - 1)
            point = start + t * (end - start)
            path.append(point)
        return path
    
    def check_collision(self, joint_positions):
        """Check if a configuration is in collision."""
        # Save current state
        current_positions = self.get_joint_positions()
        
        # Set test positions
        self.set_joint_positions(joint_positions, control=False)
        p.stepSimulation()
        
        # Check for collisions
        collision = False
        for i in range(p.getNumJoints(self.robot_id)):
            contact_points = p.getContactPoints(self.robot_id, self.robot_id, i)
            if contact_points:
                collision = True
                break
        
        # Restore original state
        self.set_joint_positions(current_positions)
        return collision
    
    def plan_joint_path(self, target_positions, num_points=50):
        """Plan a path in joint space."""
        current_positions = self.get_joint_positions()
        path = self.linear_interpolation(current_positions, target_positions, num_points)
        
        # Check path for collisions
        valid_path = []
        for positions in path:
            if not self.check_collision(positions):
                valid_path.append(positions)
            else:
                print("Warning: Collision detected in path")
                break
        
        return valid_path
    
    def execute_path(self, path, time_step=0.01):
        """Execute a planned path."""
        # Clear previous trajectory data
        self.trajectory_data = {'time': [], 'x': [], 'z': [], 'torques': []}
        
        for positions in path:
            self.set_joint_positions(positions)
            p.stepSimulation()
            self.collect_data()
            time.sleep(time_step)
    
    def inverse_kinematics(self, target_position):
        """Compute inverse kinematics for target end effector position."""
        print(f"\nComputing IK for target position: {target_position}")
        
        # Get IK solution for all joints
        all_joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.gripper_joint_index,  # Gripper joint index
            target_position,
            maxNumIterations=100,
            residualThreshold=1e-5,
            jointDamping=[0.01] * len(self.joint_indices)
        )
        
        # Print debug information
        print(f"IK solution length: {len(all_joint_angles)}")
        print(f"Joint indices: {self.joint_indices}")
        
        # Extract only the angles for our revolute joints
        revolute_joint_angles = []
        for i, idx in enumerate(self.joint_indices[:-1]): # Exclude gripper joint
            if i < len(all_joint_angles):
                revolute_joint_angles.append(all_joint_angles[i])
            else:
                print(f"Warning: IK solution missing angle for joint {idx}")
                # Use current joint angle as fallback
                current_angle = p.getJointState(self.robot_id, idx)[0]
                revolute_joint_angles.append(current_angle)
        
        print(f"Final joint angles: {revolute_joint_angles}")
        return np.array(revolute_joint_angles)

class RobotGUI:
    def __init__(self, planner):
        self.planner = planner
        self.root = tk.Tk()
        self.root.title("Robot Control")
        self.root.geometry("300x200")  # Set window size
        
        # Create input fields
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # X position input
        x_frame = ttk.Frame(frame)
        x_frame.pack(fill=tk.X, pady=5)
        ttk.Label(x_frame, text="X Position:").pack(side=tk.LEFT)
        self.x_entry = ttk.Entry(x_frame, width=10)
        self.x_entry.pack(side=tk.LEFT, padx=5)
        self.x_entry.insert(0, "0.3")
        
        # Z position input
        z_frame = ttk.Frame(frame)
        z_frame.pack(fill=tk.X, pady=5)
        ttk.Label(z_frame, text="Z Position:").pack(side=tk.LEFT)
        self.z_entry = ttk.Entry(z_frame, width=10)
        self.z_entry.pack(side=tk.LEFT, padx=5)
        self.z_entry.insert(0, "0.3")
        
        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Move", command=self.move_robot).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Plot Data", command=self.plot_data).pack(side=tk.LEFT, padx=5)
    
    def move_robot(self):
        try:
            x = float(self.x_entry.get())
            z = float(self.z_entry.get())
            target_position = [x, 0, z]  # Y is always 0 for planar motion
            
            print(f"\nMoving to position: {target_position}")
            # Get IK solution and plan path
            target_joints = self.planner.inverse_kinematics(target_position)
            path = self.planner.plan_joint_path(target_joints)
            
            # Execute path
            if path:
                self.planner.execute_path(path)
                print(f"Reached target position")
            else:
                print("Path planning failed")
                
        except ValueError:
            print("Invalid input. Please enter valid numbers.")
    
    def plot_data(self):
        self.planner.plot_data()
    
    def update(self):
        try:
            self.root.update()
            return True
        except tk.TclError:  # Changed from TkError to TclError
            return False

def demo_path_planning():
    """Demonstrate path planning capabilities."""
    # Connect to PyBullet and spawn robot
    robot_id, client = spawn_robot(create_connection=True)
    planner = PathPlanner(robot_id)
    
    # Create GUI
    gui = RobotGUI(planner)
    
    # Keep the simulation running
    try:
        while True:
            p.stepSimulation()
            if not gui.update():  # GUI was closed
                break
            time.sleep(1./240.)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
        try:
            gui.root.quit()
        except:
            pass

if __name__ == "__main__":
    demo_path_planning() 