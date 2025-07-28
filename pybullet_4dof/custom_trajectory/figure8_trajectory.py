import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os

def spawn_robot():
    """Spawn the robot in PyBullet simulation"""
    # Connect to PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load ground plane
    planeId = p.loadURDF("plane.urdf")

    # Load our robot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir, "..", "urdf", "robot.urdf")
    
    # Set initial pose
    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    
    # Load robot with fixed base
    robot = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=True)

    # Set camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=0,
        cameraPitch=0,
        cameraTargetPosition=[0, 0, 0]
    )
    
    return robot, physicsClient

class Figure8Tracker:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        # Get joint indices for the three actuated joints
        self.joint_indices = {
            'link_1_to_link_2': 2,
            'link_2_to_link_3': 3,
            'link_3_to_link_4': 4
        }
        self.gripper_joint_index = 5  # link_4_to_gripper is the end effector
        
        # Data storage for plotting
        self.time_data = []
        self.actual_x = []
        self.actual_z = []
        self.desired_x = []
        self.desired_z = []
        self.joint_torques = {name: [] for name in self.joint_indices.keys()}
        
        # Initialize plot
        self.setup_plot()
        
    def setup_plot(self):
        """Setup real-time plotting"""
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Trajectory plot
        self.ax1.set_title('End Effector Trajectory')
        self.ax1.set_xlabel('X Position (m)')
        self.ax1.set_ylabel('Z Position (m)')
        self.ax1.grid(True)
        
        # Initialize plot lines
        self.actual_line, = self.ax1.plot([], [], 'b-', label='Actual')
        self.desired_line, = self.ax1.plot([], [], 'r--', label='Desired')
        self.ax1.legend()
        
        # Torque plot
        self.ax2.set_title('Joint Torques')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Torque (N⋅m)')
        self.ax2.grid(True)
        
        # Initialize torque lines
        self.torque_lines = {}
        for name in self.joint_indices.keys():
            line, = self.ax2.plot([], [], label=name)
            self.torque_lines[name] = line
        self.ax2.legend()
        
        plt.tight_layout()
        
    def update_plots(self):
        """Update real-time plots"""
        # Update trajectory plot
        self.actual_line.set_data(self.actual_x, self.actual_z)
        self.desired_line.set_data(self.desired_x, self.desired_z)
        
        # Auto-scale trajectory plot
        if len(self.actual_x) > 0:
            all_x = self.actual_x + self.desired_x
            all_z = self.actual_z + self.desired_z
            x_min, x_max = min(all_x), max(all_x)
            z_min, z_max = min(all_z), max(all_z)
            x_margin = (x_max - x_min) * 0.1
            z_margin = (z_max - z_min) * 0.1
            self.ax1.set_xlim(x_min - x_margin, x_max + x_margin)
            self.ax1.set_ylim(z_min - z_margin, z_max + z_margin)
        
        # Update torque plot
        if len(self.time_data) > 0:
            for name, line in self.torque_lines.items():
                line.set_data(self.time_data, self.joint_torques[name])
            self.ax2.relim()
            self.ax2.autoscale_view()
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def get_end_effector_state(self):
        """Get current end effector position"""
        state = p.getLinkState(self.robot_id, self.gripper_joint_index)
        return np.array([state[0][0], state[0][2]])  # x, z coordinates
    
    def get_joint_torques(self):
        """Get current joint torques"""
        torques = {}
        for name, idx in self.joint_indices.items():
            torques[name] = p.getJointState(self.robot_id, idx)[3]
        return torques
    
    def generate_figure8_trajectory(self, x_min, x_max, z_min, z_max, num_points):
        """Generate figure-8 trajectory points"""
        A = (x_max - x_min) / 2
        B = (z_max - z_min) / 2
        x_center = (x_max + x_min) / 2
        z_center = (z_max + z_min) / 2
        
        t = np.linspace(0, 2*np.pi, num_points)
        x = A * np.sin(t) + x_center
        z = B * np.sin(2*t) + z_center
        
        return x, z
    
    def track_trajectory(self, x_min, x_max, z_min, z_max, duration=10.0):
        """Track the figure-8 trajectory"""
        # Generate trajectory points
        num_points = int(duration * 240)  # 240Hz control rate
        x_traj, z_traj = self.generate_figure8_trajectory(x_min, x_max, z_min, z_max, num_points)
        
        dt = 1.0/240.0  # Time step
        start_time = time.time()
        
        try:
            for i in range(num_points):
                current_time = time.time() - start_time
                
                # Get target position
                target_pos = [x_traj[i], 0, z_traj[i]]
                
                # Compute inverse kinematics
                joint_poses = p.calculateInverseKinematics(
                    self.robot_id,
                    self.gripper_joint_index,
                    target_pos,
                    maxNumIterations=100,
                    residualThreshold=1e-5
                )
                
                # Apply joint positions (only to the three actuated joints)
                for joint_name, joint_idx in self.joint_indices.items():
                    # Map our joint index to the IK solution index (subtract 2 since our first actuated joint starts at index 2)
                    ik_index = joint_idx - 2
                    p.setJointMotorControl2(
                        self.robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=joint_poses[ik_index],
                        force=100
                    )
                
                # Step simulation
                p.stepSimulation()
                
                # Get actual end effector position
                actual_pos = self.get_end_effector_state()
                
                # Get joint torques
                torques = self.get_joint_torques()
                
                # Store data for plotting
                self.time_data.append(current_time)
                self.actual_x.append(actual_pos[0])
                self.actual_z.append(actual_pos[1])
                self.desired_x.append(x_traj[i])
                self.desired_z.append(z_traj[i])
                for name, torque in torques.items():
                    self.joint_torques[name].append(torque)
                
                # Update plots
                if i % 10 == 0:  # Update every 10 steps for better performance
                    self.update_plots()
                
                # Maintain timing
                elapsed = time.time() - start_time
                if elapsed < (i+1)*dt:
                    time.sleep((i+1)*dt - elapsed)
        
        except KeyboardInterrupt:
            print("\nTrajectory tracking interrupted!")
        
        # Final plot update
        self.update_plots()
        plt.ioff()
        plt.show()

def main():
    # Spawn robot
    robot_id, physics_client = spawn_robot()
    
    # Create tracker
    tracker = Figure8Tracker(robot_id)
    
    # Define workspace bounds (adjust these based on robot's workspace)
    x_min, x_max = 0.1, 0.3
    z_min, z_max = 0.3, 0.5
    
    # Track trajectory
    print("Starting figure-8 trajectory tracking...")
    print("Press Ctrl+C to stop")
    tracker.track_trajectory(x_min, x_max, z_min, z_max)
    
    # Disconnect
    p.disconnect()

if __name__ == "__main__":
    main() 