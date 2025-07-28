import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import time
import os
import csv
import signal
import sys

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

    # Enable force/torque sensors and set control parameters for the actuated joints
    actuated_joints = [2, 3, 4]  # link_1_to_link_2, link_2_to_link_3, link_3_to_link_4
    for joint_idx in actuated_joints:
        p.enableJointForceTorqueSensor(robot, joint_idx, True)
        # Set joint control parameters
        p.changeDynamics(robot, joint_idx,
                        jointDamping=0.5,  # Reduced damping
                        maxJointVelocity=20.0)  # Increased max velocity
        # Set initial joint positions to a reasonable starting pose
        p.resetJointState(robot, joint_idx, targetValue=0.0)
        
    # Set camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=0,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0]
    )
    
    return robot, physicsClient

class CircularTrajectoryGUI:
    def __init__(self):
        # Set up signal handler for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Circular Trajectory Control")
        
        # Add circle counter
        self.circle_count = 0
        self.WARMUP_CIRCLES = 5  # Number of circles to run before collecting data
        
        # Initialize data collection variables first
        self.data_collection_started = False
        self.circle_completed = False
        self.phase = 0
        self.start_time = None
        self.settling_start_time = None
        self.physics_client = None
        
        # Create data directory if it doesn't exist
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data storage
        self.time_data = []
        self.actual_x = []
        self.actual_z = []
        self.desired_x = []
        self.desired_z = []
        self.error_x = []
        self.error_z = []
        self.joint_angles = {joint: [] for joint in ['link_1_to_link_2', 'link_2_to_link_3', 'link_3_to_link_4']}
        self.joint_torques = {joint: [] for joint in ['link_1_to_link_2', 'link_2_to_link_3', 'link_3_to_link_4']}
        
        # Spawn robot
        self.robot_id, self.physics_client = spawn_robot()
        
        # Get joint indices
        self.joint_indices = {
            'link_1_to_link_2': 2,
            'link_2_to_link_3': 3,
            'link_3_to_link_4': 4
        }
        self.gripper_joint_index = 5  # link_4_to_gripper is the end effector
        
        # Error settling threshold
        self.error_threshold = 0.05  # Increased threshold to 5cm
        self.settling_count = 0
        self.required_settling_count = 20  # Reduced required count
        self.settling_timeout = 5.0  # 5 seconds timeout for settling
        
        # Create control frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create input fields for circle parameters
        ttk.Label(control_frame, text="Circle Center:").grid(row=0, column=0, columnspan=2)
        ttk.Label(control_frame, text="X:").grid(row=1, column=0)
        ttk.Label(control_frame, text="Z:").grid(row=2, column=0)
        self.center_x = ttk.Entry(control_frame, width=10)
        self.center_z = ttk.Entry(control_frame, width=10)
        self.center_x.grid(row=1, column=1)
        self.center_z.grid(row=2, column=1)
        self.center_x.insert(0, "0.3")  # Default X center
        self.center_z.insert(0, "0.3")  # Default Z center
        
        ttk.Label(control_frame, text="Radius:").grid(row=3, column=0)
        self.radius = ttk.Entry(control_frame, width=10)
        self.radius.grid(row=3, column=1)
        self.radius.insert(0, "0.1")  # Default radius
        
        # Create speed control
        ttk.Label(control_frame, text="Speed:").grid(row=4, column=0)
        self.speed = ttk.Scale(control_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL)
        self.speed.grid(row=4, column=1)
        self.speed.set(1.0)  # Default speed
        
        # Create control buttons
        ttk.Button(control_frame, text="Start Motion", command=self.start_circular_motion).grid(row=5, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Stop Motion", command=self.stop_motion).grid(row=6, column=0, columnspan=2, pady=5)
        
        # Create status frame for live data
        status_frame = ttk.LabelFrame(control_frame, text="Current Status", padding="10")
        status_frame.grid(row=7, column=0, columnspan=2, pady=5, sticky="nsew")
        
        # Position labels
        ttk.Label(status_frame, text="Current Position:").grid(row=0, column=0, columnspan=2)
        ttk.Label(status_frame, text="X:").grid(row=1, column=0)
        ttk.Label(status_frame, text="Z:").grid(row=2, column=0)
        self.current_x_label = ttk.Label(status_frame, text="0.0")
        self.current_z_label = ttk.Label(status_frame, text="0.0")
        self.current_x_label.grid(row=1, column=1)
        self.current_z_label.grid(row=2, column=1)
        
        # Error labels
        ttk.Label(status_frame, text="Current Error:").grid(row=3, column=0, columnspan=2)
        ttk.Label(status_frame, text="X Error:").grid(row=4, column=0)
        ttk.Label(status_frame, text="Z Error:").grid(row=5, column=0)
        self.error_x_label = ttk.Label(status_frame, text="0.0")
        self.error_z_label = ttk.Label(status_frame, text="0.0")
        self.error_x_label.grid(row=4, column=1)
        self.error_z_label.grid(row=5, column=1)
        
        # Add status message label
        ttk.Label(status_frame, text="Status:").grid(row=6, column=0, columnspan=2)
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.grid(row=7, column=0, columnspan=2)

        # Create matplotlib figures
        self.setup_plots()
        
        # Motion control variables
        self.is_running = False
    
    def setup_plots(self):
        """Setup matplotlib plots"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(12, 8))
        
        # Trajectory plot
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title('End Effector Trajectory')
        self.ax1.set_xlabel('X Position (m)')
        self.ax1.set_ylabel('Z Position (m)')
        self.ax1.grid(True)
        self.ax1.set_aspect('equal')
        
        # Initialize plot lines for trajectory
        self.actual_line, = self.ax1.plot([], [], 'b-', label='Actual', linewidth=2)
        self.desired_line, = self.ax1.plot([], [], 'r--', label='Desired', linewidth=2)
        self.current_point, = self.ax1.plot([], [], 'rx', markersize=10)
        self.ax1.legend()
        
        # Set fixed limits for trajectory plot
        self.ax1.set_xlim(0.1, 0.5)  # Adjust based on workspace
        self.ax1.set_ylim(0.1, 0.5)  # Adjust based on workspace
        
        # Error plot
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title('Position Error')
        self.ax2.set_xlabel('Time (s)')
        self.ax2.set_ylabel('Error (m)')
        self.ax2.grid(True)
        
        # Initialize plot lines for error
        self.error_x_line, = self.ax2.plot([], [], 'r-', label='X Error', linewidth=2)
        self.error_z_line, = self.ax2.plot([], [], 'b-', label='Z Error', linewidth=2)
        self.ax2.legend()
        
        # Set fixed limits for error plot
        self.ax2.set_ylim(-0.1, 0.1)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1)
        
        plt.tight_layout()
    
    def update_plots(self):
        """Update all plots"""
        # For trajectory plot, only show current iteration
        if self.data_collection_started and not self.circle_completed:
            # Update actual trajectory line
            self.actual_line.set_data(self.actual_x, self.actual_z)
            
            # Update desired trajectory (show full circle)
            theta = np.linspace(0, 2*np.pi, 100)
            desired_x = self.center_x_val + self.radius_val * np.cos(theta)
            desired_z = self.center_z_val + self.radius_val * np.sin(theta)
            self.desired_line.set_data(desired_x, desired_z)
            
            # Update current point
            if len(self.actual_x) > 0:
                self.current_point.set_data([self.actual_x[-1]], [self.actual_z[-1]])
        else:
            # Clear the plots when not collecting data
            self.actual_line.set_data([], [])
            self.desired_line.set_data([], [])
            self.current_point.set_data([], [])
        
        # Update error plot
        if len(self.time_data) > 0:
            # Only show last 5 seconds of error data
            window = 5  # seconds
            if self.time_data[-1] > window:
                start_idx = next(i for i, t in enumerate(self.time_data) if t > self.time_data[-1] - window)
            else:
                start_idx = 0
                
            self.error_x_line.set_data(
                self.time_data[start_idx:],
                self.error_x[start_idx:]
            )
            self.error_z_line.set_data(
                self.time_data[start_idx:],
                self.error_z[start_idx:]
            )
            
            # Update time axis limits to show moving window
            self.ax2.set_xlim(
                max(0, self.time_data[-1] - window),
                max(window, self.time_data[-1])
            )
        
        self.canvas.draw()
    
    def get_end_effector_state(self):
        """Get current end effector position"""
        state = p.getLinkState(self.robot_id, self.gripper_joint_index)
        return np.array([state[0][0], state[0][2]])  # x, z coordinates
    
    def stop_motion(self):
        """Stop the circular motion"""
        self.is_running = False
    
    def update_status(self, message):
        """Update status message and print to console"""
        print(f"Status: {message}")
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\nReceived Ctrl+C. Cleaning up...")
        self.cleanup()
        sys.exit(0)
        
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'physics_client') and self.physics_client is not None:
                p.disconnect(self.physics_client)
                self.physics_client = None
        except Exception as e:
            print(f"Error during PyBullet cleanup: {str(e)}")
            
        try:
            if hasattr(self, 'root'):
                self.root.quit()
        except Exception as e:
            print(f"Error during Tkinter cleanup: {str(e)}")
            
    def save_data_to_csv(self):
        """Save collected data to CSV file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.data_dir, f"robot_data_{timestamp}.csv")
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                header = ['Time', 'Desired_X', 'Desired_Z', 'Actual_X', 'Actual_Z', 'Error_X', 'Error_Z']
                # Add joint angle headers
                header.extend([f'Angle_{joint}' for joint in self.joint_angles.keys()])
                # Add joint torque headers
                header.extend([f'Torque_{joint}' for joint in self.joint_torques.keys()])
                writer.writerow(header)
                
                # Write data
                for i in range(len(self.time_data)):
                    row = [
                        self.time_data[i],
                        self.desired_x[i],
                        self.desired_z[i],
                        self.actual_x[i],
                        self.actual_z[i],
                        self.error_x[i],
                        self.error_z[i]
                    ]
                    # Add joint angles
                    for joint in self.joint_angles.keys():
                        row.append(self.joint_angles[joint][i])
                    # Add joint torques
                    for joint in self.joint_torques.keys():
                        row.append(self.joint_torques[joint][i])
                    writer.writerow(row)
            
            print(f"\nData saved to: {filename}")
            return True
        except Exception as e:
            print(f"\nError saving data: {str(e)}")
            return False
            
    def start_circular_motion(self):
        """Start the circular motion with given parameters"""
        try:
            # Get circle parameters
            self.center_x_val = float(self.center_x.get())
            self.center_z_val = float(self.center_z.get())
            self.radius_val = float(self.radius.get())
            
            # Reset counters and flags
            self.circle_count = 0
            self.data_collection_started = False
            self.circle_completed = False
            self.phase = 0
            self.start_time = None
            
            # Start motion
            self.is_running = True
            self.update_status("Starting warmup circles...")
            self.execute_circular_motion()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid input! Please enter valid numbers.")
            
    def check_error_settled(self, x_error, z_error):
        """Check if position error has settled below threshold"""
        total_error = np.sqrt(x_error**2 + z_error**2)
        current_time = time.time()
        
        if self.settling_start_time is not None and (current_time - self.settling_start_time) > self.settling_timeout:
            self.update_status("Settling timeout reached, starting motion anyway...")
            return True
            
        if total_error < self.error_threshold:
            self.settling_count += 1
            self.update_status(f"Settling... ({self.settling_count}/{self.required_settling_count})")
        else:
            self.settling_count = 0
            self.update_status(f"Waiting to settle... (error: {total_error:.3f}m)")
            
        return self.settling_count >= self.required_settling_count
        
    def execute_circular_motion(self):
        """Execute the circular motion"""
        if not self.is_running:
            return
            
        try:
            # Get current speed setting
            speed = self.speed.get()
            
            # Calculate current target point
            x = self.center_x_val + self.radius_val * np.cos(self.phase)
            z = self.center_z_val + self.radius_val * np.sin(self.phase)
            
            # Get actual end effector position
            actual_pos = self.get_end_effector_state()
            x_error = x - actual_pos[0]
            z_error = z - actual_pos[1]
            
            # Update status labels
            self.current_x_label.config(text=f"{actual_pos[0]:.3f}")
            self.current_z_label.config(text=f"{actual_pos[1]:.3f}")
            self.error_x_label.config(text=f"{x_error:.3f}")
            self.error_z_label.config(text=f"{z_error:.3f}")
            
            # Calculate inverse kinematics with orientation constraint
            target_pos = [x, 0, z]
            target_orn = p.getQuaternionFromEuler([0, 0, 0])  # Keep end effector level
            joint_poses = p.calculateInverseKinematics(
                self.robot_id,
                self.gripper_joint_index,
                target_pos,
                target_orn,
                maxNumIterations=100,
                residualThreshold=1e-5
            )
            
            # Apply joint positions with higher gains for better tracking
            for joint_name, joint_idx in self.joint_indices.items():
                ik_index = joint_idx - 2
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[ik_index],
                    targetVelocity=0,
                    force=100,  # Reduced force for smoother motion
                    positionGain=0.5,  # Reduced gains for smoother motion
                    velocityGain=0.5
                )
            
            # Step simulation
            p.stepSimulation()
            
            # Update phase
            self.phase += 0.05 * speed
            
            # Check if a circle is completed
            if self.phase >= 2 * np.pi:
                self.circle_count += 1
                self.phase = 0  # Reset phase for next circle
                
                if self.circle_count < self.WARMUP_CIRCLES:
                    # Still in warmup phase
                    self.update_status(f"Warmup circle {self.circle_count}/{self.WARMUP_CIRCLES} completed")
                    
                elif self.circle_count == self.WARMUP_CIRCLES:
                    # Start data collection for the next (21st) circle
                    self.data_collection_started = True
                    self.start_time = time.time()
                    # Clear any previous data
                    self.time_data = []
                    self.actual_x = []
                    self.actual_z = []
                    self.desired_x = []
                    self.desired_z = []
                    self.error_x = []
                    self.error_z = []
                    self.joint_angles = {joint: [] for joint in self.joint_angles.keys()}
                    self.joint_torques = {joint: [] for joint in self.joint_torques.keys()}
                    self.update_status("Starting data collection for circle 21...")
                    
                elif self.circle_count > self.WARMUP_CIRCLES:
                    # Data collection circle completed
                    self.circle_completed = True
                    if self.save_data_to_csv():
                        self.update_status("Circle 21 completed! Data saved successfully.")
                    else:
                        self.update_status("Circle completed but failed to save data!")
                    self.stop_motion()
            
            # Collect data if we're on the 21st circle
            if self.data_collection_started and not self.circle_completed:
                current_time = time.time() - self.start_time
                
                # Store positions and errors
                self.time_data.append(current_time)
                self.desired_x.append(x)
                self.desired_z.append(z)
                self.actual_x.append(actual_pos[0])
                self.actual_z.append(actual_pos[1])
                self.error_x.append(x_error)
                self.error_z.append(z_error)
                
                # Get and store joint states
                for joint_name, joint_idx in self.joint_indices.items():
                    joint_state = p.getJointState(self.robot_id, joint_idx)
                    self.joint_angles[joint_name].append(joint_state[0])  # Joint angle
                    self.joint_torques[joint_name].append(joint_state[3])  # Applied torque
                
                # Update status with progress for data collection circle
                progress = (self.phase / (2 * np.pi)) * 100
                self.update_status(f"Recording circle 21... {progress:.1f}% complete")
            
            # Update plots every few steps for better performance
            if len(self.time_data) % 5 == 0:
                self.update_plots()
            
            # Schedule next update if still running
            if self.is_running:
                self.root.after(20, self.execute_circular_motion)
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            self.is_running = False
            
    def run(self):
        """Start the GUI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
        finally:
            self.cleanup()

if __name__ == "__main__":
    # Create and run circular trajectory GUI
    gui = CircularTrajectoryGUI()
    print("Starting Circular Trajectory Control GUI...")
    try:
        gui.run()
    except KeyboardInterrupt:
        print("\nExiting gracefully...")
    finally:
        gui.cleanup() 