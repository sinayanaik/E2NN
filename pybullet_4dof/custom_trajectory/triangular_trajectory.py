import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import os
import csv
from datetime import datetime
from scipy.optimize import minimize
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class RealTimeVisualizer:
    def __init__(self):
        # Initialize window closed flag first
        self.window_closed = False
        
        # Initialize data processing variables
        self.prev_velocities = np.zeros(3)  # One for each joint
        self.prev_accelerations = np.zeros(3)  # One for each joint
        self.velocity_history = [[], [], []]  # History buffer for each joint
        self.dt = 1.0/240.0  # Fixed time step
        self.first_update = True
        self.update_counter = 0
        self.filter_window = 5  # Window size for moving average
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Real-time Triangular Trajectory and Dynamics Visualization")
        
        # Set up window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        
        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        # Create control frame
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(12, 12))
        
        # Create subplots with GridSpec
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        # Initialize data lists
        self.target_x = []
        self.target_z = []
        self.actual_x = []
        self.actual_z = []
        self.time_steps = []
        self.step_counter = 0
        
        # Initialize joint data lists
        self.angles_joint1 = []
        self.angles_joint2 = []
        self.angles_joint3 = []
        
        self.velocities_joint1 = []
        self.velocities_joint2 = []
        self.velocities_joint3 = []
        
        self.accelerations_joint1 = []
        self.accelerations_joint2 = []
        self.accelerations_joint3 = []
        
        self.torques_joint1 = []
        self.torques_joint2 = []
        self.torques_joint3 = []
        
        # Create subplots
        # Trajectory subplot (spans two columns)
        self.ax_traj = self.fig.add_subplot(gs[0, :])
        self.ax_traj.set_xlabel('X Position (m)')
        self.ax_traj.set_ylabel('Z Position (m)')
        self.ax_traj.grid(True)
        self.ax_traj.set_title('End-Effector Trajectory')
        self.ax_traj.set_aspect('equal')
        
        # Joint angles subplot
        self.ax_angles = self.fig.add_subplot(gs[1, 0])
        self.ax_angles.set_xlabel('Time Step')
        self.ax_angles.set_ylabel('Joint Angle (rad)')
        self.ax_angles.grid(True)
        self.ax_angles.set_title('Joint Angles')
        
        # Joint velocities subplot
        self.ax_velocities = self.fig.add_subplot(gs[1, 1])
        self.ax_velocities.set_xlabel('Time Step')
        self.ax_velocities.set_ylabel('Velocity (rad/s)')
        self.ax_velocities.grid(True)
        self.ax_velocities.set_title('Joint Velocities')
        
        # Joint accelerations subplot
        self.ax_accelerations = self.fig.add_subplot(gs[2, 0])
        self.ax_accelerations.set_xlabel('Time Step')
        self.ax_accelerations.set_ylabel('Acceleration (rad/s²)')
        self.ax_accelerations.grid(True)
        self.ax_accelerations.set_title('Joint Accelerations')
        
        # Torque subplot
        self.ax_torque = self.fig.add_subplot(gs[2, 1])
        self.ax_torque.set_xlabel('Time Step')
        self.ax_torque.set_ylabel('Torque (N⋅m)')
        self.ax_torque.grid(True)
        self.ax_torque.set_title('Joint Torques')
        
        # Create plot lines
        self.target_line, = self.ax_traj.plot([], [], 'r-', label='Target')
        self.actual_line, = self.ax_traj.plot([], [], 'b--', label='Actual')
        self.ax_traj.legend()
        
        # Create lines for joint angles, velocities, accelerations, and torques
        colors = ['r', 'g', 'b']
        self.angle_lines = [self.ax_angles.plot([], [], label=f'Joint {i+1}', color=c)[0] for i, c in enumerate(colors)]
        self.velocity_lines = [self.ax_velocities.plot([], [], label=f'Joint {i+1}', color=c)[0] for i, c in enumerate(colors)]
        self.acceleration_lines = [self.ax_accelerations.plot([], [], label=f'Joint {i+1}', color=c)[0] for i, c in enumerate(colors)]
        self.torque_lines = [self.ax_torque.plot([], [], label=f'Joint {i+1}', color=c)[0] for i, c in enumerate(colors)]
        
        # Add legends
        self.ax_angles.legend()
        self.ax_velocities.legend()
        self.ax_accelerations.legend()
        self.ax_torque.legend()
        
        # Set fixed axis limits based on workspace
        self.ax_traj.set_xlim(0, 0.5)
        self.ax_traj.set_ylim(0, 0.5)
        
        # Set initial limits for time series plots
        for ax in [self.ax_angles, self.ax_velocities, self.ax_accelerations, self.ax_torque]:
            ax.set_xlim(0, 100)
        
        self.ax_angles.set_ylim(-3.14, 3.14)
        self.ax_velocities.set_ylim(-2, 2)
        self.ax_accelerations.set_ylim(-5, 5)
        self.ax_torque.set_ylim(-50, 50)
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Add PD Control Gain Sliders
        tk.Label(self.control_frame, text="PD Control Gains", font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Kp slider
        tk.Label(self.control_frame, text="Position Gain (Kp)").pack()
        self.kp_slider = tk.Scale(self.control_frame, from_=0.01, to=1.0, resolution=0.01, 
                                orient=tk.HORIZONTAL, length=200)
        self.kp_slider.set(0.10)
        self.kp_slider.pack(pady=5)
        
        # Kd slider
        tk.Label(self.control_frame, text="Velocity Gain (Kd)").pack()
        self.kd_slider = tk.Scale(self.control_frame, from_=0.1, to=2.0, resolution=0.1, 
                                orient=tk.HORIZONTAL, length=200)
        self.kd_slider.set(0.4)
        self.kd_slider.pack(pady=5)
        
        # Add Trajectory Parameters
        tk.Label(self.control_frame, text="Trajectory Parameters", font=('Arial', 12, 'bold')).pack(pady=(20,5))
        
        # Center X slider
        tk.Label(self.control_frame, text="Center X Position (m)").pack()
        self.center_x_slider = tk.Scale(self.control_frame, from_=0.1, to=0.4, resolution=0.01,
                                      orient=tk.HORIZONTAL, length=200)
        self.center_x_slider.set(0.25)
        self.center_x_slider.pack(pady=5)
        
        # Center Z slider
        tk.Label(self.control_frame, text="Center Z Position (m)").pack()
        self.center_z_slider = tk.Scale(self.control_frame, from_=0.1, to=0.4, resolution=0.01,
                                      orient=tk.HORIZONTAL, length=200)
        self.center_z_slider.set(0.20)
        self.center_z_slider.pack(pady=5)
        
        # Radius slider
        tk.Label(self.control_frame, text="Radius (m)").pack()
        self.radius_slider = tk.Scale(self.control_frame, from_=0.05, to=0.15, resolution=0.01,
                                    orient=tk.HORIZONTAL, length=200)
        self.radius_slider.set(0.10)
        self.radius_slider.pack(pady=5)
        
        # Add current values display
        self.values_frame = tk.Frame(self.control_frame)
        self.values_frame.pack(pady=10)
        
        self.kp_label = tk.Label(self.values_frame, text="Current Kp: 0.10")
        self.kp_label.pack()
        self.kd_label = tk.Label(self.values_frame, text="Current Kd: 0.4")
        self.kd_label.pack()
        
        # Add Save Data Button and Status
        tk.Label(self.control_frame, text="Data Logging", font=('Arial', 12, 'bold')).pack(pady=(20,5))
        self.trajectory_label = tk.Label(self.control_frame, text="Triangle #: 0")
        self.trajectory_label.pack()
        
        self.save_button = tk.Button(self.control_frame, text="Save Previous Triangle Data", 
                                   command=self._save_button_clicked)
        self.save_button.pack(pady=5)
        self.save_status = tk.Label(self.control_frame, text="")
        self.save_status.pack()
        
        # Data caching
        self.current_trajectory_data = []
        self.previous_trajectory_data = []
        self.current_trajectory_number = 0
        self.save_requested = False
        
        # Bind slider updates
        self.kp_slider.configure(command=self._update_labels)
        self.kd_slider.configure(command=self._update_labels)
        
        # Update the window
        self.root.update()
    
    def _update_labels(self, _=None):
        """Update the display labels and print current values"""
        kp = self.kp_slider.get()
        kd = self.kd_slider.get()
        self.kp_label.configure(text=f"Current Kp: {kp:.2f}")
        self.kd_label.configure(text=f"Current Kd: {kd:.1f}")
        
        center_x = self.center_x_slider.get()
        center_z = self.center_z_slider.get()
        radius = self.radius_slider.get()
        print(f"\nParameters updated:")
        print(f"PD Gains: Kp={kp:.2f}, Kd={kd:.1f}")
        print(f"Trajectory: Center=({center_x:.2f}, {center_z:.2f}), Radius={radius:.2f}")
    
    def _save_button_clicked(self):
        """Handle manual save button click"""
        if self.previous_trajectory_data:
            self.save_requested = True
            self.save_status.configure(text="Save requested for previous triangle...", fg="blue")
        else:
            self.save_status.configure(text="No previous triangle data to save", fg="orange")

    def get_gains(self):
        try:
            if self.window_closed:
                return None, None
            return self.kp_slider.get(), self.kd_slider.get()
        except tk.TclError:
            self.window_closed = True
            return None, None
    
    def get_trajectory_params(self):
        try:
            if self.window_closed:
                return None, None, None
            return (
                self.center_x_slider.get(),
                self.center_z_slider.get(),
                self.radius_slider.get()
            )
        except tk.TclError:
            self.window_closed = True
            return None, None, None
    
    def cache_data_point(self, data_point):
        """Cache data point for saving"""
        try:
            timestamp = data_point[0]
            target_x, target_z = data_point[1], data_point[2]
            actual_x, actual_z = data_point[3], data_point[4]
            joint_angles = data_point[5:8]
            joint_velocities = data_point[8:11]
            joint_accelerations = [
                self.accelerations_joint1[-1] if self.accelerations_joint1 else 0.0,
                self.accelerations_joint2[-1] if self.accelerations_joint2 else 0.0,
                self.accelerations_joint3[-1] if self.accelerations_joint3 else 0.0
            ]
            joint_torques = data_point[14:17]
            kp, kd = data_point[17], data_point[18]
            radius, center_x, center_z = data_point[19], data_point[20], data_point[21]
            
            modified_data = [
                timestamp,
                target_x, target_z,
                actual_x, actual_z,
                *joint_angles,
                *joint_velocities,
                *joint_accelerations,
                *joint_torques,
                kp, kd,
                radius,
                center_x,
                center_z
            ]
            
            self.current_trajectory_data.append(modified_data)
        except Exception as e:
            print(f"Error in cache_data_point: {str(e)}")
            print("Data point:", data_point)
            import traceback
            traceback.print_exc()

    def clear_cache(self):
        """Move current data to previous and clear current"""
        if self.current_trajectory_data:
            self.previous_trajectory_data = self.current_trajectory_data.copy()
            self.current_trajectory_data = []

    def save_cached_data(self, triangle_number):
        """Save manually requested data"""
        if not self.previous_trajectory_data:
            self.save_status.configure(text="No previous triangle data to save", fg="orange")
            return False
            
        kp, kd = self.get_gains()
        center_x, center_z, radius = self.get_trajectory_params()
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"triangular_trajectory_kp{kp:.2f}_kd{kd:.1f}_r{radius:.2f}_x{center_x:.2f}_z{center_z:.2f}_triangle{triangle_number}_{timestamp_str}.csv"
        
        data_folder = 'data'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        filepath = os.path.join(data_folder, filename)

        header = [
            'timestamp', 'target_x', 'target_z', 'actual_x', 'actual_z',
            'joint1_angle', 'joint2_angle', 'joint3_angle',
            'joint1_velocity', 'joint2_velocity', 'joint3_velocity',
            'joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration',
            'joint1_torque', 'joint2_torque', 'joint3_torque',
            'kp', 'kd', 'radius', 'center_x', 'center_z'
        ]
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self.previous_trajectory_data)
            
            self.save_status.configure(text=f"Manually saved triangle #{triangle_number-1}", fg="green")
            print(f"Manually saved data for triangle #{triangle_number-1} to {filename}")
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            self.save_status.configure(text="Error saving data", fg="red")
            return False

    def calculate_filtered_acceleration(self, current_velocity, joint_idx):
        """Calculate filtered acceleration using moving average of velocities"""
        try:
            self.velocity_history[joint_idx].append(current_velocity)
            
            if len(self.velocity_history[joint_idx]) > self.filter_window:
                self.velocity_history[joint_idx].pop(0)
            
            if len(self.velocity_history[joint_idx]) < 2:
                return 0.0
            
            smoothed_vel_current = np.mean(self.velocity_history[joint_idx][-2:])
            smoothed_vel_prev = np.mean(self.velocity_history[joint_idx][:-1])
            
            acceleration = (smoothed_vel_current - smoothed_vel_prev) / self.dt
            
            filtered_acc = 0.2 * acceleration + 0.8 * self.prev_accelerations[joint_idx]
            
            max_acc = 5.0
            filtered_acc = np.clip(filtered_acc, -max_acc, max_acc)
            
            return filtered_acc
            
        except Exception as e:
            print(f"Error in acceleration calculation: {str(e)}")
            return 0.0

    def update(self, target_x, target_z, actual_x, actual_z, joint_states, robot_id, actuated_joint_indices):
        """Update the visualization with PyBullet joint states"""
        if self.is_closed():
            return False
            
        try:
            self.target_x.append(target_x)
            self.target_z.append(target_z)
            self.actual_x.append(actual_x)
            self.actual_z.append(actual_z)
            
            self.step_counter += 1
            self.time_steps.append(self.step_counter)
            
            angles = []
            velocities = []
            torques = []
            accelerations = []
            
            for i, joint_index in enumerate(actuated_joint_indices):
                state = p.getJointState(robot_id, joint_index)
                angle = state[0]
                velocity = state[1]
                torque = state[3]
                
                if self.first_update:
                    acceleration = 0.0
                else:
                    acceleration = self.calculate_filtered_acceleration(velocity, i)
                
                self.prev_velocities[i] = velocity
                self.prev_accelerations[i] = acceleration
                
                angles.append(angle)
                velocities.append(velocity)
                torques.append(torque)
                accelerations.append(acceleration)
            
            if self.first_update:
                self.first_update = False
            
            self.angles_joint1.append(angles[0])
            self.angles_joint2.append(angles[1])
            self.angles_joint3.append(angles[2])
            
            self.velocities_joint1.append(velocities[0])
            self.velocities_joint2.append(velocities[1])
            self.velocities_joint3.append(velocities[2])
            
            self.accelerations_joint1.append(accelerations[0])
            self.accelerations_joint2.append(accelerations[1])
            self.accelerations_joint3.append(accelerations[2])
            
            self.torques_joint1.append(torques[0])
            self.torques_joint2.append(torques[1])
            self.torques_joint3.append(torques[2])
            
            if len(self.time_steps) > 100:
                self.time_steps = self.time_steps[-100:]
                self.target_x = self.target_x[-100:]
                self.target_z = self.target_z[-100:]
                self.actual_x = self.actual_x[-100:]
                self.actual_z = self.actual_z[-100:]
                
                self.angles_joint1 = self.angles_joint1[-100:]
                self.angles_joint2 = self.angles_joint2[-100:]
                self.angles_joint3 = self.angles_joint3[-100:]
                
                self.velocities_joint1 = self.velocities_joint1[-100:]
                self.velocities_joint2 = self.velocities_joint2[-100:]
                self.velocities_joint3 = self.velocities_joint3[-100:]
                
                self.accelerations_joint1 = self.accelerations_joint1[-100:]
                self.accelerations_joint2 = self.accelerations_joint2[-100:]
                self.accelerations_joint3 = self.accelerations_joint3[-100:]
                
                self.torques_joint1 = self.torques_joint1[-100:]
                self.torques_joint2 = self.torques_joint2[-100:]
                self.torques_joint3 = self.torques_joint3[-100:]
            
            self.update_counter += 1
            if self.update_counter % 5 == 0:
                self.target_line.set_data(self.target_x, self.target_z)
                self.actual_line.set_data(self.actual_x, self.actual_z)
                
                for i, line in enumerate(self.angle_lines):
                    line.set_data(self.time_steps, [self.angles_joint1, self.angles_joint2, self.angles_joint3][i])
                
                for i, line in enumerate(self.velocity_lines):
                    line.set_data(self.time_steps, [self.velocities_joint1, self.velocities_joint2, self.velocities_joint3][i])
                
                for i, line in enumerate(self.acceleration_lines):
                    line.set_data(self.time_steps, [self.accelerations_joint1, self.accelerations_joint2, self.accelerations_joint3][i])
                
                for i, line in enumerate(self.torque_lines):
                    line.set_data(self.time_steps, [self.torques_joint1, self.torques_joint2, self.torques_joint3][i])
                
                for ax in [self.ax_angles, self.ax_velocities, self.ax_accelerations, self.ax_torque]:
                    ax.set_xlim(self.step_counter - 100, self.step_counter)
                
                self._adjust_plot_limits()
                
                self.canvas.draw()
            
            if self.update_counter % 10 == 0:
                if hasattr(self, 'root') and self.root.winfo_exists():
                    self.root.update()
            
            return True
            
        except Exception as e:
            print(f"Error in visualization update: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _adjust_plot_limits(self):
        """Adjust y-axis limits of plots based on data"""
        try:
            def get_limits_with_margin(data_list, margin_factor=0.1, min_range=0.1):
                if not data_list or not any(data_list):
                    return -1, 1
                
                min_val = min(min(d) for d in data_list if d)
                max_val = max(max(d) for d in data_list if d)
                
                if max_val - min_val < min_range:
                    center = (max_val + min_val) / 2
                    min_val = center - min_range/2
                    max_val = center + min_range/2
                
                margin = max((max_val - min_val) * margin_factor, min_range * 0.1)
                return min_val - margin, max_val + margin
            
            angle_data = [self.angles_joint1, self.angles_joint2, self.angles_joint3]
            if any(angle_data):
                self.ax_angles.set_ylim(*get_limits_with_margin(angle_data, min_range=0.5))
            
            vel_data = [self.velocities_joint1, self.velocities_joint2, self.velocities_joint3]
            if any(vel_data):
                self.ax_velocities.set_ylim(*get_limits_with_margin(vel_data, min_range=0.2))
            
            acc_data = [self.accelerations_joint1, self.accelerations_joint2, self.accelerations_joint3]
            if any(acc_data):
                self.ax_accelerations.set_ylim(*get_limits_with_margin(acc_data, min_range=1.0))
            
            torque_data = [self.torques_joint1, self.torques_joint2, self.torques_joint3]
            if any(torque_data):
                self.ax_torque.set_ylim(*get_limits_with_margin(torque_data, min_range=5.0))
        
        except Exception as e:
            print(f"Error in _adjust_plot_limits: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_triangle_number(self, number):
        """Update triangle number"""
        self.current_trajectory_number = number
        self.trajectory_label.configure(text=f"Triangle #: {number}")
        
    def clear_data(self):
        self.target_x = []
        self.target_z = []
        self.actual_x = []
        self.actual_z = []
        self.target_line.set_data([], [])
        self.actual_line.set_data([], [])
        
        self.time_steps = []
        self.step_counter = 0
        
        self.angles_joint1 = []
        self.angles_joint2 = []
        self.angles_joint3 = []
        
        self.velocities_joint1 = []
        self.velocities_joint2 = []
        self.velocities_joint3 = []
        
        self.accelerations_joint1 = []
        self.accelerations_joint2 = []
        self.accelerations_joint3 = []
        
        self.torques_joint1 = []
        self.torques_joint2 = []
        self.torques_joint3 = []
        
        self.prev_velocities = np.zeros(3)
        self.prev_accelerations = np.zeros(3)
        self.velocity_history = [[], [], []]
        self.first_update = True
        
        for line in self.angle_lines: line.set_data([], [])
        for line in self.velocity_lines: line.set_data([], [])
        for line in self.acceleration_lines: line.set_data([], [])
        for line in self.torque_lines: line.set_data([], [])
        
        self.canvas.draw()
        
    def close(self):
        """Safely close the visualization window"""
        try:
            self.window_closed = True
            if hasattr(self, 'root') and self.root:
                self.root.quit()
                self.root.destroy()
        except:
            pass

    def is_closed(self):
        """Check if the window has been closed"""
        try:
            if not hasattr(self, 'root') or not self.root.winfo_exists():
                return True
            self.root.update()
            return self.window_closed
        except:
            return True

def setup_simulation(urdf_path):
    """
    Sets up the PyBullet simulation environment.
    """
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    
    p.loadURDF("plane.urdf")
    
    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    
    robot_id = p.loadURDF(urdf_path, startPos, startOrientation, useFixedBase=True)
    
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=0,
        cameraPitch=0,
        cameraTargetPosition=[0.25, 0, 0.25]
    )
    
    return robot_id

def get_joint_info(robot_id):
    """
    Retrieves information about the robot's joints.
    """
    actuated_joint_indices = []
    joint_names = []
    
    num_joints = p.getNumJoints(robot_id)
    end_effector_link_index = -1

    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        
        if joint_name in ["link_1_to_link_2", "link_2_to_link_3", "link_3_to_link_4"]:
            actuated_joint_indices.append(i)
            joint_names.append(joint_name)
            p.enableJointForceTorqueSensor(robot_id, i, 1)

        if joint_name == "link_4_to_gripper":
            end_effector_link_index = i

    print("Actuated Joints Found:")
    for i, name in zip(actuated_joint_indices, joint_names):
        print(f"- {name} (Index: {i})")
    print(f"End-Effector Link Index: {end_effector_link_index}\n")

    return actuated_joint_indices, end_effector_link_index

def generate_triangular_trajectory(cx, cz, radius, num_points):
    """
    Generates a list of (x, z) points for a triangular trajectory inscribed in a circle.
    """
    trajectory = []
    
    # Vertices of an equilateral triangle pointing up, inscribed in a circle
    angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
    vertices = [np.array([cx + radius * np.cos(angle), cz + radius * np.sin(angle)]) for angle in angles]
    
    # Close the loop
    vertices.append(vertices[0])

    points_per_side = num_points // 3
    
    for i in range(3):
        start_point = vertices[i]
        end_point = vertices[i+1]
        
        num_side_points = points_per_side
        if i == 2: # last side gets remaining points
            num_side_points = num_points - 2 * points_per_side

        for j in range(num_side_points):
            t = j / (num_side_points - 1) if num_side_points > 1 else 0
            point = start_point * (1 - t) + end_point * t
            trajectory.append(tuple(point))
            
    return trajectory

def custom_inverse_kinematics(target_x, target_z, link_lengths):
    """
    Custom inverse kinematics solver based on the simplified mathematical model.
    """
    l1 = link_lengths['l1']
    l2 = link_lengths['l2']
    l3 = link_lengths['l3']
    l4 = link_lengths['l4']

    def forward_kinematics_for_ik(theta):
        x = l2 * np.sin(theta[0]) + l3 * np.sin(theta[0] + theta[1]) + l4 * np.sin(theta[0] + theta[1] + theta[2])
        z = l1 + l2 * np.cos(theta[0]) + l3 * np.cos(theta[0] + theta[1]) + l4 * np.cos(theta[0] + theta[1] + theta[2])
        return x, z

    def objective(theta):
        end_x, end_z = forward_kinematics_for_ik(theta)
        return np.sqrt((end_x - target_x)**2 + (end_z - target_z)**2)
    
    theta0 = [0, 0, 0]
    bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    
    result = minimize(objective, theta0, method='SLSQP', bounds=bounds)
    
    if result.success and result.fun < 1e-3:
        return result.x
    else:
        return None

def main():
    """
    Main function to run the simulation and track the triangular trajectory.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_file = os.path.join(current_dir, "..", "urdf", "robot.urdf")
        
        link_lengths = {'l1': 0.27, 'l2': 0.15, 'l3': 0.15, 'l4': 0.10}

        if not os.path.exists(urdf_file):
            print(f"Error: '{urdf_file}' not found.")
            return

        robot_id = setup_simulation(urdf_file)
        actuated_joint_indices, ee_link_index = get_joint_info(robot_id)

        visualizer = RealTimeVisualizer()
        
        center_x, center_z, radius = visualizer.get_trajectory_params()
        num_trajectory_points = 1000
        
        trajectory_points = generate_triangular_trajectory(center_x, center_z, radius, num_trajectory_points)
        
        marker_radius = 0.0165
        marker_color = [0, 1, 0, 1]
        
        marker_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=marker_radius,
            rgbaColor=marker_color
        )
        
        gripper_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=marker_visual,
            basePosition=[0, 0, 0]
        )
        
        # Draw initial trajectory
        p.removeAllUserDebugItems()
        for i in range(len(trajectory_points) - 1):
            p1 = [trajectory_points[i][0], 0, trajectory_points[i][1]]
            p2 = [trajectory_points[i + 1][0], 0, trajectory_points[i + 1][1]]
            p.addUserDebugLine(p1, p2, lineColorRGB=[1, 0, 0], lineWidth=2)
        
        prev_params = (center_x, center_z, radius)
        triangle_count = 0
        last_time = time.time()
        dt = 1.0/240.0
        
        print("Starting simulation. Use sliders to adjust parameters.")
        print("Click 'Save Previous Triangle Data' to save data.")
        print("Press Ctrl+C or close the window to stop.")

        while True:
            try:
                if visualizer.is_closed():
                    break

                current_time = time.time()
                if current_time - last_time < dt:
                    continue
                last_time = current_time

                current_params = visualizer.get_trajectory_params()
                if current_params[0] is None:
                    break
                
                if current_params != prev_params:
                    center_x, center_z, radius = current_params
                    trajectory_points = generate_triangular_trajectory(center_x, center_z, radius, num_trajectory_points)
                    prev_params = current_params
                    
                    p.removeAllUserDebugItems()
                    for i in range(len(trajectory_points) - 1):
                        p1 = [trajectory_points[i][0], 0, trajectory_points[i][1]]
                        p2 = [trajectory_points[i + 1][0], 0, trajectory_points[i + 1][1]]
                        p.addUserDebugLine(p1, p2, lineColorRGB=[1, 0, 0], lineWidth=2)
                
                visualizer.clear_data()
                visualizer.clear_cache()
                
                for i, (target_x, target_z) in enumerate(trajectory_points):
                    if visualizer.is_closed():
                        break
                        
                    kp, kd = visualizer.get_gains()
                    if kp is None:
                        break
                    
                    joint_poses = custom_inverse_kinematics(target_x, target_z, link_lengths)

                    if joint_poses is not None:
                        for j, joint_index in enumerate(actuated_joint_indices):
                            p.setJointMotorControl2(
                                bodyUniqueId=robot_id,
                                jointIndex=joint_index,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_poses[j],
                                force=50,
                                positionGain=kp,
                                velocityGain=kd
                            )

                    for _ in range(4):
                        p.stepSimulation()
                        time.sleep(dt/4.0)

                    # Update marker position
                    link_state = p.getLinkState(robot_id, ee_link_index)
                    p.resetBasePositionAndOrientation(gripper_marker, link_state[0], link_state[1])

                    actual_pos = link_state[0]
                    joint_states = [p.getJointState(robot_id, idx) for idx in actuated_joint_indices]
                    
                    if not visualizer.update(target_x, target_z, actual_pos[0], actual_pos[2], joint_states, robot_id, actuated_joint_indices):
                        break

                    timestamp = time.time()
                    joint_states_for_data = [p.getJointState(robot_id, idx) for idx in actuated_joint_indices]
                    
                    kp, kd = visualizer.get_gains()
                    if kp is None:
                        break
                    
                    current_center_x, current_center_z, current_radius = visualizer.get_trajectory_params()
                    if current_center_x is None:
                        break
                    
                    data_point = [
                        timestamp,
                        target_x, target_z,
                        actual_pos[0], actual_pos[2],
                        joint_states_for_data[0][0],
                        joint_states_for_data[1][0],
                        joint_states_for_data[2][0],
                        joint_states_for_data[0][1],
                        joint_states_for_data[1][1],
                        joint_states_for_data[2][1],
                        0.0, 0.0, 0.0,
                        joint_states_for_data[0][3],
                        joint_states_for_data[1][3],
                        joint_states_for_data[2][3],
                        kp, kd,
                        current_radius,
                        current_center_x,
                        current_center_z
                    ]
                    visualizer.cache_data_point(data_point)

                    if i == num_trajectory_points - 1:
                        triangle_count += 1
                        visualizer.update_triangle_number(triangle_count)
                        print(f"Triangle {triangle_count} completed. Gains: Kp={kp:.2f}, Kd={kd:.1f}, "
                              f"Center: ({current_center_x:.2f}, {current_center_z:.2f}), Radius: {current_radius:.2f}")
                        
                        visualizer.clear_cache()
                        
                        if visualizer.save_requested:
                            visualizer.save_cached_data(triangle_count)
                            visualizer.save_requested = False
                            
            except Exception as e:
                print(f"Error in simulation loop: {str(e)}")
                import traceback
                traceback.print_exc()
                break

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            visualizer.close()
        except:
            pass
        try:
            p.disconnect()
        except:
            pass

if __name__ == "__main__":
    main() 