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
        self.root.title("Real-time Trajectory and Dynamics Visualization")
        
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
        
        # Initialize previous velocities for acceleration calculation
        # self.prev_velocities = np.zeros(3)  # One for each joint
        # self.prev_accelerations = np.zeros(3) # One for each joint
        # self.dt = 1.0/240.0  # Fixed time step
        # self.first_update = True
        # self.update_counter = 0
        
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
        self.kp_slider.set(0.10)  # Updated to match successful parameters
        self.kp_slider.pack(pady=5)
        
        # Kd slider
        tk.Label(self.control_frame, text="Velocity Gain (Kd)").pack()
        self.kd_slider = tk.Scale(self.control_frame, from_=0.1, to=2.0, resolution=0.1, 
                                orient=tk.HORIZONTAL, length=200)
        self.kd_slider.set(0.4)  # Updated to match successful parameters
        self.kd_slider.pack(pady=5)
        
        # Add Trajectory Parameters
        tk.Label(self.control_frame, text="Trajectory Parameters", font=('Arial', 12, 'bold')).pack(pady=(20,5))
        
        # Center X slider
        tk.Label(self.control_frame, text="Center X Position (m)").pack()
        self.center_x_slider = tk.Scale(self.control_frame, from_=0.1, to=0.4, resolution=0.01,
                                      orient=tk.HORIZONTAL, length=200)
        self.center_x_slider.set(0.25)  # Updated to match successful parameters
        self.center_x_slider.pack(pady=5)
        
        # Center Z slider
        tk.Label(self.control_frame, text="Center Z Position (m)").pack()
        self.center_z_slider = tk.Scale(self.control_frame, from_=0.1, to=0.4, resolution=0.01,
                                      orient=tk.HORIZONTAL, length=200)
        self.center_z_slider.set(0.10)  # Updated to match successful parameters
        self.center_z_slider.pack(pady=5)
        
        # Radius slider
        tk.Label(self.control_frame, text="Radius (m)").pack()
        self.radius_slider = tk.Scale(self.control_frame, from_=0.05, to=0.15, resolution=0.01,
                                    orient=tk.HORIZONTAL, length=200)
        self.radius_slider.set(0.10)  # Updated to match successful parameters
        self.radius_slider.pack(pady=5)
        
        # Add current values display
        self.values_frame = tk.Frame(self.control_frame)
        self.values_frame.pack(pady=10)
        
        self.kp_label = tk.Label(self.values_frame, text="Current Kp: 0.50")
        self.kp_label.pack()
        self.kd_label = tk.Label(self.values_frame, text="Current Kd: 0.5")
        self.kd_label.pack()
        
        # Add Save Data Button and Status
        tk.Label(self.control_frame, text="Data Logging", font=('Arial', 12, 'bold')).pack(pady=(20,5))
        self.circle_label = tk.Label(self.control_frame, text="Circle #: 0")
        self.circle_label.pack()
        self.save_button = tk.Button(self.control_frame, text="Save Current Circle Data", 
                                   command=self._save_button_clicked)
        self.save_button.pack(pady=5)
        self.save_status = tk.Label(self.control_frame, text="")
        self.save_status.pack()
        
        # Data caching
        self.current_circle_data = []
        self.previous_circle_data = []  # Store previous circle data until saved
        self.current_circle_number = 0
        self.save_requested = False
        self.auto_save_enabled = False  # Default auto-save to off
        
        # Add Auto-save Toggle
        tk.Label(self.control_frame, text="Auto Save", font=('Arial', 12, 'bold')).pack(pady=(20,5))
        self.auto_save_var = tk.BooleanVar(value=False)  # Initialize checkbox to unchecked
        self.auto_save_check = tk.Checkbutton(self.control_frame, text="Enable Auto Save",
                                             variable=self.auto_save_var,
                                             command=self._toggle_auto_save)
        self.auto_save_check.pack(pady=5)
        
        # Add Save Data Button and Status
        tk.Label(self.control_frame, text="Data Logging", font=('Arial', 12, 'bold')).pack(pady=(20,5))
        self.circle_label = tk.Label(self.control_frame, text="Circle #: 0")
        self.circle_label.pack()
        self.save_button = tk.Button(self.control_frame, text="Save Previous Circle Data", 
                                   command=self._save_button_clicked)
        self.save_button.pack(pady=5)
        self.save_status = tk.Label(self.control_frame, text="")
        self.save_status.pack()
        
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
        
        # Print current parameters to terminal
        center_x = self.center_x_slider.get()
        center_z = self.center_z_slider.get()
        radius = self.radius_slider.get()
        print(f"\nParameters updated:")
        print(f"PD Gains: Kp={kp:.2f}, Kd={kd:.1f}")
        print(f"Trajectory: Center=({center_x:.2f}, {center_z:.2f}), Radius={radius:.2f}")
    
    def _save_button_clicked(self):
        """Handle manual save button click"""
        if self.previous_circle_data:  # Only save if there's previous data
            self.save_requested = True
            self.save_status.configure(text="Save requested for previous circle...", fg="blue")
        else:
            self.save_status.configure(text="No previous circle data to save", fg="orange")

    def _toggle_auto_save(self):
        """Toggle automatic saving of circle data"""
        self.auto_save_enabled = self.auto_save_var.get()
        status = "enabled" if self.auto_save_enabled else "disabled"
        print(f"Auto-save {status}")
        self.save_status.configure(text=f"Auto-save {status}", fg="blue")

    def get_gains(self):
        return self.kp_slider.get(), self.kd_slider.get()
    
    def get_trajectory_params(self):
        return (
            self.center_x_slider.get(),
            self.center_z_slider.get(),
            self.radius_slider.get()
        )
    
    def cache_data_point(self, data_point):
        """Cache data point for saving"""
        try:
            # Extract data from data_point with proper indexing
            timestamp = data_point[0]
            target_x, target_z = data_point[1], data_point[2]
            actual_x, actual_z = data_point[3], data_point[4]
            joint_angles = data_point[5:8]  # Indices 5,6,7 are joint angles
            joint_velocities = data_point[8:11]  # Indices 8,9,10 are joint velocities
            joint_accelerations = [
                self.accelerations_joint1[-1] if self.accelerations_joint1 else 0.0,
                self.accelerations_joint2[-1] if self.accelerations_joint2 else 0.0,
                self.accelerations_joint3[-1] if self.accelerations_joint3 else 0.0
            ]
            joint_torques = data_point[14:17]  # Indices 14,15,16 are joint torques
            kp, kd = data_point[17], data_point[18]
            radius, center_x, center_z = data_point[19], data_point[20], data_point[21]
            
            # Create modified data point with all measurements
            modified_data = [
                timestamp,
                target_x, target_z,
                actual_x, actual_z,
                *joint_angles,
                *joint_velocities,
                *joint_accelerations,  # Use calculated accelerations
                *joint_torques,
                kp, kd,
                radius,
                center_x,
                center_z
            ]
            
            self.current_circle_data.append(modified_data)
        except Exception as e:
            print(f"Error in cache_data_point: {str(e)}")
            print("Data point:", data_point)
            import traceback
            traceback.print_exc()

    def clear_cache(self):
        """Move current data to previous and clear current"""
        if self.current_circle_data:  # Only if we have data to move
            self.previous_circle_data = self.current_circle_data.copy()
            self.current_circle_data = []

    def save_cached_data(self, circle_number):
        """Save manually requested data"""
        if not self.previous_circle_data:
            self.save_status.configure(text="No previous circle data to save", fg="orange")
            return False
            
        # Get current parameters for filename
        kp, kd = self.get_gains()
        center_x, center_z, radius = self.get_trajectory_params()
        
        # Create filename with parameters
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"semi_circle_trajectory_kp{kp:.2f}_kd{kd:.1f}_r{radius:.2f}_x{center_x:.2f}_z{center_z:.2f}_semicircle{circle_number}_{timestamp_str}.csv"
        
        if not os.path.exists('data'):
            os.makedirs('data')
        filepath = os.path.join('data', filename)

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
                writer.writerows(self.previous_circle_data)
            
            self.save_status.configure(text=f"Manually saved circle #{circle_number-1}", fg="green")
            print(f"Manually saved data for circle #{circle_number-1} to {filename}")
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            self.save_status.configure(text="Error saving data", fg="red")
            return False

    def auto_save_circle_data(self, circle_number):
        """Automatically save the previous circle data"""
        if not self.previous_circle_data or not self.auto_save_enabled:
            return False

        # Get current parameters for filename
        kp, kd = self.get_gains()
        center_x, center_z, radius = self.get_trajectory_params()
        
        # Create filename with parameters
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"semi_circle_trajectory_kp{kp:.2f}_kd{kd:.1f}_r{radius:.2f}_x{center_x:.2f}_z{center_z:.2f}_semicircle{circle_number}_{timestamp_str}.csv"
        
        if not os.path.exists('data'):
            os.makedirs('data')
        filepath = os.path.join('data', filename)

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
                writer.writerows(self.previous_circle_data)
            
            self.save_status.configure(text=f"Auto-saved circle #{circle_number-1}", fg="green")
            print(f"Auto-saved data for circle #{circle_number-1} to {filename}")
            return True
        except Exception as e:
            print(f"Error auto-saving data: {str(e)}")
            self.save_status.configure(text="Error auto-saving data", fg="red")
            return False

    def calculate_velocities_and_accelerations(self, joint_angles, timestamp):
        """Calculate velocities and accelerations for each joint"""
        dt = 1.0/240.0  # Fixed timestep
        if self.prev_time is None:
            self.prev_time = timestamp
            return ([0.0] * 3, [0.0] * 3)  # Return zeros for first frame
            
        # Calculate velocities
        velocities = []
        accelerations = []
        
        for i, (angle, prev_vel) in enumerate(zip(joint_angles, 
            [self.prev_velocities['joint1'], 
             self.prev_velocities['joint2'], 
             self.prev_velocities['joint3']])):
            
            # Get previous angle from stored data
            prev_angle = self.angles_joint1[-1] if i == 0 else (
                        self.angles_joint2[-1] if i == 1 else self.angles_joint3[-1])
            
            # Calculate velocity (rad/s)
            velocity = (angle - prev_angle) / dt
            
            # Calculate acceleration (rad/s²)
            acceleration = (velocity - prev_vel) / dt
            
            velocities.append(velocity)
            accelerations.append(acceleration)
            
            # Update previous velocities
            if i == 0:
                self.prev_velocities['joint1'] = velocity
            elif i == 1:
                self.prev_velocities['joint2'] = velocity
            else:
                self.prev_velocities['joint3'] = velocity
        
        self.prev_time = timestamp
        return velocities, accelerations

    def calculate_derivatives(self, joint_angles, current_time):
        """Calculate velocities and accelerations using finite differences"""
        if self.first_update:
            self.first_update = False
            self.prev_angles = joint_angles.copy()
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        
        # Use fixed time step for more stable derivatives
        dt = self.dt
        
        # Calculate velocities
        velocities = []
        for curr_angle, prev_angle in zip(joint_angles, self.prev_angles):
            vel = (curr_angle - prev_angle) / dt
            # Apply low-pass filter to smooth velocity
            vel = 0.7 * vel + 0.3 * self.prev_velocities[len(velocities)]
            velocities.append(vel)
        
        # Calculate accelerations
        accelerations = []
        for curr_vel, prev_vel in zip(velocities, self.prev_velocities):
            acc = (curr_vel - prev_vel) / dt
            # Apply low-pass filter to smooth acceleration
            acc = 0.7 * acc
            accelerations.append(acc)
        
        # Update previous values
        self.prev_angles = joint_angles.copy()
        self.prev_velocities = velocities.copy()
        
        # Limit values to prevent extreme spikes
        velocities = [max(min(v, 10.0), -10.0) for v in velocities]  # Limit to ±10 rad/s
        accelerations = [max(min(a, 50.0), -50.0) for a in accelerations]  # Limit to ±50 rad/s²
        
        return velocities, accelerations

    def calculate_filtered_acceleration(self, current_velocity, joint_idx):
        """Calculate filtered acceleration using moving average of velocities"""
        try:
            # Add current velocity to history
            self.velocity_history[joint_idx].append(current_velocity)
            
            # Keep only recent history
            if len(self.velocity_history[joint_idx]) > self.filter_window:
                self.velocity_history[joint_idx].pop(0)
            
            # Need at least 2 points to calculate acceleration
            if len(self.velocity_history[joint_idx]) < 2:
                return 0.0
            
            # Calculate smoothed velocities
            smoothed_vel_current = np.mean(self.velocity_history[joint_idx][-2:])
            smoothed_vel_prev = np.mean(self.velocity_history[joint_idx][:-1])
            
            # Calculate acceleration
            acceleration = (smoothed_vel_current - smoothed_vel_prev) / self.dt
            
            # Apply additional low-pass filter
            filtered_acc = 0.2 * acceleration + 0.8 * self.prev_accelerations[joint_idx]
            
            # Limit maximum acceleration
            max_acc = 5.0  # Maximum allowed acceleration in rad/s²
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
            # Keep trajectory history for plotting
            self.target_x.append(target_x)
            self.target_z.append(target_z)
            self.actual_x.append(actual_x)
            self.actual_z.append(actual_z)
            
            # Update time steps
            self.step_counter += 1
            self.time_steps.append(self.step_counter)
            
            # Get joint data from PyBullet states
            angles = []
            velocities = []
            torques = []
            accelerations = []
            
            # Get joint states directly from PyBullet
            for i, joint_index in enumerate(actuated_joint_indices):
                state = p.getJointState(robot_id, joint_index)
                angle = state[0]  # Position
                velocity = state[1]  # Velocity
                torque = state[3]  # Applied torque
                
                # Calculate filtered acceleration
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
            
            # Update data lists
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
            
            # Keep only last 100 points for time series plots
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
            
            # Update plots every few steps to reduce computational load
            self.update_counter += 1
            if self.update_counter % 5 == 0:  # Update every 5 steps
                # Update trajectory plot
                self.target_line.set_data(self.target_x, self.target_z)
                self.actual_line.set_data(self.actual_x, self.actual_z)
                
                # Update joint angle plots
                for i, line in enumerate(self.angle_lines):
                    line.set_data(self.time_steps, [self.angles_joint1, self.angles_joint2, self.angles_joint3][i])
                
                # Update velocity plots
                for i, line in enumerate(self.velocity_lines):
                    line.set_data(self.time_steps, [self.velocities_joint1, self.velocities_joint2, self.velocities_joint3][i])
                
                # Update acceleration plots
                for i, line in enumerate(self.acceleration_lines):
                    line.set_data(self.time_steps, [self.accelerations_joint1, self.accelerations_joint2, self.accelerations_joint3][i])
                
                # Update torque plots
                for i, line in enumerate(self.torque_lines):
                    line.set_data(self.time_steps, [self.torques_joint1, self.torques_joint2, self.torques_joint3][i])
                
                # Update x-axis limits for time series plots
                for ax in [self.ax_angles, self.ax_velocities, self.ax_accelerations, self.ax_torque]:
                    ax.set_xlim(self.step_counter - 100, self.step_counter)
                
                # Auto-adjust y-axis limits
                self._adjust_plot_limits()
                
                # Redraw canvas
                self.canvas.draw()
            
            # Process Tkinter events less frequently
            if self.update_counter % 10 == 0:  # Process events every 10 steps
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
            # Helper function to get limits with margin
            def get_limits_with_margin(data_list, margin_factor=0.1, min_range=0.1):
                if not data_list or not any(data_list):
                    return -1, 1
                
                min_val = min(min(d) for d in data_list if d)
                max_val = max(max(d) for d in data_list if d)
                
                # Ensure minimum range
                if max_val - min_val < min_range:
                    center = (max_val + min_val) / 2
                    min_val = center - min_range/2
                    max_val = center + min_range/2
                
                margin = max((max_val - min_val) * margin_factor, min_range * 0.1)
                return min_val - margin, max_val + margin
            
            # Adjust angle plot limits
            angle_data = [self.angles_joint1, self.angles_joint2, self.angles_joint3]
            if any(angle_data):
                self.ax_angles.set_ylim(*get_limits_with_margin(angle_data, min_range=0.5))
            
            # Adjust velocity plot limits
            vel_data = [self.velocities_joint1, self.velocities_joint2, self.velocities_joint3]
            if any(vel_data):
                self.ax_velocities.set_ylim(*get_limits_with_margin(vel_data, min_range=0.2))
            
            # Adjust acceleration plot limits
            acc_data = [self.accelerations_joint1, self.accelerations_joint2, self.accelerations_joint3]
            if any(acc_data):
                self.ax_accelerations.set_ylim(*get_limits_with_margin(acc_data, min_range=1.0))
            
            # Adjust torque plot limits
            torque_data = [self.torques_joint1, self.torques_joint2, self.torques_joint3]
            if any(torque_data):
                self.ax_torque.set_ylim(*get_limits_with_margin(torque_data, min_range=5.0))
        
        except Exception as e:
            print(f"Error in _adjust_plot_limits: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_semi_circle_number(self, number):
        """Update circle number and handle auto-saving"""
        self.current_circle_number = number
        self.circle_label.configure(text=f"Circle #: {number}")
        
        # Auto-save previous circle data when a new circle starts
        if number > 1:  # Don't try to save the first circle
            self.auto_save_circle_data(number)

    def clear_data(self):
        # Clear trajectory data
        self.target_x = []
        self.target_z = []
        self.actual_x = []
        self.actual_z = []
        self.target_line.set_data([], [])
        self.actual_line.set_data([], [])
        
        # Clear time steps
        self.time_steps = []
        self.step_counter = 0
        
        # Clear joint data
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
        
        # Reset processing variables
        self.prev_velocities = np.zeros(3)
        self.prev_accelerations = np.zeros(3)
        self.velocity_history = [[], [], []]  # Reset velocity history
        self.first_update = True
        
        # Clear all plot lines
        for line in self.angle_lines:
            line.set_data([], [])
        for line in self.velocity_lines:
            line.set_data([], [])
        for line in self.acceleration_lines:
            line.set_data([], [])
        for line in self.torque_lines:
            line.set_data([], [])
        
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
    link3_to_link4_index = -1
    link4_to_gripper_index = -1

    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        
        if joint_name in ["link_1_to_link_2", "link_2_to_link_3", "link_3_to_link_4"]:
            actuated_joint_indices.append(i)
            joint_names.append(joint_name)
            p.enableJointForceTorqueSensor(robot_id, i, 1)

        if joint_name == "link_4_to_gripper":
            end_effector_link_index = i
            link4_to_gripper_index = i
        elif joint_name == "link_3_to_link_4":
            link3_to_link4_index = i

    print("Actuated Joints Found:")
    for i, name in zip(actuated_joint_indices, joint_names):
        print(f"- {name} (Index: {i})")
    print(f"End-Effector Link Index: {end_effector_link_index}\n")

    return actuated_joint_indices, end_effector_link_index, link3_to_link4_index, link4_to_gripper_index

def generate_semi_circular_trajectory(cx, cy, radius, num_points):
    """
    Generates a list of (x, y) points for a bidirectional semi-circular trajectory.
    The trajectory follows the same semi-circle path forward and backward.
    """
    trajectory = []
    
    # Use half points for each direction
    points_per_direction = num_points // 2
    
    # Generate forward semi-circle points (π to 0 radians for clockwise motion)
    for i in range(points_per_direction):
        angle = math.pi * (1 - i / (points_per_direction - 1))
        x = cx + radius * math.cos(angle)
        z = cy + radius * math.sin(angle)
        trajectory.append((x, z))
    
    # Generate return semi-circle points (0 to π radians for counter-clockwise return)
    for i in range(points_per_direction):
        angle = math.pi * i / (points_per_direction - 1)
        x = cx + radius * math.cos(angle)
        z = cy + radius * math.sin(angle)
        trajectory.append((x, z))
    
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
    Main function to run the simulation and track the semi-circular trajectory.
    """
    try:
        # --- Simulation and Trajectory Parameters ---
        urdf_file = "/home/san/Public/pybullet_4dof/urdf/robot.urdf"
        
        # Link lengths derived from URDF for our custom IK solver
        link_lengths = {'l1': 0.27, 'l2': 0.15, 'l3': 0.15, 'l4': 0.10}

        if not os.path.exists(urdf_file):
            print(f"Error: '{urdf_file}' not found. Please update the URDF file path.")
            return

        # --- Setup ---
        robot_id = setup_simulation(urdf_file)
        actuated_joint_indices, ee_link_index, link3_to_link4_index, link4_to_gripper_index = get_joint_info(robot_id)

        # Initialize visualizer
        visualizer = RealTimeVisualizer()
        
        # Get initial trajectory parameters
        center_x, center_z, radius = visualizer.get_trajectory_params()
        num_trajectory_points = 1000  # Increased from 250 to 1000 for smoother motion (500 points each way)
        
        # Generate initial trajectory
        trajectory_points = generate_semi_circular_trajectory(center_x, center_z, radius, num_trajectory_points)
        
        # Create visual marker for gripper effect (ball being manipulated)
        marker_radius = 0.0165  # Size of the ball being manipulated
        marker_color = [0, 1, 0, 1]  # Green color [R, G, B, Alpha]
        
        # Create visual shape for marker
        marker_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=marker_radius,
            rgbaColor=marker_color
        )
        
        # Create the marker body (with no collision)
        gripper_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=marker_visual,
            basePosition=[0, 0, 0]
        )
        
        # Create box containers for start and end points
        box_size = [0.06, 0.08, 0.08]  # Width, Length, Height
        box_base_thickness = 0.01  # Thickness of the box base and walls
        
        # Colors for the boxes
        start_box_color = [0.8, 0.2, 0.2, 0.9]  # Red box
        end_box_color = [0.2, 0.2, 0.8, 0.9]    # Blue box
        
        # Create the container boxes (main body)
        start_box_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[box_size[0]/2, box_size[1]/2, box_base_thickness/2],
            rgbaColor=start_box_color
        )
        
        end_box_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[box_size[0]/2, box_size[1]/2, box_base_thickness/2],
            rgbaColor=end_box_color
        )
        
        # Create walls for the boxes
        wall_height = box_size[2]
        wall_thickness = box_base_thickness
        
        # Visual shapes for walls
        wall_shapes = {
            'start': [],
            'end': []
        }
        
        # Create walls for both boxes
        for box_type in ['start', 'end']:
            color = start_box_color if box_type == 'start' else end_box_color
            
            # Front wall
            wall_shapes[box_type].append(
                p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[box_size[0]/2, wall_thickness/2, wall_height/2],
                    rgbaColor=color
                )
            )
            
            # Back wall
            wall_shapes[box_type].append(
                p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[box_size[0]/2, wall_thickness/2, wall_height/2],
                    rgbaColor=color
                )
            )
            
            # Left wall
            wall_shapes[box_type].append(
                p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[wall_thickness/2, box_size[1]/2, wall_height/2],
                    rgbaColor=color
                )
            )
            
            # Right wall
            wall_shapes[box_type].append(
                p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[wall_thickness/2, box_size[1]/2, wall_height/2],
                    rgbaColor=color
                )
            )
        
        # Calculate start and end positions for the boxes
        start_point = trajectory_points[0]
        end_point = trajectory_points[num_trajectory_points // 2 - 1]
        
        # Add offset to box positions
        box_x_offset = 0.0175  # 1.75cm offset
        start_point = (start_point[0] - box_x_offset, start_point[1])  # Move red box left
        end_point = (end_point[0] + box_x_offset, end_point[1])      # Move blue box right
        
        # Create the box containers with their walls
        def create_container(base_pos, base_visual, wall_shapes):
            # Create base
            base = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=base_visual,
                basePosition=[base_pos[0], 0, box_base_thickness/2]
            )
            
            # Wall positions relative to base
            wall_positions = [
                [0, box_size[1]/2, wall_height/2],    # Front wall
                [0, -box_size[1]/2, wall_height/2],   # Back wall
                [-box_size[0]/2, 0, wall_height/2],   # Left wall
                [box_size[0]/2, 0, wall_height/2]     # Right wall
            ]
            
            # Create walls
            walls = []
            for wall_shape, rel_pos in zip(wall_shapes, wall_positions):
                wall = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=wall_shape,
                    basePosition=[
                        base_pos[0] + rel_pos[0],
                        0 + rel_pos[1],
                        rel_pos[2]
                    ]
                )
                walls.append(wall)
            
            return [base] + walls
        
        # Create both containers with their walls
        start_container = create_container(
            [start_point[0], start_point[1]], 
            start_box_visual, 
            wall_shapes['start']
        )
        
        end_container = create_container(
            [end_point[0], end_point[1]], 
            end_box_visual, 
            wall_shapes['end']
        )
        
        # Draw initial trajectory in PyBullet workspace
        for i in range(len(trajectory_points) - 1):
            p1 = [trajectory_points[i][0], 0, trajectory_points[i][1]]
            p2 = [trajectory_points[i + 1][0], 0, trajectory_points[i + 1][1]]
            p.addUserDebugLine(p1, p2, lineColorRGB=[1, 0, 0], lineWidth=2)
        
        # Variable to store previous trajectory parameters for change detection
        prev_params = (center_x, center_z, radius)
        semi_circle_count = 0
        last_time = time.time()
        dt = 1.0/240.0  # Fixed time step
        
        print(f"Starting simulation. Use the sliders to adjust PD gains and trajectory parameters in real-time.")
        print("Click 'Save Current Semi-Circle Data' button to save data for the current semi-circle.")
        print("Press Ctrl+C or close the window to stop the simulation.")

        while True:
            try:
                if visualizer.is_closed():
                    break

                current_time = time.time()
                if current_time - last_time < dt:
                    continue
                last_time = current_time

                # Check if trajectory parameters have changed
                current_params = visualizer.get_trajectory_params()
                if current_params != prev_params:
                    # Update trajectory
                    center_x, center_z, radius = current_params
                    trajectory_points = generate_semi_circular_trajectory(center_x, center_z, radius, num_trajectory_points)
                    prev_params = current_params
                    
                    # Update container positions
                    start_point = trajectory_points[0]
                    end_point = trajectory_points[num_trajectory_points // 2 - 1]
                    
                    # Apply offset to box positions
                    start_point = (start_point[0] - box_x_offset, start_point[1])
                    end_point = (end_point[0] + box_x_offset, end_point[1])
                    
                    # Update start container position
                    for i, body_id in enumerate(start_container):
                        if i == 0:  # Base
                            p.resetBasePositionAndOrientation(
                                body_id,
                                [start_point[0], 0, box_base_thickness/2],
                                [0, 0, 0, 1]
                            )
                        else:  # Walls
                            wall_idx = i - 1
                            wall_positions = [
                                [0, box_size[1]/2, wall_height/2],    # Front wall
                                [0, -box_size[1]/2, wall_height/2],   # Back wall
                                [-box_size[0]/2, 0, wall_height/2],   # Left wall
                                [box_size[0]/2, 0, wall_height/2]     # Right wall
                            ]
                            rel_pos = wall_positions[wall_idx]
                            p.resetBasePositionAndOrientation(
                                body_id,
                                [
                                    start_point[0] + rel_pos[0],
                                    0 + rel_pos[1],
                                    rel_pos[2]
                                ],
                                [0, 0, 0, 1]
                            )
                    
                    # Update end container position
                    for i, body_id in enumerate(end_container):
                        if i == 0:  # Base
                            p.resetBasePositionAndOrientation(
                                body_id,
                                [end_point[0], 0, box_base_thickness/2],
                                [0, 0, 0, 1]
                            )
                        else:  # Walls
                            wall_idx = i - 1
                            wall_positions = [
                                [0, box_size[1]/2, wall_height/2],    # Front wall
                                [0, -box_size[1]/2, wall_height/2],   # Back wall
                                [-box_size[0]/2, 0, wall_height/2],   # Left wall
                                [box_size[0]/2, 0, wall_height/2]     # Right wall
                            ]
                            rel_pos = wall_positions[wall_idx]
                            p.resetBasePositionAndOrientation(
                                body_id,
                                [
                                    end_point[0] + rel_pos[0],
                                    0 + rel_pos[1],
                                    rel_pos[2]
                                ],
                                [0, 0, 0, 1]
                            )
                    
                    # Clear and redraw trajectory
                    p.removeAllUserDebugItems()
                    for i in range(len(trajectory_points) - 1):
                        p1 = [trajectory_points[i][0], 0, trajectory_points[i][1]]
                        p2 = [trajectory_points[i + 1][0], 0, trajectory_points[i + 1][1]]
                        p.addUserDebugLine(p1, p2, lineColorRGB=[1, 0, 0], lineWidth=2)
                
                # Clear visualization at the start of each circle
                visualizer.clear_data()
                visualizer.clear_cache()
                
                for i, (target_x, target_z) in enumerate(trajectory_points):
                    if visualizer.is_closed():
                        break
                        
                    # Get current PD gains from sliders
                    kp, kd = visualizer.get_gains()
                    
                    # Calculate Inverse Kinematics using the custom function
                    joint_poses = custom_inverse_kinematics(target_x, target_z, link_lengths)

                    if joint_poses is not None:
                        # Set joint positions using the Position Controller with current gains
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

                    # Step simulation with multiple substeps for stability
                    for _ in range(4):  # 4 substeps
                        p.stepSimulation()
                        time.sleep(dt/4.0)  # Distribute the time step

                    # Only show marker during the semi-circular part
                    if i < num_trajectory_points // 2:  # Check if it's in the forward semi-circle
                        # Get positions of the two joints
                        link3_state = p.getLinkState(robot_id, link3_to_link4_index)
                        link4_state = p.getLinkState(robot_id, link4_to_gripper_index)
                        
                        # Get the direction vector between the joints
                        dir_x = link4_state[0][0] - link3_state[0][0]
                        dir_y = link4_state[0][1] - link3_state[0][1]
                        dir_z = link4_state[0][2] - link3_state[0][2]
                        
                        # Normalize the direction vector
                        length = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                        if length > 0:
                            dir_x /= length
                            dir_y /= length
                            dir_z /= length
                            
                            # Extend from the gripper joint by 0.03m
                            offset = 0.03
                            marker_pos = [
                                link4_state[0][0] + dir_x * offset,
                                link4_state[0][1] + dir_y * offset,
                                link4_state[0][2] + dir_z * offset
                            ]
                            
                            # Update marker position with gripper orientation
                            p.resetBasePositionAndOrientation(gripper_marker, marker_pos, link4_state[1])
                    else:
                        # Hide marker during straight line motion
                        hide_pos = [1000, 1000, 1000]
                        p.resetBasePositionAndOrientation(gripper_marker, hide_pos, [0, 0, 0, 1])

                    # Get current end-effector position for visualization
                    link_state = p.getLinkState(robot_id, ee_link_index)
                    actual_pos = link_state[0]
                    
                    # Get joint states
                    joint_states = [p.getJointState(robot_id, idx) for idx in actuated_joint_indices]
                    
                    # Update visualization
                    if not visualizer.update(target_x, target_z, actual_pos[0], actual_pos[2], joint_states, robot_id, actuated_joint_indices):
                        break

                    # --- Data Collection ---
                    timestamp = time.time()
                    
                    # Get joint states for data logging
                    joint_states_for_data = []
                    for joint_index in actuated_joint_indices:
                        state = p.getJointState(robot_id, joint_index)
                        joint_states_for_data.append(state)
                    
                    # Get current gains and parameters
                    kp, kd = visualizer.get_gains()
                    current_params = visualizer.get_trajectory_params()
                    current_center_x, current_center_z, current_radius = current_params
                    
                    # Cache the data point with all required values
                    data_point = [
                        timestamp,                    # 0
                        target_x, target_z,          # 1, 2
                        actual_pos[0], actual_pos[2], # 3, 4
                        joint_states_for_data[0][0],  # 5 - Joint 1 angle
                        joint_states_for_data[1][0],  # 6 - Joint 2 angle
                        joint_states_for_data[2][0],  # 7 - Joint 3 angle
                        joint_states_for_data[0][1],  # 8 - Joint 1 velocity
                        joint_states_for_data[1][1],  # 9 - Joint 2 velocity
                        joint_states_for_data[2][1],  # 10 - Joint 3 velocity
                        0.0, 0.0, 0.0,               # 11, 12, 13 - Accelerations (calculated in visualizer)
                        joint_states_for_data[0][3],  # 14 - Joint 1 torque
                        joint_states_for_data[1][3],  # 15 - Joint 2 torque
                        joint_states_for_data[2][3],  # 16 - Joint 3 torque
                        kp, kd,                      # 17, 18
                        current_radius,               # 19
                        current_center_x,             # 20
                        current_center_z              # 21
                    ]
                    visualizer.cache_data_point(data_point)

                    # Check if a circle is completed
                    if i == num_trajectory_points - 1:
                        semi_circle_count += 1
                        visualizer.update_semi_circle_number(semi_circle_count)
                        print(f"Semi-circle {semi_circle_count} completed. Gains: Kp={kp:.2f}, Kd={kd:.1f}, "
                              f"Center: ({current_center_x:.2f}, {current_center_z:.2f}), Radius: {current_radius:.2f}")
                        
                        # Move current data to previous and clear current
                        visualizer.clear_cache()
                        
                        # Check if manual save was requested
                        if visualizer.save_requested:
                            visualizer.save_cached_data(semi_circle_count)
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
        # Cleanup
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