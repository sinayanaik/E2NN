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
        self.window_closed = False
        
        self.prev_velocities = np.zeros(3)
        self.prev_accelerations = np.zeros(3)
        self.velocity_history = [[], [], []]
        self.dt = 1.0/240.0
        self.first_update = True
        self.update_counter = 0
        self.filter_window = 5
        
        self.root = tk.Tk()
        self.root.title("Real-time Figure-Eight Trajectory and Dynamics Visualization")
        
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        self.fig = Figure(figsize=(12, 12))
        
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        
        self.target_x = []
        self.target_z = []
        self.actual_x = []
        self.actual_z = []
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
        
        self.ax_traj = self.fig.add_subplot(gs[0, :])
        self.ax_traj.set_xlabel('X Position (m)')
        self.ax_traj.set_ylabel('Z Position (m)')
        self.ax_traj.grid(True)
        self.ax_traj.set_title('End-Effector Trajectory')
        self.ax_traj.set_aspect('equal')
        
        self.ax_angles = self.fig.add_subplot(gs[1, 0])
        self.ax_angles.set_xlabel('Time Step')
        self.ax_angles.set_ylabel('Joint Angle (rad)')
        self.ax_angles.grid(True)
        self.ax_angles.set_title('Joint Angles')
        
        self.ax_velocities = self.fig.add_subplot(gs[1, 1])
        self.ax_velocities.set_xlabel('Time Step')
        self.ax_velocities.set_ylabel('Velocity (rad/s)')
        self.ax_velocities.grid(True)
        self.ax_velocities.set_title('Joint Velocities')
        
        self.ax_accelerations = self.fig.add_subplot(gs[2, 0])
        self.ax_accelerations.set_xlabel('Time Step')
        self.ax_accelerations.set_ylabel('Acceleration (rad/s²)')
        self.ax_accelerations.grid(True)
        self.ax_accelerations.set_title('Joint Accelerations')
        
        self.ax_torque = self.fig.add_subplot(gs[2, 1])
        self.ax_torque.set_xlabel('Time Step')
        self.ax_torque.set_ylabel('Torque (N⋅m)')
        self.ax_torque.grid(True)
        self.ax_torque.set_title('Joint Torques')
        
        self.target_line, = self.ax_traj.plot([], [], 'r-', label='Target')
        self.actual_line, = self.ax_traj.plot([], [], 'b--', label='Actual')
        self.ax_traj.legend()
        
        colors = ['r', 'g', 'b']
        self.angle_lines = [self.ax_angles.plot([], [], label=f'Joint {i+1}', color=c)[0] for i, c in enumerate(colors)]
        self.velocity_lines = [self.ax_velocities.plot([], [], label=f'Joint {i+1}', color=c)[0] for i, c in enumerate(colors)]
        self.acceleration_lines = [self.ax_accelerations.plot([], [], label=f'Joint {i+1}', color=c)[0] for i, c in enumerate(colors)]
        self.torque_lines = [self.ax_torque.plot([], [], label=f'Joint {i+1}', color=c)[0] for i, c in enumerate(colors)]
        
        self.ax_angles.legend()
        self.ax_velocities.legend()
        self.ax_accelerations.legend()
        self.ax_torque.legend()
        
        self.ax_traj.set_xlim(0, 0.5)
        self.ax_traj.set_ylim(0, 0.5)
        
        for ax in [self.ax_angles, self.ax_velocities, self.ax_accelerations, self.ax_torque]:
            ax.set_xlim(0, 100)
        
        self.ax_angles.set_ylim(-np.pi, np.pi)
        self.ax_velocities.set_ylim(-5, 5)
        self.ax_accelerations.set_ylim(-20, 20)
        self.ax_torque.set_ylim(-50, 50)
        
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        tk.Label(self.control_frame, text="PD Control Gains", font=('Arial', 12, 'bold')).pack(pady=5)
        
        tk.Label(self.control_frame, text="Position Gain (Kp)").pack()
        self.kp_slider = tk.Scale(self.control_frame, from_=0.001, to=1.0, resolution=0.001, orient=tk.HORIZONTAL, length=200)
        self.kp_slider.set(0.02)
        self.kp_slider.pack(pady=5)
        
        tk.Label(self.control_frame, text="Velocity Gain (Kd)").pack()
        self.kd_slider = tk.Scale(self.control_frame, from_=0.001, to=1.0, resolution=0.001, orient=tk.HORIZONTAL, length=200)
        self.kd_slider.set(0.3)
        self.kd_slider.pack(pady=5)
        
        tk.Label(self.control_frame, text="Trajectory Parameters", font=('Arial', 12, 'bold')).pack(pady=(20,5))
        
        tk.Label(self.control_frame, text="Center X (X)").pack()
        self.center_x_slider = tk.Scale(self.control_frame, from_=-0.5, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, length=200)
        self.center_x_slider.set(0.20)
        self.center_x_slider.pack(pady=5)
        
        tk.Label(self.control_frame, text="Center Z (Z)").pack()
        self.center_z_slider = tk.Scale(self.control_frame, from_=-0.5, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, length=200)
        self.center_z_slider.set(0.30)
        self.center_z_slider.pack(pady=5)
        
        tk.Label(self.control_frame, text="Amplitude (A)").pack()
        self.amplitude_slider = tk.Scale(self.control_frame, from_=-0.2, to=0.2, resolution=0.01, orient=tk.HORIZONTAL, length=200)
        self.amplitude_slider.set(-0.10)
        self.amplitude_slider.pack(pady=5)

        self.values_frame = tk.Frame(self.control_frame)
        self.values_frame.pack(pady=10)
        
        self.kp_label = tk.Label(self.values_frame, text="Current Kp: 0.02")
        self.kp_label.pack()
        self.kd_label = tk.Label(self.values_frame, text="Current Kd: 0.3")
        self.kd_label.pack()
        
        tk.Label(self.control_frame, text="Data Logging", font=('Arial', 12, 'bold')).pack(pady=(20,5))
        self.trajectory_label = tk.Label(self.control_frame, text="Figure-Eight #: 0")
        self.trajectory_label.pack()
        self.save_button = tk.Button(self.control_frame, text="Save Current Figure-Eight Data", command=self._save_button_clicked)
        self.save_button.pack(pady=5)
        self.save_status = tk.Label(self.control_frame, text="")
        self.save_status.pack()
        
        self.current_trajectory_data = []
        self.previous_trajectory_data = []
        self.current_trajectory_number = 0
        self.save_requested = False
        
        self.kp_slider.configure(command=self._update_labels)
        self.kd_slider.configure(command=self._update_labels)
        self.center_x_slider.configure(command=self._update_labels)
        self.center_z_slider.configure(command=self._update_labels)
        self.amplitude_slider.configure(command=self._update_labels)
        
        self.root.update()
    
    def _update_labels(self, _=None):
        kp = self.kp_slider.get()
        kd = self.kd_slider.get()
        self.kp_label.configure(text=f"Current Kp: {kp:.2f}")
        self.kd_label.configure(text=f"Current Kd: {kd:.1f}")
        
        center_x = self.center_x_slider.get()
        center_z = self.center_z_slider.get()
        amplitude = self.amplitude_slider.get()
        print(f"\nParameters updated:")
        print(f"PD Gains: Kp={kp:.2f}, Kd={kd:.1f}")
        print(f"Trajectory: Center=({center_x:.2f}, {center_z:.2f}), Amplitude={amplitude:.2f}")
    
    def _save_button_clicked(self):
        if self.previous_trajectory_data:
            self.save_requested = True
            self.save_status.configure(text="Save requested for previous figure-eight...", fg="blue")
        else:
            self.save_status.configure(text="No previous figure-eight data to save", fg="orange")

    def get_gains(self):
        return self.kp_slider.get(), self.kd_slider.get()
    
    def get_trajectory_params(self):
        return (
            self.center_x_slider.get(),
            self.center_z_slider.get(),
            self.amplitude_slider.get()
        )
    
    def cache_data_point(self, data_point):
        self.current_trajectory_data.append(data_point)

    def clear_cache(self):
        if self.current_trajectory_data:
            self.previous_trajectory_data = self.current_trajectory_data.copy()
            self.current_trajectory_data = []

    def save_cached_data(self, trajectory_number):
        if not self.previous_trajectory_data:
            self.save_status.configure(text="No previous data to save", fg="orange")
            return False
            
        kp, kd = self.get_gains()
        center_x, center_z, amplitude = self.get_trajectory_params()
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"figure8_trajectory_kp{kp:.2f}_kd{kd:.1f}_A{amplitude:.2f}_X{center_x:.2f}_Z{center_z:.2f}_figure8_{trajectory_number}_{timestamp_str}.csv"
        
        if not os.path.exists('data'):
            os.makedirs('data')
        filepath = os.path.join('data', filename)

        header = [
            'timestamp', 'target_x', 'target_z', 'actual_x', 'actual_z',
            'joint1_angle', 'joint2_angle', 'joint3_angle',
            'joint1_velocity', 'joint2_velocity', 'joint3_velocity',
            'joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration',
            'joint1_torque', 'joint2_torque', 'joint3_torque',
            'kp', 'kd', 'amplitude_A', 'center_x', 'center_z'
        ]
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self.previous_trajectory_data)
            
            self.save_status.configure(text=f"Saved Figure-Eight #{trajectory_number-1}", fg="green")
            print(f"Saved data for Figure-Eight #{trajectory_number-1} to {filename}")
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            self.save_status.configure(text="Error saving data", fg="red")
            return False

    def calculate_filtered_acceleration(self, current_velocity, joint_idx):
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
            
            max_acc = 20.0
            filtered_acc = np.clip(filtered_acc, -max_acc, max_acc)
            
            return filtered_acc
            
        except Exception as e:
            print(f"Error in acceleration calculation: {str(e)}")
            return 0.0

    def update(self, target_x, target_z, actual_x, actual_z, joint_states, robot_id, actuated_joint_indices):
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
                angle, velocity, _, torque = state[0], state[1], state[2], state[3]
                
                acceleration = 0.0 if self.first_update else self.calculate_filtered_acceleration(velocity, i)
                
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
            
            if len(self.time_steps) > 200:
                for lst in [self.time_steps, self.target_x, self.target_z, self.actual_x, self.actual_z,
                            self.angles_joint1, self.angles_joint2, self.angles_joint3,
                            self.velocities_joint1, self.velocities_joint2, self.velocities_joint3,
                            self.accelerations_joint1, self.accelerations_joint2, self.accelerations_joint3,
                            self.torques_joint1, self.torques_joint2, self.torques_joint3]:
                    del lst[0]

            self.update_counter += 1
            if self.update_counter % 5 == 0:
                self.target_line.set_data(self.target_x, self.target_z)
                self.actual_line.set_data(self.actual_x, self.actual_z)
                
                self.angle_lines[0].set_data(self.time_steps, self.angles_joint1)
                self.angle_lines[1].set_data(self.time_steps, self.angles_joint2)
                self.angle_lines[2].set_data(self.time_steps, self.angles_joint3)

                self.velocity_lines[0].set_data(self.time_steps, self.velocities_joint1)
                self.velocity_lines[1].set_data(self.time_steps, self.velocities_joint2)
                self.velocity_lines[2].set_data(self.time_steps, self.velocities_joint3)

                self.acceleration_lines[0].set_data(self.time_steps, self.accelerations_joint1)
                self.acceleration_lines[1].set_data(self.time_steps, self.accelerations_joint2)
                self.acceleration_lines[2].set_data(self.time_steps, self.accelerations_joint3)

                self.torque_lines[0].set_data(self.time_steps, self.torques_joint1)
                self.torque_lines[1].set_data(self.time_steps, self.torques_joint2)
                self.torque_lines[2].set_data(self.time_steps, self.torques_joint3)

                for ax in [self.ax_angles, self.ax_velocities, self.ax_accelerations, self.ax_torque]:
                    ax.set_xlim(self.time_steps[0], self.time_steps[-1])
                
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
        def get_limits_with_margin(data_list, margin_factor=0.1, min_range=0.1):
            flat_list = [item for sublist in data_list for item in sublist]
            if not flat_list: return -1, 1
            min_val, max_val = min(flat_list), max(flat_list)
            
            if max_val - min_val < min_range:
                center = (max_val + min_val) / 2
                min_val, max_val = center - min_range/2, center + min_range/2
            
            margin = (max_val - min_val) * margin_factor
            return min_val - margin, max_val + margin
        
        self.ax_angles.set_ylim(*get_limits_with_margin([self.angles_joint1, self.angles_joint2, self.angles_joint3], min_range=np.pi))
        self.ax_velocities.set_ylim(*get_limits_with_margin([self.velocities_joint1, self.velocities_joint2, self.velocities_joint3], min_range=2.0))
        self.ax_accelerations.set_ylim(*get_limits_with_margin([self.accelerations_joint1, self.accelerations_joint2, self.accelerations_joint3], min_range=10.0))
        self.ax_torque.set_ylim(*get_limits_with_margin([self.torques_joint1, self.torques_joint2, self.torques_joint3], min_range=20.0))
        
    def update_trajectory_number(self, number):
        self.current_trajectory_number = number
        self.trajectory_label.configure(text=f"Figure-Eight #: {number}")

    def clear_data(self):
        self.target_x, self.target_z, self.actual_x, self.actual_z = [], [], [], []
        self.time_steps, self.step_counter = [], 0
        self.angles_joint1, self.angles_joint2, self.angles_joint3 = [], [], []
        self.velocities_joint1, self.velocities_joint2, self.velocities_joint3 = [], [], []
        self.accelerations_joint1, self.accelerations_joint2, self.accelerations_joint3 = [], [], []
        self.torques_joint1, self.torques_joint2, self.torques_joint3 = [], [], []
        
        self.prev_velocities = np.zeros(3)
        self.prev_accelerations = np.zeros(3)
        self.velocity_history = [[], [], []]
        self.first_update = True
        
        for line in self.angle_lines + self.velocity_lines + self.acceleration_lines + self.torque_lines:
            line.set_data([], [])
        self.target_line.set_data([],[])
        self.actual_line.set_data([],[])

        self.canvas.draw()
        
    def close(self):
        try:
            self.window_closed = True
            if hasattr(self, 'root') and self.root:
                self.root.quit()
                self.root.destroy()
        except: pass

    def is_closed(self):
        try:
            if not hasattr(self, 'root') or not self.root.winfo_exists(): return True
            self.root.update()
            return self.window_closed
        except: return True

def setup_simulation(urdf_path):
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(urdf_path, [0,0,0], p.getQuaternionFromEuler([0,0,0]), useFixedBase=True)
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=0, cameraPitch=0, cameraTargetPosition=[0.25, 0, 0.25])
    return robot_id

def get_joint_info(robot_id):
    actuated_joint_indices = []
    num_joints = p.getNumJoints(robot_id)
    end_effector_link_index = -1
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        
        if info[2] == p.JOINT_REVOLUTE:
            actuated_joint_indices.append(i)
            p.enableJointForceTorqueSensor(robot_id, i, 1)

        if joint_name == 'link_4_to_gripper':
            end_effector_link_index = i
            
    # We only care about the first 3 joints for this 3-DOF arm
    actuated_joint_indices = [j for j in actuated_joint_indices if p.getJointInfo(robot_id, j)[1].decode('utf-8') in ["link_1_to_link_2", "link_2_to_link_3", "link_3_to_link_4"]]
            
    return actuated_joint_indices, end_effector_link_index

def generate_figure8_trajectory(X, Z, A, num_points):
    trajectory = []
    for i in range(num_points):
        t = -np.pi + (2 * np.pi * i / num_points)
        x = A * np.cos(t) * np.sin(t) + X
        z = A * np.cos(t) + Z
        trajectory.append((x, z))
    return trajectory

def custom_inverse_kinematics(target_x, target_z, link_lengths):
    l1, l2, l3, l4 = link_lengths['l1'], link_lengths['l2'], link_lengths['l3'], link_lengths['l4']

    def forward_kinematics_for_ik(theta):
        x = l2 * np.sin(theta[0]) + l3 * np.sin(theta[0] + theta[1]) + l4 * np.sin(theta[0] + theta[1] + theta[2])
        z = l1 + l2 * np.cos(theta[0]) + l3 * np.cos(theta[0] + theta[1]) + l4 * np.cos(theta[0] + theta[1] + theta[2])
        return x, z

    def objective(theta):
        end_x, end_z = forward_kinematics_for_ik(theta)
        return np.sqrt((end_x - target_x)**2 + (end_z - target_z)**2)
    
    result = minimize(objective, [0,0,0], method='SLSQP', bounds=[(-np.pi, np.pi)]*3)
    
    if result.success and result.fun < 1e-3:
        return result.x
    return None

def main():
    try:
        urdf_file = "/home/san/Public/pybullet_4dof/urdf/robot.urdf"
        if not os.path.exists(urdf_file):
            print(f"Error: '{urdf_file}' not found.")
            return

        robot_id = setup_simulation(urdf_file)
        actuated_joint_indices, ee_link_index = get_joint_info(robot_id)
        link_lengths = {'l1': 0.27, 'l2': 0.15, 'l3': 0.15, 'l4': 0.10}

        visualizer = RealTimeVisualizer()
        
        center_x, center_z, amplitude = visualizer.get_trajectory_params()
        num_trajectory_points = 1000
        trajectory_points = generate_figure8_trajectory(center_x, center_z, amplitude, num_trajectory_points)
        
        # Draw initial trajectory
        for i in range(len(trajectory_points) - 1):
            p1 = [trajectory_points[i][0], 0, trajectory_points[i][1]]
            p2 = [trajectory_points[i + 1][0], 0, trajectory_points[i + 1][1]]
            p.addUserDebugLine(p1, p2, lineColorRGB=[1, 0, 0], lineWidth=5)
        
        prev_params = (center_x, center_z, amplitude)
        trajectory_count = 0
        dt = 1.0/240.0
        
        print("Starting simulation...")

        while not visualizer.is_closed():
            current_params = visualizer.get_trajectory_params()
            if current_params != prev_params:
                center_x, center_z, amplitude = current_params
                trajectory_points = generate_figure8_trajectory(center_x, center_z, amplitude, num_trajectory_points)
                prev_params = current_params
                
                p.removeAllUserDebugItems()
                for i in range(len(trajectory_points) - 1):
                    p1 = [trajectory_points[i][0], 0, trajectory_points[i][1]]
                    p2 = [trajectory_points[i + 1][0], 0, trajectory_points[i + 1][1]]
                    p.addUserDebugLine(p1, p2, lineColorRGB=[1, 0, 0], lineWidth=2)
            
            visualizer.clear_data()
            
            for i, (target_x, target_z) in enumerate(trajectory_points):
                if visualizer.is_closed(): break
                    
                kp, kd = visualizer.get_gains()
                
                joint_poses = custom_inverse_kinematics(target_x, target_z, link_lengths)

                if joint_poses is not None:
                    for j, joint_index in enumerate(actuated_joint_indices):
                        p.setJointMotorControl2(bodyUniqueId=robot_id, jointIndex=joint_index, controlMode=p.POSITION_CONTROL,
                                            targetPosition=joint_poses[j], force=50, positionGain=kp, velocityGain=kd)

                p.stepSimulation()
                time.sleep(dt)

                link_state = p.getLinkState(robot_id, ee_link_index)
                actual_pos = link_state[0]
                
                joint_states = [p.getJointState(robot_id, idx) for idx in actuated_joint_indices]
                
                if not visualizer.update(target_x, target_z, actual_pos[0], actual_pos[2], joint_states, robot_id, actuated_joint_indices):
                    break

                # Data Collection
                timestamp = time.time()
                joint_angles = [s[0] for s in joint_states]
                joint_velocities = [s[1] for s in joint_states]
                joint_torques = [s[3] for s in joint_states]
                joint_accelerations = [
                    visualizer.accelerations_joint1[-1] if visualizer.accelerations_joint1 else 0.0,
                    visualizer.accelerations_joint2[-1] if visualizer.accelerations_joint2 else 0.0,
                    visualizer.accelerations_joint3[-1] if visualizer.accelerations_joint3 else 0.0,
                ]

                data_point = [
                    timestamp, target_x, target_z, actual_pos[0], actual_pos[2],
                    *joint_angles,
                    *joint_velocities,
                    *joint_accelerations,
                    *joint_torques,
                    kp, kd, amplitude, center_x, center_z
                ]
                visualizer.cache_data_point(data_point)

                if i == num_trajectory_points - 1:
                    trajectory_count += 1
                    visualizer.update_trajectory_number(trajectory_count)
                    print(f"Figure-Eight {trajectory_count} completed.")
                    
                    visualizer.clear_cache()
                    
                    if visualizer.save_requested:
                        visualizer.save_cached_data(trajectory_count)
                        visualizer.save_requested = False
                        
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            visualizer.close()
        except: pass
        try:
            p.disconnect()
        except: pass

if __name__ == "__main__":
    main()
