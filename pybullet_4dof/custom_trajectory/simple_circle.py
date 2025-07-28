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
        self.root = tk.Tk()
        self.root.title("Real-time Trajectory and Torque Visualization")
        
        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        # Create control frame
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Create matplotlib figure with three subplots
        self.fig = Figure(figsize=(10, 10))
        
        # Trajectory subplot
        self.ax_traj = self.fig.add_subplot(311)
        self.ax_traj.set_xlabel('X Position (m)')
        self.ax_traj.set_ylabel('Z Position (m)')
        self.ax_traj.grid(True)
        self.ax_traj.set_title('End-Effector Trajectory')
        
        # Torque subplot
        self.ax_torque = self.fig.add_subplot(312)
        self.ax_torque.set_xlabel('Time Step')
        self.ax_torque.set_ylabel('Torque (N⋅m)')
        self.ax_torque.grid(True)
        self.ax_torque.set_title('Joint Torques')
        
        # Joint angles subplot
        self.ax_angles = self.fig.add_subplot(313)
        self.ax_angles.set_xlabel('Time Step')
        self.ax_angles.set_ylabel('Joint Angle (rad)')
        self.ax_angles.grid(True)
        self.ax_angles.set_title('Joint Angles')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Initialize trajectory data lists
        self.target_x = []
        self.target_z = []
        self.actual_x = []
        self.actual_z = []
        
        # Initialize torque data lists
        self.time_steps = []
        self.torques_joint1 = []
        self.torques_joint2 = []
        self.torques_joint3 = []
        
        # Initialize joint angle data lists
        self.angles_joint1 = []
        self.angles_joint2 = []
        self.angles_joint3 = []
        self.step_counter = 0
        
        # Create plot lines for trajectory
        self.target_line, = self.ax_traj.plot([], [], 'r-', label='Target')
        self.actual_line, = self.ax_traj.plot([], [], 'b--', label='Actual')
        self.ax_traj.legend()
        
        # Create plot lines for torques
        self.torque_lines = [
            self.ax_torque.plot([], [], label=f'Joint {i+1}')[0]
            for i in range(3)
        ]
        self.ax_torque.legend()
        
        # Create plot lines for joint angles
        self.angle_lines = [
            self.ax_angles.plot([], [], label=f'Joint {i+1}')[0]
            for i in range(3)
        ]
        self.ax_angles.legend()
        
        # Set fixed axis limits based on workspace
        self.ax_traj.set_xlim(0, 0.5)
        self.ax_traj.set_ylim(0, 0.5)
        self.ax_traj.set_aspect('equal')
        
        # Set initial torque plot limits
        self.ax_torque.set_xlim(0, 100)
        self.ax_torque.set_ylim(-50, 50)
        
        # Set initial joint angle plot limits
        self.ax_angles.set_xlim(0, 100)
        self.ax_angles.set_ylim(-3.14, 3.14)  # ±π range for joint angles
        
        # Add PD Control Gain Sliders
        tk.Label(self.control_frame, text="PD Control Gains", font=('Arial', 12, 'bold')).pack(pady=5)
        
        # Kp slider
        tk.Label(self.control_frame, text="Position Gain (Kp)").pack()
        self.kp_slider = tk.Scale(self.control_frame, from_=0.01, to=1.0, resolution=0.01, 
                                orient=tk.HORIZONTAL, length=200)
        self.kp_slider.set(0.50)  # Updated default value
        self.kp_slider.pack(pady=5)
        
        # Kd slider
        tk.Label(self.control_frame, text="Velocity Gain (Kd)").pack()
        self.kd_slider = tk.Scale(self.control_frame, from_=0.1, to=2.0, resolution=0.1, 
                                orient=tk.HORIZONTAL, length=200)
        self.kd_slider.set(0.5)  # Updated default value
        self.kd_slider.pack(pady=5)
        
        # Add Trajectory Parameters
        tk.Label(self.control_frame, text="Trajectory Parameters", font=('Arial', 12, 'bold')).pack(pady=(20,5))
        
        # Center X slider
        tk.Label(self.control_frame, text="Center X Position (m)").pack()
        self.center_x_slider = tk.Scale(self.control_frame, from_=0.1, to=0.4, resolution=0.01,
                                      orient=tk.HORIZONTAL, length=200)
        self.center_x_slider.set(0.30)  # Updated default value
        self.center_x_slider.pack(pady=5)
        
        # Center Z slider
        tk.Label(self.control_frame, text="Center Z Position (m)").pack()
        self.center_z_slider = tk.Scale(self.control_frame, from_=0.1, to=0.4, resolution=0.01,
                                      orient=tk.HORIZONTAL, length=200)
        self.center_z_slider.set(0.30)  # Updated default value
        self.center_z_slider.pack(pady=5)
        
        # Radius slider
        tk.Label(self.control_frame, text="Radius (m)").pack()
        self.radius_slider = tk.Scale(self.control_frame, from_=0.05, to=0.15, resolution=0.01,
                                    orient=tk.HORIZONTAL, length=200)
        self.radius_slider.set(0.05)  # Updated default value
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
        self.current_circle_number = 0
        self.save_requested = False
        
        # Bind slider updates
        self.kp_slider.configure(command=self._update_labels)
        self.kd_slider.configure(command=self._update_labels)
    
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
        self.save_requested = True
        self.save_status.configure(text="Save requested...", fg="blue")
    
    def get_gains(self):
        return self.kp_slider.get(), self.kd_slider.get()
    
    def get_trajectory_params(self):
        return (
            self.center_x_slider.get(),
            self.center_z_slider.get(),
            self.radius_slider.get()
        )
    
    def cache_data_point(self, data_point):
        self.current_circle_data.append(data_point)
    
    def clear_cache(self):
        self.current_circle_data = []
    
    def save_cached_data(self, circle_number):
        if not self.current_circle_data:
            return False
            
        # Get current parameters for filename
        kp, kd = self.get_gains()
        center_x, center_z, radius = self.get_trajectory_params()
        
        # Create filename with parameters
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trajectory_kp{kp:.2f}_kd{kd:.1f}_r{radius:.2f}_x{center_x:.2f}_z{center_z:.2f}_circle{circle_number}_{timestamp_str}.csv"
        
        if not os.path.exists('data'):
            os.makedirs('data')
        filepath = os.path.join('data', filename)

        header = [
            'timestamp', 'target_x', 'target_z', 'actual_x', 'actual_z',
            'joint1_angle', 'joint2_angle', 'joint3_angle',
            'joint1_torque', 'joint2_torque', 'joint3_torque',
            'kp', 'kd', 'radius', 'center_x', 'center_z'
        ]
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.current_circle_data)
        
        self.save_status.configure(text=f"Saved circle #{circle_number}", fg="green")
        return True
        
    def update(self, target_x, target_z, actual_x, actual_z, joint_torques, joint_angles):
        # Update trajectory data
        self.target_x.append(target_x)
        self.target_z.append(target_z)
        self.actual_x.append(actual_x)
        self.actual_z.append(actual_z)
        
        # Update torque and angle data
        self.step_counter += 1
        self.time_steps.append(self.step_counter)
        
        # Update torques
        self.torques_joint1.append(joint_torques[0])
        self.torques_joint2.append(joint_torques[1])
        self.torques_joint3.append(joint_torques[2])
        
        # Update angles
        self.angles_joint1.append(joint_angles[0])
        self.angles_joint2.append(joint_angles[1])
        self.angles_joint3.append(joint_angles[2])
        
        # Keep only last 100 points for time series plots
        if len(self.time_steps) > 100:
            self.time_steps = self.time_steps[-100:]
            self.torques_joint1 = self.torques_joint1[-100:]
            self.torques_joint2 = self.torques_joint2[-100:]
            self.torques_joint3 = self.torques_joint3[-100:]
            self.angles_joint1 = self.angles_joint1[-100:]
            self.angles_joint2 = self.angles_joint2[-100:]
            self.angles_joint3 = self.angles_joint3[-100:]
            
            # Update x-axis limits to show moving window
            self.ax_torque.set_xlim(self.step_counter - 100, self.step_counter)
            self.ax_angles.set_xlim(self.step_counter - 100, self.step_counter)
        
        # Update trajectory plot
        self.target_line.set_data(self.target_x, self.target_z)
        self.actual_line.set_data(self.actual_x, self.actual_z)
        
        # Update torque plots
        torque_data = [self.torques_joint1, self.torques_joint2, self.torques_joint3]
        for line, torques in zip(self.torque_lines, torque_data):
            line.set_data(self.time_steps, torques)
        
        # Update joint angle plots
        angle_data = [self.angles_joint1, self.angles_joint2, self.angles_joint3]
        for line, angles in zip(self.angle_lines, angle_data):
            line.set_data(self.time_steps, angles)
        
        # Adjust y-axis limits of torque plot if needed
        all_torques = self.torques_joint1 + self.torques_joint2 + self.torques_joint3
        if all_torques:
            min_torque = min(all_torques)
            max_torque = max(all_torques)
            margin = (max_torque - min_torque) * 0.1
            self.ax_torque.set_ylim(min_torque - margin, max_torque + margin)
        
        # Adjust y-axis limits of angle plot if needed
        all_angles = self.angles_joint1 + self.angles_joint2 + self.angles_joint3
        if all_angles:
            min_angle = min(all_angles)
            max_angle = max(all_angles)
            margin = (max_angle - min_angle) * 0.1
            self.ax_angles.set_ylim(min_angle - margin, max_angle + margin)
        
        # Redraw canvas
        self.canvas.draw()
        
        # Process Tkinter events
        self.root.update()
    
    def update_circle_number(self, number):
        self.current_circle_number = number
        self.circle_label.configure(text=f"Circle #: {number}")
        
    def clear_data(self):
        # Clear trajectory data
        self.target_x = []
        self.target_z = []
        self.actual_x = []
        self.actual_z = []
        self.target_line.set_data([], [])
        self.actual_line.set_data([], [])
        
        # Clear torque data
        self.time_steps = []
        self.torques_joint1 = []
        self.torques_joint2 = []
        self.torques_joint3 = []
        
        # Clear angle data
        self.angles_joint1 = []
        self.angles_joint2 = []
        self.angles_joint3 = []
        self.step_counter = 0
        
        # Clear all lines
        for line in self.torque_lines:
            line.set_data([], [])
        for line in self.angle_lines:
            line.set_data([], [])
        
        self.canvas.draw()
        
    def close(self):
        self.root.destroy()

def setup_simulation(urdf_path):
    """
    Sets up the PyBullet simulation environment.
    Connects to the physics server, sets gravity, loads the ground plane,
    and loads the robot from the provided URDF file.

    Args:
        urdf_path (str): The file path to the robot's URDF file.

    Returns:
        int: The unique ID of the loaded robot.
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
        cameraPitch=
        0,
        cameraTargetPosition=[0.25, 0, 0.25]
    )
    
    return robot_id

def get_joint_info(robot_id):
    """
    Retrieves information about the robot's joints, focusing on the actuated ones.

    Args:
        robot_id (int): The unique ID of the robot.

    Returns:
        tuple: A tuple containing:
            - list: Indices of the three actuated arm joints.
            - int: The link index of the end-effector.
    """
    actuated_joint_indices = []
    joint_names = []
    
    num_joints = p.getNumJoints(robot_id)
    end_effector_link_index = -1

    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode('utf-8')
        
        # Identify the three actuated joints
        if joint_name in ["link_1_to_link_2", "link_2_to_link_3", "link_3_to_link_4"]:
            actuated_joint_indices.append(i)
            joint_names.append(joint_name)
            # Enable force/torque sensor for data logging
            p.enableJointForceTorqueSensor(robot_id, i, 1)

        # Identify the end-effector link (gripper base)
        if joint_name == "link_4_to_gripper":
            end_effector_link_index = i

    print("Actuated Joints Found:")
    for i, name in zip(actuated_joint_indices, joint_names):
        print(f"- {name} (Index: {i})")
    print(f"End-Effector Link Index: {end_effector_link_index}\n")

    return actuated_joint_indices, end_effector_link_index

def generate_circular_trajectory(cx, cy, radius, num_points):
    """
    Generates a list of (x, y) points for a circular trajectory.

    Args:
        cx (float): X-coordinate of the circle's center.
        cy (float): Y-coordinate of the circle's center (used as Z in the simulation).
        radius (float): The radius of the circle.
        num_points (int): The number of points to generate for the trajectory.

    Returns:
        list: A list of (x, y) tuples representing the trajectory points.
    """
    trajectory = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = cx + radius * math.cos(angle)
        z = cy + radius * math.sin(angle)
        trajectory.append((x, z))
    return trajectory

def custom_inverse_kinematics(target_x, target_z, link_lengths):
    """
    Custom inverse kinematics solver based on the simplified mathematical model.
    This solver uses optimization to find the joint angles.

    Args:
        target_x (float): The target X-coordinate for the end-effector.
        target_z (float): The target Z-coordinate for the end-effector.
        link_lengths (dict): A dictionary containing the lengths of the robot links.

    Returns:
        numpy.ndarray: An array of the three calculated joint angles, or None if no solution is found.
    """
    l1 = link_lengths['l1']
    l2 = link_lengths['l2']
    l3 = link_lengths['l3']
    l4 = link_lengths['l4']

    def forward_kinematics_for_ik(theta):
        # This is a helper function for the optimizer
        x = l2 * np.sin(theta[0]) + l3 * np.sin(theta[0] + theta[1]) + l4 * np.sin(theta[0] + theta[1] + theta[2])
        z = l1 + l2 * np.cos(theta[0]) + l3 * np.cos(theta[0] + theta[1]) + l4 * np.cos(theta[0] + theta[1] + theta[2])
        return x, z

    def objective(theta):
        # The objective is to minimize the distance between the calculated and target end-effector positions
        end_x, end_z = forward_kinematics_for_ik(theta)
        return np.sqrt((end_x - target_x)**2 + (end_z - target_z)**2)
    
    # Initial guess for the joint angles
    theta0 = [0, 0, 0]
    
    # Joint limits [-pi, pi] for all three joints
    bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    
    # Use scipy's optimizer to find the best joint angles
    result = minimize(objective, theta0, method='SLSQP', bounds=bounds)
    
    if result.success and result.fun < 1e-3: # Check if a good solution was found
        return result.x
    else:
        # print(f"Warning: IK solution not found or has high error for target ({target_x:.2f}, {target_z:.2f})")
        return None


def main():
    """
    Main function to run the simulation, track the trajectory, and log data.
    """
    # --- Simulation and Trajectory Parameters ---
    urdf_file = "/home/san/Public/pybullet_4dof/urdf/robot.urdf"
    
    # Link lengths derived from URDF for our custom IK solver
    link_lengths = {'l1': 0.27, 'l2': 0.15, 'l3': 0.15, 'l4': 0.10}

    if not os.path.exists(urdf_file):
        print(f"Error: '{urdf_file}' not found. Please update the URDF file path.")
        return

    # --- Setup ---
    robot_id = setup_simulation(urdf_file)
    actuated_joint_indices, ee_link_index = get_joint_info(robot_id)

    # Initialize visualizer
    visualizer = RealTimeVisualizer()
    
    # Get initial trajectory parameters
    center_x, center_z, radius = visualizer.get_trajectory_params()
    num_trajectory_points = 500
    
    # Generate initial trajectory
    trajectory_points = generate_circular_trajectory(center_x, center_z, radius, num_trajectory_points)
    
    # Draw initial trajectory in PyBullet workspace
    for i in range(num_trajectory_points):
        p1 = [trajectory_points[i][0], 0, trajectory_points[i][1]]
        p2 = [trajectory_points[(i + 1) % num_trajectory_points][0], 0, trajectory_points[(i + 1) % num_trajectory_points][1]]
        p.addUserDebugLine(p1, p2, lineColorRGB=[1, 0, 0], lineWidth=2)
    
    # Variable to store previous trajectory parameters for change detection
    prev_params = (center_x, center_z, radius)

    circle_count = 0
    
    print(f"Starting simulation. Use the sliders to adjust PD gains and trajectory parameters in real-time.")
    print("Click 'Save Current Circle Data' button to save data for the current circle.")
    print("Press Ctrl+C or close the window to stop the simulation.")

    try:
        # --- Main Control Loop ---
        while True:
            # Check if trajectory parameters have changed
            current_params = visualizer.get_trajectory_params()
            if current_params != prev_params:
                # Update trajectory
                center_x, center_z, radius = current_params
                trajectory_points = generate_circular_trajectory(center_x, center_z, radius, num_trajectory_points)
                prev_params = current_params
                
                # Clear the debug lines
                p.removeAllUserDebugItems()
                
                # Draw new trajectory
                for i in range(num_trajectory_points):
                    p1 = [trajectory_points[i][0], 0, trajectory_points[i][1]]
                    p2 = [trajectory_points[(i + 1) % num_trajectory_points][0], 0, trajectory_points[(i + 1) % num_trajectory_points][1]]
                    p.addUserDebugLine(p1, p2, lineColorRGB=[1, 0, 0], lineWidth=2)
            
            # Clear visualization at the start of each circle
            visualizer.clear_data()
            visualizer.clear_cache()
            
            for i, (target_x, target_z) in enumerate(trajectory_points):
                
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

                p.stepSimulation()
                time.sleep(1./240.)

                # Get current end-effector position for visualization
                link_state = p.getLinkState(robot_id, ee_link_index)
                actual_pos = link_state[0]
                
                # Get current joint torques
                joint_torques = []
                for joint_index in actuated_joint_indices:
                    state = p.getJointState(robot_id, joint_index)
                    joint_torques.append(state[3])  # Torque is at index 3
                
                # Get current joint angles
                joint_angles = []
                for joint_index in actuated_joint_indices:
                    state = p.getJointState(robot_id, joint_index)
                    joint_angles.append(state[0]) # Angle is at index 0
                
                # Update visualization
                visualizer.update(target_x, target_z, actual_pos[0], actual_pos[2], joint_torques, joint_angles)

                # --- Data Collection ---
                timestamp = time.time()
                
                joint_angles_for_data = []
                joint_torques_for_data = []
                for joint_index in actuated_joint_indices:
                    state = p.getJointState(robot_id, joint_index)
                    joint_angles_for_data.append(state[0]) # Angle is at index 0
                    joint_torques_for_data.append(state[3])  # Torque is at index 3
                
                # Cache the data point
                data_point = [
                    timestamp,
                    target_x, target_z,
                    actual_pos[0], actual_pos[2],
                    *joint_angles_for_data,
                    *joint_torques_for_data,
                    kp, kd,
                    radius,
                    center_x,
                    center_z
                ]
                visualizer.cache_data_point(data_point)

                # Check if a circle is completed
                if i == num_trajectory_points - 1:
                    circle_count += 1
                    visualizer.update_circle_number(circle_count)
                    print(f"Circle {circle_count} completed. Gains: Kp={kp:.2f}, Kd={kd:.1f}, "
                          f"Center: ({center_x:.2f}, {center_z:.2f}), Radius: {radius:.2f}")
                    
                    # Check if save was requested
                    if visualizer.save_requested:
                        visualizer.save_cached_data(circle_count)
                        visualizer.save_requested = False

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    finally:
        # Close visualizer
        visualizer.close()
        p.disconnect()

if __name__ == "__main__":
    main()
