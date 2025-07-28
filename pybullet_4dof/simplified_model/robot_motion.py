import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.optimize import minimize
from typing import List, Tuple, Optional

class RobotModel:
    def __init__(self):
        # Link lengths from URDF
        self.l1: float = 0.27  # Link 1 length (fixed vertical)
        self.l2: float = 0.15  # Link 2 length
        self.l3: float = 0.15  # Link 3 length
        self.l4: float = 0.10  # Link 4 length
        self.l_gripper: float = 0.00  # Gripper length
        
        # Joint limits from URDF (only for moving joints)
        self.joint_limits = np.array([
            [-3.14, 3.14],  # Joint 2
            [-3.14, 3.14],  # Joint 3
            [-3.14, 3.14]   # Joint 4
        ])
        
    def forward_kinematics(self, theta: List[float]) -> np.ndarray:
        """
        Calculate forward kinematics for the planar robot (X-Z plane)
        theta: list of 3 joint angles [θ2, θ3, θ4] for the moving joints
        Returns: list of points defining the robot's configuration
        """
        # Base position
        x: List[float] = [0.0]  # X coordinates
        z: List[float] = [0.0]  # Z coordinates
        
        # Link 1 - Fixed vertical
        x.append(0.0)
        z.append(self.l1)
        
        # Current angle (starts at 0 since first link is vertical)
        current_angle = 0.0
        
        # Link 2
        current_angle += theta[0]  # First moving joint (θ2)
        x.append(x[-1] + self.l2 * np.sin(current_angle))
        z.append(z[-1] + self.l2 * np.cos(current_angle))
        
        # Link 3
        current_angle += theta[1]  # Second moving joint (θ3)
        x.append(x[-1] + self.l3 * np.sin(current_angle))
        z.append(z[-1] + self.l3 * np.cos(current_angle))
        
        # Link 4
        current_angle += theta[2]  # Third moving joint (θ4)
        x.append(x[-1] + self.l4 * np.sin(current_angle))
        z.append(z[-1] + self.l4 * np.cos(current_angle))
        
        # Gripper
        x.append(x[-1] + self.l_gripper * np.sin(current_angle))
        z.append(z[-1] + self.l_gripper * np.cos(current_angle))
        
        return np.array([x, z])
    
    def plot_robot(self, ax: Axes, theta: List[float], color: str = 'blue') -> None:
        """Plot the robot in a given configuration in X-Z plane"""
        points = self.forward_kinematics(theta)
        x, z = points
        
        # Plot links
        ax.plot(x, z, color=color, linewidth=2)
        
        # Plot joints as circles (excluding base and first joint which is fixed)
        ax.scatter(x[2:-1], z[2:-1], color='red', s=50)  # Only moving joints
        
        # Plot base joint in black (fixed)
        ax.scatter(x[1], z[1], color='black', s=50)
        
        # Plot end-effector
        ax.scatter(x[-1], z[-1], color='green', s=50)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True)
        
        # Label axes
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')

def inverse_kinematics(robot, target_x, target_z):
    """
    Solve inverse kinematics to reach target position
    Returns: joint angles [θ2, θ3, θ4]
    """
    def objective(theta):
        # Get end effector position for given joint angles
        points = robot.forward_kinematics(theta)
        end_x, end_z = points[0, -1], points[1, -1]
        
        # Calculate distance to target
        return np.sqrt((end_x - target_x)**2 + (end_z - target_z)**2)
    
    # Initial guess (vertical configuration)
    theta0 = [0, 0, 0]
    
    # Joint limits from robot
    bounds = robot.joint_limits
    
    # Solve optimization problem
    result = minimize(objective, theta0, method='SLSQP', bounds=bounds)
    
    if result.success:
        return result.x
    else:
        raise ValueError("Could not find valid joint angles for target position")

class RobotControlGUI:
    def __init__(self):
        self.robot = RobotModel()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Robot Arm Control")
        
        # Create left frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create coordinate input fields
        ttk.Label(control_frame, text="Start Position:").grid(row=0, column=0, columnspan=2)
        ttk.Label(control_frame, text="X:").grid(row=1, column=0)
        ttk.Label(control_frame, text="Z:").grid(row=2, column=0)
        self.start_x = ttk.Entry(control_frame, width=10)
        self.start_z = ttk.Entry(control_frame, width=10)
        self.start_x.grid(row=1, column=1)
        self.start_z.grid(row=2, column=1)
        
        ttk.Label(control_frame, text="End Position:").grid(row=3, column=0, columnspan=2)
        ttk.Label(control_frame, text="X:").grid(row=4, column=0)
        ttk.Label(control_frame, text="Z:").grid(row=5, column=0)
        self.end_x = ttk.Entry(control_frame, width=10)
        self.end_z = ttk.Entry(control_frame, width=10)
        self.end_x.grid(row=4, column=1)
        self.end_z.grid(row=5, column=1)
        
        # Create move button
        ttk.Button(control_frame, text="Move Robot", command=self.move_robot).grid(row=6, column=0, columnspan=2)
        
        # Create matplotlib figure
        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1)
        
        # Set up the plot
        self.setup_plot()
        
        # Initial robot configuration
        self.current_angles = [0, 0, 0]
        self.update_plot()
    
    def setup_plot(self):
        """Set up the matplotlib plot"""
        self.ax.set_xlim([-0.8, 0.8])
        self.ax.set_ylim([0, 0.8])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Z (m)')
        self.ax.set_title('3-DOF Planar Robot Arm')
        self.ax.grid(True)
    
    def update_plot(self):
        """Update the robot visualization"""
        self.ax.clear()
        self.setup_plot()
        self.robot.plot_robot(self.ax, self.current_angles)
        self.canvas.draw()
    
    def move_robot(self):
        """Move the robot from start to end position"""
        try:
            # Get target positions
            start_x = float(self.start_x.get())
            start_z = float(self.start_z.get())
            end_x = float(self.end_x.get())
            end_z = float(self.end_z.get())
            
            # Check if positions are within workspace
            max_reach = self.robot.l2 + self.robot.l3 + self.robot.l4 + self.robot.l_gripper
            if abs(start_x) > max_reach or abs(end_x) > max_reach or \
               start_z > max_reach or end_z > max_reach or \
               start_z < 0 or end_z < 0:
                messagebox.showerror("Error", "Position out of workspace!")
                return
            
            # Get joint angles for start and end positions
            start_angles = inverse_kinematics(self.robot, start_x, start_z)
            end_angles = inverse_kinematics(self.robot, end_x, end_z)
            
            # Create trajectory
            num_steps = 50
            trajectory = []
            for i in range(num_steps):
                t = i / (num_steps - 1)
                angles = start_angles * (1 - t) + end_angles * t
                trajectory.append(angles)
            
            # Execute trajectory
            for angles in trajectory:
                self.current_angles = angles
                self.update_plot()
                self.root.update()
                plt.pause(0.01)
            
        except ValueError as e:
            messagebox.showerror("Error", str(e))
        except Exception as e:
            messagebox.showerror("Error", "Invalid input!")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

class CircularTrajectoryGUI:
    def __init__(self):
        self.robot = RobotModel()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Circular Trajectory Control")
        
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
        self.center_x.insert(0, "0.2")  # Default values
        self.center_z.insert(0, "0.4")
        
        ttk.Label(control_frame, text="Radius:").grid(row=3, column=0)
        self.radius = ttk.Entry(control_frame, width=10)
        self.radius.grid(row=3, column=1)
        self.radius.insert(0, "0.1")  # Default value
        
        # Create control buttons
        ttk.Button(control_frame, text="Start Motion", command=self.start_circular_motion).grid(row=4, column=0, columnspan=2, pady=5)
        ttk.Button(control_frame, text="Stop Motion", command=self.stop_motion).grid(row=5, column=0, columnspan=2, pady=5)
        
        # Create matplotlib figure
        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1)
        
        # Motion control variables
        self.is_running = False
        self.current_angles = [0, 0, 0]
        
        # Set up the plot
        self.setup_plot()
        self.update_plot()
    
    def setup_plot(self):
        """Set up the matplotlib plot"""
        self.ax.set_xlim([-0.8, 0.8])
        self.ax.set_ylim([0, 0.8])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Z (m)')
        self.ax.set_title('Circular Trajectory Control')
        self.ax.grid(True)
    
    def update_plot(self):
        """Update the robot visualization"""
        self.ax.clear()
        self.setup_plot()
        self.robot.plot_robot(self.ax, self.current_angles)
        
        # If motion is active, plot the target circle
        if hasattr(self, 'center_x_val'):
            circle = plt.Circle(
                (float(self.center_x_val), float(self.center_z_val)),
                float(self.radius_val),
                fill=False,
                linestyle='--',
                color='red'
            )
            self.ax.add_patch(circle)
        
        self.canvas.draw()
    
    def stop_motion(self):
        """Stop the circular motion"""
        self.is_running = False
    
    def start_circular_motion(self):
        """Start the circular motion with given parameters"""
        try:
            # Get circle parameters
            self.center_x_val = float(self.center_x.get())
            self.center_z_val = float(self.center_z.get())
            self.radius_val = float(self.radius.get())
            
            # Validate parameters
            max_reach = self.robot.l2 + self.robot.l3 + self.robot.l4 + self.robot.l_gripper
            
            # Check if circle is within workspace
            if (abs(self.center_x_val) + self.radius_val > max_reach or 
                self.center_z_val + self.radius_val > max_reach or 
                self.center_z_val - self.radius_val < 0):
                messagebox.showerror("Error", "Circle trajectory is outside workspace!")
                return
            
            # Start motion in a separate thread to keep GUI responsive
            self.is_running = True
            self.execute_circular_motion()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid input! Please enter valid numbers.")
    
    def execute_circular_motion(self):
        """Execute the circular motion"""
        if not self.is_running:
            return
            
        try:
            # Generate points on circle
            t = np.linspace(0, 2*np.pi, 100)
            x = self.center_x_val + self.radius_val * np.cos(t)
            z = self.center_z_val + self.radius_val * np.sin(t)
            
            # Move through points
            for i in range(len(t)):
                if not self.is_running:
                    break
                    
                try:
                    # Calculate inverse kinematics for this point
                    self.current_angles = inverse_kinematics(self.robot, x[i], z[i])
                    self.update_plot()
                    self.root.update()
                    plt.pause(0.02)
                    
                except ValueError:
                    messagebox.showerror("Error", "Point unreachable!")
                    self.is_running = False
                    break
            
            # If we completed the circle and are still running, continue with another circle
            if self.is_running:
                self.root.after(10, self.execute_circular_motion)
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.is_running = False
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def simulate_wave_motion():
    """Simulate a waving motion of the robot"""
    robot = RobotModel()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set axis limits
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([0, 0.8])
    
    # Set title
    ax.set_title('3-DOF Planar Robot Arm Simulation')
    
    # Generate waving motion
    t = np.linspace(0, 2*np.pi, 50)
    for angle in t:
        ax.clear()
        
        # Set limits and labels
        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([0, 0.8])
        ax.set_title('3-DOF Planar Robot Arm Simulation')
        
        # Calculate joint angles for waving motion
        theta = [
            0.5 + 0.3 * np.sin(angle),  # Joint 2
            -0.5 + 0.3 * np.sin(angle), # Joint 3
            0.3 * np.sin(angle)         # Joint 4
        ]
        
        # Plot robot
        robot.plot_robot(ax, theta)
        
        # Draw and pause
        plt.draw()
        plt.pause(0.01)
    
    plt.show()

def demonstrate_workspace():
    """Demonstrate robot's workspace by showing multiple configurations"""
    robot = RobotModel()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set axis limits
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([0, 0.8])
    
    # Set title
    ax.set_title('Robot Workspace Demonstration')
    
    # Generate different configurations
    configs = [
        [0, 0, 0],                    # Home position
        [0.5, -0.5, 0],              # Reaching forward
        [0.8, -0.8, 0],              # Reaching up
        [-0.5, -0.5, 0.5],           # Reaching down
        [0.8, -0.3, 0]               # Extended position
    ]
    
    # Plot each configuration with different colors
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for theta, color in zip(configs, colors):
        robot.plot_robot(ax, theta, color=color)
    
    plt.show()

if __name__ == "__main__":
    # Ask user which mode to run
    print("Choose mode:")
    print("1. Point-to-Point Control")
    print("2. Circular Trajectory")
    print("3. Wave Motion")
    print("4. Workspace Demonstration")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == "1":
        # Create and run point-to-point GUI
        gui = RobotControlGUI()
        print("Starting Point-to-Point Control GUI...")
        gui.run()
    
    elif choice == "2":
        # Create and run circular trajectory GUI
        gui = CircularTrajectoryGUI()
        print("Starting Circular Trajectory Control GUI...")
        gui.run()
    
    elif choice == "3":
        # Run wave motion simulation
        print("Simulating wave motion...")
        simulate_wave_motion()
    
    elif choice == "4":
        # Run workspace demonstration
        print("Demonstrating workspace...")
        demonstrate_workspace()
    
    else:
        print("Invalid choice!") 