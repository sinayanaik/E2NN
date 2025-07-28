import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from robot_motion import RobotModel, inverse_kinematics

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
        self.center_x.insert(0, "0.25")  # Default X center
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
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=1)
        
        # Motion control variables
        self.is_running = False
        self.current_angles = [0, 0, 0]
        self.phase = 0  # Track circle phase
        
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
        self.ax.set_aspect('equal')
    
    def update_plot(self):
        """Update the robot visualization"""
        self.ax.clear()
        self.setup_plot()
        self.robot.plot_robot(self.ax, self.current_angles)
        
        # If motion is active, plot the target circle
        if hasattr(self, 'center_x_val'):
            circle = plt.Circle(
                (self.center_x_val, self.center_z_val),
                self.radius_val,
                fill=False,
                linestyle='--',
                color='red'
            )
            self.ax.add_patch(circle)
            
            # Plot current target point
            x = self.center_x_val + self.radius_val * np.cos(self.phase)
            z = self.center_z_val + self.radius_val * np.sin(self.phase)
            self.ax.plot(x, z, 'rx', markersize=10)
        
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
            
            # Start motion
            self.is_running = True
            self.execute_circular_motion()
            
        except ValueError:
            messagebox.showerror("Error", "Invalid input! Please enter valid numbers.")
    
    def execute_circular_motion(self):
        """Execute the circular motion"""
        if not self.is_running:
            return
            
        try:
            # Get current speed setting
            speed = self.speed.get()
            
            # Update phase based on speed
            self.phase = (self.phase + 0.05 * speed) % (2 * np.pi)
            
            # Calculate current target point
            x = self.center_x_val + self.radius_val * np.cos(self.phase)
            z = self.center_z_val + self.radius_val * np.sin(self.phase)
            
            try:
                # Calculate inverse kinematics for this point
                self.current_angles = inverse_kinematics(self.robot, x, z)
                self.update_plot()
                
            except ValueError:
                messagebox.showerror("Error", "Point unreachable!")
                self.is_running = False
                return
            
            # Schedule next update if still running
            if self.is_running:
                self.root.after(20, self.execute_circular_motion)
                
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.is_running = False
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    # Create and run circular trajectory GUI
    gui = CircularTrajectoryGUI()
    print("Starting Circular Trajectory Control GUI...")
    gui.run() 