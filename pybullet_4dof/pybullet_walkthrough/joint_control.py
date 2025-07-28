import pybullet as p
import numpy as np
from spawn_robot import spawn_robot
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque
import threading

def rad2deg(rad):
    return rad * 180.0 / np.pi

def deg2rad(deg):
    return deg * np.pi / 180.0

class JointDataPlotter:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.data = {}
        self.time_points = deque(maxlen=max_points)
        self.start_time = time.time()
        
        # Initialize time points
        current_time = 0
        for _ in range(max_points):
            self.time_points.append(current_time)
            current_time += 1/240.0  # Match simulation rate
    
    def initialize_joint(self, joint_name):
        if joint_name not in self.data:
            self.data[joint_name] = {
                'position': deque([0] * self.max_points, maxlen=self.max_points),
                'target_position': deque([0] * self.max_points, maxlen=self.max_points),
                'coordinates': {
                    'x': deque([0] * self.max_points, maxlen=self.max_points),
                    'z': deque([0] * self.max_points, maxlen=self.max_points)
                }
            }
    
    def update_data(self, t, joint_name, current_pos, target_pos, coordinates):
        self.initialize_joint(joint_name)
        self.time_points.append(t)
        
        self.data[joint_name]['position'].append(current_pos)
        self.data[joint_name]['target_position'].append(target_pos)
        self.data[joint_name]['coordinates']['x'].append(coordinates[0])
        self.data[joint_name]['coordinates']['z'].append(coordinates[2])

def create_display_window():
    root = tk.Tk()
    root.title("Joint Control Dashboard")
    root.geometry("1200x800")
    
    style = ttk.Style()
    style.configure("Joint.TLabelframe", padding=10)
    style.configure("Joint.TLabel", padding=5)
    
    info_frame = ttk.Frame(root, padding="10")
    info_frame.grid(row=0, column=0, sticky="nsew")
    
    plot_frame = ttk.Frame(root, padding="10")
    plot_frame.grid(row=0, column=1, sticky="nsew")
    
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=2)
    root.grid_rowconfigure(0, weight=1)
    
    return root, info_frame, plot_frame

def create_plots(plot_frame):
    figs = {}
    axes = {}
    canvases = {}
    
    plot_types = ['Position Tracking', 'X-Z Trajectory']
    
    for i, plot_type in enumerate(plot_types):
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=i, column=0, sticky="nsew", padx=5, pady=5)
        plot_frame.grid_rowconfigure(i, weight=1)
        
        ax.set_title(plot_type)
        ax.grid(True)
        
        figs[plot_type] = fig
        axes[plot_type] = ax
        canvases[plot_type] = canvas
    
    return figs, axes, canvases

def update_plots(plotter, axes, canvases):
    time_data = list(plotter.time_points)
    if not time_data:
        return
    
    # Clear all plots
    for ax in axes.values():
        ax.clear()
        ax.grid(True)
    
    # Update all plots
    for joint_name, joint_data in plotter.data.items():
        positions = list(joint_data['position'])
        targets = list(joint_data['target_position'])
        
        if len(positions) == len(time_data):
            # Position Tracking plot
            axes['Position Tracking'].plot(time_data, positions, label=f'{joint_name} Current')
            axes['Position Tracking'].plot(time_data, targets, '--', label=f'{joint_name} Target')
            
            # Coordinates plot
            coord_x = list(joint_data['coordinates']['x'])
            coord_z = list(joint_data['coordinates']['z'])
            if len(coord_x) == len(coord_z):
                axes['X-Z Trajectory'].plot(coord_x, coord_z, 'o-', label=joint_name, markersize=2)
    
    # Set titles and labels
    axes['Position Tracking'].set_title('Position Tracking')
    axes['Position Tracking'].set_xlabel('Time (s)')
    axes['Position Tracking'].set_ylabel('Position (degrees)')
    axes['Position Tracking'].legend(loc='upper right')
    
    axes['X-Z Trajectory'].set_title('X-Z Trajectory')
    axes['X-Z Trajectory'].set_xlabel('X Position (m)')
    axes['X-Z Trajectory'].set_ylabel('Z Position (m)')
    axes['X-Z Trajectory'].legend(loc='upper right')
    
    # Update all canvases
    for canvas in canvases.values():
        canvas.draw()

def setup_joint_sliders(robot):
    joint_params = {}
    revolute_joints = {}
    
    num_joints = p.getNumJoints(robot)
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        
        if joint_type == p.JOINT_REVOLUTE:
            revolute_joints[joint_name] = i
            
            lower_limit_rad = joint_info[8]
            upper_limit_rad = joint_info[9]
            
            lower_limit_deg = rad2deg(lower_limit_rad)
            upper_limit_deg = rad2deg(upper_limit_rad)
            
            slider = p.addUserDebugParameter(
                f"{joint_name} (deg)",
                lower_limit_deg,
                upper_limit_deg,
                0
            )
            
            joint_params[joint_name] = {
                'slider': slider,
                'limits_rad': (lower_limit_rad, upper_limit_rad),
                'limits_deg': (lower_limit_deg, upper_limit_deg)
            }
            
            print(f"\nJoint {i}: {joint_name}")
            print(f"  Type: Revolute")
            print(f"  Limits (deg): [{lower_limit_deg:.1f}, {upper_limit_deg:.1f}]")
    
    return joint_params, revolute_joints

def update_display(joint_params, revolute_joints, info_frame, plotter, axes, canvases, t):
    # Clear previous widgets
    for widget in info_frame.winfo_children():
        widget.destroy()
    
    # Update joint state displays
    for name, params in joint_params.items():
        joint_idx = revolute_joints[name]
        
        # Get current joint state
        state = p.getJointState(robot, joint_idx)
        current_pos_rad = state[0]
        current_pos_deg = rad2deg(current_pos_rad)
        
        # Get target position
        target_pos_deg = p.readUserDebugParameter(params['slider'])
        
        # Get coordinates
        link_state = p.getLinkState(robot, joint_idx)
        link_pos = link_state[0]
        
        # Create joint frame
        joint_frame = ttk.LabelFrame(info_frame, text=name, style="Joint.TLabelframe")
        joint_frame.pack(fill="x", padx=5, pady=5)
        
        # Add labels with joint information
        ttk.Label(joint_frame,
                 text=f"Current Position: {current_pos_deg:.1f}°",
                 style="Joint.TLabel").pack(anchor="w")
        
        ttk.Label(joint_frame,
                 text=f"Target Position: {target_pos_deg:.1f}°",
                 style="Joint.TLabel").pack(anchor="w")
        
        ttk.Label(joint_frame,
                 text=f"Coordinates: X={link_pos[0]:.3f}, Z={link_pos[2]:.3f} m",
                 style="Joint.TLabel").pack(anchor="w")
        
        # Update plotter data
        plotter.update_data(t, name, current_pos_deg, target_pos_deg, link_pos)
    
    update_plots(plotter, axes, canvases)

def control_joints_with_sliders(robot, joint_params, revolute_joints):
    # Create display window
    root, info_frame, plot_frame = create_display_window()
    figs, axes, canvases = create_plots(plot_frame)
    plotter = JointDataPlotter()
    
    t = 0  # Time variable
    try:
        while True:
            # Update joint states based on sliders
            for name, params in joint_params.items():
                joint_idx = revolute_joints[name]
                target_pos_deg = p.readUserDebugParameter(params['slider'])
                target_pos_rad = deg2rad(target_pos_deg)
                
                p.setJointMotorControl2(
                    robot,
                    joint_idx,
                    p.POSITION_CONTROL,
                    target_pos_rad,
                    force=100
                )
            
            # Update display and plots
            update_display(joint_params, revolute_joints, info_frame, plotter, axes, canvases, t)
            
            p.stepSimulation()
            time.sleep(1./240.)  # 240 Hz simulation
            t += 1./240.
            
            # Update Tkinter window
            root.update()
            
    except KeyboardInterrupt:
        # Clean up debug items
        for params in joint_params.values():
            p.removeUserDebugItem(params['slider'])
        root.destroy()
        p.disconnect()
    except tk.TclError:  # Handle window closing
        # Clean up debug items
        for params in joint_params.values():
            p.removeUserDebugItem(params['slider'])
        p.disconnect()

def main():
    # Spawn robot and get the robot ID
    global robot  # Make robot accessible to all functions
    robot, physics_client = spawn_robot()
    
    # Setup joint sliders
    joint_params, revolute_joints = setup_joint_sliders(robot)
    
    # Start control loop
    control_joints_with_sliders(robot, joint_params, revolute_joints)

if __name__ == "__main__":
    main() 