# Joint State Display in PyBullet - Real-Time Visualization and Data Logging

## Overview

The system provides:
1. Real-time numerical display of joint states
2. Live plotting of joint data
3. CSV data logging for post-analysis
4. Automatic joint movement for testing

## Code Implementation

### Core Classes

#### 1. DataLogger Class
```python
import csv
import os
from datetime import datetime

class DataLogger:
    def __init__(self):
        os.makedirs('data', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f'data/robot_data_{timestamp}.csv'
        self.headers = ['time', 'joint_name', 'position', 'velocity', 
                       'force_x', 'force_y', 'force_z',
                       'torque', 'coord_x', 'coord_z']
        
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
    
    def log_data(self, t, joint_name, position, velocity, forces, torque, coordinates):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                t,
                joint_name,
                position,
                velocity,
                forces[0] if len(forces) > 0 else 0,
                forces[1] if len(forces) > 1 else 0,
                forces[2] if len(forces) > 2 else 0,
                torque,
                coordinates[0],
                coordinates[2]
            ])
```

#### 2. JointPlotter Class
```python
from collections import deque
import time

class JointPlotter:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.data = {}
        self.time_points = deque(maxlen=max_points)
        self.start_time = time.time()
        self.logger = DataLogger()
        
        # Initialize time points
        current_time = 0
        for _ in range(max_points):
            self.time_points.append(current_time)
            current_time += 1/30.0  # 30Hz update rate
    
    def initialize_joint(self, joint_name):
        if joint_name not in self.data:
            self.data[joint_name] = {
                'position': deque([0] * self.max_points, maxlen=self.max_points),
                'velocity': deque([0] * self.max_points, maxlen=self.max_points),
                'forces': {
                    'x': deque([0] * self.max_points, maxlen=self.max_points),
                    'y': deque([0] * self.max_points, maxlen=self.max_points),
                    'z': deque([0] * self.max_points, maxlen=self.max_points)
                },
                'torques': {
                    'x': deque([0] * self.max_points, maxlen=self.max_points),
                    'y': deque([0] * self.max_points, maxlen=self.max_points),
                    'z': deque([0] * self.max_points, maxlen=self.max_points)
                },
                'coordinates': {
                    'x': deque([0] * self.max_points, maxlen=self.max_points),
                    'z': deque([0] * self.max_points, maxlen=self.max_points)
                }
            }
    
    def update_data(self, t, joint_name, position, velocity, forces, torques, coordinates):
        self.initialize_joint(joint_name)
        self.logger.log_data(t, joint_name, position, velocity, forces, torques, coordinates)
        
        self.time_points.append(t)
        self.data[joint_name]['position'].append(position)
        self.data[joint_name]['velocity'].append(velocity)
        
        self.data[joint_name]['forces']['x'].append(forces[0] if len(forces) > 0 else 0)
        self.data[joint_name]['forces']['y'].append(forces[1] if len(forces) > 1 else 0)
        self.data[joint_name]['forces']['z'].append(forces[2] if len(forces) > 2 else 0)
        
        self.data[joint_name]['torques']['x'].append(torques)
        self.data[joint_name]['torques']['y'].append(0)
        self.data[joint_name]['torques']['z'].append(0)
        
        self.data[joint_name]['coordinates']['x'].append(coordinates[0])
        self.data[joint_name]['coordinates']['z'].append(coordinates[2])
```

### Setup Functions

#### 1. Joint Setup
```python
def setup_joint_display(robot):
    joint_params = {}
    revolute_joints = {}
    
    num_joints = p.getNumJoints(robot)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            revolute_joints[joint_name] = i
            joint_params[joint_name] = {
                'type': joint_type,
                'limits': (joint_info[8], joint_info[9])
            }
    
    return joint_params, revolute_joints
```

#### 2. Display Creation
```python
def create_display_window():
    root = tk.Tk()
    root.title("Robot Joint Dashboard")
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

def create_plots(plot_frame, joint_names):
    figs = {}
    axes = {}
    canvases = {}
    
    plot_types = ['Position/Velocity', 'Forces', 'Torques', 'Coordinates']
    
    for i, plot_type in enumerate(plot_types):
        fig = Figure(figsize=(6, 3), dpi=100)
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
```

### Update Functions

#### 1. Display Updates
```python
def update_display(robot, joint_params, revolute_joints, root, info_frame, 
                  plotter, figs, axes, canvases, t):
    # Move joints
    move_joints(robot, joint_params, revolute_joints, t)
    
    # Clear previous widgets
    for widget in info_frame.winfo_children():
        widget.destroy()
    
    # Update joint state displays
    for name, params in joint_params.items():
        joint_idx = revolute_joints[name]
        
        state = p.getJointState(robot, joint_idx)
        current_pos = state[0]
        current_vel = state[1]
        reaction_forces = state[2]
        motor_torque = state[3]
        
        link_state = p.getLinkState(robot, joint_idx)
        link_pos = link_state[0]
        
        if params['type'] == p.JOINT_REVOLUTE:
            current_pos = rad2deg(current_pos)
            unit = "°"
        else:
            unit = "m"
        
        lower, upper = params['limits']
        if params['type'] == p.JOINT_REVOLUTE:
            lower = rad2deg(lower)
            upper = rad2deg(upper)
        range_val = upper - lower
        position_percent = ((current_pos - lower) / range_val) * 100 if range_val != 0 else 0
        
        # Create joint frame
        joint_frame = ttk.LabelFrame(info_frame, text=name, style="Joint.TLabelframe")
        joint_frame.pack(fill="x", padx=5, pady=5)
        
        # Add labels with joint information
        ttk.Label(joint_frame, 
                 text=f"Position: {current_pos:.1f}{unit} [{position_percent:.1f}%]",
                 style="Joint.TLabel").pack(anchor="w")
        
        ttk.Label(joint_frame,
                 text=f"Limits: [{lower:.1f}{unit} to {upper:.1f}{unit}]",
                 style="Joint.TLabel").pack(anchor="w")
        
        ttk.Label(joint_frame,
                 text=f"Coordinates: X={link_pos[0]:.3f}, Z={link_pos[2]:.3f} m",
                 style="Joint.TLabel").pack(anchor="w")
        
        ttk.Label(joint_frame,
                 text=f"Velocity: {current_vel:.1f} {'rad/s' if params['type'] == p.JOINT_REVOLUTE else 'm/s'}",
                 style="Joint.TLabel").pack(anchor="w")
        
        if len(reaction_forces) >= 3:
            ttk.Label(joint_frame,
                     text=f"Forces: [{reaction_forces[0]:.1f}, {reaction_forces[1]:.1f}, {reaction_forces[2]:.1f}] N",
                     style="Joint.TLabel").pack(anchor="w")
        
        ttk.Label(joint_frame,
                 text=f"Motor Torque: {motor_torque:.1f} N⋅m",
                 style="Joint.TLabel").pack(anchor="w")
        
        # Update plotter data
        plotter.update_data(
            t,
            name,
            current_pos,
            current_vel,
            reaction_forces[:3] if len(reaction_forces) >= 3 else [0, 0, 0],
            motor_torque,
            link_pos
        )
    
    update_plots(plotter, axes, canvases)
    root.update()

def update_plots(plotter, axes, canvases):
    time_data = list(plotter.time_points)
    if not time_data:
        return
    
    # Clear all plots
    for ax in axes.values():
        ax.clear()
        ax.grid(True)
    
    # Set fixed y-axis limits
    axes['Position/Velocity'].set_ylim(-180, 180)
    axes['Forces'].set_ylim(-10, 10)
    axes['Torques'].set_ylim(-5, 5)
    
    # Update all plots
    for joint_name, joint_data in plotter.data.items():
        positions = list(joint_data['position'])
        velocities = list(joint_data['velocity'])
        
        if len(positions) == len(time_data):
            # Position and Velocity plot
            axes['Position/Velocity'].plot(time_data, positions, label=f'{joint_name} Pos')
            axes['Position/Velocity'].plot(time_data, velocities, label=f'{joint_name} Vel')
            
            # Forces plot
            forces_x = list(joint_data['forces']['x'])
            forces_y = list(joint_data['forces']['y'])
            forces_z = list(joint_data['forces']['z'])
            axes['Forces'].plot(time_data, forces_x, label=f'{joint_name} X')
            axes['Forces'].plot(time_data, forces_y, label=f'{joint_name} Y')
            axes['Forces'].plot(time_data, forces_z, label=f'{joint_name} Z')
            
            # Torques plot
            torques_x = list(joint_data['torques']['x'])
            axes['Torques'].plot(time_data, torques_x, label=f'{joint_name}')
            
            # Coordinates plot
            coord_x = list(joint_data['coordinates']['x'])
            coord_z = list(joint_data['coordinates']['z'])
            if len(coord_x) == len(coord_z):
                axes['Coordinates'].plot(coord_x, coord_z, 'o-', label=joint_name, markersize=2)
    
    # Set titles and labels
    axes['Position/Velocity'].set_title('Position and Velocity')
    axes['Position/Velocity'].set_xlabel('Time (s)')
    axes['Position/Velocity'].set_ylabel('Position (deg) / Velocity (rad/s)')
    axes['Position/Velocity'].legend(loc='upper right')
    
    axes['Forces'].set_title('Forces')
    axes['Forces'].set_xlabel('Time (s)')
    axes['Forces'].set_ylabel('Force (N)')
    axes['Forces'].legend(loc='upper right')
    
    axes['Torques'].set_title('Motor Torque')
    axes['Torques'].set_xlabel('Time (s)')
    axes['Torques'].set_ylabel('Torque (N⋅m)')
    axes['Torques'].legend(loc='upper right')
    
    axes['Coordinates'].set_title('X-Z Trajectory')
    axes['Coordinates'].set_xlabel('X Position (m)')
    axes['Coordinates'].set_ylabel('Z Position (m)')
    axes['Coordinates'].legend(loc='upper right')
    
    # Update all canvases
    for canvas in canvases.values():
        canvas.draw()
```

#### 2. Joint Control
```python
def move_joints(robot, joint_params, revolute_joints, t):
    for name, params in joint_params.items():
        joint_idx = revolute_joints[name]
        if params['type'] == p.JOINT_REVOLUTE:
            freq = 0.5 + revolute_joints[name] * 0.2
            amplitude = 45  # degrees
            position = amplitude * math.sin(freq * t)
            p.setJointMotorControl2(robot, joint_idx, p.POSITION_CONTROL, 
                                  targetPosition=math.radians(position))
```

### Utility Functions
```python
def rad2deg(rad):
    return rad * 180.0 / np.pi
```

### Main Function
```python
def main():
    # Spawn robot
    robot, physics_client = spawn_robot()
    
    # Setup joint displays
    joint_params, revolute_joints = setup_joint_display(robot)
    
    # Start display loop
    display_joint_states(robot, joint_params, revolute_joints)

if __name__ == "__main__":
    main()
```

## Data File Format

### CSV Structure
```
time,joint_name,position,velocity,force_x,force_y,force_z,torque,coord_x,coord_z
0.000,shoulder,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
0.033,shoulder,1.2,0.5,-0.1,0.0,9.8,0.2,0.1,0.5
```

## Usage Example
```python
from joint_display import setup_joint_display, display_joint_states
from spawn_robot import spawn_robot

# Get robot
robot, _ = spawn_robot()

# Setup and start visualization
joint_params, revolute_joints = setup_joint_display(robot)
display_joint_states(robot, joint_params, revolute_joints)
```

## Data Analysis Example
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data/robot_data_20250704_114948.csv')

# Analyze specific joint
joint_data = data[data['joint_name'] == 'shoulder']

# Plot position over time
plt.figure(figsize=(10, 6))
plt.plot(joint_data['time'], joint_data['position'])
plt.title('Shoulder Joint Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (degrees)')
plt.grid(True)
plt.show()
``` 