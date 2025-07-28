# Joint Control in PyBullet - Interactive Control and Visualization

This guide explains how to control your robot's joints using interactive sliders while visualizing the motion in real-time plots.

## System Overview

The system consists of three main components:
1. PyBullet sliders for joint control
2. Real-time data visualization window
3. Data plotting system

## Components

### 1. Control Interface
- Sliders for each joint
- Shows position in degrees
- Real-time control
- Automatic limit handling

### 2. Visualization Window
The display window shows:
1. **Numerical Display** (Left Panel):
   - Current position
   - Target position
   - Joint coordinates

2. **Real-time Plots** (Right Panel):
   - Position tracking
   - X-Z trajectory

## Code Structure

### 1. Data Management Class
```python
class JointDataPlotter:
    def __init__(self, max_points=100):
        """Initialize data buffers for plotting"""
        self.max_points = max_points
        self.data = {}
        self.time_points = deque(maxlen=max_points)
        self.start_time = time.time()
        
        # Pre-fill time points for smooth start
        current_time = 0
        for _ in range(max_points):
            self.time_points.append(current_time)
            current_time += 1/240.0  # Match simulation rate
    
    def initialize_joint(self, joint_name):
        """Create data structures for a new joint"""
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
        """Update data buffers with new values"""
        self.initialize_joint(joint_name)
        self.time_points.append(t)
        
        self.data[joint_name]['position'].append(current_pos)
        self.data[joint_name]['target_position'].append(target_pos)
        self.data[joint_name]['coordinates']['x'].append(coordinates[0])
        self.data[joint_name]['coordinates']['z'].append(coordinates[2])
```

### 2. Display Creation
```python
def create_display_window():
    """Create the main visualization window"""
    root = tk.Tk()
    root.title("Joint Control Dashboard")
    root.geometry("1200x800")
    
    # Configure styles
    style = ttk.Style()
    style.configure("Joint.TLabelframe", padding=10)
    style.configure("Joint.TLabel", padding=5)
    
    # Create frames
    info_frame = ttk.Frame(root, padding="10")
    info_frame.grid(row=0, column=0, sticky="nsew")
    
    plot_frame = ttk.Frame(root, padding="10")
    plot_frame.grid(row=0, column=1, sticky="nsew")
    
    # Configure grid weights
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=2)
    root.grid_rowconfigure(0, weight=1)
    
    return root, info_frame, plot_frame
```

### 3. Plot Creation
```python
def create_plots(plot_frame):
    """Set up matplotlib plots in the window"""
    figs = {}
    axes = {}
    canvases = {}
    
    plot_types = ['Position Tracking', 'X-Z Trajectory']
    
    for i, plot_type in enumerate(plot_types):
        # Create figure and axis
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create canvas and add to frame
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=i, column=0, sticky="nsew")
        
        # Configure grid
        plot_frame.grid_rowconfigure(i, weight=1)
        
        # Store references
        figs[plot_type] = fig
        axes[plot_type] = ax
        canvases[plot_type] = canvas
    
    return figs, axes, canvases
```

### 4. Display Updates
```python
def update_display(joint_params, revolute_joints, info_frame, plotter, axes, canvases, t):
    """Update the display with current joint states"""
    # Clear old widgets
    for widget in info_frame.winfo_children():
        widget.destroy()
    
    # Update each joint's display
    for name, params in joint_params.items():
        joint_idx = revolute_joints[name]
        
        # Get current state
        state = p.getJointState(robot, joint_idx)
        current_pos_rad = state[0]
        current_pos_deg = rad2deg(current_pos_rad)
        target_pos_deg = p.readUserDebugParameter(params['slider'])
        
        # Get coordinates
        link_state = p.getLinkState(robot, joint_idx)
        link_pos = link_state[0]
        
        # Create display frame
        joint_frame = ttk.LabelFrame(info_frame, text=name)
        joint_frame.pack(fill="x", padx=5, pady=5)
        
        # Add information labels
        ttk.Label(joint_frame,
                 text=f"Current Position: {current_pos_deg:.1f}°").pack(anchor="w")
        ttk.Label(joint_frame,
                 text=f"Target Position: {target_pos_deg:.1f}°").pack(anchor="w")
        ttk.Label(joint_frame,
                 text=f"Coordinates: X={link_pos[0]:.3f}, Z={link_pos[2]:.3f} m").pack(anchor="w")
        
        # Update plot data
        plotter.update_data(t, name, current_pos_deg, target_pos_deg, link_pos)
    
    # Update plots
    update_plots(plotter, axes, canvases)
```

### 5. Plot Updates
```python
def update_plots(plotter, axes, canvases):
    """Update all plot displays"""
    time_data = list(plotter.time_points)
    if not time_data:
        return
    
    # Clear old plots
    for ax in axes.values():
        ax.clear()
        ax.grid(True)
    
    # Update each joint's plots
    for joint_name, joint_data in plotter.data.items():
        positions = list(joint_data['position'])
        targets = list(joint_data['target_position'])
        
        if len(positions) == len(time_data):
            # Position tracking plot
            axes['Position Tracking'].plot(time_data, positions, 
                                        label=f'{joint_name} Current')
            axes['Position Tracking'].plot(time_data, targets, '--', 
                                        label=f'{joint_name} Target')
            
            # Trajectory plot
            coord_x = list(joint_data['coordinates']['x'])
            coord_z = list(joint_data['coordinates']['z'])
            if len(coord_x) == len(coord_z):
                axes['X-Z Trajectory'].plot(coord_x, coord_z, 'o-', 
                                         label=joint_name, markersize=2)
    
    # Set labels and titles
    axes['Position Tracking'].set_title('Position Tracking')
    axes['Position Tracking'].set_xlabel('Time (s)')
    axes['Position Tracking'].set_ylabel('Position (degrees)')
    axes['Position Tracking'].legend(loc='upper right')
    
    axes['X-Z Trajectory'].set_title('X-Z Trajectory')
    axes['X-Z Trajectory'].set_xlabel('X Position (m)')
    axes['X-Z Trajectory'].set_ylabel('Z Position (m)')
    axes['X-Z Trajectory'].legend(loc='upper right')
    
    # Update display
    for canvas in canvases.values():
        canvas.draw()
```

## Understanding the Plots

### 1. Position Tracking Plot
- **X-axis**: Time (seconds)
- **Y-axis**: Joint position (degrees)
- **Solid lines**: Current position
- **Dashed lines**: Target position
- Shows how well joints follow commands

### 2. X-Z Trajectory Plot
- **X-axis**: X position (meters)
- **Y-axis**: Z position (meters)
- Shows joint movement paths
- Useful for:
  - Checking workspace coverage
  - Verifying motion paths
  - Detecting anomalies

## Performance Considerations

### Update Rates
- Simulation: 240 Hz
- Display updates: 240 Hz
- Plot buffer: 100 points
- Smooth real-time visualization

### Memory Management
- Circular buffers for data
- Limited history length
- Efficient plot updates
- Automatic cleanup

## Usage Example

```python
from joint_control import setup_joint_sliders, control_joints_with_sliders
from spawn_robot import spawn_robot

def main():
    # Create robot
    robot, _ = spawn_robot()
    
    # Setup control and visualization
    joint_params, revolute_joints = setup_joint_sliders(robot)
    
    # Start interactive control
    control_joints_with_sliders(robot, joint_params, revolute_joints)

if __name__ == "__main__":
    main()
```

## Tips for Use

### 1. Smooth Control
- Move sliders gradually
- Watch position tracking
- Monitor trajectory paths
- Check for overshoots

### 2. Performance
- Keep window visible
- Avoid rapid slider changes
- Let plots update naturally
- Close when finished

### 3. Data Analysis
- Watch position tracking
- Compare target vs actual
- Look for patterns
- Check coordinate paths

## Next Steps
1. Add data recording
2. Implement custom trajectories
3. Add velocity plotting
4. Create motion sequences
5. Add force monitoring 