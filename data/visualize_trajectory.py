import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import os
from datetime import datetime
import tkinter as tk

def create_stats_window(stats, kp, kd, center_x, center_z, radius=None, amplitude=None):
    """Create a separate window for statistics"""
    stats_window = tk.Tk()
    stats_window.title("Joint Statistics")
    
    frame = tk.Frame(stats_window, padx=10, pady=10)
    frame.pack(expand=True, fill='both')
    
    if radius is not None:
        param_text = f"Parameters: Kp={kp:.2f}, Kd={kd:.1f}, Radius={radius:.2f}m, Center=({center_x:.2f}, {center_z:.2f})m"
    else:
        param_text = f"Parameters: Kp={kp:.2f}, Kd={kd:.1f}, Amplitude={amplitude:.2f}, Center=({center_x:.2f}, {center_z:.2f})m"

    param_label = tk.Label(frame, text=param_text, font=('Arial', 10, 'bold'))
    param_label.pack(pady=(0, 10))
    
    columns = ['Joint', 'Metric', 'Mean', 'Std', 'RMS/Range']
    table_frame = tk.Frame(frame)
    table_frame.pack(expand=True, fill='both')
    
    for i, col in enumerate(columns):
        label = tk.Label(table_frame, text=col, font=('Arial', 10, 'bold'), 
                        relief='solid', padx=5, pady=2)
        label.grid(row=0, column=i, sticky='nsew')
    
    row_idx = 1
    for i, joint_stats in enumerate(stats, 1):
        for metric, values in joint_stats.items():
            tk.Label(table_frame, text=f'Joint {i}', padx=5, pady=2).grid(row=row_idx, column=0)
            tk.Label(table_frame, text=metric, padx=5, pady=2).grid(row=row_idx, column=1)
            
            if metric == 'Angle':
                tk.Label(table_frame, text=f'{values["Mean"]:.3f}', padx=5, pady=2).grid(row=row_idx, column=2)
                tk.Label(table_frame, text=f'{values["Std"]:.3f}', padx=5, pady=2).grid(row=row_idx, column=3)
                tk.Label(table_frame, text=f'{values["Max"]-values["Min"]:.3f}', padx=5, pady=2).grid(row=row_idx, column=4)
            else:
                tk.Label(table_frame, text=f'{values["Mean"]:.3f}', padx=5, pady=2).grid(row=row_idx, column=2)
                tk.Label(table_frame, text=f'{values["Std"]:.3f}', padx=5, pady=2).grid(row=row_idx, column=3)
                tk.Label(table_frame, text=f'{values["RMS"]:.3f}', padx=5, pady=2).grid(row=row_idx, column=4)
            row_idx += 1
    
    for i in range(5):
        table_frame.grid_columnconfigure(i, weight=1)
    
    return stats_window

def create_view_mode_window(callback):
    """Create a window to select trajectory view mode"""
    mode_window = tk.Tk()
    mode_window.title("Select View Mode")
    
    frame = tk.Frame(mode_window, padx=20, pady=20)
    frame.pack(expand=True, fill='both')
    
    tk.Label(frame, text="Select Trajectory View Mode:", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
    
    mode_var = tk.StringVar(value="complete")
    
    tk.Radiobutton(frame, text="Complete Motion (Forward + Return)", 
                   variable=mode_var, value="complete").pack(anchor='w', pady=5)
    tk.Radiobutton(frame, text="Forward Motion Only", 
                   variable=mode_var, value="forward").pack(anchor='w', pady=5)
    tk.Radiobutton(frame, text="Return Motion Only", 
                   variable=mode_var, value="return").pack(anchor='w', pady=5)
    
    tk.Button(frame, text="OK", command=lambda: [callback(mode_var.get()), mode_window.destroy()]).pack(pady=(10, 0))
    
    return mode_window

def add_navigation_buttons(fig):
    """Add navigation buttons for resetting views"""
    ax_reset = plt.axes([0.8, 0.01, 0.1, 0.03])
    button_reset = Button(ax_reset, 'Reset Views', color='lightgoldenrodyellow')
    
    def reset_views(event):
        for ax in fig.get_axes():
            ax.autoscale()
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()
    
    button_reset.on_clicked(reset_views)
    return button_reset

def visualize_trajectory_data(csv_file):
    """
    Visualize trajectory and dynamics data from a CSV file.
    """
    try:
        df = pd.read_csv(csv_file)
        
        is_figure8 = 'figure8' in os.path.basename(csv_file).lower()

        numeric_columns = [
            'target_x', 'target_z', 'actual_x', 'actual_z',
            'joint1_angle', 'joint2_angle', 'joint3_angle',
            'joint1_velocity', 'joint2_velocity', 'joint3_velocity',
            'joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration',
            'joint1_torque', 'joint2_torque', 'joint3_torque'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        time_steps = np.arange(len(df))
        
        def update_plots(view_mode):
            print(f"Creating visualization in {view_mode} mode...")
            plt.ion()
            
            plt.rcParams.update({'font.size': 10, 'toolbar': 'toolmanager'})
            fig = plt.figure(figsize=(16, 16))
            
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.5, wspace=0.35,
                                top=0.85, bottom=0.05, left=0.1, right=0.95)
            
            kp = df['kp'].iloc[0]
            kd = df['kd'].iloc[0]
            center_x = df['center_x'].iloc[0]
            center_z = df['center_z'].iloc[0]
            
            if is_figure8:
                amplitude = df['amplitude_A'].iloc[0]
                radius = None
                view_mode = "complete" 
            else:
                amplitude = None
                radius = df['radius'].iloc[0]
            
            mid_point = len(df) // 2
            
            if view_mode == "forward":
                data_slice = slice(0, mid_point)
                time_slice = time_steps[:mid_point]
                subtitle = "(Forward Motion Only)"
            elif view_mode == "return":
                data_slice = slice(mid_point, None)
                time_slice = time_steps[mid_point:]
                subtitle = "(Return Motion Only)"
            else:  # complete
                data_slice = slice(None)
                time_slice = time_steps
                subtitle = "(Forward: solid, Return: transparent)" if not is_figure8 else "(Figure-Eight Trajectory)"

            ax1 = fig.add_subplot(gs[0, 0])
            
            if view_mode == "complete" and not is_figure8:
                ax1.plot(df['target_x'].values[:mid_point], df['target_z'].values[:mid_point], 'r-', label='Target (Forward)', linewidth=2)
                ax1.plot(df['actual_x'].values[:mid_point], df['actual_z'].values[:mid_point], 'b--', label='Actual (Forward)', linewidth=2)
                ax1.plot(df['target_x'].values[mid_point:], df['target_z'].values[mid_point:], 'r-', alpha=0.5, label='Target (Return)', linewidth=2)
                ax1.plot(df['actual_x'].values[mid_point:], df['actual_z'].values[mid_point:], 'b--', alpha=0.5, label='Actual (Return)', linewidth=2)
            else:
                ax1.plot(df['target_x'].iloc[data_slice], df['target_z'].iloc[data_slice], 'r-', label='Target', linewidth=2)
                ax1.plot(df['actual_x'].iloc[data_slice], df['actual_z'].iloc[data_slice], 'b--', label='Actual', linewidth=2)
            
            ax1.set_xlabel('X Position (m)', labelpad=10)
            ax1.set_ylabel('Z Position (m)', labelpad=10)
            ax1.set_title(f'End-Effector Trajectory\n{subtitle}', pad=20, y=1.0)
            ax1.grid(True)
            ax1.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
            ax1.axis('equal')

            ax2 = fig.add_subplot(gs[0, 1])
            position_error = np.sqrt((df['target_x'].iloc[data_slice] - df['actual_x'].iloc[data_slice])**2 + (df['target_z'].iloc[data_slice] - df['actual_z'].iloc[data_slice])**2)
            ax2.plot(time_slice, position_error, 'k-', linewidth=2)
            ax2.set_xlabel('Time Step', labelpad=10)
            ax2.set_ylabel('Error (m)', labelpad=10)
            ax2.set_title('Position Error', pad=20, y=1.0)
            ax2.grid(True)
            
            mean_error = np.nanmean(position_error); max_error = np.nanmax(position_error)
            stats_text = f'Mean Error: {mean_error:.4f} m\nMax Error: {max_error:.4f} m'
            ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3), verticalalignment='top', horizontalalignment='right')
            
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.plot(time_slice, df['joint1_angle'].iloc[data_slice], 'r-', label='Joint 1', linewidth=2)
            ax3.plot(time_slice, df['joint2_angle'].iloc[data_slice], 'g-', label='Joint 2', linewidth=2)
            ax3.plot(time_slice, df['joint3_angle'].iloc[data_slice], 'b-', label='Joint 3', linewidth=2)
            ax3.set_xlabel('Time Step', labelpad=10); ax3.set_ylabel('Angle (rad)', labelpad=10)
            ax3.set_title('Joint Angles', pad=20, y=1.0); ax3.grid(True); ax3.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
            
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.plot(time_slice, df['joint1_velocity'].iloc[data_slice], 'r-', label='Joint 1', linewidth=2)
            ax4.plot(time_slice, df['joint2_velocity'].iloc[data_slice], 'g-', label='Joint 2', linewidth=2)
            ax4.plot(time_slice, df['joint3_velocity'].iloc[data_slice], 'b-', label='Joint 3', linewidth=2)
            ax4.set_xlabel('Time Step', labelpad=10); ax4.set_ylabel('Velocity (rad/s)', labelpad=10)
            ax4.set_title('Joint Velocities', pad=20, y=1.0); ax4.grid(True); ax4.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
            
            ax5 = fig.add_subplot(gs[2, 0])
            ax5.plot(time_slice, df['joint1_acceleration'].iloc[data_slice], 'r-', label='Joint 1', linewidth=2)
            ax5.plot(time_slice, df['joint2_acceleration'].iloc[data_slice], 'g-', label='Joint 2', linewidth=2)
            ax5.plot(time_slice, df['joint3_acceleration'].iloc[data_slice], 'b-', label='Joint 3', linewidth=2)
            ax5.set_xlabel('Time Step', labelpad=10); ax5.set_ylabel('Acceleration (rad/s²)', labelpad=10)
            ax5.set_title('Joint Accelerations', pad=20, y=1.0); ax5.grid(True); ax5.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
            
            ax6 = fig.add_subplot(gs[2, 1])
            ax6.plot(time_slice, df['joint1_torque'].iloc[data_slice], 'r-', label='Joint 1', linewidth=2)
            ax6.plot(time_slice, df['joint2_torque'].iloc[data_slice], 'g-', label='Joint 2', linewidth=2)
            ax6.plot(time_slice, df['joint3_torque'].iloc[data_slice], 'b-', label='Joint 3', linewidth=2)
            ax6.set_xlabel('Time Step', labelpad=10); ax6.set_ylabel('Torque (N⋅m)', labelpad=10)
            ax6.set_title('Joint Torques', pad=20, y=1.0); ax6.grid(True); ax6.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
            
            stats = []
            for joint in range(1, 4):
                stats.append({
                    'Angle': {'Mean': np.nanmean(df[f'joint{joint}_angle'].iloc[data_slice]), 'Std': np.nanstd(df[f'joint{joint}_angle'].iloc[data_slice]), 'Max': np.nanmax(df[f'joint{joint}_angle'].iloc[data_slice]), 'Min': np.nanmin(df[f'joint{joint}_angle'].iloc[data_slice])},
                    'Velocity': {'Mean': np.nanmean(df[f'joint{joint}_velocity'].iloc[data_slice]), 'Std': np.nanstd(df[f'joint{joint}_velocity'].iloc[data_slice]), 'RMS': np.sqrt(np.nanmean(df[f'joint{joint}_velocity'].iloc[data_slice]**2))},
                    'Acceleration': {'Mean': np.nanmean(df[f'joint{joint}_acceleration'].iloc[data_slice]), 'Std': np.nanstd(df[f'joint{joint}_acceleration'].iloc[data_slice]), 'RMS': np.sqrt(np.nanmean(df[f'joint{joint}_acceleration'].iloc[data_slice]**2))},
                    'Torque': {'Mean': np.nanmean(df[f'joint{joint}_torque'].iloc[data_slice]), 'Std': np.nanstd(df[f'joint{joint}_torque'].iloc[data_slice]), 'RMS': np.sqrt(np.nanmean(df[f'joint{joint}_torque'].iloc[data_slice]**2))}
                })
            
            if is_figure8:
                title = f'Trajectory Analysis\nKp={kp:.2f}, Kd={kd:.1f}, Amplitude={amplitude:.2f}, Center=({center_x:.2f}, {center_z:.2f})m'
            else:
                title = f'Trajectory Analysis\nKp={kp:.2f}, Kd={kd:.1f}, Radius={radius:.2f}m, Center=({center_x:.2f}, {center_z:.2f})m'
            plt.suptitle(title, y=0.98, fontsize=12)
            
            stats_window = create_stats_window(stats, kp, kd, center_x, center_z, radius=radius, amplitude=amplitude)
            
            plt.show()
            stats_window.mainloop()
        
        if is_figure8:
            update_plots('complete')
        else:
            mode_window = create_view_mode_window(update_plots)
            mode_window.mainloop()
        
    except Exception as e:
        print(f"Error while processing file: {str(e)}")
        import traceback
        traceback.print_exc()
        return

def main():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        print("Scanning for trajectory files...")
        csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv') and 'trajectory' in f.lower()]
        
        if not csv_files:
            print(f"\nNo trajectory CSV files found in the directory: {script_dir}")
            return
            
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(script_dir, x)), reverse=True)
        
        while True:
            print("\nAvailable data files (sorted by newest first):")
            for i, file in enumerate(csv_files, 1):
                mod_time = os.path.getmtime(os.path.join(script_dir, file))
                mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
                file_size = os.path.getsize(os.path.join(script_dir, file)) / 1024
                print(f"{i}. [{mod_time_str}] {file} ({file_size:.1f} KB)")
            
            try:
                choice = input("\nEnter the number of the file to visualize (or 0 to exit): ").strip()
                
                if not choice:
                    print("Please enter a number.")
                    continue
                    
                choice = int(choice)
                if choice == 0: break
                if 1 <= choice <= len(csv_files):
                    csv_file = os.path.join(script_dir, csv_files[choice - 1])
                    print(f"\nVisualizing {os.path.basename(csv_file)}...")
                    visualize_trajectory_data(csv_file)
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(csv_files)}")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
            except Exception as e:
                print(f"Error: {str(e)}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 