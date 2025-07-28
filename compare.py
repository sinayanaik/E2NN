import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import re
try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False

def calculate_metrics(y_true, y_pred):
    metrics = {}
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['NMSE'] = np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    metrics['R²'] = r2_score(y_true, y_pred)
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['MaxAE'] = np.max(np.abs(y_true - y_pred))
    return metrics

def find_ground_truth_file():
    """Finds the latest trajectory csv file in the data directory."""
    try:
        data_dir = 'data'
        if not os.path.exists(data_dir):
            return None
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'trajectory' in f]
        if not csv_files:
            return None
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
        return os.path.join(data_dir, csv_files[0])
    except FileNotFoundError:
        return None

class ComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Comparison Tool")
        
        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        # --- Top Frame for Controls ---
        controls_frame = tk.Frame(self.main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(controls_frame, text="Number of Predictions to Compare:").pack(side=tk.LEFT, padx=(0, 5))
        self.num_files_entry = tk.Entry(controls_frame, width=5)
        self.num_files_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.num_files_entry.insert(0, "2")
        
        tk.Button(controls_frame, text="Set Up", command=self.setup_file_entries).pack(side=tk.LEFT)

        # --- Frame for dynamic file/legend entries ---
        self.file_entries_frame = tk.Frame(self.main_frame)
        self.file_entries_frame.pack(fill=tk.X, pady=(0, 10))
        
        # --- Frame for Metrics ---
        metrics_outer_frame = tk.LabelFrame(self.main_frame, text="Metrics to Display", padx=10, pady=10)
        metrics_outer_frame.pack(fill=tk.X)
        
        self.metrics_vars = {
            'MSE': tk.BooleanVar(value=True),
            'NMSE': tk.BooleanVar(value=False),
            'R²': tk.BooleanVar(value=True),
            'MAE': tk.BooleanVar(value=False),
            'MaxAE': tk.BooleanVar(value=False)
        }

        metrics_frame = tk.Frame(metrics_outer_frame)
        metrics_frame.pack(fill=tk.X)
        for name, var in self.metrics_vars.items():
            tk.Checkbutton(metrics_frame, text=name, variable=var).pack(side=tk.LEFT, expand=True)
        
        # --- Plot Button ---
        tk.Button(self.main_frame, text="Generate Comparison Plot", command=self.generate_plot).pack(pady=10, fill=tk.X, ipady=5)

        self.file_widgets = []
        self.setup_file_entries() # Initial setup

    def setup_file_entries(self):
        for widgets in self.file_widgets:
            widgets['frame'].destroy()
        self.file_widgets = []

        try:
            num_files = int(self.num_files_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")
            return

        for i in range(num_files):
            frame = tk.Frame(self.file_entries_frame)
            frame.pack(fill=tk.X, pady=2)
            
            label = tk.Label(frame, text=f"Prediction {i+1}:")
            label.pack(side=tk.LEFT, padx=(0, 5))

            path_var = tk.StringVar()
            entry = tk.Entry(frame, textvariable=path_var, width=50)
            entry.pack(side=tk.LEFT, expand=True, fill=tk.X)

            button = tk.Button(frame, text="Browse...", command=lambda v=path_var: self.browse_file(v))
            button.pack(side=tk.LEFT, padx=(5, 10))
            
            tk.Label(frame, text="Legend Name:").pack(side=tk.LEFT)
            legend_var = tk.StringVar()
            legend_entry = tk.Entry(frame, textvariable=legend_var, width=20)
            legend_entry.pack(side=tk.LEFT)

            self.file_widgets.append({'frame': frame, 'label': label, 'path_var': path_var, 'entry': entry, 'button': button, 'legend_var': legend_var, 'legend_entry': legend_entry})

    def browse_file(self, path_variable):
        filepath = filedialog.askopenfilename(
            title="Select Prediction CSV File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filepath:
            path_variable.set(filepath)

    def generate_plot(self):
        gt_path = find_ground_truth_file()
        if not gt_path:
            messagebox.showerror("Error", "Could not find the ground truth trajectory file in `data/`.")
            return

        prediction_data = []
        for widgets in self.file_widgets:
            path = widgets['path_var'].get()
            legend = widgets['legend_var'].get()
            if path and legend:
                prediction_data.append({'path': path, 'legend': legend})
        
        if not prediction_data:
            messagebox.showerror("Error", "Please select at least one prediction file and provide a legend name.")
            return

        ground_truth_df = pd.read_csv(gt_path)
        gt_torques = ground_truth_df[['joint1_torque', 'joint2_torque', 'joint3_torque']]

        # Determine the scope of the plot
        all_dfs = [pd.read_csv(p['path'], index_col='original_index') for p in prediction_data]
        max_index = 0
        for df in all_dfs:
            if not df.empty:
                max_index = max(max_index, df.index.max())
        
        gt_torques_sliced = gt_torques.loc[:max_index]

        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
        # High-contrast color cycle for plots
        colors = ['#E6194B', '#3CB44B', '#4363D8', '#F58231', '#911EB4', '#42D4F4', '#F032E6', '#BFEF45', '#FABED4', '#469990']

        for j in range(3):
            axes[j].plot(gt_torques_sliced.index, gt_torques_sliced.iloc[:, j], label='Ground Truth', color='black', linewidth=1.5)

        for i, (p_data, df) in enumerate(zip(prediction_data, all_dfs)):
            if df.empty:
                continue

            pred_cols = [c for c in df.columns if c.startswith('pred_')]
            y_pred = df[pred_cols]
            
            y_true_for_metrics = gt_torques.loc[df.index].values

            metrics = calculate_metrics(y_true_for_metrics, y_pred.values)
            
            legend_label = p_data['legend']
            metrics_to_display = []
            for name, var in self.metrics_vars.items():
                if var.get():
                    metrics_to_display.append(f"{name}={metrics[name]:.3f}")
            if metrics_to_display:
                legend_label += f" ({', '.join(metrics_to_display)})"

            color = colors[i % len(colors)]
            for j in range(3):
                axes[j].plot(df.index, y_pred.iloc[:, j], label=legend_label, color=color, linewidth=1.5)

        for i, title in enumerate(['Joint 1 Torque', 'Joint 2 Torque', 'Joint 3 Torque']):
            axes[i].set_title(title)
            axes[i].legend(fontsize='small')
            axes[i].grid(True)
        axes[2].set_xlabel("Time Step")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle('Model Predictions vs. Ground Truth')
        
        if MPLCURSORS_AVAILABLE:
            mplcursors.cursor(hover=True)

        plt.show()

if __name__ == '__main__':
    root = tk.Tk()
    app = ComparisonApp(root)
    root.mainloop() 