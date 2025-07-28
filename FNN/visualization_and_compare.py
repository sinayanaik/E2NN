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
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'R²': r2}

def find_ground_truth_file():
    """Finds the latest trajectory csv file in the ../data directory."""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(data_dir):
             # Fallback for running from a different directory
            data_dir = 'data'
            if not os.path.exists(data_dir):
                return None

        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'trajectory' in f]
        if not csv_files:
            return None
        # Get the newest file
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
        return os.path.join(data_dir, csv_files[0])
    except FileNotFoundError:
        return None

def visualize_and_compare():
    root = tk.Tk()
    root.title("FNN Result Visualizer")

    main_frame = tk.Frame(root, padx=10, pady=10)
    main_frame.pack(expand=True, fill=tk.BOTH)
    
    metrics_vars = {
        'MSE': tk.BooleanVar(value=True),
        'MAE': tk.BooleanVar(value=False),
        'R²': tk.BooleanVar(value=True)
    }

    def load_prediction_files():
        gt_path = find_ground_truth_file()
        if not gt_path:
            messagebox.showerror("Error", "Could not find the ground truth trajectory file in `../data/` or `data/`.")
            return

        filepaths = filedialog.askopenfilenames(
            title="Select Prediction CSV Files",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if not filepaths:
            return
            
        ground_truth_df = pd.read_csv(gt_path)
        gt_torques = ground_truth_df[['joint1_torque', 'joint2_torque', 'joint3_torque']]

        # First, determine the maximum index from all prediction files
        all_dfs = [pd.read_csv(f, index_col='original_index') for f in filepaths]
        max_index = 0
        for df in all_dfs:
            if not df.empty:
                max_index = max(max_index, df.index.max())
        
        # Slice ground truth based on the max index of predictions
        gt_torques_sliced = gt_torques.loc[:max_index]

        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
        
        # Color cycle for plots
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
        
        # Plot ground truth once
        for j in range(3):
            axes[j].plot(gt_torques_sliced.index, gt_torques_sliced.iloc[:, j], label='Ground Truth', color='black', linewidth=1.5)

        for i, df in enumerate(all_dfs):
            if df.empty:
                continue

            pred_cols = [c for c in df.columns if c.startswith('pred_')]
            y_pred = df[pred_cols]
            
            # Ground truth for metrics should be sliced to match the current prediction df's index
            y_true_for_metrics = gt_torques.loc[df.index].values

            metrics = calculate_metrics(y_true_for_metrics, y_pred.values)
            run_name = os.path.basename(os.path.dirname(filepaths[i]))
            
            # Use regex to extract hyperparameters for legend
            match = re.search(r"lr([\d.]+)_bs(\d+)_layers([\dx]+)", run_name)
            if match:
                lr, bs, layers = match.groups()
                legend_label = f"lr={lr}, bs={bs}, layers={layers}"
            else:
                legend_label = f"Run {i+1}"
            
            metrics_to_display = []
            for name, var in metrics_vars.items():
                if var.get():
                    metrics_to_display.append(f"{name}={metrics[name]:.3f}")
            if metrics_to_display:
                legend_label += f" ({', '.join(metrics_to_display)})"

            # Plotting predictions
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

    # GUI Layout
    tk.Label(main_frame, text="Display Metrics in Legend:").pack(anchor='w', pady=(0,5))
    
    metrics_frame = tk.Frame(main_frame)
    metrics_frame.pack(fill=tk.X)
    for name, var in metrics_vars.items():
        tk.Checkbutton(metrics_frame, text=name, variable=var).pack(side=tk.LEFT, expand=True)
    
    tk.Button(main_frame, text="Load Prediction CSVs to Compare", command=load_prediction_files).pack(pady=20, fill=tk.X, ipady=5)

    root.mainloop()

if __name__ == '__main__':
    visualize_and_compare() 