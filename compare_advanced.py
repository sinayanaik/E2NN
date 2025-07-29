import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import altair as alt
import webbrowser
import tempfile

def find_default_prediction_files():
    """Finds the latest prediction CSV from E2NN, FNN, and RNN directories."""
    model_dirs = ['E2NN', 'FNN', 'RNN']
    latest_files = []

    for model_dir in model_dirs:
        if not os.path.isdir(model_dir):
            continue
        
        latest_file = None
        latest_mtime = 0
        
        for root, _, files in os.walk(model_dir):
            for file in files:
                if 'predictions' in file and file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        mtime = os.path.getmtime(file_path)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            latest_file = file_path
                    except OSError:
                        continue # File might not exist anymore
        
        if latest_file:
            # Use model_dir as legend name
            latest_files.append({'path': latest_file, 'legend': model_dir})
            
    return latest_files

def calculate_metrics(y_true, y_pred):
    metrics = {}
    # Use multioutput='raw_values' to get per-joint scores
    metrics['MSE'] = mean_squared_error(y_true, y_pred, multioutput='raw_values')
    
    # Calculate NMSE per joint
    y_true_mean = y_true.mean(axis=0, keepdims=True)
    numerator = np.sum((y_true - y_pred)**2, axis=0)
    denominator = np.sum((y_true - y_true_mean)**2, axis=0)
    # Avoid division by zero if a ground truth signal is constant
    metrics['NMSE'] = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)

    metrics['R²'] = r2_score(y_true, y_pred, multioutput='raw_values')
    metrics['MAE'] = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    
    # Calculate MaxAE per joint
    metrics['MaxAE'] = np.max(np.abs(y_true - y_pred), axis=0)
    return metrics

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
        
        self.default_files = find_default_prediction_files()
        if self.default_files:
            self.num_files_entry.insert(0, str(len(self.default_files)))
        else:
            self.num_files_entry.insert(0, "2")

        tk.Button(controls_frame, text="Set Up", command=self.setup_file_entries).pack(side=tk.LEFT)

        # --- Ground Truth File Selection ---
        gt_frame = tk.Frame(self.main_frame, pady=5)
        gt_frame.pack(fill=tk.X)
        
        tk.Label(gt_frame, text="Ground Truth File:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.gt_path_var = tk.StringVar()
        gt_entry = tk.Entry(gt_frame, textvariable=self.gt_path_var, width=70)
        gt_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        browse_button = tk.Button(gt_frame, text="Browse...", command=self.browse_gt_file)
        browse_button.pack(side=tk.LEFT, padx=(5, 0))

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

    def browse_gt_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Ground Truth CSV File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
            initialdir='data'
        )
        if filepath:
            self.gt_path_var.set(filepath)

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

        if hasattr(self, 'default_files') and self.default_files:
            for i, widget_set in enumerate(self.file_widgets):
                if i < len(self.default_files):
                    widget_set['path_var'].set(self.default_files[i]['path'])
                    widget_set['legend_var'].set(self.default_files[i]['legend'])
        
        self.default_files = []

    def browse_file(self, path_variable):
        filepath = filedialog.askopenfilename(
            title="Select Prediction CSV File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filepath:
            path_variable.set(filepath)

    def generate_plot(self):
        gt_path = self.gt_path_var.get()
        if not gt_path or not os.path.exists(gt_path):
            messagebox.showerror("Error", "Please select a valid ground truth trajectory file.")
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

        try:
            ground_truth_df = pd.read_csv(gt_path)
            gt_torque_cols = ['joint1_torque', 'joint2_torque', 'joint3_torque']
            gt_torques = ground_truth_df[gt_torque_cols]

            all_pred_dfs = [pd.read_csv(p['path'], index_col='original_index') for p in prediction_data]
        except Exception as e:
            messagebox.showerror("Error Reading Files", f"Failed to read or process CSV files.\nError: {e}")
            return

        max_index = 0
        for df in all_pred_dfs:
            if not df.empty:
                max_index = max(max_index, df.index.max())
        
        gt_torques_sliced = gt_torques.loc[:max_index].copy()
        gt_torques_sliced['Time Step'] = gt_torques_sliced.index

        all_data_to_plot = []

        gt_melted = gt_torques_sliced.melt(
            id_vars=['Time Step'], 
            value_vars=gt_torque_cols, 
            var_name='Joint', 
            value_name='Torque'
        )
        gt_melted['Model'] = 'Ground Truth'
        gt_melted['Joint'] = gt_melted['Joint'].str.replace('_torque', '').str.replace('joint', 'Joint ')
        all_data_to_plot.append(gt_melted)

        for i, (p_data, pred_df) in enumerate(zip(prediction_data, all_pred_dfs)):
            if pred_df.empty:
                continue
            
            pred_df_copy = pred_df.copy()
            pred_cols = [c for c in pred_df_copy.columns if c.startswith('pred_')]
            if len(pred_cols) != 3:
                messagebox.showwarning("Warning", f"Prediction file '{p_data['path']}' has {len(pred_cols)} prediction columns, expected 3. Skipping.")
                continue

            y_true_for_metrics = gt_torques.loc[pred_df_copy.index].values
            y_pred = pred_df_copy[pred_cols].values
            metrics_per_joint = calculate_metrics(y_true_for_metrics, y_pred)

            pred_df_copy['Time Step'] = pred_df_copy.index
            pred_df_renamed = pred_df_copy[pred_cols].rename(columns={
                pred_cols[0]: 'Joint 1',
                pred_cols[1]: 'Joint 2',
                pred_cols[2]: 'Joint 3'
            })
            pred_df_renamed['Time Step'] = pred_df_copy.index
            
            pred_melted = pred_df_renamed.melt(
                id_vars=['Time Step'],
                var_name='Joint',
                value_name='Torque'
            )
            pred_melted['Model'] = p_data['legend']
            
            metrics_df_data = []
            for j in range(3):
                joint_name = f'Joint {j+1}'
                metrics_for_joint = {'Joint': joint_name}
                for name, var in self.metrics_vars.items():
                    if var.get():
                        metrics_for_joint[name] = metrics_per_joint[name][j]
                metrics_df_data.append(metrics_for_joint)
            
            metrics_df = pd.DataFrame(metrics_df_data)
            pred_melted_with_metrics = pd.merge(pred_melted, metrics_df, on='Joint')
            all_data_to_plot.append(pred_melted_with_metrics)

        if len(all_data_to_plot) <= 1: # Only ground truth is present
            messagebox.showinfo("Info", "No valid prediction data to plot.")
            return
            
        plot_df = pd.concat(all_data_to_plot, ignore_index=True)

        tooltip_items = [
            alt.Tooltip('Time Step:Q'),
            alt.Tooltip('Torque:Q', format='.4f'),
            alt.Tooltip('Model:N')
        ]
        
        for name, var in self.metrics_vars.items():
            if var.get():
                tooltip_items.append(alt.Tooltip(f'{name}:Q', format='.6f'))

        base_chart = alt.Chart(plot_df).mark_line().encode(
            x='Time Step:Q',
            y=alt.Y('Torque:Q', title='Torque'),
            color=alt.Color('Model:N', scale=alt.Scale(scheme='category10')),
            tooltip=tooltip_items
        )

        gt_chart = base_chart.transform_filter(alt.datum.Model == 'Ground Truth')
        pred_chart = base_chart.transform_filter(alt.datum.Model != 'Ground Truth')
        
        layered_base = alt.layer(gt_chart, pred_chart)

        joint_names = sorted(plot_df['Joint'].unique())

        if not joint_names:
            messagebox.showerror("Error", "No data to plot.")
            return

        plot_files = []
        try:
            for joint in joint_names:
                joint_chart = layered_base.transform_filter(
                    alt.datum.Joint == joint
                ).properties(
                    width=1200,
                    height=700,
                    title=f'{joint} Torque'
                ).configure_view(
                    continuousWidth=1300,
                    continuousHeight=1300
                ).interactive()

                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'_{joint.replace(" ", "_")}.html') as f:
                    joint_chart.save(f.name, format='html')
                    plot_files.append(f.name)
        
            for file_path in plot_files:
                webbrowser.open(f'file://{os.path.realpath(file_path)}', new=2)

        except Exception as e:
            messagebox.showerror("Plotting Error", f"Failed to generate or open plots.\nError: {e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = ComparisonApp(root)
    root.mainloop() 