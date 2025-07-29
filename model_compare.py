import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, colorchooser, messagebox
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import altair as alt
from altair_viewer import show
import webbrowser
import os
import vl_convert as vlc

# Define model architectures
class FNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FNN, self).__init__()
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# --- New E2NN Definitions ---
class InertiaNN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 9)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, q):
        return self.net(q).reshape(-1, 3, 3)

class CoriolisVecNN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 3)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, q, qdot):
        x = torch.cat([q, qdot], dim=1)
        return self.net(x)

class GravityNN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 3)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)
    
    def forward(self, q):
        return self.net(q)

class E2NN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.inertia_net = InertiaNN(hidden_size)
        self.coriolis_net = CoriolisVecNN(hidden_size)
        self.gravity_net = GravityNN(hidden_size)

    def forward(self, q, q_dot, q_ddot):
        M_q = self.inertia_net(q)
        C_q_qdot = self.coriolis_net(q, q_dot)
        G_q = self.gravity_net(q)
        m_q_ddot = torch.bmm(M_q, q_ddot.unsqueeze(-1)).squeeze(-1)
        tau = m_q_ddot + C_q_qdot + G_q
        return tau

def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
    return np.array(xs)

class ComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Comparison Tool")
        
        self.ground_truth_path = tk.StringVar()
        self.model_entries = []

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(sticky=(tk.W, tk.E, tk.N, tk.S))

        # Ground Truth Selection
        gt_frame = ttk.LabelFrame(main_frame, text="Ground Truth Data", padding="10")
        gt_frame.grid(sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(gt_frame, text="CSV Path:").pack(side=tk.LEFT)
        ttk.Entry(gt_frame, textvariable=self.ground_truth_path, width=80).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(gt_frame, text="Browse", command=self.browse_gt).pack(side=tk.LEFT)

        # Model Selection
        self.models_frame = ttk.LabelFrame(main_frame, text="Models to Compare", padding="10")
        self.models_frame.grid(sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(self.models_frame, text="Add Model", command=self.add_model_entry).pack()

        # Define a list of visually distinct colors
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.color_index = 0

        # Controls
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(control_frame, text="Generate Comparison Plot", command=self.generate_plot).pack()

    def browse_gt(self):
        path = filedialog.askopenfilename(title="Select Ground Truth CSV", filetypes=[("CSV files", "*.csv")])
        if path: self.ground_truth_path.set(path)

    def add_model_entry(self):
        entry_frame = ttk.Frame(self.models_frame)
        entry_frame.pack(fill=tk.X, pady=2)

        model_path = tk.StringVar()
        legend_name = tk.StringVar()
        color = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1

        ttk.Label(entry_frame, text="Model Path (.pth):").pack(side=tk.LEFT)
        entry = ttk.Entry(entry_frame, textvariable=model_path, width=60)
        entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(entry_frame, text="Browse", command=lambda p=model_path: self.browse_model(p)).pack(side=tk.LEFT)
        
        ttk.Label(entry_frame, text="Legend:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(entry_frame, textvariable=legend_name, width=20).pack(side=tk.LEFT)
        
        self.model_entries.append({'path': model_path, 'legend': legend_name, 'color': color, 'frame': entry_frame})

    def browse_model(self, path_var):
        path = filedialog.askopenfilename(title="Select Model File", filetypes=[("PyTorch models", "*.pth")])
        if path: path_var.set(path)

    def generate_plot(self):
        try:
            df_gt = pd.read_csv(self.ground_truth_path.get())
            features = ['joint1_angle', 'joint2_angle', 'joint3_angle', 'joint1_velocity', 'joint2_velocity', 'joint3_velocity', 'joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration']
            targets = ['joint1_torque', 'joint2_torque', 'joint3_torque']

            all_predictions = []
            for entry in self.model_entries:
                model_path = entry['path'].get()
                legend = entry['legend'].get()
                
                # Load architecture from file
                arch_path = os.path.join(os.path.dirname(model_path), "architecture_plus_hyperparameters.txt")
                params = {}
                with open(arch_path, 'r') as f:
                    for line in f:
                        key, value = line.strip().split(': ')
                        params[key] = value

                model_type = params.pop('model_type')
                
                sl = 0 # Sequence length for non-RNN models
                if model_type == 'FNN':
                    hidden_sizes = [int(s) for s in params['hidden_layers'].split(',')]
                    model = FNN(len(features), hidden_sizes, len(targets))
                elif model_type == 'RNN':
                    hs = int(params['hidden_size'])
                    nl = int(params['num_layers'])
                    sl = int(params['seq_length'])
                    model = RNN(len(features), hs, nl, len(targets))
                elif model_type == 'E2NN':
                    hs = int(params['hidden_size'])
                    model = E2NN(hidden_size=hs)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                model.load_state_dict(torch.load(model_path, weights_only=True))
                model.eval()

                # --- Data Preparation and Prediction ---
                scaler_X_q = StandardScaler().fit(df_gt[['joint1_angle', 'joint2_angle', 'joint3_angle']].values)
                scaler_X_q_dot = StandardScaler().fit(df_gt[['joint1_velocity', 'joint2_velocity', 'joint3_velocity']].values)
                scaler_X_q_ddot = StandardScaler().fit(df_gt[['joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration']].values)
                
                df_gt_q_scaled = scaler_X_q.transform(df_gt[['joint1_angle', 'joint2_angle', 'joint3_angle']].values)
                df_gt_q_dot_scaled = scaler_X_q_dot.transform(df_gt[['joint1_velocity', 'joint2_velocity', 'joint3_velocity']].values)
                df_gt_q_ddot_scaled = scaler_X_q_ddot.transform(df_gt[['joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration']].values)

                with torch.no_grad():
                    if model_type == 'E2NN':
                        q_in = torch.tensor(df_gt_q_scaled, dtype=torch.float32)
                        q_dot_in = torch.tensor(df_gt_q_dot_scaled, dtype=torch.float32)
                        q_ddot_in = torch.tensor(df_gt_q_ddot_scaled, dtype=torch.float32)
                        predictions = model(q_in, q_dot_in, q_ddot_in).numpy()
                    else:
                        # Standard FNN/RNN prediction
                        df_gt_scaled = np.hstack((df_gt_q_scaled, df_gt_q_dot_scaled, df_gt_q_ddot_scaled))
                        X_in = create_sequences(df_gt_scaled, sl) if sl > 0 else df_gt_scaled
                        
                        pred_scaled = model(torch.tensor(X_in, dtype=torch.float32)).numpy()
                        
                        scaler_y = StandardScaler().fit(df_gt[targets].values)
                        predictions = scaler_y.inverse_transform(pred_scaled)
                
                pred_df = pd.DataFrame(predictions, columns=[f"predicted_{t}" for t in targets])
                all_predictions.append({'legend': legend, 'preds': pred_df, 'seq_len': sl})

            # Generate a plot for each joint
            all_metrics = {}
            chart_paths = []
            self.charts = [] # To store chart objects for saving

            for i, joint_target in enumerate(targets):
                plot_data = pd.DataFrame({
                    'timestamp': df_gt.index,
                    'Torque': df_gt[joint_target],
                    'Legend': 'Ground Truth'
                })
                
                metrics_summary = []
                all_metrics[joint_target] = {}
                
                color_domain = ['Ground Truth']
                color_range = ['#000000'] # Black for ground truth

                for pred_data in all_predictions:
                    legend = pred_data['legend']
                    preds = pred_data['preds']
                    seq_len = pred_data['seq_len']

                    # Find the corresponding color from model_entries
                    model_color = next((m['color'] for m in self.model_entries if m['legend'].get() == legend), '#000000')
                    color_domain.append(legend)
                    color_range.append(model_color)
                    
                    pred_col = f"predicted_{joint_target}"
                    
                    temp_df = pd.DataFrame({
                        'timestamp': df_gt.index[seq_len:] if seq_len > 0 else df_gt.index,
                        'Torque': preds[pred_col],
                        'Legend': legend
                    })
                    plot_data = pd.concat([plot_data, temp_df])

                    # Calculate metrics
                    y_true = df_gt[joint_target].values[seq_len:] if seq_len > 0 else df_gt[joint_target].values
                    y_pred = preds[pred_col].values
                    
                    mse = np.mean((y_true - y_pred)**2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y_true - y_pred))
                    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
                    
                    all_metrics[joint_target][legend] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
                    metrics_summary.append(f"{legend} (RMSE: {rmse:.3f}, R²: {r2:.3f})")

                chart_title = f"Torque Comparison for {joint_target.replace('_', ' ').title()}"
                chart_subtitle = " | ".join(metrics_summary)

                chart = alt.Chart(plot_data).mark_line(point=False).encode(
                    x='timestamp:Q',
                    y='Torque:Q',
                    color=alt.Color('Legend:N', scale=alt.Scale(domain=color_domain, range=color_range)),
                ).properties(
                    title={'text': chart_title, 'subtitle': chart_subtitle}
                ).interactive()

                chart_path = f'comparison_{joint_target}.html'
                chart.save(chart_path)
                chart_paths.append(chart_path)
                self.charts.append(chart) # Save chart object

            self.display_metrics_window(all_metrics)
            for path in chart_paths:
                webbrowser.open('file://' + os.path.realpath(path))

        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to generate plot: {e}")

    def display_metrics_window(self, metrics):
        win = tk.Toplevel(self.root)
        win.title("Performance Metrics")
        
        tree = ttk.Treeview(win, columns=("Model", "Joint", "RMSE", "R²", "MAE", "MSE"), show='headings')
        tree.heading("Model", text="Model")
        tree.heading("Joint", text="Joint")
        tree.heading("RMSE", text="RMSE")
        tree.heading("R²", text="R²")
        tree.heading("MAE", text="MAE")
        tree.heading("MSE", text="MSE")

        for joint, models in metrics.items():
            for model, values in models.items():
                tree.insert("", "end", values=(
                    model, 
                    joint.replace('_', ' ').title(),
                    f"{values['RMSE']:.4f}",
                    f"{values['R²']:.4f}",
                    f"{values['MAE']:.4f}",
                    f"{values['MSE']:.4f}"
                ))
        
        tree.pack(expand=True, fill='both', side=tk.TOP)
        
        save_button = ttk.Button(win, text="Save Plots as PNG", command=self.save_plots_as_png)
        save_button.pack(side=tk.BOTTOM, pady=10)

    def save_plots_as_png(self):
        if not self.charts:
            messagebox.showwarning("No Plots", "No plots have been generated yet.")
            return

        save_dir = filedialog.askdirectory(title="Select Directory to Save PNGs")
        if not save_dir:
            return
            
        try:
            for i, chart in enumerate(self.charts):
                joint_name = chart.title.text.split(" for ")[-1].replace(" ", "_")
                filename = os.path.join(save_dir, f"comparison_{joint_name}.png")
                
                # Use vl-convert to save the chart as a PNG
                png_data = vlc.vegalite_to_png(chart.to_json(), scale=2)
                with open(filename, "wb") as f:
                    f.write(png_data)
            messagebox.showinfo("Save Complete", f"All plots saved to:\n{save_dir}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save PNGs: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ComparisonApp(root)
    root.mainloop() 