import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, colorchooser, messagebox
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import webbrowser
import traceback
import matplotlib.pyplot as plt

# --- Model Architectures (Simplified) ---
class FNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = [nn.Linear(s_in, s_out) for s_in, s_out in zip([input_size] + hidden_sizes, hidden_sizes + [output_size])]
        self.net = nn.Sequential(*[l for layer in layers[:-1] for l in (layer, nn.ReLU())] + [layers[-1]])
    def forward(self, x): return self.net(x)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x); return self.fc(out[:, -1, :])

class E2NN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        def _build_net(i, o, h):
            return nn.Sequential(nn.Linear(i, h), nn.LayerNorm(h), nn.Tanh(), nn.Dropout(0.1), nn.Linear(h,h), nn.LayerNorm(h), nn.Tanh(), nn.Linear(h,o))
        self.inertia_net, self.coriolis_net, self.gravity_net = _build_net(3,9,hidden_size), _build_net(6,3,hidden_size), _build_net(3,3,hidden_size)
    def forward(self, q, q_dot, q_ddot):
        M_q = self.inertia_net(q).reshape(-1, 3, 3)
        return torch.bmm(M_q, q_ddot.unsqueeze(-1)).squeeze(-1) + self.coriolis_net(torch.cat([q, q_dot], dim=1)) + self.gravity_net(q)

def create_sequences(data, seq_length):
    return np.array([data[i:(i + seq_length)] for i in range(len(data) - seq_length)])

# --- Main Application ---
class ErrorPlotApp:
    def __init__(self, root):
        self.root = root; self.root.title("Error Plot Tool")
        
        self.ground_truth_path, self.model_entries, self.plot_prefix = tk.StringVar(), [], tk.StringVar(value="error")
        
        # --- Settings with default values ---
        self.plot_settings = {'width': tk.DoubleVar(value=16), 'height': tk.DoubleVar(value=9),
                              'x_title': tk.StringVar(value='Timestamp'), 'y_title': tk.StringVar(value='Error (Nm)'),
                              'axis_title_size': tk.IntVar(value=16), 'label_size': tk.IntVar(value=14), 
                              'legend_size': tk.IntVar(value=10)}

        main_frame = ttk.Frame(self.root, padding="10"); main_frame.grid(sticky="nsew"); self.root.columnconfigure(0, weight=1)
        self.setup_data_ui(main_frame)
        self.setup_settings_ui(main_frame)
        ttk.Button(main_frame, text="Generate Error Plot", command=self.generate_plot, style="Accent.TButton").grid(row=3, column=0, pady=10)
        ttk.Style(self.root).configure("Accent.TButton", foreground="white", background="black")

    def setup_data_ui(self, p):
        gt_frame = ttk.LabelFrame(p, text="Ground Truth Data", padding="10"); gt_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(gt_frame, text="CSV Path:").pack(side=tk.LEFT)
        ttk.Entry(gt_frame, textvariable=self.ground_truth_path).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        browse_btn = ttk.Button(gt_frame, text="Browse", command=lambda: self.ground_truth_path.set(filedialog.askopenfilename(title="Select Ground Truth CSV", filetypes=[("CSV files", "*.csv")])))
        browse_btn.pack(side=tk.LEFT)
        
        self.models_frame = ttk.LabelFrame(p, text="Models to Compare", padding="10"); self.models_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ttk.Button(self.models_frame, text="Add Model", command=self.add_model_entry).pack()
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    def setup_settings_ui(self, p):
        sf = ttk.Frame(p, padding="5"); sf.grid(row=2, column=0, sticky="ew", padx=5, pady=5); sf.columnconfigure((0,1), weight=1)
        
        psf = ttk.LabelFrame(sf, text="Plot Settings", padding=10); psf.grid(row=0, column=0, sticky="nsew", padx=5); psf.columnconfigure(1, weight=1)
        
        settings_vars = [("Filename Prefix:", self.plot_prefix), ("Plot Width:", self.plot_settings['width']), ("Plot Height:", self.plot_settings['height']),
                         ("X-Axis Title:", self.plot_settings['x_title']), ("Y-Axis Title:", self.plot_settings['y_title']), 
                         ("Axis Title Size:", self.plot_settings['axis_title_size']),
                         ("Tick Label Size:", self.plot_settings['label_size']), 
                         ("Legend Size:", self.plot_settings['legend_size'])]
        for i, (label, var) in enumerate(settings_vars):
            ttk.Label(psf, text=label).grid(row=i, column=0, sticky="w", padx=2, pady=2)
            ttk.Entry(psf, textvariable=var, width=15).grid(row=i, column=1, sticky="ew", padx=2)

    def add_model_entry(self):
        ef = ttk.Frame(self.models_frame); ef.pack(fill=tk.X, pady=5, padx=5)
        color = self.colors[len(self.model_entries) % len(self.colors)]
        
        entry_vars = {'path': tk.StringVar(), 'legend': tk.StringVar(), 'color': tk.StringVar(value=color), 
                      'line_style': tk.StringVar(value='solid'), 'line_width': tk.DoubleVar(value=1.8)}

        ttk.Label(ef, text="Model:").pack(side=tk.LEFT); ttk.Entry(ef, textvariable=entry_vars['path'], width=30).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(ef, text="Browse", command=lambda v=entry_vars: self.browse_model(v)).pack(side=tk.LEFT)
        ttk.Label(ef, text="Legend:").pack(side=tk.LEFT, padx=(10,2)); ttk.Entry(ef, textvariable=entry_vars['legend'], width=15).pack(side=tk.LEFT)
        
        cl = ttk.Label(ef, text="      ", background=color, relief="solid"); cl.pack(side=tk.LEFT, padx=(10, 2), ipady=2)
        ttk.Button(ef, text="Color", command=lambda v=entry_vars['color'], l=cl: self.choose_color(v, l)).pack(side=tk.LEFT)
        
        style_dd = ttk.Combobox(ef, textvariable=entry_vars['line_style'], values=['solid', 'dashed', 'dotted', 'dashdot'], width=7, state="readonly")
        style_dd.pack(side=tk.LEFT, padx=5); style_dd.set('solid')
        
        ttk.Label(ef, text="Width:").pack(side=tk.LEFT); ttk.Entry(ef, textvariable=entry_vars['line_width'], width=4).pack(side=tk.LEFT)
        self.model_entries.append(entry_vars)

    def choose_color(self, color_var, color_label):
        chosen_color = colorchooser.askcolor(color=color_var.get())[1]
        if chosen_color: color_var.set(chosen_color); color_label.config(background=chosen_color)

    def browse_model(self, entry_vars):
        path = filedialog.askopenfilename(title="Select Model File", filetypes=[("PyTorch models", "*.pth")])
        if path:
            entry_vars['path'].set(path)
            try: entry_vars['legend'].set(os.path.basename(os.path.dirname(os.path.dirname(path))))
            except: entry_vars['legend'].set(os.path.basename(path).rsplit('.', 1)[0])

    def generate_plot(self):
        try:
            df_gt = pd.read_csv(self.ground_truth_path.get(), engine='python')
            targets = ['joint1_torque', 'joint2_torque', 'joint3_torque']
            all_predictions = self.predict_models(df_gt, targets)
            
            # Create single error plot with all joints
            fig, axes = plt.subplots(3, 1, figsize=(self.plot_settings['width'].get(), self.plot_settings['height'].get() * 2), sharex=True)
            if not isinstance(axes, np.ndarray): axes = [axes]
            
            for i, (ax, joint_target) in enumerate(zip(axes, targets)):
                self.plot_error_curves(ax, df_gt, all_predictions, joint_target, i)
            
            # Shared legend at the top
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(labels), frameon=False, fontsize=self.plot_settings['legend_size'].get())
            fig.tight_layout(rect=[0, 0.03, 1, 0.99])
            
            chart_path = f"{(self.plot_prefix.get() or 'error')}_all_joints.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            webbrowser.open('file://' + os.path.realpath(chart_path))

        except Exception as e:
            messagebox.showerror("Error", f"Plot generation failed: {e}\n{traceback.format_exc()}"); traceback.print_exc()

    def predict_models(self, df_gt, targets):
        predictions = []
        features = [c for c in df_gt.columns if any(p in c for p in ['angle', 'velocity', 'acceleration'])]
        for entry in self.model_entries:
            with open(os.path.join(os.path.dirname(entry['path'].get()), "architecture_plus_hyperparameters.txt")) as f:
                params = dict(line.strip().split(': ') for line in f)
            model_type, sl = params.pop('model_type'), int(params.get('seq_length', 0))
            
            if model_type == 'FNN': model = FNN(len(features), [int(s) for s in params['hidden_layers'].split(',')], len(targets))
            elif model_type == 'RNN': model = RNN(len(features), int(params['hidden_size']), int(params['num_layers']), len(targets))
            else: model = E2NN(int(params['hidden_size']))
            model.load_state_dict(torch.load(entry['path'].get(), weights_only=True)); model.eval()

            scalers = {p: StandardScaler().fit(df_gt[[c for c in features if p in c]]) for p in ['angle', 'velocity', 'acceleration']}
            scaled_features = np.hstack([scalers[p].transform(df_gt[[c for c in features if p in c]]) for p in ['angle', 'velocity', 'acceleration']])
            
            with torch.no_grad():
                if model_type == 'E2NN':
                    preds = model(*[torch.tensor(scaled_features[:, i*3:(i+1)*3], dtype=torch.float32) for i in range(3)]).numpy()
                else:
                    X_in = create_sequences(scaled_features, sl) if sl > 0 else scaled_features
                    preds = StandardScaler().fit(df_gt[targets]).inverse_transform(model(torch.tensor(X_in, dtype=torch.float32)).numpy())
            predictions.append({'preds': pd.DataFrame(preds, columns=[f"predicted_{t}" for t in targets]), 'seq_len': sl, 'entry': entry})
        return predictions

    def plot_error_curves(self, ax, df_gt, all_predictions, joint_target, joint_idx):
        gt_values = df_gt[joint_target].values
        
        for pred_data in all_predictions:
            entry, preds, seq_len = pred_data['entry'], pred_data['preds'], pred_data['seq_len']
            ts = df_gt.index[seq_len:] if seq_len > 0 else df_gt.index
            pred_values = preds[f"predicted_{joint_target}"].values
            gt_aligned = gt_values[seq_len:] if seq_len > 0 else gt_values
            
            # Calculate error (ground truth - prediction)
            error = gt_aligned - pred_values
            
            ax.plot(ts, error, label=entry['legend'].get(), color=entry['color'].get(), 
                    linestyle=entry['line_style'].get(), lw=entry['line_width'].get())

        ax.set_ylabel(f"{joint_target.replace('_', ' ').title()} Error (Nm)", fontsize=self.plot_settings['axis_title_size'].get())
        ax.tick_params(axis='both', which='major', labelsize=self.plot_settings['label_size'].get())
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Add zero line for reference
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        
        if joint_idx == 2:  # Only bottom subplot gets x-label
            ax.set_xlabel(self.plot_settings['x_title'].get(), fontsize=self.plot_settings['axis_title_size'].get())

if __name__ == "__main__":
    root = tk.Tk(); app = ErrorPlotApp(root); root.mainloop() 