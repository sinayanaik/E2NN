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
from matplotlib.patches import Ellipse, ConnectionPatch

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
class ComparisonApp:
    def __init__(self, root):
        self.root = root; self.root.title("Model Comparison Tool")
        
        self.ground_truth_path, self.model_entries, self.plot_prefix = tk.StringVar(), [], tk.StringVar(value="comparison")
        # Ground truth line settings
        self.ground_truth_settings = {'color': tk.StringVar(value='black'), 'line_style': tk.StringVar(value='solid'), 'line_width': tk.DoubleVar(value=2.5)}
        
        # --- Settings with default values ---
        self.zoom_center_x, self.zoom_x_range = tk.DoubleVar(value=650), tk.DoubleVar(value=25)
        self.inset_radius = tk.DoubleVar(value=0.22)
        # Individual joint magnifier positions
        self.joint1_pos = {'x': tk.DoubleVar(value=0.75), 'y': tk.DoubleVar(value=0.25)}
        self.joint2_pos = {'x': tk.DoubleVar(value=0.55), 'y': tk.DoubleVar(value=0.65)}
        self.joint3_pos = {'x': tk.DoubleVar(value=0.55), 'y': tk.DoubleVar(value=0.55)}
        self.plot_settings = {'width': tk.DoubleVar(value=32), 'height': tk.DoubleVar(value=18),
                              'x_title': tk.StringVar(value='Timestamp'), 'y_title': tk.StringVar(value='τ (Nm)'),
                              'axis_title_size': tk.IntVar(value=50), 'label_size': tk.IntVar(value=50), 
                              'legend_size': tk.IntVar(value=50), 'dpi': tk.IntVar(value=100)}

        main_frame = ttk.Frame(self.root, padding="10"); main_frame.grid(sticky="nsew"); self.root.columnconfigure(0, weight=1)
        self.setup_data_ui(main_frame)
        self.setup_settings_ui(main_frame)
        ttk.Button(main_frame, text="Generate Comparison Plot", command=self.generate_plot, style="Accent.TButton").grid(row=3, column=0, pady=10)
        ttk.Style(self.root).configure("Accent.TButton", foreground="white", background="black")

    def setup_data_ui(self, p):
        gt_frame = ttk.LabelFrame(p, text="Ground Truth Data", padding="10"); gt_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # CSV path row
        path_frame = ttk.Frame(gt_frame); path_frame.pack(fill=tk.X, pady=2)
        ttk.Label(path_frame, text="CSV Path:").pack(side=tk.LEFT)
        ttk.Entry(path_frame, textvariable=self.ground_truth_path).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        browse_btn = ttk.Button(path_frame, text="Browse", command=lambda: self.ground_truth_path.set(filedialog.askopenfilename(title="Select Ground Truth CSV", filetypes=[("CSV files", "*.csv")])))
        browse_btn.pack(side=tk.LEFT)
        
        # Ground truth line settings row
        gt_settings_frame = ttk.Frame(gt_frame); gt_settings_frame.pack(fill=tk.X, pady=2)
        ttk.Label(gt_settings_frame, text="Color:").pack(side=tk.LEFT)
        
        gt_color_label = ttk.Label(gt_settings_frame, text="      ", background='black', relief="solid")
        gt_color_label.pack(side=tk.LEFT, padx=(5, 2), ipady=2)
        ttk.Button(gt_settings_frame, text="Color", command=lambda: self.choose_ground_truth_color(gt_color_label)).pack(side=tk.LEFT)
        
        ttk.Label(gt_settings_frame, text="Style:").pack(side=tk.LEFT, padx=(10, 2))
        gt_style_dd = ttk.Combobox(gt_settings_frame, textvariable=self.ground_truth_settings['line_style'], 
                                   values=['solid', 'dashed', 'dotted', 'dashdot'], width=7, state="readonly")
        gt_style_dd.pack(side=tk.LEFT, padx=2); gt_style_dd.set('solid')
        
        ttk.Label(gt_settings_frame, text="Width:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(gt_settings_frame, textvariable=self.ground_truth_settings['line_width'], width=4).pack(side=tk.LEFT)
        
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
                         ("Legend Size:", self.plot_settings['legend_size']), ("DPI:", self.plot_settings['dpi'])]
        for i, (label, var) in enumerate(settings_vars):
            ttk.Label(psf, text=label).grid(row=i, column=0, sticky="w", padx=2, pady=2)
            ttk.Entry(psf, textvariable=var, width=15).grid(row=i, column=1, sticky="ew", padx=2)

        mf = ttk.LabelFrame(sf, text="Magnifier Settings", padding=10); mf.grid(row=0, column=1, sticky="nsew", padx=5); mf.columnconfigure(1, weight=1)
        
        # General magnifier settings
        ttk.Label(mf, text="Zoom At Timestamp:").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(mf, textvariable=self.zoom_center_x, width=10).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Label(mf, text="Zoom Width:").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(mf, textvariable=self.zoom_x_range, width=10).grid(row=1, column=1, sticky="ew", padx=2)
        ttk.Label(mf, text="Magnifier Size:").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(mf, textvariable=self.inset_radius, width=10).grid(row=2, column=1, sticky="ew", padx=2)
        
        # Joint-specific positions
        ttk.Label(mf, text="Joint1 X Pos:").grid(row=3, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(mf, textvariable=self.joint1_pos['x'], width=10).grid(row=3, column=1, sticky="ew", padx=2)
        ttk.Label(mf, text="Joint1 Y Pos:").grid(row=4, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(mf, textvariable=self.joint1_pos['y'], width=10).grid(row=4, column=1, sticky="ew", padx=2)
        
        ttk.Label(mf, text="Joint2 X Pos:").grid(row=5, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(mf, textvariable=self.joint2_pos['x'], width=10).grid(row=5, column=1, sticky="ew", padx=2)
        ttk.Label(mf, text="Joint2 Y Pos:").grid(row=6, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(mf, textvariable=self.joint2_pos['y'], width=10).grid(row=6, column=1, sticky="ew", padx=2)
        
        ttk.Label(mf, text="Joint3 X Pos:").grid(row=7, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(mf, textvariable=self.joint3_pos['x'], width=10).grid(row=7, column=1, sticky="ew", padx=2)
        ttk.Label(mf, text="Joint3 Y Pos:").grid(row=8, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(mf, textvariable=self.joint3_pos['y'], width=10).grid(row=8, column=1, sticky="ew", padx=2)

    def add_model_entry(self):
        ef = ttk.Frame(self.models_frame); ef.pack(fill=tk.X, pady=5, padx=5)
        color = self.colors[len(self.model_entries) % len(self.colors)]
        
        entry_vars = {'path': tk.StringVar(), 'legend': tk.StringVar(), 'color': tk.StringVar(value=color), 
                      'line_style': tk.StringVar(value='solid'), 'line_width': tk.DoubleVar(value=6.5)}

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
    
    def choose_ground_truth_color(self, color_label):
        chosen_color = colorchooser.askcolor(color=self.ground_truth_settings['color'].get())[1]
        if chosen_color: 
            self.ground_truth_settings['color'].set(chosen_color)
            color_label.config(background=chosen_color)

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
            
            # --- Individual Plots ---
            for joint_target in targets:
                fig, ax = plt.subplots(figsize=(self.plot_settings['width'].get(), self.plot_settings['height'].get()))
                self.plot_all_curves(ax, df_gt, all_predictions, joint_target)
                self.create_circular_magnifier(fig, ax, df_gt, all_predictions, joint_target)
                # Set custom Y-axis label for individual plots
                joint_number = joint_target.replace('joint', '').replace('_torque', '')
                ax.set_ylabel(f"$\\tau_{{{joint_number}}}$ (Nm)", fontsize=self.plot_settings['axis_title_size'].get())
                plt.tight_layout(pad=1.0)
                dpi_value = self.plot_settings['dpi'].get()
                # Save PNG
                chart_path_png = f"{(self.plot_prefix.get() or 'comparison')}_{joint_target}.png"
                plt.savefig(chart_path_png, dpi=dpi_value, bbox_inches='tight')
                # Save SVG
                chart_path_svg = f"{(self.plot_prefix.get() or 'comparison')}_{joint_target}.svg"
                plt.savefig(chart_path_svg, dpi=dpi_value, bbox_inches='tight', format='svg')
                plt.close(fig)
                webbrowser.open('file://' + os.path.realpath(chart_path_png))

            # --- Combined Plot ---
            n_joints = len(targets)
            fig_combined, axes_combined = plt.subplots(n_joints, 1, figsize=(self.plot_settings['width'].get(), self.plot_settings['height'].get() * n_joints * 0.6), sharex=True)
            if n_joints == 1: axes_combined = [axes_combined]

            for i, ax in enumerate(axes_combined):
                joint_target = targets[i]
                self.plot_all_curves(ax, df_gt, all_predictions, joint_target, draw_legend=False)
                self.create_circular_magnifier(fig_combined, ax, df_gt, all_predictions, joint_target)
                # Set custom Y-axis label for combined plot
                joint_number = joint_target.replace('joint', '').replace('_torque', '')
                ax.set_ylabel(f"$\\tau_{{{joint_number}}}$ (Nm)", fontsize=self.plot_settings['axis_title_size'].get())
                if i < n_joints - 1: ax.set_xlabel('')
            
            handles, labels = axes_combined[0].get_legend_handles_labels()
            fig_combined.subplots_adjust(top=0.92, left=0.02, bottom=0.08)
            fig_combined.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                                ncol=len(labels), frameon=False, fontsize=self.plot_settings['legend_size'].get())
            
            dpi_value = self.plot_settings['dpi'].get()
            # Save PNG
            combined_path_png = f"{(self.plot_prefix.get() or 'comparison')}_all_joints.png"
            plt.savefig(combined_path_png, dpi=dpi_value, bbox_inches='tight')
            # Save SVG
            combined_path_svg = f"{(self.plot_prefix.get() or 'comparison')}_all_joints.svg"
            plt.savefig(combined_path_svg, dpi=dpi_value, bbox_inches='tight', format='svg')
            plt.close(fig_combined)
            webbrowser.open('file://' + os.path.realpath(combined_path_png))

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

    def plot_all_curves(self, ax, df_gt, all_predictions, joint_target, draw_legend=True):
        ax.plot(df_gt.index, df_gt[joint_target], label='GROUND TRUTH', 
                color=self.ground_truth_settings['color'].get(), 
                linestyle=self.ground_truth_settings['line_style'].get(), 
                lw=self.ground_truth_settings['line_width'].get(), zorder=10)
        for pred_data in all_predictions:
            entry, preds, seq_len = pred_data['entry'], pred_data['preds'], pred_data['seq_len']
            ts = df_gt.index[seq_len:] if seq_len > 0 else df_gt.index
            ax.plot(ts, preds[f"predicted_{joint_target}"].values, label=entry['legend'].get(), color=entry['color'].get(), 
                    linestyle=entry['line_style'].get(), lw=entry['line_width'].get())

        ax.set_xlabel(self.plot_settings['x_title'].get(), fontsize=self.plot_settings['axis_title_size'].get())
        ax.set_ylabel(self.plot_settings['y_title'].get(), fontsize=self.plot_settings['axis_title_size'].get())
        ax.tick_params(axis='both', which='major', labelsize=self.plot_settings['label_size'].get())
        if draw_legend:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=len(all_predictions) + 1,
                      fontsize=self.plot_settings['legend_size'].get(), frameon=False)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    def create_circular_magnifier(self, fig, ax, df_gt, all_predictions, joint_target):
        radius = self.inset_radius.get()
        zoom_center, zoom_range = self.zoom_center_x.get(), self.zoom_x_range.get()
        zoom_x1, zoom_x2 = zoom_center - zoom_range / 2, zoom_center + zoom_range / 2
        
        # Get joint-specific position
        if 'joint1' in joint_target:
            cx, cy = self.joint1_pos['x'].get(), self.joint1_pos['y'].get()
        elif 'joint2' in joint_target:
            cx, cy = self.joint2_pos['x'].get(), self.joint2_pos['y'].get()
        elif 'joint3' in joint_target:
            cx, cy = self.joint3_pos['x'].get(), self.joint3_pos['y'].get()
        else:
            cx, cy = 0.75, 0.35  # default position
        
        # Create circular inset (perfect circle)
        axins = ax.inset_axes([cx - radius, cy - radius, 2 * radius, 2 * radius], transform=ax.transAxes, zorder=12)
        
        # Plot data in inset
        for pred_data in all_predictions + [{'entry': {'color': self.ground_truth_settings['color'], 'line_style': self.ground_truth_settings['line_style'], 'line_width': self.ground_truth_settings['line_width']}, 'preds': df_gt.rename(columns={joint_target: f"predicted_{joint_target}"}), 'seq_len': 0}]:
            entry, preds, seq_len = pred_data['entry'], pred_data['preds'], pred_data['seq_len']
            ts = df_gt.index[seq_len:] if seq_len > 0 else df_gt.index
            axins.plot(ts, preds[f"predicted_{joint_target}"].values, color=entry['color'].get(), linestyle=entry['line_style'].get(), lw=entry['line_width'].get())

        zoom_mask = (df_gt.index >= zoom_x1) & (df_gt.index <= zoom_x2)
        if not np.any(zoom_mask): return
        min_y, max_y = np.min(df_gt.loc[zoom_mask, joint_target]), np.max(df_gt.loc[zoom_mask, joint_target])
        y_padding = (max_y - min_y) * 0.1
        axins.set_xlim(zoom_x1, zoom_x2); axins.set_ylim(min_y - y_padding, max_y + y_padding)
        
        axins.set_facecolor('white'); axins.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
        for spine in axins.spines.values(): spine.set_visible(False)
        
        # Perfect circle clipping
        from matplotlib.patches import Circle
        clip_circle = Circle((0.5, 0.5), 0.5, transform=axins.transAxes, facecolor='none', edgecolor='none')
        axins.add_patch(clip_circle)
        for artist in axins.get_children(): artist.set_clip_path(clip_circle)

        # Perfect circle border
        border = Circle((cx, cy), radius, transform=ax.transAxes, facecolor='none', edgecolor='black', lw=1.5, zorder=13)
        ax.add_patch(border)
        
        # Highlight rectangle on main plot
        rect_patch = plt.Rectangle((zoom_x1, min_y), zoom_x2 - zoom_x1, max_y - min_y, fc='grey', ec='black', alpha=0.15, ls='--')
        ax.add_patch(rect_patch)
        
        # Connection lines
        for x_coord, corner in [(zoom_x1, 'left'), (zoom_x2, 'right')]:
            xyA = (x_coord, min_y + (max_y-min_y)/2)
            angle = np.deg2rad(225 if corner == 'left' else -45)
            xyB = (cx + radius * np.cos(angle), cy + radius * np.sin(angle))
            ax.add_artist(ConnectionPatch(xyA=xyA, xyB=xyB, coordsA=ax.transData, coordsB=ax.transAxes, ec="gray", lw=1, ls='--', zorder=12))

if __name__ == "__main__":
    root = tk.Tk(); app = ComparisonApp(root); root.mainloop()
