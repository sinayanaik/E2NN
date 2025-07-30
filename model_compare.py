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
        self.plot_prefix = tk.StringVar(value="comparison")
        self.zoom_start = tk.DoubleVar(value=460)
        self.zoom_end = tk.DoubleVar(value=520)

        # New plot settings dictionary
        self.plot_settings = {
            'x_title': tk.StringVar(value='Timestamp'),
            'y_title': tk.StringVar(value='Torque (Nm)'),
            'title_size': tk.IntVar(value=16),
            'label_size': tk.IntVar(value=12),
            'legend_size': tk.IntVar(value=10)
        }

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)

        # Ground Truth Selection
        gt_frame = ttk.LabelFrame(main_frame, text="Ground Truth Data", padding="10")
        gt_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(gt_frame, text="CSV Path:").pack(side=tk.LEFT)
        ttk.Entry(gt_frame, textvariable=self.ground_truth_path, width=80).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(gt_frame, text="Browse", command=self.browse_gt).pack(side=tk.LEFT)

        # Model Selection
        self.models_frame = ttk.LabelFrame(main_frame, text="Models to Compare", padding="10")
        self.models_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(self.models_frame, text="Add Model", command=self.add_model_entry).pack()

        # Define a list of visually distinct colors
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.color_index = 0

        # --- New Plot Customization Frame ---
        customization_frame = ttk.LabelFrame(main_frame, text="Plot Customization", padding="10")
        customization_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Filename and Zoom
        top_settings_frame = ttk.Frame(customization_frame)
        top_settings_frame.pack(fill=tk.X, pady=2)
        ttk.Label(top_settings_frame, text="Plot Filename Prefix:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(top_settings_frame, textvariable=self.plot_prefix, width=20).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(top_settings_frame, text="Zoom Start:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(top_settings_frame, textvariable=self.zoom_start, width=8).pack(side=tk.LEFT)
        ttk.Label(top_settings_frame, text="Zoom End:").pack(side=tk.LEFT, padx=(5, 5))
        ttk.Entry(top_settings_frame, textvariable=self.zoom_end, width=8).pack(side=tk.LEFT, padx=(0, 10))

        # Titles
        title_settings_frame = ttk.Frame(customization_frame)
        title_settings_frame.pack(fill=tk.X, pady=2)
        ttk.Label(title_settings_frame, text="X-Axis Title:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(title_settings_frame, textvariable=self.plot_settings['x_title']).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(title_settings_frame, text="Y-Axis Title:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(title_settings_frame, textvariable=self.plot_settings['y_title']).pack(side=tk.LEFT, padx=(0, 10))

        # Font Sizes
        font_settings_frame = ttk.Frame(customization_frame)
        font_settings_frame.pack(fill=tk.X, pady=2)
        ttk.Label(font_settings_frame, text="Title Size:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(font_settings_frame, textvariable=self.plot_settings['title_size'], width=5).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(font_settings_frame, text="Label Size:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(font_settings_frame, textvariable=self.plot_settings['label_size'], width=5).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(font_settings_frame, text="Legend Size:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(font_settings_frame, textvariable=self.plot_settings['legend_size'], width=5).pack(side=tk.LEFT, padx=(0, 10))

        # Controls
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(control_frame, text="Generate Comparison Plot", command=self.generate_plot).pack()

    def browse_gt(self):
        path = filedialog.askopenfilename(title="Select Ground Truth CSV", filetypes=[("CSV files", "*.csv")])
        if path: self.ground_truth_path.set(path)

    def add_model_entry(self):
        entry_frame = ttk.Frame(self.models_frame)
        entry_frame.pack(fill=tk.X, pady=5, padx=5)

        model_path = tk.StringVar()
        legend_name = tk.StringVar()
        
        color_val = self.colors[self.color_index % len(self.colors)]
        self.color_index += 1
        color_var = tk.StringVar(value=color_val)
        line_style_var = tk.StringVar(value='solid')

        # Model Path
        ttk.Label(entry_frame, text="Model:").pack(side=tk.LEFT)
        entry = ttk.Entry(entry_frame, textvariable=model_path, width=40)
        entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(entry_frame, text="Browse", command=lambda p=model_path, l=legend_name: self.browse_model(p, l)).pack(side=tk.LEFT)
        
        # Legend
        ttk.Label(entry_frame, text="Legend:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(entry_frame, textvariable=legend_name, width=15).pack(side=tk.LEFT)
        
        # Color Chooser
        color_label = ttk.Label(entry_frame, text="      ", background=color_val, relief="solid", borderwidth=1)
        color_label.pack(side=tk.LEFT, padx=(10, 2), ipady=2)
        def choose_color_for_entry():
            chosen_color = colorchooser.askcolor(color=color_var.get())[1]
            if chosen_color:
                color_var.set(chosen_color)
                color_label.config(background=chosen_color)
        ttk.Button(entry_frame, text="Color", command=choose_color_for_entry).pack(side=tk.LEFT)

        # Line Style
        ttk.Label(entry_frame, text="Style:").pack(side=tk.LEFT, padx=(10, 2))
        style_dropdown = ttk.Combobox(entry_frame, textvariable=line_style_var, values=['solid', 'dashed', 'dotted', 'dotdash'], width=7, state="readonly")
        style_dropdown.pack(side=tk.LEFT)
        style_dropdown.set('solid')

        self.model_entries.append({'path': model_path, 'legend': legend_name, 'color': color_var, 'line_style': line_style_var, 'frame': entry_frame})

    def browse_model(self, path_var, legend_var):
        path = filedialog.askopenfilename(title="Select Model File", filetypes=[("PyTorch models", "*.pth")])
        if path:
            path_var.set(path)
            # Auto-fill legend name with a more descriptive one from the parent directory
            try:
                # e.g., .../1_very_good/run_.../model.pth -> "1_very_good"
                parent_dir_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
                legend_var.set(parent_dir_name)
            except Exception:
                # Fallback to filename without extension
                legend_var.set(os.path.basename(path).rsplit('.', 1)[0])

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

            # Get global plot settings
            x_title = self.plot_settings['x_title'].get()
            y_title = self.plot_settings['y_title'].get()
            title_size = self.plot_settings['title_size'].get()
            label_size = self.plot_settings['label_size'].get()
            legend_size = self.plot_settings['legend_size'].get()

            for i, joint_target in enumerate(targets):
                # --- Restructured Plotting Logic ---
                plot_data_list = []
                metrics_summary = []
                all_metrics[joint_target] = {}

                # Setup color and dash scales
                color_domain = ['GROUND TRUTH']
                color_range = ['#000000']
                altair_dash_map = {'solid': [], 'dashed': [8, 4], 'dotted': [2, 2], 'dotdash': [8, 4, 2, 4]}
                dash_domain = ['GROUND TRUTH']
                dash_range = [altair_dash_map['solid']]

                # Add Ground Truth data
                gt_temp_df = pd.DataFrame({
                    'timestamp': df_gt.index,
                    'Torque': df_gt[joint_target],
                    'Legend': 'GROUND TRUTH'
                })
                plot_data_list.append(gt_temp_df)
                
                # Process each model's prediction
                for pred_data in all_predictions:
                    legend = pred_data['legend']
                    preds = pred_data['preds']
                    seq_len = pred_data['seq_len']
                    
                    # Get model-specific style settings
                    model_entry = next((m for m in self.model_entries if m['legend'].get() == legend), None)
                    if not model_entry: continue

                    color_domain.append(legend)
                    color_range.append(model_entry['color'].get())
                    dash_domain.append(legend)
                    dash_range.append(altair_dash_map.get(model_entry['line_style'].get(), []))

                    pred_col = f"predicted_{joint_target}"
                    temp_df = pd.DataFrame({
                        'timestamp': df_gt.index[seq_len:] if seq_len > 0 else df_gt.index,
                        'Torque': preds[pred_col],
                        'Legend': legend
                    })
                    plot_data_list.append(temp_df)

                    # Calculate metrics
                    y_true = df_gt[joint_target].values[seq_len:] if seq_len > 0 else df_gt[joint_target].values
                    y_pred = preds[pred_col].values
                    mse = np.mean((y_true - y_pred)**2); rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(y_true - y_pred))
                    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
                    all_metrics[joint_target][legend] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
                    metrics_summary.append(f"{legend} (RMSE: {rmse:.3f}, R²: {r2:.3f})")

                plot_data = pd.concat(plot_data_list)
                chart_title = f"Torque Comparison for {joint_target.replace('_', ' ').title()}"
                chart_subtitle = " | ".join(metrics_summary)

                base_chart = alt.Chart(plot_data).mark_line(point=False).encode(
                    x=alt.X('timestamp:Q', title=x_title),
                    y=alt.Y('Torque:Q', title=y_title),
                    color=alt.Color('Legend:N', 
                                  scale=alt.Scale(domain=color_domain, range=color_range),
                                  legend=alt.Legend(title="Model")),
                    strokeDash=alt.StrokeDash('Legend:N', 
                                            scale=alt.Scale(domain=dash_domain, range=dash_range)),
                    order='timestamp:Q'
                ).properties(
                    title={'text': chart_title, 'subtitle': chart_subtitle},
                    width=800
                )

                # --- Zoomed View Logic ---
                try:
                    zoom_x_start = self.zoom_start.get()
                    zoom_x_end = self.zoom_end.get()
                    if zoom_x_start >= zoom_x_end:
                        messagebox.showwarning("Invalid Range", "Zoom Start must be less than Zoom End. Using default values.")
                        zoom_x_start, zoom_x_end = 460, 520
                except (tk.TclError, ValueError):
                    messagebox.showwarning("Invalid Input", "Invalid zoom range values. Please enter numbers. Using default values.")
                    zoom_x_start, zoom_x_end = 460, 520

                # Rectangle to highlight the zoom area on the main chart
                zoom_area_highlight = alt.Chart(pd.DataFrame([
                    {'x': zoom_x_start, 'x2': zoom_x_end}
                ])).mark_rect(
                    opacity=0.2, color='gray'
                ).encode(x='x', x2='x2')

                # We create a new chart object to avoid property inheritance issues
                # Filter data for the zoom view explicitly for robustness
                zoomed_plot_data = plot_data[
                    (plot_data['timestamp'] >= zoom_x_start) & 
                    (plot_data['timestamp'] <= zoom_x_end)
                ].copy()

                zoomed_chart = alt.Chart(zoomed_plot_data).mark_line(point=False).encode(
                    x=alt.X('timestamp:Q', 
                            title=f"Zoomed View (Timestamp {zoom_x_start} to {zoom_x_end})"),
                    y=alt.Y('Torque:Q', scale=alt.Scale(zero=False), title=y_title),
                    color=alt.Color('Legend:N', scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
                    strokeDash=alt.StrokeDash('Legend:N', scale=alt.Scale(domain=dash_domain, range=dash_range), legend=None),
                    order='timestamp:Q'
                ).properties(
                    width=800,
                    height=200
                )

                # Layer the main chart with its highlight
                main_chart_with_highlight = base_chart + zoom_area_highlight

                # Combine the main chart (with highlight) and the zoomed chart vertically
                # Apply configurations to the final concatenated chart
                final_chart = alt.vconcat(
                    main_chart_with_highlight.interactive(),
                    zoomed_chart.interactive()
                ).configure_title(
                    fontSize=title_size
                ).configure_axis(
                    labelFontSize=label_size,
                    titleFontSize=label_size
                ).configure_legend(
                    titleFontSize=label_size,
                    labelFontSize=legend_size
                )
                
                prefix = self.plot_prefix.get().strip()
                if not prefix:
                    prefix = "comparison"
                chart_path = f'{prefix}_{joint_target}.html'
                final_chart.save(chart_path)
                chart_paths.append(chart_path)
                self.charts.append(final_chart) # Save chart object

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
                joint_name = chart.title['text'].split(" for ")[-1].replace(" ", "_")
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