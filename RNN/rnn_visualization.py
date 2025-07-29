import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    return mse, rmse, mae, r2

class VisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RNN Model Visualization")
        self.root.geometry("1000x800")

        self.file_path = tk.StringVar()

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # File Selection Frame
        file_frame = ttk.LabelFrame(main_frame, text="Select Prediction File", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Label(file_frame, text="Predictions CSV:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.file_path, width=80).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        file_frame.columnconfigure(1, weight=1)

        ttk.Button(file_frame, text="Load and Visualize", command=self.visualize).grid(row=1, column=1, pady=5)

        # Visualization and Metrics Frame
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_frame.rowconfigure(1, weight=1)
        
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        
        self.metrics_text = tk.Text(results_frame, height=10, width=40)
        self.metrics_text.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

    def browse_file(self):
        path = filedialog.askopenfilename(
            initialdir=".",
            title="Select a predictions CSV file",
            filetypes=(("CSV files", "*.csv"), ("all files", "*.*"))
        )
        if path:
            self.file_path.set(path)

    def visualize(self):
        try:
            df = pd.read_csv(self.file_path.get())
            
            targets = ['joint1_torque', 'joint2_torque', 'joint3_torque']
            predicted_targets = [f'predicted_{t}' for t in targets]

            y_true = df[targets].values
            y_pred = df[predicted_targets].values

            self.fig.clear()
            for i, (target, pred_target) in enumerate(zip(targets, predicted_targets)):
                ax = self.fig.add_subplot(len(targets), 1, i+1)
                ax.plot(df.index, df[target], label=f'Actual {target}')
                ax.plot(df.index, df[pred_target], label=f'Predicted {target}', linestyle='--')
                ax.set_title(f'{target.replace("_", " ").title()} vs. Prediction')
                ax.set_ylabel('Torque (Nm)')
                ax.legend()
                ax.grid(True)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
            mse, rmse, mae, r2 = calculate_metrics(y_true, y_pred)
            metrics_str = (
                f"Overall Performance Metrics:\n\n"
                f"MSE: {mse:.4f}\n"
                f"RMSE: {rmse:.4f}\n"
                f"MAE: {mae:.4f}\n"
                f"R-squared (R²): {r2:.4f}"
            )
            self.metrics_text.delete('1.0', tk.END)
            self.metrics_text.insert(tk.END, metrics_str)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load or visualize data: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = VisualizationApp(root)
    root.mainloop() 