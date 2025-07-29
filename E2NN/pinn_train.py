import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
import numpy as np

# --- Model Definitions provided by user ---
class InertiaNN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 9)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, q):
        return self.net(q).reshape(-1, 3, 3)

class CoriolisVecNN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, q, qdot):
        x = torch.cat([q, qdot], dim=1)
        return self.net(x)

class GravityNN(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

# TrainingApp Class
class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Structured E2NN Model Training")
        self.root.geometry("1200x800")

        self.data_path = tk.StringVar()
        self.hyperparameters = {
            'learning_rate': tk.DoubleVar(value=0.001),
            'batch_size': tk.IntVar(value=128),
            'epochs': tk.IntVar(value=500),
            'hidden_size': tk.IntVar(value=128),
            'patience': tk.IntVar(value=20)
        }
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.create_widgets()

    def create_widgets(self):
        data_frame = ttk.LabelFrame(self.main_frame, text="Data Loading", padding="10")
        data_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Label(data_frame, text="Dataset Path:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(data_frame, textvariable=self.data_path, width=80).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(data_frame, text="Browse", command=self.browse_data).grid(row=0, column=2, padx=5)
        data_frame.columnconfigure(1, weight=1)

        hp_frame = ttk.LabelFrame(self.main_frame, text="Hyperparameters", padding="10")
        hp_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        ttk.Label(hp_frame, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(hp_frame, textvariable=self.hyperparameters['learning_rate']).grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(hp_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(hp_frame, textvariable=self.hyperparameters['batch_size']).grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Label(hp_frame, text="Epochs:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(hp_frame, textvariable=self.hyperparameters['epochs']).grid(row=2, column=1, sticky=tk.W, pady=2)
        ttk.Label(hp_frame, text="Hidden Size (per sub-net):").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(hp_frame, textvariable=self.hyperparameters['hidden_size']).grid(row=3, column=1, sticky=tk.W, pady=2)
        ttk.Label(hp_frame, text="Patience:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(hp_frame, textvariable=self.hyperparameters['patience']).grid(row=4, column=1, sticky=tk.W, pady=2)
        
        train_control_frame = ttk.Frame(self.main_frame, padding="10")
        train_control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(train_control_frame, text="Start Training", command=self.start_training).pack(side=tk.LEFT)
        self.save_button = ttk.Button(train_control_frame, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="10")
        self.viz_frame.grid(row=1, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)
        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.ax_loss = self.fig.add_subplot(211)
        self.ax_pred = self.fig.add_subplot(212)
        
        self.metrics_frame = ttk.LabelFrame(self.main_frame, text="Performance Metrics", padding="10")
        self.metrics_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.metrics_text = tk.Text(self.metrics_frame, height=8, width=50)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

    def browse_data(self):
        path = filedialog.askopenfilename(initialdir="../data", title="Select a Trajectory CSV file", filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
        if path: self.data_path.set(path)

    def start_training(self):
        try:
            self.df_full = pd.read_csv(self.data_path.get())
            
            features_q = ['joint1_angle', 'joint2_angle', 'joint3_angle']
            features_q_dot = ['joint1_velocity', 'joint2_velocity', 'joint3_velocity']
            features_q_ddot = ['joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration']
            self.targets = ['joint1_torque', 'joint2_torque', 'joint3_torque']

            X_q, X_q_dot, X_q_ddot = self.df_full[features_q].values, self.df_full[features_q_dot].values, self.df_full[features_q_ddot].values
            y = self.df_full[self.targets].values

            X_train_q, X_test_q, X_train_q_dot, X_test_q_dot, X_train_q_ddot, X_test_q_ddot, y_train, y_test = train_test_split(
                X_q, X_q_dot, X_q_ddot, y, test_size=0.2, random_state=42
            )

            self.scaler_q = StandardScaler().fit(X_train_q)
            self.scaler_q_dot = StandardScaler().fit(X_train_q_dot)
            self.scaler_q_ddot = StandardScaler().fit(X_train_q_ddot)

            train_dataset = TensorDataset(
                torch.tensor(self.scaler_q.transform(X_train_q), dtype=torch.float32),
                torch.tensor(self.scaler_q_dot.transform(X_train_q_dot), dtype=torch.float32),
                torch.tensor(self.scaler_q_ddot.transform(X_train_q_ddot), dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            self.test_dataset = TensorDataset(
                torch.tensor(self.scaler_q.transform(X_test_q), dtype=torch.float32),
                torch.tensor(self.scaler_q_dot.transform(X_test_q_dot), dtype=torch.float32),
                torch.tensor(self.scaler_q_ddot.transform(X_test_q_ddot), dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.float32)
            )

            train_loader = DataLoader(train_dataset, batch_size=self.hyperparameters['batch_size'].get(), shuffle=True)
            test_loader = DataLoader(self.test_dataset, batch_size=self.hyperparameters['batch_size'].get(), shuffle=False)

            self.model = E2NN(hidden_size=self.hyperparameters['hidden_size'].get())
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters['learning_rate'].get())
            early_stopping = EarlyStopping(patience=self.hyperparameters['patience'].get(), verbose=True)

            train_losses, val_losses = [], []
            for epoch in range(self.hyperparameters['epochs'].get()):
                self.model.train()
                epoch_train_loss = 0
                for q, q_dot, q_ddot, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(q, q_dot, q_ddot)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                train_losses.append(epoch_train_loss / len(train_loader))

                self.model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for q, q_dot, q_ddot, y_batch in test_loader:
                        outputs = self.model(q, q_dot, q_ddot)
                        loss = criterion(outputs, y_batch)
                        epoch_val_loss += loss.item()
                val_losses.append(epoch_val_loss / len(test_loader))
                
                print(f"Epoch [{epoch+1}/{self.hyperparameters['epochs'].get()}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
                early_stopping(val_losses[-1])
                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break

            self.plot_results(train_losses, val_losses)
            self.calculate_and_display_metrics()
            messagebox.showinfo("Training Complete", "E2NN model training has finished.")
            self.save_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during training: {e}")

    def plot_results(self, train_losses, val_losses):
        self.ax_loss.clear()
        self.ax_loss.plot(train_losses, label='Training Loss')
        self.ax_loss.plot(val_losses, label='Validation Loss')
        self.ax_loss.set_title('Model Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend()
        self.ax_loss.grid(True)
        
        self.model.eval()
        with torch.no_grad():
            q_full = torch.tensor(self.scaler_q.transform(self.df_full[['joint1_angle', 'joint2_angle', 'joint3_angle']].values), dtype=torch.float32)
            q_dot_full = torch.tensor(self.scaler_q_dot.transform(self.df_full[['joint1_velocity', 'joint2_velocity', 'joint3_velocity']].values), dtype=torch.float32)
            q_ddot_full = torch.tensor(self.scaler_q_ddot.transform(self.df_full[['joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration']].values), dtype=torch.float32)
            predictions = self.model(q_full, q_dot_full, q_ddot_full).numpy()
        
        y_true_full = self.df_full[self.targets].values
        
        self.ax_pred.clear()
        self.ax_pred.plot(y_true_full[:, 0], label='Actual Torque Joint 1', alpha=0.7)
        self.ax_pred.plot(predictions[:, 0], label='Predicted Torque Joint 1', linestyle='--', alpha=0.7)
        self.ax_pred.set_title('Torque Prediction on Full Dataset (Joint 1)')
        self.ax_pred.set_xlabel('Time Step')
        self.ax_pred.set_ylabel('Torque (Nm)')
        self.ax_pred.legend()
        self.ax_pred.grid(True)
        self.fig.tight_layout()
        self.canvas.draw()
        
    def calculate_and_display_metrics(self):
        self.model.eval()
        with torch.no_grad():
            q, q_dot, q_ddot, y_true_tensor = self.test_dataset.tensors
            predictions = self.model(q, q_dot, q_ddot).numpy()
        
        y_true = y_true_tensor.numpy()
        mse = np.mean((y_true - predictions)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - predictions))
        r2 = 1 - (np.sum((y_true - predictions)**2) / np.sum((y_true - np.mean(y_true, axis=0))**2))

        metrics_str = (f"MSE: {mse:.4f}\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR-squared (R²): {r2:.4f}\n")
        self.metrics_text.delete('1.0', tk.END)
        self.metrics_text.insert(tk.END, metrics_str)

    def save_results(self):
        lr, bs, hs = self.hyperparameters['learning_rate'].get(), self.hyperparameters['batch_size'].get(), self.hyperparameters['hidden_size'].get()
        folder_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lr{lr}_bs{bs}_hs{hs}"
        save_path = filedialog.askdirectory(initialdir=".", title="Select Folder to Save Results")
        if not save_path: return
        run_path = os.path.join(save_path, folder_name)
        os.makedirs(run_path, exist_ok=True)
        
        try:
            self.fig.savefig(os.path.join(run_path, f"plots_lr{lr}_bs{bs}.png"))
            torch.save(self.model.state_dict(), os.path.join(run_path, f"model_lr{lr}_bs{bs}.pth"))
            
            with open(os.path.join(run_path, "architecture_plus_hyperparameters.txt"), "w") as f:
                f.write(f"model_type: E2NN\n")
                f.write(f"hidden_size: {hs}\n")

            with open(os.path.join(run_path, "performance_metrics.txt"), "w") as f:
                f.write(self.metrics_text.get('1.0', tk.END))
            
            q_full = torch.tensor(self.scaler_q.transform(self.df_full[['joint1_angle', 'joint2_angle', 'joint3_angle']].values), dtype=torch.float32)
            q_dot_full = torch.tensor(self.scaler_q_dot.transform(self.df_full[['joint1_velocity', 'joint2_velocity', 'joint3_velocity']].values), dtype=torch.float32)
            q_ddot_full = torch.tensor(self.scaler_q_ddot.transform(self.df_full[['joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration']].values), dtype=torch.float32)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(q_full, q_dot_full, q_ddot_full).numpy()

            pred_df = pd.DataFrame(predictions, columns=[f'predicted_{t}' for t in self.targets])
            result_df = pd.concat([self.df_full, pred_df], axis=1)
            result_df.to_csv(os.path.join(run_path, f"predictions_lr{lr}_bs{bs}.csv"), index=False)
            
            messagebox.showinfo("Save Complete", f"Results saved successfully to:\n{run_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop() 