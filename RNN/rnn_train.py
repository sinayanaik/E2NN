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

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, _ = self.rnn(x)
        # out shape: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out

def create_sequences(input_data, output_data, seq_length):
    xs, ys = [], []
    for i in range(len(input_data) - seq_length):
        x = input_data[i:(i + seq_length)]
        y = output_data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RNN Model Training")
        self.root.geometry("1200x800")

        self.data_path = tk.StringVar()
        self.hyperparameters = {
            'learning_rate': tk.DoubleVar(value=0.001),
            'batch_size': tk.IntVar(value=32),
            'epochs': tk.IntVar(value=100),
            'hidden_size': tk.IntVar(value=128),
            'num_layers': tk.IntVar(value=2),
            'seq_length': tk.IntVar(value=10)
        }

        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.create_widgets()

    def create_widgets(self):
        # Data Loading Frame
        data_frame = ttk.LabelFrame(self.main_frame, text="Data Loading", padding="10")
        data_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Label(data_frame, text="Dataset Path:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(data_frame, textvariable=self.data_path, width=80).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(data_frame, text="Browse", command=self.browse_data).grid(row=0, column=2, padx=5)
        data_frame.columnconfigure(1, weight=1)

        # Hyperparameters Frame
        hp_frame = ttk.LabelFrame(self.main_frame, text="Hyperparameters", padding="10")
        hp_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        ttk.Label(hp_frame, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(hp_frame, textvariable=self.hyperparameters['learning_rate']).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(hp_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(hp_frame, textvariable=self.hyperparameters['batch_size']).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(hp_frame, text="Epochs:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(hp_frame, textvariable=self.hyperparameters['epochs']).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(hp_frame, text="Hidden Size:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(hp_frame, textvariable=self.hyperparameters['hidden_size']).grid(row=3, column=1, sticky=tk.W, pady=2)

        ttk.Label(hp_frame, text="Num Layers:").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(hp_frame, textvariable=self.hyperparameters['num_layers']).grid(row=4, column=1, sticky=tk.W, pady=2)

        ttk.Label(hp_frame, text="Sequence Length:").grid(row=5, column=0, sticky=tk.W, pady=2)
        ttk.Entry(hp_frame, textvariable=self.hyperparameters['seq_length']).grid(row=5, column=1, sticky=tk.W, pady=2)

        # Training Control Frame
        train_control_frame = ttk.Frame(self.main_frame, padding="10")
        train_control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(train_control_frame, text="Start Training", command=self.start_training).pack(side=tk.LEFT)
        self.save_button = ttk.Button(train_control_frame, text="Save Results", command=self.save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=10)

        # Visualization Frame (rest is same as FNN)
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="10")
        self.viz_frame.grid(row=1, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(2, weight=1)

        self.fig = Figure(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.ax_loss = self.fig.add_subplot(211)
        self.ax_pred = self.fig.add_subplot(212)

        # Performance Metrics Frame
        self.metrics_frame = ttk.LabelFrame(self.main_frame, text="Performance Metrics", padding="10")
        self.metrics_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.metrics_text = tk.Text(self.metrics_frame, height=8, width=50)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

    def browse_data(self):
        file_path = filedialog.askopenfilename(
            initialdir="../data",
            title="Select a Trajectory CSV file",
            filetypes=(("CSV files", "*.csv"), ("all files", "*.*"))
        )
        if file_path:
            self.data_path.set(file_path)

    def start_training(self):
        try:
            df = pd.read_csv(self.data_path.get())
            self.features = ['joint1_angle', 'joint2_angle', 'joint3_angle', 
                             'joint1_velocity', 'joint2_velocity', 'joint3_velocity',
                             'joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration']
            self.targets = ['joint1_torque', 'joint2_torque', 'joint3_torque']
            
            X = df[self.features].values
            y = df[self.targets].values
            
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()

            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y)
            
            seq_length = self.hyperparameters['seq_length'].get()
            X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)
            
            X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
            test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

            train_loader = DataLoader(train_dataset, batch_size=self.hyperparameters['batch_size'].get(), shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.hyperparameters['batch_size'].get(), shuffle=False)

            model = RNN(input_size=len(self.features), 
                        hidden_size=self.hyperparameters['hidden_size'].get(), 
                        num_layers=self.hyperparameters['num_layers'].get(), 
                        output_size=len(self.targets))

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.hyperparameters['learning_rate'].get())

            train_losses, val_losses = [], []
            for epoch in range(self.hyperparameters['epochs'].get()):
                model.train()
                epoch_train_loss = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                train_losses.append(epoch_train_loss / len(train_loader))
                
                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in test_loader:
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        epoch_val_loss += loss.item()
                val_losses.append(epoch_val_loss / len(test_loader))

                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.hyperparameters["epochs"].get()}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
            
            self.model = model
            self.df_full = df
            self.X_test_seq = X_test
            self.y_test_seq = y_test
            self.plot_results(train_losses, val_losses)
            self.calculate_and_display_metrics()
            messagebox.showinfo("Training Complete", "Model training has finished successfully.")
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
        
        # Create sequences from the full dataset for plotting
        X_full_scaled = self.scaler_X.transform(self.df_full[self.features].values)
        y_full_scaled = self.scaler_y.transform(self.df_full[self.targets].values)
        X_full_seq, y_true_full_seq_scaled = create_sequences(X_full_scaled, y_full_scaled, self.hyperparameters['seq_length'].get())

        with torch.no_grad():
            predictions_scaled = self.model(torch.tensor(X_full_seq, dtype=torch.float32)).numpy()

        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        y_true_full = self.scaler_y.inverse_transform(y_true_full_seq_scaled)
        
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
            predictions_scaled = self.model(torch.tensor(self.X_test_seq, dtype=torch.float32)).numpy()
        
        y_pred = self.scaler_y.inverse_transform(predictions_scaled)
        y_true = self.scaler_y.inverse_transform(self.y_test_seq)
        
        mse = np.mean((y_true - y_pred)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

        metrics_str = (
            f"Mean Squared Error (MSE): {mse:.4f}\n"
            f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
            f"Mean Absolute Error (MAE): {mae:.4f}\n"
            f"R-squared (R²): {r2:.4f}\n"
        )
        self.metrics_text.delete('1.0', tk.END)
        self.metrics_text.insert(tk.END, metrics_str)

    def save_results(self):
        lr = self.hyperparameters['learning_rate'].get()
        bs = self.hyperparameters['batch_size'].get()
        hs = self.hyperparameters['hidden_size'].get()
        nl = self.hyperparameters['num_layers'].get()
        sl = self.hyperparameters['seq_length'].get()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        folder_name = f"run_{timestamp}_lr{lr}_bs{bs}_hs{hs}_nl{nl}_sl{sl}"
        
        save_path = filedialog.askdirectory(initialdir=".", title="Select Folder to Save Results")
        
        if not save_path: return

        run_path = os.path.join(save_path, folder_name)
        os.makedirs(run_path, exist_ok=True)

        try:
            self.fig.savefig(os.path.join(run_path, f"plots_lr{lr}_bs{bs}.png"))
            torch.save(self.model.state_dict(), os.path.join(run_path, f"model_lr{lr}_bs{bs}.pth"))
            
            # Save architecture and hyperparameters
            with open(os.path.join(run_path, "architecture_plus_hyperparameters.txt"), "w") as f:
                f.write(f"model_type: RNN\n")
                f.write(f"hidden_size: {self.hyperparameters['hidden_size'].get()}\n")
                f.write(f"num_layers: {self.hyperparameters['num_layers'].get()}\n")
                f.write(f"seq_length: {self.hyperparameters['seq_length'].get()}\n")

            with open(os.path.join(run_path, "performance_metrics.txt"), "w") as f:
                f.write(self.metrics_text.get('1.0', tk.END))

            df = pd.read_csv(self.data_path.get())
            X_full_scaled = self.scaler_X.transform(df[self.features].values)
            y_full_scaled = self.scaler_y.transform(df[self.targets].values)
            X_full_seq, _ = create_sequences(X_full_scaled, y_full_scaled, self.hyperparameters['seq_length'].get())

            self.model.eval()
            with torch.no_grad():
                predictions_scaled = self.model(torch.tensor(X_full_seq, dtype=torch.float32)).numpy()
            
            predictions = self.scaler_y.inverse_transform(predictions_scaled)
            pred_df = pd.DataFrame(predictions, columns=[f'predicted_{t}' for t in self.targets])
            
            # Align predictions with original data
            result_df = df.iloc[self.hyperparameters['seq_length'].get():].reset_index(drop=True)
            result_df = pd.concat([result_df, pred_df], axis=1)
            result_df.to_csv(os.path.join(run_path, f"predictions_lr{lr}_bs{bs}.csv"), index=False)
            
            messagebox.showinfo("Save Complete", f"Results saved successfully to:\n{run_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop() 