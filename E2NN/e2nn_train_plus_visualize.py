import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader

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
        
        # M(q) * q_ddot
        # M_q is [batch, 3, 3], q_ddot is [batch, 3] -> [batch, 3, 1]
        m_q_ddot = torch.bmm(M_q, q_ddot.unsqueeze(-1)).squeeze(-1) # -> [batch, 3]

        # Euler-Lagrange equation
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

def train_model(hyperparameters, data_path, stroke_option, test_split_ratio):
    print("Starting training with the following hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")
    
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}_hs{hyperparameters['hidden_size']}_{stroke_option.replace(' ', '')}"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), run_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    if stroke_option == 'Forward Stroke Only':
        df = df.iloc[:500]

    features_q = ['joint1_angle', 'joint2_angle', 'joint3_angle']
    features_q_dot = ['joint1_velocity', 'joint2_velocity', 'joint3_velocity']
    features_q_ddot = ['joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration']
    targets = ['joint1_torque', 'joint2_torque', 'joint3_torque']

    X_q = df[features_q].values
    X_q_dot = df[features_q_dot].values
    X_q_ddot = df[features_q_ddot].values
    y = df[targets].values

    indices = np.arange(len(df))
    
    X_train_q, X_test_q, X_train_q_dot, X_test_q_dot, X_train_q_ddot, X_test_q_ddot, y_train, y_test, train_indices, test_indices = train_test_split(
        X_q, X_q_dot, X_q_ddot, y, indices, test_size=test_split_ratio, random_state=42
    )

    scaler_q = StandardScaler()
    X_train_q = scaler_q.fit_transform(X_train_q)
    X_test_q = scaler_q.transform(X_test_q)
    
    scaler_q_dot = StandardScaler()
    X_train_q_dot = scaler_q_dot.fit_transform(X_train_q_dot)
    X_test_q_dot = scaler_q_dot.transform(X_test_q_dot)

    scaler_q_ddot = StandardScaler()
    X_train_q_ddot = scaler_q_ddot.fit_transform(X_train_q_ddot)
    X_test_q_ddot = scaler_q_ddot.transform(X_test_q_ddot)

    X_train_q_tensor = torch.tensor(X_train_q, dtype=torch.float32)
    X_train_q_dot_tensor = torch.tensor(X_train_q_dot, dtype=torch.float32)
    X_train_q_ddot_tensor = torch.tensor(X_train_q_ddot, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_q_tensor = torch.tensor(X_test_q, dtype=torch.float32)
    X_test_q_dot_tensor = torch.tensor(X_test_q_dot, dtype=torch.float32)
    X_test_q_ddot_tensor = torch.tensor(X_test_q_ddot, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_q_tensor, X_train_q_dot_tensor, X_train_q_ddot_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    
    test_dataset = TensorDataset(X_test_q_tensor, X_test_q_dot_tensor, X_test_q_ddot_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

    model = E2NN(hidden_size=hyperparameters['hidden_size'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'])
    early_stopping = EarlyStopping(patience=hyperparameters['patience'], verbose=True)

    train_losses, val_losses = [], []
    for epoch in range(hyperparameters['epochs']):
        model.train()
        epoch_train_loss = 0.0
        for q_batch, q_dot_batch, q_ddot_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(q_batch, q_dot_batch, q_ddot_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * q_batch.size(0)
        
        train_losses.append(epoch_train_loss / len(train_loader.dataset))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for q_val_batch, q_dot_val_batch, q_ddot_val_batch, y_val_batch in test_loader:
                val_outputs = model(q_val_batch, q_dot_val_batch, q_ddot_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
                epoch_val_loss += val_loss.item() * q_val_batch.size(0)
        
        epoch_val_loss /= len(test_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{hyperparameters['epochs']}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {epoch_val_loss:.4f}")

        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    model_name = f"model_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}.pth"
    torch.save(model.state_dict(), os.path.join(output_dir, model_name))
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plot_name = f"loss_plot_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}.png"
    plt.savefig(os.path.join(output_dir, plot_name))
    plt.close()
    
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_q_tensor, X_test_q_dot_tensor, X_test_q_ddot_tensor).numpy()
        train_predictions = model(X_train_q_tensor, X_train_q_dot_tensor, X_train_q_ddot_tensor).numpy()

    y_test_numpy = y_test
    mse = np.mean((y_test_numpy - test_predictions)**2)
    mae = np.mean(np.abs(y_test_numpy - test_predictions))
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test_numpy, test_predictions)

    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
        f.write("Hyperparameters:\n")
        for key, value in hyperparameters.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"  data_path: {os.path.basename(data_path)}\n")
        f.write(f"  stroke_option: {stroke_option}\n")
        f.write(f"  test_split_ratio: {test_split_ratio}\n")
        f.write("\n")
        f.write("Performance Metrics:\n")
        f.write(f"  Mean Squared Error (MSE): {mse}\n")
        f.write(f"  Mean Absolute Error (MAE): {mae}\n")
        f.write(f"  R-squared (R2): {r2}\n")

    all_predictions = np.vstack([train_predictions, test_predictions])
    all_indices = np.hstack([train_indices, test_indices])
    
    # Create a DataFrame for predictions with original indices
    pred_df = pd.DataFrame(all_predictions, index=all_indices, columns=[f'pred_{t}' for t in targets]).sort_index()

    # Combine with original features
    full_df = df.join(pred_df)
    
    predictions_filename = f"predictions_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}.csv"
    full_df.to_csv(os.path.join(output_dir, predictions_filename), index_label='original_index')
    
    messagebox.showinfo("Training Complete", f"Model and results saved to:\n{output_dir}")

def hyperparameter_gui():
    root = tk.Tk()
    root.title("E2NN Hyperparameters")
    
    params = {}
    
    tk.Label(root, text="Learning Rate (e.g., 0.001):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    params['lr'] = tk.Entry(root); params['lr'].grid(row=0, column=1, padx=5, pady=2); params['lr'].insert(0, "0.001")
    
    tk.Label(root, text="Batch Size (e.g., 32):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    params['batch_size'] = tk.Entry(root); params['batch_size'].grid(row=1, column=1, padx=5, pady=2); params['batch_size'].insert(0, "32")

    tk.Label(root, text="Epochs (e.g., 500):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    params['epochs'] = tk.Entry(root); params['epochs'].grid(row=2, column=1, padx=5, pady=2); params['epochs'].insert(0, "500")
    
    tk.Label(root, text="Hidden Size (e.g., 64):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
    params['hidden_size'] = tk.Entry(root); params['hidden_size'].grid(row=3, column=1, padx=5, pady=2); params['hidden_size'].insert(0, "64")
    
    tk.Label(root, text="Early Stopping Patience (e.g., 20):").grid(row=4, column=0, sticky="w", padx=5, pady=2)
    params['patience'] = tk.Entry(root); params['patience'].grid(row=4, column=1, padx=5, pady=2); params['patience'].insert(0, "20")

    tk.Label(root, text="Data Stroke:").grid(row=5, column=0, sticky="w", padx=5, pady=2)
    stroke_options = ["Complete Stroke", "Forward Stroke Only"]
    params['stroke_option'] = tk.StringVar(root); params['stroke_option'].set(stroke_options[0])
    tk.OptionMenu(root, params['stroke_option'], *stroke_options).grid(row=5, column=1, sticky="ew", padx=5, pady=2)

    tk.Label(root, text="Test Split Ratio (e.g., 0.2):").grid(row=6, column=0, sticky="w", padx=5, pady=2)
    params['test_split_ratio'] = tk.Entry(root); params['test_split_ratio'].grid(row=6, column=1, padx=5, pady=2); params['test_split_ratio'].insert(0, "0.2")

    data_path_var = tk.StringVar()
    tk.Label(root, text="Data File:").grid(row=7, column=0, sticky="w", padx=5, pady=2)
    tk.Entry(root, textvariable=data_path_var, width=40).grid(row=7, column=1, padx=5, pady=2)
    
    def select_file():
        path = filedialog.askopenfilename(title="Select Trajectory CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if path:
            data_path_var.set(path)
            
    tk.Button(root, text="Browse...", command=select_file).grid(row=7, column=2, padx=5, pady=2)

    def start_training():
        data_path = data_path_var.get()
        if not data_path:
            messagebox.showerror("Error", "Please select a data file.")
            return

        try:
            hyperparameters = {
                'lr': float(params['lr'].get()),
                'batch_size': int(params['batch_size'].get()),
                'epochs': int(params['epochs'].get()),
                'hidden_size': int(params['hidden_size'].get()),
                'patience': int(params['patience'].get())
            }
            stroke_option = params['stroke_option'].get()
            test_split_ratio = float(params['test_split_ratio'].get())

            root.withdraw()
            train_model(hyperparameters, data_path, stroke_option, test_split_ratio)
            root.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your hyperparameter values.\n{e}")
            root.deiconify()
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred during training:\n{e}")
            root.deiconify()
            
    tk.Button(root, text="Start Training", command=start_training).grid(row=8, columnspan=3, pady=10)
    root.mainloop()

if __name__ == '__main__':
    hyperparameter_gui() 