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
from sklearn.metrics import r2_score

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

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

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

def create_sequences(input_data, output_data, seq_length):
    xs, ys = [], []
    for i in range(len(input_data) - seq_length):
        x = input_data[i:(i + seq_length)]
        y = output_data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model(hyperparameters, data_path, stroke_option, test_split_ratio):
    print("Starting training with the following hyperparameters:")
    for key, value in hyperparameters.items():
        print(f"  {key}: {value}")

    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}_seq{hyperparameters['sequence_length']}_{stroke_option.replace(' ', '')}"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), run_name)
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    if stroke_option == 'Forward Stroke Only':
        df = df.iloc[:500]

    features = [
        'joint1_angle', 'joint2_angle', 'joint3_angle',
        'joint1_velocity', 'joint2_velocity', 'joint3_velocity',
        'joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration'
    ]
    targets = ['joint1_torque', 'joint2_torque', 'joint3_torque']

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    X, y = create_sequences(df[features].values, df[targets].values, hyperparameters['sequence_length'])
    
    original_indices = np.arange(hyperparameters['sequence_length'], len(df))
    
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, original_indices, test_size=test_split_ratio, random_state=42
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

    model = RNN(
        input_size=len(features),
        hidden_size=hyperparameters['hidden_size'],
        num_layers=hyperparameters['num_layers'],
        output_size=len(targets)
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'])
    early_stopping = EarlyStopping(patience=hyperparameters['patience'], verbose=True)

    train_losses, val_losses = [], []
    for epoch in range(hyperparameters['epochs']):
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        train_losses.append(epoch_train_loss / len(train_loader))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in test_loader:
                val_outputs = model(X_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
                epoch_val_loss += val_loss.item()
        
        val_losses.append(epoch_val_loss / len(test_loader))
        print(f"Epoch [{epoch+1}/{hyperparameters['epochs']}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        
        early_stopping(val_losses[-1])
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model_name = f"model_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}.pth"
    torch.save(model.state_dict(), os.path.join(output_dir, model_name))

    plot_name = f"loss_plot_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}.png"
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, plot_name))
    plt.close()

    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train_tensor).numpy()
        test_predictions = model(X_test_tensor).numpy()

    gt_path = find_ground_truth_file()
    if gt_path:
        ground_truth_df = pd.read_csv(gt_path)
        gt_torques = ground_truth_df[['joint1_torque', 'joint2_torque', 'joint3_torque']]
        
        if stroke_option == 'Forward Stroke Only':
            gt_torques = gt_torques.iloc[:500]

        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
        
        axes[0].plot(gt_torques.index, gt_torques.iloc[:, 0], label='Ground Truth', color='black', linewidth=1.5)
        axes[1].plot(gt_torques.index, gt_torques.iloc[:, 1], label='_nolegend_', color='black', linewidth=1.5)
        axes[2].plot(gt_torques.index, gt_torques.iloc[:, 2], label='_nolegend_', color='black', linewidth=1.5)

        all_indices = np.concatenate([train_indices, test_indices])
        all_predictions = np.concatenate([train_predictions, test_predictions])
        plot_df = pd.DataFrame(data=all_predictions, index=all_indices, columns=['p1', 'p2', 'p3']).sort_index()
        
        axes[0].plot(plot_df.index, plot_df['p1'], label='Prediction', color='red')
        axes[1].plot(plot_df.index, plot_df['p2'], label='_nolegend_', color='red')
        axes[2].plot(plot_df.index, plot_df['p3'], label='_nolegend_', color='red')

        handles, labels = axes[0].get_legend_handles_labels()
        for i, title in enumerate(['Joint 1 Torque', 'Joint 2 Torque', 'Joint 3 Torque']):
            axes[i].set_title(title)
            axes[i].legend(handles=handles, labels=labels)
            axes[i].grid(True)
        axes[2].set_xlabel("Time Step")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle('Model Predictions vs. Ground Truth')
        pred_plot_name = f"prediction_plot_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}.png"
        plt.savefig(os.path.join(output_dir, pred_plot_name))
        plt.close(fig)
    
    mse = np.mean((y_test - test_predictions)**2)
    mae = np.mean(np.abs(y_test - test_predictions))
    r2 = r2_score(y_test, test_predictions)

    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
        f.write(f"Mean Squared Error (MSE): {mse}\n")
        f.write(f"Mean Absolute Error (MAE): {mae}\n")
        f.write(f"R-squared (R2): {r2}\n")

    train_df = pd.DataFrame(data=np.hstack([X_train[:, -1, :], y_train, train_predictions]),
                            columns=features + [f'actual_{t}' for t in targets] + [f'pred_{t}' for t in targets],
                            index=train_indices)
    train_df['set'] = 'train'

    test_df = pd.DataFrame(data=np.hstack([X_test[:, -1, :], y_test, test_predictions]),
                           columns=features + [f'actual_{t}' for t in targets] + [f'pred_{t}' for t in targets],
                           index=test_indices)
    test_df['set'] = 'test'
    
    predictions_df = pd.concat([train_df, test_df]).sort_index()
    predictions_filename = f"predictions_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}.csv"
    predictions_df.to_csv(os.path.join(output_dir, predictions_filename), index=True, index_label='original_index')

    messagebox.showinfo("Training Complete", f"Model and results saved to:\n{output_dir}")

def hyperparameter_gui():
    root = tk.Tk()
    root.title("RNN Hyperparameters")

    params = {}
    
    tk.Label(root, text="Learning Rate (e.g., 0.001):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    params['lr'] = tk.Entry(root); params['lr'].grid(row=0, column=1, padx=5, pady=2); params['lr'].insert(0, "0.001")
    
    tk.Label(root, text="Batch Size (e.g., 32):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    params['batch_size'] = tk.Entry(root); params['batch_size'].grid(row=1, column=1, padx=5, pady=2); params['batch_size'].insert(0, "32")

    tk.Label(root, text="Epochs (e.g., 100):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    params['epochs'] = tk.Entry(root); params['epochs'].grid(row=2, column=1, padx=5, pady=2); params['epochs'].insert(0, "100")
    
    tk.Label(root, text="Hidden Size (e.g., 50):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
    params['hidden_size'] = tk.Entry(root); params['hidden_size'].grid(row=3, column=1, padx=5, pady=2); params['hidden_size'].insert(0, "50")
    
    tk.Label(root, text="Num Layers (e.g., 2):").grid(row=4, column=0, sticky="w", padx=5, pady=2)
    params['num_layers'] = tk.Entry(root); params['num_layers'].grid(row=4, column=1, padx=5, pady=2); params['num_layers'].insert(0, "2")
    
    tk.Label(root, text="Sequence Length (e.g., 10):").grid(row=5, column=0, sticky="w", padx=5, pady=2)
    params['sequence_length'] = tk.Entry(root); params['sequence_length'].grid(row=5, column=1, padx=5, pady=2); params['sequence_length'].insert(0, "10")

    tk.Label(root, text="Early Stopping Patience (e.g., 10):").grid(row=6, column=0, sticky="w", padx=5, pady=2)
    params['patience'] = tk.Entry(root); params['patience'].grid(row=6, column=1, padx=5, pady=2); params['patience'].insert(0, "10")

    tk.Label(root, text="Data Stroke:").grid(row=7, column=0, sticky="w", padx=5, pady=2)
    stroke_options = ["Complete Stroke", "Forward Stroke Only"]
    params['stroke_option'] = tk.StringVar(root); params['stroke_option'].set(stroke_options[0])
    tk.OptionMenu(root, params['stroke_option'], *stroke_options).grid(row=7, column=1, sticky="ew", padx=5, pady=2)

    tk.Label(root, text="Test Split Ratio (e.g., 0.2):").grid(row=8, column=0, sticky="w", padx=5, pady=2)
    params['test_split_ratio'] = tk.Entry(root); params['test_split_ratio'].grid(row=8, column=1, padx=5, pady=2); params['test_split_ratio'].insert(0, "0.2")

    data_path_var = tk.StringVar()
    tk.Label(root, text="Data File:").grid(row=9, column=0, sticky="w", padx=5, pady=2)
    tk.Entry(root, textvariable=data_path_var, width=40).grid(row=9, column=1, padx=5, pady=2)
    def select_file():
        path = filedialog.askopenfilename(title="Select Trajectory CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if path:
            data_path_var.set(path)
    tk.Button(root, text="Browse...", command=select_file).grid(row=9, column=2, padx=5, pady=2)

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
                'num_layers': int(params['num_layers'].get()),
                'sequence_length': int(params['sequence_length'].get()),
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

    tk.Button(root, text="Start Training", command=start_training).grid(row=10, columnspan=3, pady=10)
    root.mainloop()

if __name__ == '__main__':
    hyperparameter_gui() 