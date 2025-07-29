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

class FNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation_fn):
        super(FNN, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(activation_fn())
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(activation_fn())
            
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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
        if key == 'activation_fn':
            print(f"  {key}: {value.__name__}")
        else:
            print(f"  {key}: {value}")

    # Create a unique directory for this run
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}_layers{'x'.join(map(str, hyperparameters['hidden_layers']))}_{stroke_option.replace(' ', '')}"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    if not data_path:
        messagebox.showerror("Error", "No data file provided.")
        return
    
    df = pd.read_csv(data_path)

    if stroke_option == 'Forward Stroke Only':
        df = df.iloc[:500]

    # Prepare data
    features = [
        'joint1_angle', 'joint2_angle', 'joint3_angle',
        'joint1_velocity', 'joint2_velocity', 'joint3_velocity',
        'joint1_acceleration', 'joint2_acceleration', 'joint3_acceleration'
    ]
    targets = ['joint1_torque', 'joint2_torque', 'joint3_torque']

    X = df[features].values
    y = df[targets].values

    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, np.arange(len(X)), test_size=test_split_ratio, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

    # Model, loss, and optimizer
    model = FNN(
        input_size=len(features),
        output_size=len(targets),
        hidden_layers=hyperparameters['hidden_layers'],
        activation_fn=hyperparameters['activation_fn']
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'])
    early_stopping = EarlyStopping(patience=hyperparameters['patience'], verbose=True)

    # Training loop
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
            epoch_train_loss += loss.item() * X_batch.size(0)

        train_losses.append(epoch_train_loss / len(train_loader.dataset))

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_val_batch, y_val_batch in test_loader:
                val_outputs = model(X_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
                epoch_val_loss += val_loss.item() * X_val_batch.size(0)

        epoch_val_loss /= len(test_loader.dataset)
        val_losses.append(epoch_val_loss)


        print(f"Epoch [{epoch+1}/{hyperparameters['epochs']}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {epoch_val_loss:.4f}")

        early_stopping(epoch_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
            
    # Save model
    model_name = f"model_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}.pth"
    torch.save(model.state_dict(), os.path.join(output_dir, model_name))

    # Save plots
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

    # Performance metrics
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train_tensor).numpy()
        test_predictions = model(X_test_tensor).numpy()
        
    # Generate and save prediction plot
    gt_path = find_ground_truth_file()
    if gt_path:
        ground_truth_df = pd.read_csv(gt_path)
        gt_torques = ground_truth_df[['joint1_torque', 'joint2_torque', 'joint3_torque']]

        # Slice ground truth if forward stroke only
        if stroke_option == 'Forward Stroke Only':
            gt_torques = gt_torques.iloc[:500]

        fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
        
        # Plot ground truth
        axes[0].plot(gt_torques.index, gt_torques.iloc[:, 0], label='Ground Truth', color='black', linewidth=1.5)
        axes[1].plot(gt_torques.index, gt_torques.iloc[:, 1], label='_nolegend_', color='black', linewidth=1.5)
        axes[2].plot(gt_torques.index, gt_torques.iloc[:, 2], label='_nolegend_', color='black', linewidth=1.5)

        # Combine and sort all predictions for a single plot line
        all_indices = np.concatenate([train_indices, test_indices])
        all_predictions = np.concatenate([train_predictions, test_predictions])
        plot_df = pd.DataFrame(data=all_predictions, index=all_indices, columns=['p1', 'p2', 'p3']).sort_index()
        
        # Plot single prediction line
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
    
    y_test_numpy = y_test_tensor.numpy()
    mse = np.mean((y_test_numpy - test_predictions)**2)
    mae = np.mean(np.abs(y_test_numpy - test_predictions))
    r2 = r2_score(y_test_numpy, test_predictions)

    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
        f.write("Hyperparameters:\n")
        for key, value in hyperparameters.items():
            if hasattr(value, '__name__'):
                f.write(f"  {key}: {value.__name__}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write(f"  data_path: {os.path.basename(data_path)}\n")
        f.write(f"  stroke_option: {stroke_option}\n")
        f.write(f"  test_split_ratio: {test_split_ratio}\n")
        f.write("\n")
        f.write("Performance Metrics:\n")
        f.write(f"  Mean Squared Error (MSE): {mse}\n")
        f.write(f"  Mean Absolute Error (MAE): {mae}\n")
        f.write(f"  R-squared (R2): {r2}\n")

    # Save predictions for both train and test sets
    y_train_numpy = y_train_tensor.numpy()
    
    X_train_unscaled = scaler.inverse_transform(X_train_tensor.numpy())
    X_test_unscaled = scaler.inverse_transform(X_test_tensor.numpy())

    train_df = pd.DataFrame(data=np.hstack([X_train_unscaled, y_train_numpy, train_predictions]),
                            columns=features + [f'actual_{t}' for t in targets] + [f'pred_{t}' for t in targets],
                            index=train_indices)
    train_df['set'] = 'train'

    test_df = pd.DataFrame(data=np.hstack([X_test_unscaled, y_test_numpy, test_predictions]),
                           columns=features + [f'actual_{t}' for t in targets] + [f'pred_{t}' for t in targets],
                           index=test_indices)
    test_df['set'] = 'test'

    
    predictions_df = pd.concat([train_df, test_df]).sort_index()
    predictions_filename = f"predictions_lr{hyperparameters['lr']}_bs{hyperparameters['batch_size']}.csv"
    predictions_df.to_csv(os.path.join(output_dir, predictions_filename), index=True, index_label='original_index')


    messagebox.showinfo("Training Complete", f"Model and results saved to:\n{output_dir}")

def hyperparameter_gui():
    root = tk.Tk()
    root.title("FNN Hyperparameters")

    params = {}
    
    tk.Label(root, text="Learning Rate (e.g., 0.001):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    params['lr'] = tk.Entry(root); params['lr'].grid(row=0, column=1, padx=5, pady=2); params['lr'].insert(0, "0.001")
    
    tk.Label(root, text="Batch Size (e.g., 32):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    params['batch_size'] = tk.Entry(root); params['batch_size'].grid(row=1, column=1, padx=5, pady=2); params['batch_size'].insert(0, "32")

    tk.Label(root, text="Epochs (e.g., 100):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    params['epochs'] = tk.Entry(root); params['epochs'].grid(row=2, column=1, padx=5, pady=2); params['epochs'].insert(0, "100")
    
    tk.Label(root, text="Hidden Layers (e.g., 64,32):").grid(row=3, column=0, sticky="w", padx=5, pady=2)
    params['hidden_layers'] = tk.Entry(root); params['hidden_layers'].grid(row=3, column=1, padx=5, pady=2); params['hidden_layers'].insert(0, "64,32")
    
    tk.Label(root, text="Activation Function:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
    activations = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}
    params['activation_fn'] = tk.StringVar(root); params['activation_fn'].set("ReLU")
    tk.OptionMenu(root, params['activation_fn'], *activations.keys()).grid(row=4, column=1, sticky="ew", padx=5, pady=2)

    tk.Label(root, text="Early Stopping Patience (e.g., 10):").grid(row=5, column=0, sticky="w", padx=5, pady=2)
    params['patience'] = tk.Entry(root); params['patience'].grid(row=5, column=1, padx=5, pady=2); params['patience'].insert(0, "10")

    # Add stroke selection
    tk.Label(root, text="Data Stroke:").grid(row=6, column=0, sticky="w", padx=5, pady=2)
    stroke_options = ["Complete Stroke", "Forward Stroke Only"]
    params['stroke_option'] = tk.StringVar(root); params['stroke_option'].set(stroke_options[0])
    tk.OptionMenu(root, params['stroke_option'], *stroke_options).grid(row=6, column=1, sticky="ew", padx=5, pady=2)

    # Add train/test split ratio
    tk.Label(root, text="Test Split Ratio (e.g., 0.2):").grid(row=7, column=0, sticky="w", padx=5, pady=2)
    params['test_split_ratio'] = tk.Entry(root); params['test_split_ratio'].grid(row=7, column=1, padx=5, pady=2); params['test_split_ratio'].insert(0, "0.2")

    # Add file selection
    data_path_var = tk.StringVar()
    tk.Label(root, text="Data File:").grid(row=8, column=0, sticky="w", padx=5, pady=2)
    tk.Entry(root, textvariable=data_path_var, width=40).grid(row=8, column=1, padx=5, pady=2)
    def select_file():
        path = filedialog.askopenfilename(title="Select Trajectory CSV File", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if path:
            data_path_var.set(path)
    tk.Button(root, text="Browse...", command=select_file).grid(row=8, column=2, padx=5, pady=2)


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
                'hidden_layers': [int(x.strip()) for x in params['hidden_layers'].get().split(',')],
                'activation_fn': activations[params['activation_fn'].get()],
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

    tk.Button(root, text="Start Training", command=start_training).grid(row=9, columnspan=3, pady=10)
    root.mainloop()

if __name__ == '__main__':
    hyperparameter_gui() 