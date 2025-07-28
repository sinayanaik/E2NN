import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Define the neural network architecture
class TorquePredictor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3):
        super(TorquePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x, hidden=None):
        # x shape: (batch, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        # Forward pass through LSTM
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Use only the last output for prediction
        last_output = lstm_out[:, -1, :]
        
        # Forward pass through fully connected layers
        output = self.fc(last_output)
        return output, hidden

# Custom dataset class
class TorqueDataset(Dataset):
    def __init__(self, X, y, sequence_length=10):
        self.sequence_length = sequence_length
        
        # Create sequences
        self.X_sequences = []
        self.y_sequences = []
        
        for i in range(len(X) - sequence_length):
            self.X_sequences.append(X[i:i + sequence_length])
            self.y_sequences.append(y[i + sequence_length])
        
        self.X_sequences = torch.FloatTensor(np.array(self.X_sequences))
        self.y_sequences = torch.FloatTensor(np.array(self.y_sequences))
        
    def __len__(self):
        return len(self.X_sequences)
    
    def __getitem__(self, idx):
        return self.X_sequences[idx], self.y_sequences[idx]

def train_model(model, train_loader, criterion, optimizer, num_epochs=1000):
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs, _ = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses

def main():
    # Load and preprocess data
    data_path = '../data/semi_circle_trajectory_kp0.10_kd0.4_r0.10_x0.25_z0.10_semicircle5_20250722_125106.csv'
    data = pd.read_csv(data_path)
    
    # Take only the forward semi-circle (first half of data)
    n_points = len(data) // 2
    data = data.iloc[:n_points]
    
    # Prepare input features (joint angles) and targets (torques)
    X = data[['joint1_angle', 'joint2_angle', 'joint3_angle']].values
    y = data[['joint1_torque', 'joint2_torque', 'joint3_torque']].values
    
    # Scale the data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    # Create dataset and dataloader
    sequence_length = 10
    dataset = TorqueDataset(X_scaled, y_scaled, sequence_length)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    model = TorquePredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    losses = train_model(model, train_loader, criterion, optimizer)
    
    # Save the model and scalers
    torch.save({
        'model_state_dict': model.state_dict(),
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'sequence_length': sequence_length
    }, 'torque_model.pth')
    
    # Save losses for comparison
    np.save('training_loss.npy', np.array(losses))
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss over Epochs (RNN)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    
    # Make predictions and visualize results
    model.eval()
    with torch.no_grad():
        # Create sequences for the entire dataset
        X_sequences = []
        for i in range(len(X_scaled) - sequence_length):
            X_sequences.append(X_scaled[i:i + sequence_length])
        X_sequences = torch.FloatTensor(np.array(X_sequences))
        
        # Generate predictions
        y_pred_scaled, _ = model(X_sequences)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.numpy())
        
        # Actual values (excluding first sequence_length points)
        y_actual = y[sequence_length:]
    
    # Calculate MSE for each joint
    mse_joint1 = np.mean((y_actual[:, 0] - y_pred[:, 0])**2)
    mse_joint2 = np.mean((y_actual[:, 1] - y_pred[:, 1])**2)
    mse_joint3 = np.mean((y_actual[:, 2] - y_pred[:, 2])**2)
    
    # Plot predictions vs actual
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # Joint 1 Torque
    ax1.plot(y_actual[:, 0], label='Actual', alpha=0.7)
    ax1.plot(y_pred[:, 0], label='Predicted', alpha=0.7)
    ax1.set_title(f'Joint 1 Torque (MSE: {mse_joint1:.6f})')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Torque (Nm)')
    ax1.legend()
    ax1.grid(True)
    
    # Joint 2 Torque
    ax2.plot(y_actual[:, 1], label='Actual', alpha=0.7)
    ax2.plot(y_pred[:, 1], label='Predicted', alpha=0.7)
    ax2.set_title(f'Joint 2 Torque (MSE: {mse_joint2:.6f})')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Torque (Nm)')
    ax2.legend()
    ax2.grid(True)
    
    # Joint 3 Torque
    ax3.plot(y_actual[:, 2], label='Actual', alpha=0.7)
    ax3.plot(y_pred[:, 2], label='Predicted', alpha=0.7)
    ax3.set_title(f'Joint 3 Torque (MSE: {mse_joint3:.6f})')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Torque (Nm)')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()

if __name__ == "__main__":
    main() 