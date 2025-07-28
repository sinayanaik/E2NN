import torch
import numpy as np
from RNN.train_torque_model import TorquePredictor

def load_model(model_path='torque_model.pth'):
    """Load the trained model and scalers."""
    checkpoint = torch.load(model_path)
    
    model = TorquePredictor()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    X_scaler = checkpoint['X_scaler']
    y_scaler = checkpoint['y_scaler']
    sequence_length = checkpoint['sequence_length']
    
    return model, X_scaler, y_scaler, sequence_length

def predict_torques(joint_angles, model, X_scaler, y_scaler):
    """Predict torques for given joint angles."""
    # Reshape input if single sample
    if len(joint_angles.shape) == 1:
        joint_angles = joint_angles.reshape(1, -1)
    
    # Scale input
    X_scaled = X_scaler.transform(joint_angles)
    
    # Convert to tensor and predict
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        y_pred_scaled = model(X_tensor)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    return y_pred

def main():
    # Load the model
    model, X_scaler, y_scaler, sequence_length = load_model()
    
    # Example usage
    test_angles = np.array([0.5, 0.3, 0.2])  # Example joint angles
    predicted_torques = predict_torques(test_angles, model, X_scaler, y_scaler)
    
    print("Test Joint Angles:", test_angles)
    print("Predicted Torques:")
    print("  Joint 1:", predicted_torques[0][0], "Nm")
    print("  Joint 2:", predicted_torques[0][1], "Nm")
    print("  Joint 3:", predicted_torques[0][2], "Nm")

if __name__ == "__main__":
    main() 