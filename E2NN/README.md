# E2NN for Robotic Arm Torque Prediction

This directory contains a complete workflow for training and evaluating an Equation-Embedded Neural Network (E2NN) to predict the joint torques of a robotic arm. This physics-informed approach learns the underlying dynamics of the system.

## Overview

The model is based on the Euler-Lagrange equation of motion:
\[ \tau = M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) \]

The E2NN consists of three specialized sub-networks:
1.  **Inertia Network (`InertiaNN`)**: Learns the 3x3 mass matrix \(M(q)\).
2.  **Coriolis Network (`CoriolisVecNN`)**: Learns the 3x1 Coriolis and centrifugal effects vector \(C(q, \dot{q})\dot{q}\).
3.  **Gravity Network (`GravityNN`)**: Learns the 3x1 gravity vector \(G(q)\).

The workflow is divided into two main Python scripts:

1.  **`e2nn_train_plus_visualize.py`**: A script with a graphical user interface (GUI) to configure, train, and save an E2NN model.
2.  **`visualization_and_compare.py`**: A tool for visually comparing the performance of multiple saved models against the ground truth data.

---

## The Workflow

1.  **Train a Model**: Run `e2nn_train_plus_visualize.py`. Use the GUI to set your desired hyperparameters, select the input data file, and start the training.
2.  **Review Results**: After training, a new folder is created in the `E2NN/` directory with the model, plots, metrics, and predictions.
3.  **Compare Models**: Run `visualization_and_compare.py` to load one or more `predictions.csv` files and compare them against the ground truth.

---

## `e2nn_train_plus_visualize.py`: The Training Script

### GUI and Hyperparameters

-   **Learning Rate**: Step size for the Adam optimizer (e.g., `0.001`).
-   **Batch Size**: Number of samples per training iteration.
-   **Epochs**: Maximum number of training passes.
-   **Hidden Size**: The number of neurons in the hidden layers of the sub-networks.
-   **Early Stopping Patience**: Stops training if validation loss doesn't improve for this many epochs.
-   **Data Stroke**: Train on `Complete Stroke` or `Forward Stroke Only`.
-   **Test Split Ratio**: The proportion of data for the test set (e.g., `0.2`).
-   **Data File**: Dialog to select the input trajectory CSV file.

### Advanced Functionalities

-   **Early Stopping**: Prevents overfitting by stopping training when validation loss plateaus.
-   **Data Scaling**: Input features (`q`, `q_dot`, `q_ddot`) are scaled using `StandardScaler`.

### Output Folder Structure

Each run creates a directory: `run_{timestamp}_lr{...}_bs{...}_hs{...}_{stroke_option}`

Inside each folder:
-   **`model_... .pth`**: The saved PyTorch model.
-   **`loss_plot_... .png`**: Training and validation loss plot.
-   **`performance_metrics.txt`**: Final MSE, MAE, and R² on the test set.
-   **`predictions_... .csv`**: Predictions for both training and test sets, with original data.

---

## `visualization_and_compare.py`: The Comparison Tool

This script is for in-depth analysis of trained models.

### GUI and Plotting

-   **Display Metrics in Legend**: Checkboxes to show MSE, MAE, R² in the plot legend.
-   **Load Prediction CSVs**: Select one or more `predictions.csv` files to compare.

### Advanced Functionalities

-   **Automatic Ground Truth**: Finds and loads the latest trajectory data from `../data/`.
-   **Intelligent Plotting**: Slices the ground truth plot to match the prediction data's scope.
-   **Smart Legends**: Automatically generated legends from folder names show key hyperparameters.
-   **Dynamic Plots**: If `mplcursors` is installed (`pip install mplcursors`), plots are interactive.
-   **Clear Visual Style**: Ground truth is a black line; each model has a unique color. 