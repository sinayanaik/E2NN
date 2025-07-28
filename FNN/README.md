# FNN for Robotic Arm Torque Prediction

This directory contains a complete workflow for training and evaluating a Feedforward Neural Network (FNN) to predict the joint torques of a robotic arm based on its kinematic state (joint angles, velocities, and accelerations).

## Overview

The workflow is divided into two main Python scripts:

1.  **`fnn_train_plus_visualize.py`**: A script with a graphical user interface (GUI) to configure, train, and save an FNN model. It automates the process of creating unique output directories for each training run and saves the model, performance metrics, and visualizations.
2.  **`visualization_and_compare.py`**: A powerful tool for visually comparing the performance of multiple saved models against the ground truth data. It features an intuitive GUI and generates clear, interactive plots.

---

## The Workflow

1.  **Train a Model**: Run `fnn_train_plus_visualize.py`. Use the GUI to set your desired hyperparameters, select the input data file (e.g., from the `../data` directory), and start the training.
2.  **Review Results**: Once training is complete, a new folder will be created within the `FNN/` directory. This folder contains the trained model, performance plots, metrics, and detailed predictions.
3.  **Compare Models**: Run `visualization_and_compare.py`. Use the GUI to load the `predictions.csv` files from one or more of the run folders you created. The script will generate a detailed plot comparing the performance of each model against the original ground truth data.

---

## `fnn_train_plus_visualize.py`: The Training Script

This script is the starting point for creating your torque prediction model.

### GUI and Hyperparameters

When you run the script, a GUI will appear, allowing you to configure the training process with the following options:
-   **Learning Rate**: The step size for the Adam optimizer (e.g., `0.001`).
-   **Batch Size**: The number of samples processed in each training iteration.
-   **Epochs**: The maximum number of times the training algorithm will pass through the entire dataset.
-   **Hidden Layers**: The architecture of the neural network, defined as a comma-separated list of neuron counts (e.g., `64,32` creates two hidden layers with 64 and 32 neurons).
-   **Activation Function**: The non-linear function used in the hidden layers (ReLU, Tanh, or Sigmoid).
-   **Early Stopping Patience**: An advanced feature to prevent overfitting. Training will stop if the validation loss does not improve for this number of consecutive epochs.
-   **Data Stroke**: Allows you to train on the full dataset (`Complete Stroke`) or only the first 500 data points (`Forward Stroke Only`).
-   **Test Split Ratio**: The proportion of the dataset to be reserved for testing (e.g., `0.2` for a 20% test set).
-   **Data File**: A dialog to select the input trajectory CSV file.

### Advanced Functionalities

-   **Early Stopping**: The script monitors the validation loss and automatically stops the training if the model's performance on the test set ceases to improve, saving time and preventing overfitting.
-   **Data Scaling**: Input features are automatically scaled using `StandardScaler` from scikit-learn, which is a best practice for training neural networks.

### Output Folder Structure

Each training run creates a uniquely named directory to keep your experiments organized. The naming convention is:
`run_{timestamp}_lr{learning_rate}_bs{batch_size}_layers{layer_structure}_{stroke_option}`

Inside each run folder, you will find:
-   **`model_... .pth`**: The saved state of the trained PyTorch model.
-   **`loss_plot_... .png`**: A plot showing the training and validation loss over epochs.
-   **`prediction_plot_... .png`**: A static plot comparing the model's predictions (for the entire dataset) against the ground truth, providing an immediate visual assessment of performance.
-   **`performance_metrics.txt`**: A text file containing the final performance metrics (MSE, MAE, R²) calculated on the test set.
-   **`predictions_... .csv`**: A detailed CSV file containing the model's predictions for both the training and test sets, aligned with the original data using an `original_index` column.

---

## `visualization_and_compare.py`: The Comparison Tool

This script is designed for in-depth analysis and comparison of your trained models.

### GUI and Plotting Details

The GUI is streamlined for ease of use:
-   **Display Metrics in Legend**: Checkboxes allow you to select which performance metrics (MSE, MAE, R²) are displayed directly in the plot legend for each model.
-   **Load Prediction CSVs to Compare**: This button opens a file dialog, allowing you to select one or more `predictions.csv` files from your various training runs.

### Advanced Functionalities

-   **Automatic Ground Truth**: The script automatically finds and loads the latest trajectory data file from the `../data/` directory to use as the ground truth. You don't need to select it manually.
-   **Intelligent Plotting**: The ground truth plot is automatically sliced to match the scope of the loaded predictions. If you load a model trained only on the "Forward Stroke," the plot will correctly display the ground truth for only that portion.
-   **Smart Legends**: The legends are automatically generated by parsing the folder name of each loaded CSV. This displays the key hyperparameters (learning rate, batch size, layers) for each model, making comparison effortless.
-   **Dynamic Plots**: If you have the `mplcursors` library installed (`pip install mplcursors`), the plots become interactive. Simply hover your mouse over any data point to see its exact coordinates.
-   **Clear Visual Style**: The ground truth is always plotted as a solid black line. Each loaded model is assigned a unique color from a high-contrast palette, with its predictions rendered as a clean line, making it easy to distinguish between different models and the ground truth. 