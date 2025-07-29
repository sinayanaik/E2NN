import pandas as pd
import altair as alt
import os
import tkinter as tk
from tkinter import filedialog

def create_torque_plot(file_path):
    """
    Loads trajectory data from a CSV file and creates an interactive plot of joint torques over time.
    The plot is saved as an HTML file in the same directory as the input file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        df = pd.read_csv(file_path)

        # Identify timestamp column
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'time' in df.columns:
            time_col = 'time'
        else:
            print("Error: No 'timestamp' or 'time' column found in the CSV.")
            return

        # Identify torque columns
        torque_cols = sorted([col for col in df.columns if 'torque' in col and 'commanded' not in col])
        if not torque_cols:
            print("Error: No torque columns found. Looking for columns with 'torque' in the name.")
            return
        
        print(f"Found torque columns: {torque_cols}")

        # Melt the torque columns for plotting
        df_melted = df.melt(
            id_vars=[time_col],
            value_vars=torque_cols,
            var_name='Joint',
            value_name='Torque'
        )

        # Create the interactive line chart
        chart = alt.Chart(df_melted).mark_line().encode(
            x=alt.X(f'{time_col}:Q', title='Time (s)'),
            y=alt.Y('Torque:Q', title='Torque (Nm)'),
            color=alt.Color('Joint:N', title='Joint'),
            tooltip=[
                alt.Tooltip(f'{time_col}:Q', title='Time (s)', format='.3f'),
                alt.Tooltip('Joint:N', title='Joint'),
                alt.Tooltip('Torque:Q', title='Torque (Nm)', format='.4f')
            ]
        ).properties(
            title=f'Joint Torques Over Time for {os.path.basename(file_path)}'
        ).interactive()

        # Save the chart as an HTML file in the same directory as the input file.
        output_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = os.path.join(output_dir, f"{base_name}_torques.html")
        
        chart.save(output_filename)

        print(f"Chart saved to {output_filename}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open a file dialog to select the CSV file
    csv_file = filedialog.askopenfilename(
        title="Select a CSV file with trajectory data",
        initialdir=os.path.join(os.getcwd(), 'data'), # Start in the 'data' directory
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    
    if csv_file:
        create_torque_plot(csv_file)
    else:
        print("No file selected. Exiting.")