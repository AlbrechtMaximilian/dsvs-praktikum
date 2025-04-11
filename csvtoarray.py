import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def csv_to_numpy(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Select and rename relevant columns
    df = df.rename(columns={
        "Time (s)": 'time',
        "Acceleration x (m/s^2)": 'ax',
        "Acceleration y (m/s^2)": 'ay',
        "Acceleration z (m/s^2)": 'az'
    })

    # Extract relevant columns
    df = df[['time', 'ax', 'ay', 'az']]

    # Convert to NumPy array (transpose to get 4 rows)
    return df.values.T

