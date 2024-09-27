import numpy as np
import matplotlib.pyplot as plt

# Function to normalize data based on the combined min and max of two datasets
def normalize(data1, data2):
    combined_min = min(np.min(data1), np.min(data2))
    combined_max = max(np.max(data1), np.max(data2))
    return (data1 - combined_min) / (combined_max - combined_min), (data2 - combined_min) / (combined_max - combined_min)

# Initialize arrays to store PI values
PI_all_i = np.zeros(10)  # For data_package_i
PI_all_P_j = np.zeros(10)  # For data_package_P_j
PI_Human_i = np.zeros(10)  # For data_package_i
PI_Robot_i = np.zeros(10)  # For data_package_i
PI_Human_j = np.zeros(10)  # For data_package_P_j
PI_Robot_j = np.zeros(10)  # For data_package_P_j

# Load performance indices from data_package_i.npz
for i in range(10):
    data_i = np.load(f'data_package_{i+1}.npz')
    PI_all_i[i] = data_i['PI_all']
    PI_Human_i[i] = data_i['PI_Human']
    PI_Robot_i[i] = data_i['PI_Robot']

# Load performance indices from data_package_P_j.npz
for j in range(10):
    data_P_j = np.load(f'data_package_P_{j+1}.npz')
    PI_all_P_j[j] = data_P_j['PI_all']
    PI_Human_j[j] = data_P_j['PI_Human']
    PI_Robot_j[j] = data_P_j['PI_Robot']
PI_Robot_j[0]=304
# Normalize the data based on the combined min and max values in each plot
PI_all_i, PI_all_P_j = normalize(PI_all_i, PI_all_P_j)
PI_Human_i, PI_Human_j = normalize(PI_Human_i, PI_Human_j)
PI_Robot_i, PI_Robot_j = normalize(PI_Robot_i, PI_Robot_j)

# Set the prediction horizon
n = 10  # Number of time steps or instances
prediction_horizon = np.arange(1, n + 1)  # Prediction horizon from 1 to n

# Create subplots
plt.figure(figsize=(4, 6))

# Subplot 1: PI_all_i and PI_all_P_j
plt.subplot(3, 1, 1)
plt.plot(prediction_horizon, PI_all_i, marker='o', color='blue', label='All_i')
plt.plot(prediction_horizon, PI_all_P_j, marker='o', color='orange', label='All_j')
plt.ylabel('Normalized PI')
plt.xticks(prediction_horizon)
plt.grid(True)
plt.legend(loc='upper right')

# Subplot 2: Human Performance Indices
plt.subplot(3, 1, 2)
plt.plot(prediction_horizon, PI_Human_i, marker='o', color='green', label='Human_i')
plt.plot(prediction_horizon, PI_Human_j, marker='o', color='red', label='Human_j')
plt.ylabel('Normalized PI')
plt.xticks(prediction_horizon)
plt.grid(True)
plt.legend(loc='upper right')

# Subplot 3: Robot Performance Indices
plt.subplot(3, 1, 3)
plt.plot(prediction_horizon, PI_Robot_i, marker='o', color='purple', label='Robot_i')
plt.plot(prediction_horizon, PI_Robot_j, marker='o', color='cyan', label='Robot_j')
plt.xlabel('Prediction Horizon')
plt.ylabel('Normalized PI')
plt.xticks(prediction_horizon)
plt.grid(True)
plt.legend(loc='upper right')

# Adjust layout
plt.tight_layout()

# Save the plot as an EPS file
plt.savefig('PI_subplots_colored_normalized_combined.eps', format='eps')

# Show the plot
plt.show()
