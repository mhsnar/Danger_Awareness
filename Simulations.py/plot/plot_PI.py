import numpy as np
import matplotlib.pyplot as plt




# Initialize arrays to store PI_all values
PI_all_i = np.zeros(10)  # For data_package_i
PI_all_P_j = np.zeros(10)  # For data_package_P_j
PI_Human_i= np.zeros(10)  # For data_package_P_j
PI_Robot_i= np.zeros(10)  # For data_package_P_j
PI_Human_j= np.zeros(10)  # For data_package_P_j
PI_Robot_j= np.zeros(10)  # For data_package_P_j

# Load PI_all from data_package_i.npz
for i in range(10):
    data_i = np.load(f'data_package_{i+1}.npz')
    PI_all_i[i] = data_i['PI_all']
    PI_Human_i[i] = data_i['PI_Human']
    PI_Robot_i[i] = data_i['PI_Robot']

# Load PI_all from data_package_P_j.npz
for j in range(10):
    data_P_j = np.load(f'data_package_P_{j+1}.npz')
    PI_all_P_j[j] = data_P_j['PI_all']
    PI_Human_j[j] = data_P_j['PI_Human']
    PI_Robot_j[j] = data_P_j['PI_Robot']




# Sample data (replace these with your actual data)
n = 10  # Number of time steps or instances

# Initialize arrays to hold the current Performance Indices
PI_2D = PI_all_i
PI_2D_MP = PI_all_P_j

# If you want to plot the data, you can do so with the previous code
prediction_horizon = np.arange(1, n + 1)  # Assuming prediction horizon from 1 to n
plt.figure(figsize=(6, 2.5))
plt.plot(prediction_horizon, PI_2D,marker='o', label='PI_all_i')
plt.plot(prediction_horizon, PI_2D_MP, marker='o', label='PI_all_j')
plt.plot(prediction_horizon, PI_Human_i,marker='o', label='PI_Human_i')
plt.plot(prediction_horizon, PI_Human_j, marker='o', label='PI_Human_j')
plt.plot(prediction_horizon, PI_Robot_i, marker='o',label='PI_Robot_i')
plt.plot(prediction_horizon, PI_Robot_j, marker='o', label='PI_Robot_j')

# Add title and labels
# plt.title('Performance Index vs Prediction Horizon')
plt.xlabel('Prediction Horizon')
plt.ylabel('Performance Index')
plt.xticks(prediction_horizon)
plt.grid(True)

# Position legend at the top right
plt.legend(loc='upper right')

# Set y-axis limits based on data range
# plt.ylim(400, max(max(PI_2D), max(PI_2D_MP)))

# Save the plot as an EPS file


# Show the plot
plt.show()
plt.savefig('PI.eps', format='eps')
