import numpy as np
import matplotlib.pyplot as plt

# Initialize arrays to store PI values
PI_all_i = np.zeros(7)  # For data_package_i
PI_all_P_j = np.zeros(7)  # For data_package_P_j
PI_Human_i = np.zeros(7)  # For data_package_i
PI_Robot_i = np.zeros(7)  # For data_package_i
PI_Human_j = np.zeros(7)  # For data_package_P_j
PI_Robot_j = np.zeros(7)  # For data_package_P_j

# Load performance indices from data_package_i.npz
for i in range(7):
    data_i = np.load(f'data_package_{i+1}.npz')
    PI_all_i[i] = data_i['PI_all']
    PI_Human_i[i] = data_i['PI_Human']
    PI_Robot_i[i] = data_i['PI_Robot']

# Load performance indices from data_package_P_j.npz
for j in range(7):
    data_P_j = np.load(f'data_package_P_{j+1}.npz')
    PI_all_P_j[j] = data_P_j['PI_all']
    PI_Human_j[j] = data_P_j['PI_Human']
    PI_Robot_j[j] = data_P_j['PI_Robot']

# Set the prediction horizon
n = 7  # Number of time steps or instances
prediction_horizon = np.arange(1, n + 1)  # Prediction horizon from 1 to n

# Create subplots
plt.figure(figsize=(4, 5))

# Subplot 1: PI_all_i and PI_all_P_j
plt.subplot(3, 1, 1)
plt.plot(prediction_horizon, PI_all_i, marker='o', color='blue', label='Not Predictive')
plt.plot(prediction_horizon, PI_all_P_j, marker='o', color='purple', label='Predictive')
# plt.title('PI: Data Package I vs. Data Package P')
plt.ylabel('$PI_{Sum}$')
plt.xticks(prediction_horizon)
plt.grid(True)
plt.legend(loc='upper center', bbox_to_anchor=(.5, 1.35),ncol=2)

# Subplot 2: Human Performance Indices
plt.subplot(3, 1, 2)
plt.plot(prediction_horizon, PI_Human_i, marker='o', color='blue')
plt.plot(prediction_horizon, PI_Human_j, marker='o', color='purple')
# plt.title('Human PI')
plt.ylabel('$PI_{Human}$')
plt.xticks(prediction_horizon)
plt.grid(True)


# Subplot 3: Robot Performance Indices
plt.subplot(3, 1, 3)
plt.plot(prediction_horizon, PI_Robot_i, marker='o', color='blue' )
plt.plot(prediction_horizon, PI_Robot_j, marker='o', color='purple')
# plt.title('Robot PI')
plt.xlabel('$Prediction Horizon$')
plt.ylabel('$PI_{Robot}$')
plt.xticks(prediction_horizon)
plt.grid(True)


# Adjust layout
plt.tight_layout()

# Save the plot as an EPS file
plt.savefig('PI.eps', format='eps')

# Show the plot
plt.show()
