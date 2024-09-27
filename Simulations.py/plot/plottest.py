import numpy as np
import matplotlib.pyplot as plt

# Example data (replace with your actual data)
prediction_horizon = np.arange(1, 11)
PI_all_i = np.array([450, 460, 470, 480, 490, 495, 500, 510, 520, 530])
PI_all_j = np.array([440, 450, 455, 460, 470, 480, 490, 495, 500, 510])
PI_Human_i = np.array([150, 160, 155, 160, 165, 160, 170, 175, 180, 185])
PI_Human_j = np.array([145, 150, 155, 152, 155, 160, 162, 165, 168, 170])
PI_Robot_i = np.array([300, 310, 320, 330, 340, 345, 350, 360, 370, 380])
PI_Robot_j = np.array([290, 300, 310, 320, 330, 335, 340, 350, 360, 370])

# Adding stochastic variance (standard deviation) to simulate stochastic data
np.random.seed(42)
stdev = np.array([10, 15, 12, 18, 13, 14, 10, 17, 16, 12])  # Simulated variability

def add_noise(data, stdev):
    return data + np.random.randn(*data.shape) * stdev

# Create stochastic versions of the data
PI_all_i_stochastic = add_noise(PI_all_i, stdev)
PI_all_j_stochastic = add_noise(PI_all_j, stdev)
PI_Human_i_stochastic = add_noise(PI_Human_i, stdev)
PI_Human_j_stochastic = add_noise(PI_Human_j, stdev)
PI_Robot_i_stochastic = add_noise(PI_Robot_i, stdev)
PI_Robot_j_stochastic = add_noise(PI_Robot_j, stdev)

# Plotting
plt.figure(figsize=(10, 6))

# Plot the mean lines and shaded variance (stochasticity)
plt.plot(prediction_horizon, PI_all_i, label="PI_all_i", color='blue')
plt.fill_between(prediction_horizon, PI_all_i - stdev, PI_all_i + stdev, color='blue', alpha=0.2)

plt.plot(prediction_horizon, PI_all_j, label="PI_all_j", color='orange')
plt.fill_between(prediction_horizon, PI_all_j - stdev, PI_all_j + stdev, color='orange', alpha=0.2)

plt.plot(prediction_horizon, PI_Human_i, label="PI_Human_i", color='green')
plt.fill_between(prediction_horizon, PI_Human_i - stdev, PI_Human_i + stdev, color='green', alpha=0.2)

plt.plot(prediction_horizon, PI_Human_j, label="PI_Human_j", color='red')
plt.fill_between(prediction_horizon, PI_Human_j - stdev, PI_Human_j + stdev, color='red', alpha=0.2)

plt.plot(prediction_horizon, PI_Robot_i, label="PI_Robot_i", color='purple')
plt.fill_between(prediction_horizon, PI_Robot_i - stdev, PI_Robot_i + stdev, color='purple', alpha=0.2)

plt.plot(prediction_horizon, PI_Robot_j, label="PI_Robot_j", color='brown')
plt.fill_between(prediction_horizon, PI_Robot_j - stdev, PI_Robot_j + stdev, color='brown', alpha=0.2)

# Labeling
plt.xlabel('Prediction Horizon')
plt.ylabel('Performance Index')
plt.legend(loc='upper left')
plt.grid(True)
plt.title('Performance Index vs Prediction Horizon with Stochastic Variations')

plt.show()
