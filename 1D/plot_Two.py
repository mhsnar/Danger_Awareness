import numpy as np
import matplotlib.pyplot as plt

# Example initialization for the live plot
time_steps = 100
time = np.linspace(0, 10, time_steps)  # Time axis for live plot
P_t = np.linspace(0.5, 1, time_steps)  # Scalar parameter that increases over time

# Example data for probability distributions
Prediction_Horizon = 5
Nc = np.arange(-5, 6, 1)
P = np.random.rand(Prediction_Horizon, len(Nc), 1)  # Dummy data
x_line = np.linspace(-5, 5, 100)  # Example x values for vertical lines

# Set up the plot with subplots
plt.ion()  # Turn on interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Subplot 1: Dynamic Line Plot
line1, = ax1.plot([], [], 'r-', label='$P_t(\\beta=1)$')
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
ax1.set_xlabel('Time')
ax1.set_ylabel('$P_t(\\beta=1)$')  # Y-axis label in LaTeX
ax1.set_title('Dynamic Line Extension Based on Scalar Parameter')
ax1.grid(True)
ax1.legend(loc='upper right')

# Subplot 2: Probability Distributions with Vertical Line
lines = []
for i in range(Prediction_Horizon):
    line, = ax2.plot([], [], label=f'$P(x_H[ {i+1}])$')
    lines.append(line)

ax2.set_xlabel('$N_c$')
ax2.set_ylabel('Prob. Dist. $P(x_H)$')
ax2.set_title('Probability Distributions for Different Prediction Horizons')
ax2.grid(True)
ax2.set_xticks(np.arange(-5, 6, 1))

# Set fixed axis limits based on expected data ranges
ax2.set_xlim(-5, 5)
ax2.set_ylim(0, 1)  # Assuming probability values between 0 and 1

# Create the vertical line object with a label for the legend
vertical_line, = ax2.plot([], [], color='black', linestyle=(0, (5, 5)), linewidth=2, label='Current Position')

# Fix the legend in the upper right corner
ax2.legend(loc='upper right')

# Start the loop to simulate updating data
for i in range(100):  # Loop to simulate updating data
    # Update P for the second subplot
    P = np.random.rand(Prediction_Horizon, len(Nc), 1)  # Dummy data

    # Update the line data for the first subplot
    scalar_value = P_t[i]
    line1.set_data(time[:i+1], P_t[:i+1])
    
    # Update the line data for the second subplot
    for j, line in enumerate(lines):
        line.set_data(Nc, P[j].flatten())
    
    # Update the position of the vertical line in the second subplot
    vertical_line.set_data([x_line[i % len(x_line)], x_line[i % len(x_line)]], [0, 1])

    # Update each subplot
    ax1.relim()  # Recalculate limits for the first subplot
    ax1.autoscale_view()  # Rescale the view limits for the first subplot
    ax2.relim()  # Recalculate limits for the second subplot
    ax2.autoscale_view()  # Rescale the view limits for the second subplot

    plt.draw()  # Update the figure
    plt.pause(0.1)  # Pause to allow the plot to update

plt.ioff()  # Turn off interactive mode
plt.show()
