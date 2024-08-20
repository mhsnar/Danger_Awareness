import numpy as np
import matplotlib.pyplot as plt

# Example initialization for the live plot
time_steps = 100
x = np.linspace(0, 10, time_steps)  # Time axis for live plot
P_t = np.linspace(0.5, 1, time_steps)  # Scalar parameter that increases over time
P_th = 0.2
deltaT = 0.5
n = 20

# Example data for probability distributions
Prediction_Horizon = 5
Nc = np.arange(-5, 6, 1)
P = np.random.rand(Prediction_Horizon, len(Nc), 1)  # Dummy data
x_line = np.linspace(-5, 5, 100)  # Example x values for vertical lines

# Set up the figure with subplots
plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(13, 5))

# Create a GridSpec layout
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1,3], height_ratios=[1, 1])

# First column: Two plots (one above the other)
ax0 = fig.add_subplot(gs[:, 0])  # Moving dots spanning both rows
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax2.axhline(y=P_th, color='r', linestyle=(0, (4, 3)), linewidth=2, label='$P_{th}$')


# Second column: One plot spanning both rows
ax3 = fig.add_subplot(gs[:, 2])


# Subplot 0: Moving dots positions
dot1, = ax0.plot([], [], 'ro', label='Human')  # Red dot
dot2, = ax0.plot([], [], 'bo', label='Robot')  # Blue dot
ax0.set_xlim(-5,5)
ax0.set_ylim(-5, 5)
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
# ax0.set_title('Moving Dots')
ax0.legend()
ax0.grid(True)
# Subplot 1: Live Plot of Scalar Parameter (Version 1)
line1, = ax1.plot([], [], 'r-', label='$P_t(\\beta=1)$')
ax1.set_xlim(0,deltaT*n)
ax1.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
ax1.set_xlabel('Time')
ax1.set_ylabel('$P_t(\\beta=1)$')  # Y-axis label in LaTeX
ax1.grid(True)
ax1.legend(loc='upper right')

# Subplot 2: Live Plot of Scalar Parameter (Version 2)
line2, = ax2.plot([], [], 'b-')
ax2.set_xlim(0, deltaT*n)
ax2.set_ylim(0, P_th+P_th*0.005)  # Set y-axis limits from 0 to 1
ax2.set_xlabel('Time')
ax2.set_ylabel('Collision Prob.')  # Y-axis label in LaTeX
ax2.grid(True)
ax2.legend(loc='upper right')

# Subplot 3: Probability Distributions
# Create the vertical line object with a label for the legend
vertical_line, = ax3.plot([], [], color='black', linestyle=(0, (4, 3)), linewidth=2, label='Current Position')

# Initialize line objects for each prediction horizon
lines = [ax3.plot([], [], label=f'$P(x_H[ {i+1}])$')[0] for i in range(Prediction_Horizon)]

ax3.set_xlabel('$Grid Cells$')
ax3.set_ylabel('Prob. Dist. $P(x_H)$')
ax3.grid(True)
ax3.set_xticks(np.arange(-5, 6, 1))

# Set fixed axis limits based on expected data ranges
ax3.set_xlim(-5, 5)
ax3.set_ylim(0, 1)  # Assuming probability values between 0 and 1


# Move the legend of the third subplot outside the box, at the top
ax3.legend(loc='upper right')
# Start the loop to simulate updating data
for i in range(100):  # Loop to simulate updating data
    # Update P for the third subplot
    P = np.random.rand(Prediction_Horizon, len(Nc), 1)  # Dummy data

    # Update the line data for the first subplot
    line1.set_data(x[:i+1], P_t[:i+1])
    
    # Update the line data for the second subplot
    line2.set_data(x[:i+1], P_t[:i+1])
    
    # Update the line data for the third subplot
    for j, line in enumerate(lines):
        line.set_data(Nc, P[j].flatten())
    
    # Update the position of the vertical line in the third subplot
    vertical_line.set_data([x_line[i % len(x_line)], x_line[i % len(x_line)]], [0, 1])

    # Update moving dots positions
    dot1.set_data(x[i % len(x)], np.sin(x[i % len(x)]))  # Example: sine wave for dot 1
    dot2.set_data(x[i % len(x)], np.cos(x[i % len(x)]))  # Example: cosine wave for dot 2

    # Update each subplot
    ax0.relim()  # Recalculate limits for the moving dots subplot
    ax0.autoscale_view()  # Rescale the view limits for the moving dots subplot
    ax1.relim()  # Recalculate limits for the first subplot
    ax1.autoscale_view()  # Rescale the view limits for the first subplot
    ax2.relim()  # Recalculate limits for the second subplot
    ax2.autoscale_view()  # Rescale the view limits for the second subplot
    ax3.relim()  # Recalculate limits for the third subplot
    ax3.autoscale_view()  # Rescale the view limits for the third subplot

    plt.draw()  # Update the figure
    plt.pause(0.1)  # Pause to allow the plot to update

plt.ioff()  # Turn off interactive mode
plt.show()
