import numpy as np
import matplotlib.pyplot as plt

# Example data
Prediction_Horizon = 5
Nc = np.arange(-5, 6, 1)
P = np.random.rand(Prediction_Horizon, len(Nc), 1)  # Dummy data
x = np.linspace(-5, 5, 100)  # Example x values for vertical lines

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize plot objects that will be updated
lines = []
for i in range(Prediction_Horizon):
    line, = ax.plot([], [], label=f'$P(x_H[ {i+1}])$')
    lines.append(line)

# Set up the plot details
ax.set_xlabel('$N_c$')
ax.set_ylabel('Prob. Dist. $P(x_H)$')
ax.set_title('Probability Distributions for Different Prediction Horizons')
ax.grid(True)
ax.set_xticks(np.arange(-5, 6, 1))

# Set fixed axis limits based on expected data ranges
ax.set_xlim(-5, 5)
ax.set_ylim(0, 1)  # Assuming probability values between 0 and 1

# Create a single vertical line object
vertical_line, = ax.plot([], [], color='black', linestyle=(0, (5, 5)), linewidth=2)

# Fix the legend in the upper right corner (or choose another location)
ax.legend(loc='upper right')

# Start the loop
for i in range(100):  # Loop to simulate updating data
    P = np.random.rand(Prediction_Horizon, len(Nc), 1)  # Update P with new data

    for j, line in enumerate(lines):
        line.set_data(Nc, P[j].flatten())

    # Update the position of the vertical line
    vertical_line.set_data([x[i % len(x)], x[i % len(x)]], [0, 1])

    # Redraw the plot without affecting the data lines
    ax.relim()   # Recalculate limits if needed
    ax.autoscale_view()  # Rescale the view limits
    plt.draw()  # Update the figure
    plt.pause(0.1)  # Pause to allow the plot to update

plt.ioff()  # Turn off interactive mode
plt.show()
