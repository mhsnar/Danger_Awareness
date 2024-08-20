import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

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
P = np.random.rand(Prediction_Horizon, len(Nc), len(Nc))  # Dummy data for 2D probability matrix
x_line = np.linspace(-5, 5, 100)  # Example x values for vertical lines

# Scalar data for human and robot actions
human_action_data = [-2, -1, 0, 1, 2]
robot_action_data = [0, 1, 2]

# Set up the figure with subplots
plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(15, 5))

# Create a GridSpec layout
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 3, 1], height_ratios=[1, 1])

# First column: Two plots (one above the other)
ax0 = fig.add_subplot(gs[:, 0])  # Moving dots spanning both rows
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax2.axhline(y=P_th, color='r', linestyle=(0, (4, 3)), linewidth=2, label='$P_{th}$')

# Second column: Live Plot of Probability Distributions spanning both rows
ax3 = fig.add_subplot(gs[:, 2])

# Fourth column: Human and Robot Actions
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[1, 3])

# Subplot 0: Moving dots positions
dot1, = ax0.plot([], [], 'ro', label='Human')  # Red dot
dot2, = ax0.plot([], [], 'bo', label='Robot')  # Blue dot
ax0.set_xlim(-5, 5)
ax0.set_ylim(-5, 5)
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
ax0.legend()
ax0.grid(True)

# Subplot 1: Live Plot of Scalar Parameter (Version 1)
line1, = ax1.plot([], [], 'r-', label='$P_t(\\beta=1)$')
ax1.set_xlim(0, deltaT * n)
ax1.set_ylim(0, 1)  # Set y-axis limits from 0 to 1
ax1.set_xlabel('Time')
ax1.set_ylabel('$P_t(\\beta=1)$')  # Y-axis label in LaTeX
ax1.grid(True)
ax1.legend(loc='upper right')

# Subplot 2: Live Plot of Scalar Parameter (Version 2)
line2, = ax2.plot([], [], 'b-')
ax2.set_xlim(0, deltaT * n)
ax2.set_ylim(0, P_th + P_th * 0.005)  # Set y-axis limits from 0 to 1
ax2.set_xlabel('Time')
ax2.set_ylabel('Collision Prob.')  # Y-axis label in LaTeX
ax2.grid(True)
ax2.legend(loc='upper right')

# Set up custom colormap for probability distribution
max_value = np.max(P)
min_value = np.min(P[P > 0])  # Minimum non-zero value

# Normalize the data to the range [0, 1]
P_normalized = (P - min_value) / (max_value - min_value)
P_normalized = np.clip(P_normalized, 0, 1)  # Clip negative values to 0

# Create a custom colormap with white for zeros, light blue for small values, and black for the maximum
colors = [(1, 1, 1), (0.678, 0.847, 0.902), (0, 0, 0)]  # White, light blue, black
n_bins = 100  # Number of bins for color gradient
cmap_name = 'custom_blue_to_black'
cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Second column: Live Plot of Probability Distributions
image = ax3.imshow(P_normalized[0], extent=[Nc[0], Nc[-1], Nc[0], Nc[-1]], origin='lower',
                   cmap=cm, interpolation='nearest')
ax3.set_xlabel('$N_c$')
ax3.set_ylabel('$N_c$')
ax3.set_title('Live Probability Distribution')
cbar = plt.colorbar(image, ax=ax3, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Normalized Probability Value')

# Human's Action Box
ax4.set_title("Human's Action")
human_actions = ['Running Backward', 'Walking Backward', 'Stop', 'Walking Forward', 'Running Forward']
circles_human = []

# Create a rectangle around the Human's Action box
rect_human = Rectangle((0.0, 0.05), 1.05, 0.95, fill=False, edgecolor='black', lw=2)
ax4.add_patch(rect_human)

for idx, action in enumerate(human_actions):
    ax4.text(0.2, 1 - (idx + 1) * 0.15, action, verticalalignment='center', fontsize=10)
    circle = plt.Circle((0.1, 1 - (idx + 1) * 0.15), 0.05, color='white', ec='black')
    ax4.add_patch(circle)
    circles_human.append(circle)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

# Robot's Action Box
ax5.set_title("Robot's Action")
robot_actions = ['Zero Speed', 'Half Speed', 'Full Speed']
circles_robot = []

# Create a rectangle around the Robot's Action box
rect_robot = Rectangle((0.0, 0.05), .9, .9, fill=False, edgecolor='black', lw=2)
ax5.add_patch(rect_robot)

for idx, action in enumerate(robot_actions):
    ax5.text(0.2, 1 - (idx + 1) * 0.25, action, verticalalignment='center', fontsize=10)
    circle = plt.Circle((0.1, 1 - (idx + 1) * 0.25), 0.05, color='white', ec='black')
    ax5.add_patch(circle)
    circles_robot.append(circle)

ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')

# Start the loop to simulate updating data
for i in range(100):
    P = np.random.rand(Prediction_Horizon, len(Nc), len(Nc))  # Dummy data for probability distribution
    P_normalized = P 
    P_normalized = np.clip(P_normalized, 0, 1)  # Clip negative values to 0
    image.set_data(P_normalized[i % Prediction_Horizon])

    line1.set_data(x[:i+1], P_t[:i+1])
    line2.set_data(x[:i+1], P_t[:i+1])

    dot1.set_data(x[i % len(x)], np.sin(x[i % len(x)]))
    dot2.set_data(x[i % len(x)], np.cos(x[i % len(x)]))

    # Update Human's Action Circles
    human_action_value = human_action_data[i % len(human_action_data)]
    for idx, circle in enumerate(circles_human):
        if idx - 2 == human_action_value:
            circle.set_color('black')
        else:
            circle.set_color('white')

    # Update Robot's Action Circles
    robot_action_value = robot_action_data[i % len(robot_action_data)]
    for idx, circle in enumerate(circles_robot):
        if idx == robot_action_value:
            circle.set_color('black')
        else:
            circle.set_color('white')

    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()
