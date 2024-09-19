import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch

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
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 3, 1], height_ratios=[1, 1], wspace=0.4)

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
dot1, = ax0.plot([], [], 'go', label='Human')  # Red dot
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

# Human's Action Box
ax4.set_title("Human's Velocity")

# Create a rectangle around the Human's Action box
rect_human = FancyBboxPatch((0.0, 0.0), 1, 1, boxstyle="round,pad=0.1", edgecolor='black', linewidth=2)
ax4.add_patch(rect_human)

velocity_text_human = ax4.text(0.5, 0.1, '', verticalalignment='center', horizontalalignment='center',
                               fontsize=12, color='black', bbox=dict(boxstyle="round,pad=0.5", edgecolor='black', facecolor='none'))
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

# Robot's Action Box
ax5.set_title("Robot's Velocity")

# Create a rectangle around the Robot's Action box
rect_robot = FancyBboxPatch((0.0, 0.0), 1, 1, boxstyle="round,pad=0.1", edgecolor='black', linewidth=2)
ax5.add_patch(rect_robot)

velocity_text_robot = ax5.text(0.5, 0.1, '', verticalalignment='center', horizontalalignment='center',
                                fontsize=12, color='black', bbox=dict(boxstyle="round,pad=0.5", edgecolor='black', facecolor='none'))
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')

# Start the loop to simulate updating data
for i in range(100):
    P = np.random.rand(Prediction_Horizon, len(Nc), len(Nc))  # Dummy data for probability distribution
    P_normalized = P 
    P_normalized = np.clip(P_normalized, 0, 1)  # Clip negative values to 0
    image.set_data(P_normalized[i % Prediction_Horizon])


        # Update the black square representing the actual position on ax3
    for artist in ax3.patches:
        artist.remove()  # Remove the previous square

    # Add the new black square at the current position (x_H[0, i], x_H[1, i])
    actual_position_square = FancyBboxPatch(
        (x_H[0, i] - 0.5, x_H[1, i] - 0.5), 1, 1,  # Position and size of the square
        boxstyle="round,pad=0.1", edgecolor='black', facecolor='black', linewidth=1, alpha=0.7
    )
    ax3.add_patch(actual_position_square)

    line1.set_data(x[:i+1], P_t[:i+1])
    line2.set_data(x[:i+1], P_t[:i+1])

    dot1.set_data(x[i % len(x)], np.sin(x[i % len(x)]))
    dot2.set_data(x[i % len(x)], np.cos(x[i % len(x)]))

    # Update Velocity Texts
    Vx = np.sin(x[i % len(x)])
    Vy = np.cos(x[i % len(x)])
    norm = np.sqrt(Vx**2 + Vy**2)
    angle = np.arctan2(Vy, Vx)

    # Clear previous arrows and dots
    for artist in ax4.patches:
        artist.remove()
    for artist in ax5.patches:
        artist.remove()
    ax4.plot([], [], 'bo')  # Reset the dots in the Human's Action box
    ax5.plot([], [], 'go')  # Reset the dots in the Robot's Action box

    # Draw velocity vector for human
    start_point_human = (0.5, 0.6)
    end_point_human = (0.5 + 0.4 * norm * np.cos(angle), 0.6 + 0.4 * norm * np.sin(angle))
    arrow_human = FancyArrowPatch(start_point_human, end_point_human,
                                  mutation_scale=10, arrowstyle='->', color='blue', linewidth=2, transform=ax4.transAxes)
    ax4.add_patch(arrow_human)
    ax4.plot(*start_point_human, 'bo')  # Dot at the start of the velocity vector

    # Draw velocity vector for robot
    start_point_robot = (0.5, 0.6)
    end_point_robot = (0.5 + 0.4 * norm * np.cos(angle), 0.6 + 0.4 * norm * np.sin(angle))
    arrow_robot = FancyArrowPatch(start_point_robot, end_point_robot,
                                  mutation_scale=10, arrowstyle='->', color='green', linewidth=2, transform=ax5.transAxes)
    ax5.add_patch(arrow_robot)
    ax5.plot(*start_point_robot, 'go')  # Dot at the start of the velocity vector

    # Update the velocity text
    velocity_text = f'V = [{Vx:.2f}  {Vy:.2f}]'  # Horizontally oriented
    velocity_text_human.set_text(velocity_text)
    velocity_text_robot.set_text(velocity_text)

    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()
