import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

# Second column: One plot spanning both rows
ax3 = fig.add_subplot(gs[:, 2])

# Subplot 0: Moving dots positions
dot1, = ax0.plot([], [], 'ro', label='Human')  # Red dot
dot2, = ax0.plot([], [], 'bo', label='Robot')  # Blue dot
ax0.set_xlim(-5,5)
ax0.set_ylim(-5, 5)
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
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
vertical_line, = ax3.plot([], [], color='black', linestyle=(0, (4, 3)), linewidth=2, label='Current Position')

# Initialize line objects for each prediction horizon
lines = [ax3.plot([], [], label=f'$P(x_H[ {i+1}])$')[0] for i in range(Prediction_Horizon)]

ax3.set_xlabel('$Grid Cells$')
ax3.set_ylabel('Prob. Dist. $P(x_H)$')
ax3.grid(True)
ax3.set_xticks(np.arange(-5, 6, 1))

ax3.set_xlim(-5, 5)
ax3.set_ylim(0, 1)  # Assuming probability values between 0 and 1
ax3.legend(loc='upper right')

# Fourth column: Human and Robot Actions
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[1, 3])

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
    P = np.random.rand(Prediction_Horizon, len(Nc), 1)  # Dummy data
    line1.set_data(x[:i+1], P_t[:i+1])
    line2.set_data(x[:i+1], P_t[:i+1])
    
    for j, line in enumerate(lines):
        line.set_data(Nc, P[j].flatten())
    
    vertical_line.set_data([x_line[i % len(x_line)], x_line[i % len(x_line)]], [0, 1])

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

    

    ax0.relim()
    ax0.autoscale_view()
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    ax3.relim()
    ax3.autoscale_view()

    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()
