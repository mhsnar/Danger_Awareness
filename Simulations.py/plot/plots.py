import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

u_app_H = np.load('u_app_H_P.npy')
P_t_all = np.load('P_t_all_P.npy')
time = np.load('time_P.npy')
P_xH = np.load('P_xH_P.npy')
P = P_xH
P_Coll = np.load('P_Coll.npy')
x_H = np.load('x_H_P.npy')
x_R = np.load('x_R_P.npy')
u_app_R = np.load('u_app_R_P.npy')
deltaT = 0.2
n = 100
P_th = 0.1
i = 10
human_action_value_x = u_app_H[0, i % u_app_H.shape[1]]
human_action_value_y = u_app_H[1, i % u_app_H.shape[1]]

robot_action_value_x = u_app_R[0, i % u_app_R.shape[1]]
robot_action_value_y = u_app_R[1, i % u_app_R.shape[1]]

# Create the offline version of the plot
time = np.linspace(0, n*deltaT, P_t_all.shape[0]) 
fig = plt.figure(figsize=(10, 5))  # Adjusted figure size for fewer columns

# Create a GridSpec layout
gs = fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[1, 1], wspace=0.4)

# First column: Two plots (one above the other)
ax0 = fig.add_subplot(gs[:, 0])  # Moving dots spanning both rows

# Subplot 0: Moving dots positions and trajectories
dot1, = ax0.plot([], [], 'go', label='Human')  # Green dot
dot2, = ax0.plot([], [], 'bo', label='Robot')  # Blue dot
traj_human, = ax0.plot([], [], 'g-', label='Human Trajectory')  # Green line for trajectory
traj_robot, = ax0.plot([], [], 'b-', label='Robot Trajectory')  # Blue line for trajectory

# Plotting initial data
ax0.plot(x_H[0, :], x_H[1, :], 'g-', label='Human Trajectory')  # Human trajectory
ax0.plot(x_R[0, :], x_R[1, :], 'b-', label='Robot Trajectory')  # Robot trajectory

ax0.set_xlim(-5, 5)
ax0.set_ylim(-10, 10)
ax0.set_xlabel('X')
ax0.set_ylabel('Y')
ax0.legend()
ax0.grid(True)

# Probability Distributions Plot
ax1 = fig.add_subplot(gs[:, 1])
combined_P = np.mean(P, axis=0)  # Average over prediction horizon
image = ax1.imshow(combined_P, extent=[-5.5, 5.5, -5.5, 5.5], origin='lower', interpolation='nearest')
ax1.set_xlabel('$N_c$')
ax1.set_ylabel('$N_c$')
ax1.set_title('Probability Distribution')
ax1.grid(True)
ax1.minorticks_on()
ax1.grid(which='minor', linestyle=':', linewidth=0.5)
plt.colorbar(image, ax=ax1, orientation='vertical', fraction=0.02, pad=0.04)

# Static dots on the Moving Dots plot
ax0.plot([x_H[0, -1]], [x_H[1, -1]], 'go', label='Human')
ax0.plot([x_R[0, -1]], [x_R[1, -1]], 'bo', label='Robot')

# Adding the final probability distribution square at the last position
actual_position_square = plt.Rectangle(
    (x_H[0, -1] - 0.25, x_H[1, -1] - 0.25), 0.5, 0.5,
    facecolor='black'
)
ax1.add_patch(actual_position_square)

# Final plot
plt.show()
