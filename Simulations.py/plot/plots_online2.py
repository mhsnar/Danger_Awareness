import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

# Load datasets
datasets = {
    'Dataset 1': {
        'u_app_H': np.load('u_app_H.npy'),
        'P_t_all': np.load('P_t_all.npy'),
        'time': np.load('time.npy'),
        'P_xH_all': np.load('P_xH_all.npy'),
        'P': np.load('P_xH_all.npy'),
        'P_Coll': np.load('P_Coll.npy'),
        'x_H': np.load('x_H.npy'),
        'x_R': np.load('x_R.npy'),
        'u_app_R': np.load('u_app_R.npy')
    },
    'Dataset 2': {
        'u_app_H': np.load('u_app_H_P.npy'),
        'P_t_all': np.load('P_t_all_P.npy'),
        'time': np.load('time_P.npy'),
        'P_xH_all': np.load('P_xH_all_P.npy'),
        'P': np.load('P_xH_all_P.npy'),
        'x_H': np.load('x_H_P.npy'),
        'x_R': np.load('x_R_P.npy'),
        'u_app_R': np.load('u_app_R_P.npy')
    },
    'Dataset 3': {
        'u_app_H': np.load('u_app_H_MP.npy'),
        'P_t_all': np.load('P_t_all_MP.npy'),
        'time': np.load('time_MP.npy'),
        'P_xH_all': np.load('P_xH_all_MP.npy'),
        'P': np.load('P_xH_all_MP.npy'),
        'x_H': np.load('x_H_MP.npy'),
        'x_R': np.load('x_R_MP.npy'),
        'u_app_R': np.load('u_app_R_MP.npy')
    }
}

# Create the figure with transposed grid layout (2x2)
fig, axs = plt.subplots(2, 2, figsize=(7, 9), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [4, 2], 'hspace': 0.3, 'wspace': 0.5})
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
# Placeholders for plots
images = []
robot_dots = []
human_dots = []
comfort_circles = []
colors = [(1, 1, 1), (0.678, 0.847, 0.902), (0, 0, 0)]  # White, light blue, black

n_bins = 10000  # Number of bins for color gradient
cmap_name = 'custom_blue_to_black'
cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Initialize the transposed plots (switch rows and columns)
for idx, (title, data) in enumerate(datasets.items()):
    if idx > 1:  # Skip the third column and row
        break
    
    # First row (top) - Plot Human and Robot Trajectories
    axs[0, idx].plot(data['x_H'][0, :], data['x_H'][1, :], 'g-', label='Human Trajectory')
    axs[0, idx].plot(data['x_R'][0, :], data['x_R'][1, :], 'b-', label='Robot Trajectory')
    axs[0, idx].plot([data['x_H'][0, 0]], [data['x_H'][1, 0]], 'go', label='Human Position')
    axs[0, idx].plot([data['x_R'][0, 0]], [data['x_R'][1, 0]], 'bo', label='Robot Position')

    comfort_circle = Circle((data['x_H'][0, 0], data['x_H'][1, 0]), 1.5, color='red', fill=True, alpha=1)
    axs[0, idx].add_patch(comfort_circle)
    comfort_circles.append(comfort_circle)
    axs[0, idx].set_xlim(-5, 5)
    axs[0, idx].set_ylim(-10, 10)
    axs[0, idx].set_xlabel('X')
    axs[0, idx].set_ylabel('Y')
    axs[0, idx].grid(True)

    if idx == 0:  # Add legend to the first plot in the first row
        legend = axs[0, idx].legend(loc='upper center', bbox_to_anchor=(1.2, 1.14), ncol=4)
    
    # Second row (bottom) - Plot P_xH_all slices in a loop
    for j in range(data['P_xH_all'][0].shape[0]):  # Loop over slices for frame 0
        axs[1, idx].imshow(data['P_xH_all'][0][j], extent=[-5.5, 5.5, -5.5, 5.5], 
                           origin='lower', interpolation='nearest', cmap=cm, alpha=0.5)

    cbar = fig.colorbar(axs[1, idx].images[-1], ax=axs[1, idx], orientation='horizontal', fraction=0.02, pad=0.02)
    cbar.ax.set_position([0.1, 0.365, 0.8, 0.015])
    axs[1, idx].set_xlabel('$N_c$')
    axs[1, idx].set_ylabel('$N_c$')
    axs[1, idx].locator_params(axis='x', nbins=20)  # Increase number of ticks on X-axis
    axs[1, idx].locator_params(axis='y', nbins=20) 

    # Enable grid with new settings
    axs[1, idx].grid(True)

    robot_dot, = axs[1, idx].plot([], [], 'bo', markersize=4)
    robot_dots.append(robot_dot)
    human_dot, = axs[1, idx].plot([], [], 'go', markersize=4)
    human_dots.append(human_dot)
    images.append(axs[1, idx].images[-1])

# Function to update plots
frame_count = 0  # Track number of frames updated
max_frames = 10  # Maximum number of frames to run

def update(frame):
    global frame_count
    if frame_count >= max_frames:
        ani.event_source.stop()  # Stop the animation

    for idx, (title, data) in enumerate(datasets.items()):
        if idx > 1:  # Skip the third column and row
            break
        
        # Update trajectories
        axs[0, idx].lines[-2].set_data([data['x_H'][0, frame]], [data['x_H'][1, frame]])
        axs[0, idx].lines[-1].set_data([data['x_R'][0, frame]], [data['x_R'][1, frame]])

        comfort_circles[idx].center = (data['x_H'][0, frame], data['x_H'][1, frame])
        
        # Plot each slice in P_xH_all[frame] on top of each other
        for j in range(data['P_xH_all'][frame].shape[0]):
            axs[1, idx].imshow(data['P_xH_all'][frame][j], extent=[-5.5, 5.5, -5.5, 5.5], 
                               origin='lower', interpolation='nearest', cmap=cm, alpha=0.5)

        # Update robot and human dots
        robot_dot_x = data['x_R'][0, frame]
        robot_dot_y = data['x_R'][1, frame]
        robot_dots[idx].set_data([robot_dot_x], [robot_dot_y])

        human_dot_x = data['x_H'][0, frame]
        human_dot_y = data['x_H'][1, frame]
        human_dots[idx].set_data([human_dot_x], [human_dot_y])

        if frame == max_frames-1:  # Check if it's the last frame
            plt.savefig('Simulation.eps', format='eps')  # Save the figure as EPS

    frame_count += 1  # Increment the frame count

# Setup animation with an interval of 1000ms
ani = FuncAnimation(fig, update, frames=max_frames, interval=100, blit=False)

# Show the plot
plt.tight_layout()
plt.show()

