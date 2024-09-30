import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.colors as mcolors

# Function to load data from NPZ files
def load_data_from_npz(npz_filename):
    data = np.load(npz_filename)
    return {
        'u_app_H': data['u_app_H'],
        'P_t_all': data['P_t_all'],
        'time': data['time'],
        'P_xH_all': data['P_xH_all'],
        'x_H': data['x_H'],
        'x_R': data['x_R'],
        'u_app_R': data['u_app_R']
    }

# Load the datasets
data1 = load_data_from_npz('experiment_data.npz')
data2 = load_data_from_npz('experiment_data1.npz')

datasets = {'Data 1': data1, 'Data 2': data2}

# Create the figure with a transposed grid layout (6 rows, 2 columns)
fig, axs = plt.subplots(6, 2, figsize=(6, 120),
                        gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [4, 2, 2, 2, 2, 2], 'hspace': 0.3, 'wspace': 0.5})
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

# Initialize the transposed plots
for idx, (title, data) in enumerate(datasets.items()):
    # First row - Plot Human and Robot Trajectories
    axs[0, idx].plot(data['x_H'][0, :], data['x_H'][1, :], 'g-', label='Human Trajectory')
    axs[0, idx].plot(data['x_R'][0, :], data['x_R'][1, :], 'b-', label='Robot Trajectory')
    axs[0, idx].plot([data['x_H'][0, 0]], [data['x_H'][1, 0]], 'go', label='Human Position')
    axs[0, idx].plot([data['x_R'][0, 0]], [data['x_R'][1, 0]], 'bo', label='Robot Position')

    # Add comfort circles
    comfort_circle = Circle((data['x_H'][0, 0], data['x_H'][1, 0]), 0.5, color='red', fill=False, alpha=1)
    axs[0, idx].add_patch(comfort_circle)
    comfort_circles.append(comfort_circle)
    axs[0, idx].set_xlim(-5, 5)
    axs[0, idx].set_ylim(-10, 10)

    axs[0, idx].set_xlabel('X', fontsize=10)
    axs[0, idx].set_ylabel('Y', fontsize=10)
    axs[0, idx].grid(True)

    if idx == 0:
        axs[0, idx].legend(loc='upper center', bbox_to_anchor=(1.2, 1.14), ncol=4, fontsize=8.5)

    # Plot P_xH_all slices
    for row in range(1, 6):  # Rows 1 to 5 (0-indexed)
        axs[row, idx].imshow(data['P_xH_all'][0][row - 1], extent=[-5.5, 5.5, -5.5, 5.5], 
                             origin='lower', interpolation='nearest', cmap=cm, alpha=0.5)

        axs[5, idx].set_xlabel('$N_c$')
        axs[row, 0].set_ylabel('$N_c$')
        axs[row, 1].set_ylabel(r'$N_R={}$'.format(row), fontsize=12)
        axs[row, 1].yaxis.set_label_coords(-.7, 0.5)
        axs[row, 1].yaxis.label.set_rotation(0)
        axs[row, idx].locator_params(axis='x', nbins=20)
        axs[row, idx].locator_params(axis='y', nbins=20)
        axs[row, idx].grid(True)

        robot_dot, = axs[row, idx].plot([], [], 'bo', markersize=4)
        robot_dots.append(robot_dot)
        human_dot, = axs[row, idx].plot([], [], 'go', markersize=4)
        human_dots.append(human_dot)
        images.append(axs[row, idx].images[-1])

# Function to update plots
frame_count = 0
max_frames = 11

def update(frame):
    global frame_count
    if frame_count >= max_frames:
        ani.event_source.stop()

    for idx, (title, data) in enumerate(datasets.items()):
        # Update trajectories
        axs[0, idx].lines[-2].set_data([data['x_H'][0, frame]], [data['x_H'][1, frame]])
        axs[0, idx].lines[-1].set_data([data['x_R'][0, frame]], [data['x_R'][1, frame]])

        comfort_circles[idx].center = (data['x_H'][0, frame], data['x_H'][1, frame])

        for row in range(1, 6):
            axs[row, idx].imshow(data['P_xH_all'][frame][row - 1], extent=[-5.5, 5.5, -5.5, 5.5], 
                                 origin='lower', interpolation='nearest', cmap=cm, alpha=0.5)

            # Update robot and human dots
            robot_dot_x = data['x_R'][0, frame]
            robot_dot_y = data['x_R'][1, frame]
            robot_dots[(idx * 5) + (row - 1)].set_data([robot_dot_x], [robot_dot_y])

            human_dot_x = data['x_H'][0, frame]
            human_dot_y = data['x_H'][1, frame]
            human_dots[(idx * 5) + (row - 1)].set_data([human_dot_x], [human_dot_y])

        if frame == max_frames - 1:
            plt.savefig('Experiment.eps', format='eps')

    frame_count += 1

# Setup animation
ani = FuncAnimation(fig, update, frames=max_frames, interval=1000, blit=False)

# Show the plot
plt.tight_layout()
plt.show()
