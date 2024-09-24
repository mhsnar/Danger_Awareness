import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

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

# Create the figure with transposed grid layout
fig, axs = plt.subplots(3, 3, figsize=(10, 10), gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [4, 2, 1], 'hspace': 0.5,'wspace': 0.5})

# Placeholders for plots
images = []
robot_dots = []
distance_texts = []
comfort_circles = []
min_distance_values = []

# Initialize the transposed plots (switch rows and columns)
for idx, (title, data) in enumerate(datasets.items()):
    # Second row (current column) becomes first row:
    axs[0, idx].plot(data['x_H'][0, :], data['x_H'][1, :], 'g-', label='Human Trajectory')
    axs[0, idx].plot(data['x_R'][0, :], data['x_R'][1, :], 'b-', label='Robot Trajectory')
    axs[0, idx].plot([data['x_H'][0, 0]], [data['x_H'][1, 0]], 'go', label='Current Human Position')
    axs[0, idx].plot([data['x_R'][0, 0]], [data['x_R'][1, 0]], 'bo', label='Current Robot Position')
    
    comfort_circle = Circle((data['x_H'][0, 0], data['x_H'][1, 0]), 1.5, color='red', fill=True, alpha=0.3)
    axs[0, idx].add_patch(comfort_circle)
    comfort_circles.append(comfort_circle)
    axs[0, idx].set_xlim(-5, 5)
    axs[0, idx].set_ylim(-10, 10)
    axs[0, idx].set_xlabel('X')
    axs[0, idx].set_ylabel('Y')
    axs[0, idx].grid(True)
    
    if idx == 1:
        axs[1, idx].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

    # Third row (current column) becomes second row:
    P_sample = np.mean(data['P_xH_all'][0], axis=0)
    image = axs[1, idx].imshow(P_sample, extent=[-5.5, 5.5, -5.5, 5.5], origin='lower', interpolation='nearest', cmap='viridis')
    fig.colorbar(image, ax=axs[1, idx], orientation='vertical', fraction=0.02, pad=0.02)
    axs[1, idx].set_xlabel('$N_c$')
    axs[1, idx].set_ylabel('$N_c$')
    axs[1, idx].grid(True)
    
    robot_dot, = axs[1, idx].plot([], [], 'ro', markersize=8)
    robot_dots.append(robot_dot)
    images.append(image)

    # First row (current column) becomes third row:
    initial_distance = np.linalg.norm(data['x_R'][:, 0] - data['x_H'][:, 0])
    min_distance = np.min([np.linalg.norm(data['x_R'][:, i] - data['x_H'][:, i]) for i in range(data['x_H'].shape[1])])
    text = axs[2, idx].text(0.5, 0.6, f"Current: {initial_distance:.2f}", ha='center', va='center', fontsize=12, transform=axs[2, idx].transAxes)
    min_text = axs[2, idx].text(0.5, 0.4, f"Min: {min_distance:.2f}", ha='center', va='center', fontsize=12, transform=axs[2, idx].transAxes)
    axs[2, idx].axis('off')
    distance_texts.append(text)
    min_distance_values.append(min_distance)
for idx in range(3):  # Assuming there are 3 rows
    axs[idx, 1].set_facecolor('white')
# Function to update plots
def update(frame):
    for idx, (title, data) in enumerate(datasets.items()):
        axs[0, idx].lines[-2].set_data([data['x_H'][0, frame]], [data['x_H'][1, frame]])
        axs[0, idx].lines[-1].set_data([data['x_R'][0, frame]], [data['x_R'][1, frame]])

        comfort_circles[idx].center = (data['x_H'][0, frame], data['x_H'][1, frame])
        axs[1, idx].cla()
        
        # Plot each P_xH_all[frame][j]
        for j in range(data['P_xH_all'][frame].shape[0]):
            axs[1, idx].imshow(data['P_xH_all'][frame][j])

        robot_dot_x = data['x_R'][0, frame]
        robot_dot_y = data['x_R'][1, frame]
        robot_dots[idx].set_data([robot_dot_x], [robot_dot_y])

        current_distance = np.linalg.norm(data['x_R'][:, frame] - data['x_H'][:, frame])
        distance_texts[idx].set_text(f"Current: {current_distance:.2f}")

# Setup animation
ani = FuncAnimation(fig, update, frames=np.arange(0, datasets['Dataset 1']['x_H'].shape[1]), interval=1000)

# Show the plot
plt.tight_layout()
plt.show()
