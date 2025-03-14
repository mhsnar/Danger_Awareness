# Required libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import zipfile
svdvsdvzsv= np.load('xperiment_data1.npz')
svdvsdvzsv=svdvsdvzsv['P_xH_all']

# np.save('P_xH_all.npy', svdvsdvzsv)
def load_data_from_npz_files(num_packages=2):
    datasets = {}

    for i in range(num_packages):
        data_i = np.load(f'xperiment_data{i}.npz')
        
        datasets[f'Data {i + 1}'] = {
            'u_app_H': data_i['u_app_H'],
            'P_t_all': data_i['P_t_all'],
            'time': data_i['time'],
            'P_xH_all': data_i['P_xH_all'],
            'P': data_i['P_xH_all'],  # 'P' is the same as 'P_xH_all'
            'x_H': data_i['x_H'],
            'x_R': data_i['x_R'],
            'u_app_R': data_i['u_app_R']
        }

    return datasets

# Usage
datasets = load_data_from_npz_files()

datasets['Data 1']['x_H']=datasets['Data 1']['x_H'].T
datasets['Data 1']['x_R']=datasets['Data 1']['x_R'].T
# datasets['Data 1']['P_xH_all']=np.zeros(( datasets['Data 1']['x_H'].shape[1],1,21,21))
datasets['Data 1']['P_xH_all']=datasets['Data 1']['P_xH_all']

datasets['Data 2']['x_H']=datasets['Data 2']['x_H'].T
datasets['Data 2']['x_R']=datasets['Data 2']['x_R'].T
# datasets['Data 1']['P_xH_all']=np.zeros(( datasets['Data 1']['x_H'].shape[1],1,21,21))
datasets['Data 2']['P_xH_all']=datasets['Data 2']['P_xH_all']

# print(scscs[0,:])
# print(np.max(svsv))
# print(svsv[0,0,1,0,0])
# Access the data for the first dataset
data1 = datasets['Data 1']
data2 = datasets['Data 2']

# # Load the datasets
# data1 = load_data_from_nzp('experiment_data.nzp')
# data2 = load_data_from_nzp('experiment_data1.nzp')
# svs=data1['x_H']
# svs=data1['x_R']
datasets = {'Data 1': data1, 'Data 2': data2}



# Create the figure with a grid layout for two rows and two columns
fig, axs = plt.subplots(2, 2, figsize=(4, 6), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [4, 2], 'hspace': 0.3, 'wspace': 0.5})
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

# Initialize the plots for the first two rows
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

    # Second row - Plot P_xH_all slices
    axs[1, idx].imshow(data['P_xH_all'][0][0], extent=[-5.5, 5.5, -5.5, 5.5], origin='lower', interpolation='nearest', cmap=cm, alpha=0.5)
    axs[1, idx].set_xlabel('$N_c$')
    axs[1, 0].set_ylabel('$N_c$')
    axs[1, 1].set_ylabel(r'$N_R={}$'.format(1), fontsize=12)
    axs[1, 1].yaxis.set_label_coords(-.7, 0.5)
    axs[1, 1].yaxis.label.set_rotation(0)
    axs[1, idx].locator_params(axis='x', nbins=20)
    axs[1, idx].locator_params(axis='y', nbins=20)
    axs[1, idx].grid(True)

    robot_dot, = axs[1, idx].plot([], [], 'bo', markersize=4)
    robot_dots.append(robot_dot)
    human_dot, = axs[1, idx].plot([], [], 'go', markersize=4)
    human_dots.append(human_dot)
    images.append(axs[1, idx].images[-1])

# Function to update plots
frame_count = 0
max_frames = data['x_H'].shape[1]

def update(frame):
    global frame_count
    if frame_count >= max_frames:
        ani.event_source.stop()

    for idx, (title, data) in enumerate(datasets.items()):
        # Update trajectories
        axs[0, idx].lines[-2].set_data([data['x_H'][0, frame]], [data['x_H'][1, frame]])
        axs[0, idx].lines[-1].set_data([data['x_R'][0, frame]], [data['x_R'][1, frame]])

        comfort_circles[idx].center = (data['x_H'][0, frame], data['x_H'][1, frame])

        # Update P_xH_all slices
        axs[1, idx].imshow(data['P_xH_all'][frame][0], extent=[-5.5, 5.5, -5.5, 5.5], 
                           origin='lower', interpolation='nearest', cmap=cm, alpha=0.5)

        # Update robot and human dots
        robot_dot_x = data['x_R'][0, frame]
        robot_dot_y = data['x_R'][1, frame]
        robot_dots[idx].set_data([robot_dot_x], [robot_dot_y])

        human_dot_x = data['x_H'][0, frame]
        human_dot_y = data['x_H'][1, frame]
        human_dots[idx].set_data([human_dot_x], [human_dot_y])

        if frame == max_frames - 1:
            plt.savefig('Experiment.eps', format='eps')

    frame_count += 1

# Setup animation
ani = FuncAnimation(fig, update, frames=max_frames, interval=1000, blit=False)

# Show the plot
plt.tight_layout()
plt.show()
