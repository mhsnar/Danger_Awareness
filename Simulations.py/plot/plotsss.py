import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
deltaT = 0.5
n = 20
P_th = 0.1
i = 2  # This controls which sample to select from P_xH_all

fig = plt.figure(figsize=(24, 10))  # Adjusted figure size for third column

# Create a GridSpec layout
gs = fig.add_gridspec(3, 3, width_ratios=[1, 2, 1], height_ratios=[1, 1, 1], wspace=0.05)

# Iterate through datasets
for idx, (title, data) in enumerate(datasets.items()):
    time = np.linspace(0, n * deltaT, data['P_t_all'].shape[0])
    
    # First column: Two plots (one above the other)
    ax0 = fig.add_subplot(gs[idx, 0])  # Moving dots spanning both rows
    
    # Plot trajectories and current positions (same as before)
    ax0.plot(data['x_H'][0, :], data['x_H'][1, :], 'g-', label='Human Trajectory')
    ax0.plot(data['x_R'][0, :], data['x_R'][1, :], 'b-', label='Robot Trajectory')
    ax0.plot([data['x_H'][0, -1]], [data['x_H'][1, -1]], 'go', label='Current Human Position')
    ax0.plot([data['x_R'][0, -1]], [data['x_R'][1, -1]], 'bo', label='Current Robot Position')
    
    if idx == 0:
        ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-10, 10)
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.grid(True)
    
    # Second column: Probability Distributions Plot
    ax1 = fig.add_subplot(gs[idx, 1])
    P_sample = np.mean(data['P_xH_all'][i], axis=0)  # Average across the first dimension

    # Change colormap by specifying the 'cmap' argument
    image = ax1.imshow(P_sample, extent=[-5.5, 5.5, -5.5, 5.5], origin='lower', interpolation='nearest', cmap='plasma')
    ax1.set_xlabel('$N_c$')
    ax1.set_ylabel('$N_c$')
    ax1.grid(True)
    ax1.minorticks_on()
    ax1.grid(which='minor', linestyle=':', linewidth=0.5)
    plt.colorbar(image, ax=ax1, orientation='vertical', fraction=0.02, pad=0.04)

    # Third column: Display only the step with the minimum distance
    ax2 = fig.add_subplot(gs[idx, 2])
    
    # Compute the norm of distance between each point of x_R and x_H
    distances = [np.linalg.norm(data['x_R'][:, i] - data['x_H'][:, i]) for i in range(data['x_R'].shape[1])]
    
    # Find the index of the minimum distance
    min_step = np.argmin(distances)
    min_distance = distances[min_step]
    
    # Create a table showing only the minimum distance step
    row_labels = [f"Min Step: {min_step}"]
    cell_text = [[f"{min_distance:.2f}"]]  # Format the distance to 2 decimal places
    
    # Display the table in the subplot
    table = ax2.table(cellText=cell_text, rowLabels=row_labels, colLabels=['Min Distance'], cellLoc='center', loc='center')
    table.scale(1, 1.5)  # Scale the table to fit better
    ax2.axis('off')  # Hide the axis for the table

# Final plot
plt.tight_layout()
plt.show()
