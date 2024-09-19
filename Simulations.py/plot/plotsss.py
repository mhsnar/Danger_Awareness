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
        'P_xH_all': np.load('P_xH_all.npy'),  # Changed to P_xH_all
        'P': np.load('P_xH_all.npy'),  # Changed to P_xH_all
        'P_Coll': np.load('P_Coll.npy'),
        'x_H': np.load('x_H.npy'),
        'x_R': np.load('x_R.npy'),
        'u_app_R': np.load('u_app_R.npy')
    },
    'Dataset 2': {
        'u_app_H': np.load('u_app_H_P.npy'),
        'P_t_all': np.load('P_t_all_P.npy'),
        'time': np.load('time_P.npy'),
        'P_xH_all': np.load('P_xH_all_P.npy'),  # Changed to P_xH_all
        'P': np.load('P_xH_all_P.npy'),  # Changed to P_xH_all
        'x_H': np.load('x_H_P.npy'),
        'x_R': np.load('x_R_P.npy'),
        'u_app_R': np.load('u_app_R_P.npy')
    },
    'Dataset 3': {
        'u_app_H': np.load('u_app_H_MP.npy'),
        'P_t_all': np.load('P_t_all_MP.npy'),
        'time': np.load('time_MP.npy'),
        'P_xH_all': np.load('P_xH_all_MP.npy'),  # Changed to P_xH_all
        'P': np.load('P_xH_all_MP.npy'),  # Changed to P_xH_all
        'x_H': np.load('x_H_MP.npy'),
        'x_R': np.load('x_R_MP.npy'),
        'u_app_R': np.load('u_app_R_MP.npy')
    }
}

deltaT = 0.5
n = 20
P_th = 0.1
i = 10  # This controls which sample to select from P_xH_all

fig = plt.figure(figsize=(18, 10))  # Adjusted figure size

# Create a GridSpec layout
gs = fig.add_gridspec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1], wspace=0.05)

# Iterate through datasets
for idx, (title, data) in enumerate(datasets.items()):
    time = np.linspace(0, n*deltaT, data['P_t_all'].shape[0])
    
    # First column: Two plots (one above the other)
    ax0 = fig.add_subplot(gs[idx, 0])  # Moving dots spanning both rows
    
    # Plot trajectories and current positions
    if idx == 0:  # Only add labels for the first dataset
        ax0.plot(data['x_H'][0, :], data['x_H'][1, :], 'g-', label='Human Trajectory')  # Human trajectory
        ax0.plot(data['x_R'][0, :], data['x_R'][1, :], 'b-', label='Robot Trajectory')  # Robot trajectory
        ax0.plot([data['x_H'][0, -1]], [data['x_H'][1, -1]], 'go', label='Current Human Position')
        ax0.plot([data['x_R'][0, -1]], [data['x_R'][1, -1]], 'bo', label='Current Robot Position')
        ax0.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    else:  # For other datasets, just plot without labels
        ax0.plot(data['x_H'][0, :], data['x_H'][1, :], 'g-')
        ax0.plot(data['x_R'][0, :], data['x_R'][1, :], 'b-')
        ax0.plot([data['x_H'][0, -1]], [data['x_H'][1, -1]], 'go')
        ax0.plot([data['x_R'][0, -1]], [data['x_R'][1, -1]], 'bo')

    ax0.set_xlim(-5, 5)
    ax0.set_ylim(-10, 10)
    ax0.set_xlabel('X')
    ax0.set_ylabel('Y')
    ax0.grid(True)
    
    # Second column: Probability Distributions Plot
    ax1 = fig.add_subplot(gs[idx, 1])
    
    # Plot the ith sample of P_xH_all
    P_sample = np.mean(data['P_xH_all'][i], axis=0)  # Average across the first dimension


    image = ax1.imshow(P_sample, extent=[-5.5, 5.5, -5.5, 5.5], origin='lower', interpolation='nearest')

    ax1.set_xlabel('$N_c$')
    ax1.set_ylabel('$N_c$')
    ax1.grid(True)
    ax1.minorticks_on()
    ax1.grid(which='minor', linestyle=':', linewidth=0.5)
    plt.colorbar(image, ax=ax1, orientation='vertical', fraction=0.9, pad=0.04)

# Final plot
plt.tight_layout()
plt.show()
