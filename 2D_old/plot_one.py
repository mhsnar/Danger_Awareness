import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load the array from the file
P = np.load('P.npy')

# Get the shape of the data
Prediction_Horizon, Nc_rows, Nc_cols = P.shape

# Define the range for Nc (assuming symmetric range)
Nc = np.linspace(-5, 5, Nc_rows)

# Find the maximum value in the entire P array
max_value = np.max(P)
min_value = np.min(P[P > 0])  # Minimum non-zero value

# Normalize the data to the range [0, 1]
P_normalized = (P - min_value) / (max_value - min_value)

# Clip negative values to 0 (this handles cases where P == min_value)
P_normalized = np.clip(P_normalized, 0, 1)

# Create a custom colormap with white for zeros, light blue for small values, and black for the maximum
colors = [(1, 1, 1), (0.678, 0.847, 0.902), (0, 0, 0)]  # White, light blue, black
n_bins = 100  # Number of bins for color gradient
cmap_name = 'custom_blue_to_black'
cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Set up the figure and axes for a 2x3 grid (with an additional subplot for combined predictions)
fig, axs = plt.subplots(2, 3, figsize=(10, 5))
fig.tight_layout(pad=3.0)

# Iterate over each prediction horizon and subplot
for i in range(Prediction_Horizon):
    row, col = divmod(i, 3)  # Determine the row and column index for subplot
    ax = axs[row, col]  # Select the subplot
    
    # Plot each normalized prediction horizon in a subplot
    cax = ax.imshow(P_normalized[i], extent=[Nc[0], Nc[-1], Nc[0], Nc[-1]], origin='lower',
                    cmap=cm, interpolation='nearest')
    
    # Add title and labels for each subplot
    ax.set_title(f'Prediction Horizon {i+1}')
    ax.set_xlabel('$N_c$')
    ax.set_ylabel('$N_c$')

# Combine all predictions into one plot by averaging over all prediction horizons
combined_predictions = np.mean(P_normalized, axis=0)
combined_ax = axs[1, 2]  # Select the sixth subplot

# Plot the combined predictions in the sixth subplot
cax_combined = combined_ax.imshow(combined_predictions, extent=[Nc[0], Nc[-1], Nc[0], Nc[-1]], origin='lower',
                                  cmap=cm, interpolation='nearest')

# Add title and labels for the combined plot
combined_ax.set_title('Combined Predictions')
combined_ax.set_xlabel('$N_c$')
combined_ax.set_ylabel('$N_c$')

# Add a single colorbar for all subplots
cbar = plt.colorbar(cax_combined, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Normalized Probability Value')

# Show the plot
plt.show()
