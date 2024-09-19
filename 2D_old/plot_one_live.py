import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create the Nc array as a 2D coordinate matrix
Nc_values = np.array([-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
X, Y = np.meshgrid(Nc_values, Nc_values)
coordinates_matrix = np.empty((Nc_values.shape[0], Nc_values.shape[0]), dtype=object)

for i in range(Nc_values.shape[0]):
    for j in range(Nc_values.shape[0]):
        coordinates_matrix[i, j] = np.array([[X[i, j]], [Y[i, j]]])

# Set Nc to coordinates_matrix
Nc = coordinates_matrix

# Load the array from the file (use a random array for this example)
P = np.random.rand(5, Nc.shape[0], Nc.shape[1])

# Get the shape of the data
Prediction_Horizon, Nc_rows, Nc_cols = P.shape

# Find the maximum and minimum non-zero values in the entire P array
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

# Set up the figure with subplots
plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(15, 5))

# Create a GridSpec layout
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 3, 1], height_ratios=[1, 1])

# Second column: Live Plot of Probability Distributions
ax3 = fig.add_subplot(gs[:, 2])
image = ax3.imshow(P_normalized[0], extent=[Nc_values[0], Nc_values[-1], Nc_values[0], Nc_values[-1]], origin='lower',
                   cmap=cm, interpolation='nearest')
ax3.set_xlabel('$N_c$')
ax3.set_ylabel('$N_c$')
ax3.set_title('Live Probability Distribution')
cbar = plt.colorbar(image, ax=ax3, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Normalized Probability Value')

# Start the loop to simulate updating data
for i in range(100):
    # Update the P array with new data (or reload from file)
    P = np.random.rand(Prediction_Horizon, Nc_rows, Nc_cols)  # Example data
    
    # Update the probability distribution plot
    P_normalized = (P - min_value) / (max_value - min_value)
    P_normalized = np.clip(P_normalized, 0, 1)  # Clip negative values to 0
    combined_P = np.mean(P_normalized, axis=0)  # Average over all prediction horizons
    image.set_data(combined_P)
    
    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()
