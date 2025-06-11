
import numpy as np
# import cvxpy as cp
# from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch

# Load datasets from npz files
datasets = {
    '1': np.load('xperiment_data1.npz'),
    '2': np.load('xperiment_data_Concern.npz'),
    '3': np.load('xperiment_data_Unconcern.npz')
}

print("Available datasets: 1, 2, 3")
dataset_choice = input("Please select a dataset (1/2/3): ")

if dataset_choice in datasets:
    data_i = datasets[dataset_choice]
    # Extract arrays from the npz file
    u_app_H = data_i['u_app_H'][:,:,0].T
    P_t_all = data_i['P_t_all'][0,:,0]
    time = data_i['time']
    P_xH_all = data_i['P_xH_all']
    P = data_i['P_xH_all']  # Same as P_xH_all
    x_H = data_i['x_H'].T
    x_R = data_i['x_R'].T
    u_app_R = data_i['u_app_R'].T
else:
    print("Invalid dataset choice. Please select a valid dataset (1, 2, or 3).")
    exit()

print(u_app_H)

# count = np.sum(P_xH_all[11,:,:,:] > 0)
#------------------------------------------
# Robot Model
n = x_R.shape[1]
Prediction_Horizon = 2
Prediction_Horizon_H=2

Safe_Distance=2
deltaT=0.5

Signal="off" # Signal could be "on" or "off"
Human="Concerned"  # Human could be "Concerned" or "Unconcerned"





#------------------------------------------
## Contorller
#INPUTS=x_H ,X_R
#Initials=
betas=np.array([0,
                1])



if Human=="Concerned":
    beta=1
elif Human=="Unconcerned":
    beta=0




P_t=np.array([.5,
                .5])
# self.Nc = np.array([-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
Nc = np.array([ -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
# Nc=np.array([-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0])
# Create a meshgrid for X and Y coordinates
# Create a meshgrid for X and Y coordinates
X, Y = np.meshgrid(Nc, Nc)

# Combine X and Y into a 2D coordinate matrix
# Flatten X and Y to create 2D matrices
coordinates_matrix = np.empty((Nc.shape[0], Nc.shape[0]), dtype=object)

for i in range(Nc.shape[0]):
    for j in range(Nc.shape[0]):
        coordinates_matrix[i, j] = np.array([[X[i, j]], [Y[i, j]]])
Nc=coordinates_matrix        



P_th = np.array([0.1]).reshape(-1,1)  


 



# # Generate the estimation and noise samples
mean = 0  # Zero mean for the Gaussian noise
covariance = 2  # Example covariance (which is the variance in 1D)
std_deviation = np.sqrt(covariance)  # Standard deviation
num_samples = 1  # Number of samples
tolerance=1e-5
#plot
time = np.linspace(0, n*deltaT, n) 
plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(15, 5))
# Create a GridSpec layout
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 3, 1], height_ratios=[1, 1],wspace=0.4)
# First column: Two plots (one above the other)
ax0 = fig.add_subplot(gs[:, 0])  # Moving dots spanning both rows
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax2.axhline(y=P_th, color='r', linestyle=(0, (4, 3)), linewidth=2, label='$P_{th}$')
# Second column: One plot spanning both rows
# ax3 = fig.add_subplot(gs[:, 2])
# Subplot 0: Moving dots positions
dot1, = ax0.plot([], [], 'go', label='Human')  # Green dot
dot2, = ax0.plot([], [], 'bo', label='Robot')  # Blue dot
ax0.set_xlim(-5,5)
ax0.set_ylim(-10, 10)
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
ax2.set_ylim(-.001, P_th+P_th*0.005)  # Set y-axis limits from 0 to 1
# ax2.set_ylim(-.001, 1)  # Set y-axis limits from 0 to 1
ax2.set_xlabel('Time')
ax2.set_ylabel('Collision Prob.')  # Y-axis label in LaTeX
ax2.grid(True)
ax2.legend(loc='upper right')

# Subplot 3: Probability Distributions
# Set up custom colormap for probability distribution
P = np.random.rand(Prediction_Horizon, Nc.shape[0], Nc.shape[1])
# Get the shape of the data

# Find the maximum and minimum non-zero values in the entire P array
max_value = np.max(P)
min_value = np.min(P[P > 0])  # Minimum non-zero value

# Normalize the data to the range [0, 1]
P_normalized = (P - min_value) / (max_value - min_value)
P_normalized = np.clip(P_normalized, 0, 1)  # Clip negative values to 0

# Create a custom colormap with white for zeros, light blue for small values, and black for the maximum
colors = [(1, 1, 1), (0.678, 0.847, 0.902), (0, 0, 0)]  # White, light blue, black

n_bins = 10000  # Number of bins for color gradient
cmap_name = 'custom_blue_to_black'
cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Set up the figure with subplots
# plt.ion()  # Turn on interactive mode
# fig = plt.figure(figsize=(15, 5))

# Create a GridSpec layout
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 3, 1], height_ratios=[1, 1])

# Second column: Live Plot of Probability Distributions
ax3 = fig.add_subplot(gs[:, 2])


image = ax3.imshow(P_normalized[0], extent=[-5.5, 5.5, -5.5, 5.5], origin='lower',
                   cmap=cm, interpolation='nearest')
ax3.set_xlabel('$N_c$')
ax3.set_ylabel('$N_c$')
ax3.set_title('Live Probability Distribution')
ax3.grid(True)
ax3.minorticks_on()
ax3.grid(which='minor', linestyle=':', linewidth=0.5)  # Minor grid lines

cbar = plt.colorbar(image, ax=ax3, orientation='vertical', fraction=0.02, pad=0.04)

# Fourth column: Human and Robot Actions
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[1, 3])


# Human's Action Box
ax4.set_title("Human's Velocity")

# Create a rectangle around the Human's Action box
rect_human = FancyBboxPatch((0.0, 0.0), 1, 1, boxstyle="round,pad=0.1", edgecolor='black', linewidth=2)
ax4.add_patch(rect_human)

velocity_text_human = ax4.text(0.5, 0.1, '', verticalalignment='center', horizontalalignment='center',
                               fontsize=12, color='black', bbox=dict(boxstyle="round,pad=0.5", edgecolor='black', facecolor='none'))
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

# Robot's Action Box
ax5.set_title("Robot's Velocity")

# Create a rectangle around the Robot's Action box
rect_robot = FancyBboxPatch((0.0, 0.0), 1, 1, boxstyle="round,pad=0.1", edgecolor='black', linewidth=2)
ax5.add_patch(rect_robot)

velocity_text_robot = ax5.text(0.5, 0.1, '', verticalalignment='center', horizontalalignment='center',
                                fontsize=12, color='black', bbox=dict(boxstyle="round,pad=0.5", edgecolor='black', facecolor='none'))
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')

# Variable to store if the dashed line and text were already added
line_added = False

#-----------------------------------------------------------------------------------------------

flag=0

P_Coll = np.zeros((n, 1))

for i in range(n):
    constraints=[]
    x_H0=x_H[:, i][:, np.newaxis]  
    x_R0=x_R[:, i][:, np.newaxis]  
    

    #Plot
    scalar_value = P_t_all[i]
    line1.set_data(time[:i+1], P_t_all[:i+1])
    # Update the line data for the second subplot
    line2.set_data(time[:i+1], P_Coll[:i+1])
    P_xH=P_xH_all[i,:,:,:]
    P_normalized = P_xH 
    P_normalized = np.clip(P_normalized, 0, 1)  # Clip negative values to 0
    combined_P = np.mean(P_normalized, axis=0)  # Average over all prediction horizons
    # image.set_data(P_normalized[0,:,:])
    image.set_data(combined_P)
    
    # Update the black square representing the actual position on ax3
    for artist in ax3.patches:
        artist.remove()  # Remove the previous square

    # Add the new black square at the current position (x_H[0, i], x_H[1, i])
    actual_position_square = plt.Rectangle(
        (x_H[0, i]-.25 , x_H[1, i]-.25 ), .5, .5,  # Position and size of the square
         facecolor='black'
    )
    ax3.add_patch(actual_position_square)

    # Update Human's Action Circles
    human_action_value_x = u_app_H[0, i ]
    human_action_value_y = u_app_H[1, i ]

    robot_action_value_x = u_app_R[0, i ]
    robot_action_value_y = u_app_R[1, i ]

    # Update Velocity Texts
    Vx = human_action_value_x
    Vy = human_action_value_y
    norm = np.sqrt(Vx**2 + Vy**2)
    angle = np.arctan2(Vy, Vx)

    # Clear previous arrows and dots
    for artist in ax4.patches:
        artist.remove()
    
    ax4.plot([], [], 'go')  # Reset the dots in the Human's Action box


    # Draw velocity vector for human
    # start_point_human = (0.5, 0.6)
    # end_point_human = (0.5 + 0.4 * norm * np.cos(angle), 0.6 + 0.4 * norm * np.sin(angle))
    # arrow_human = FancyArrowPatch(start_point_human, end_point_human,
    #                               mutation_scale=10, arrowstyle='->', color='green', linewidth=2, transform=ax4.transAxes)
    # ax4.add_patch(arrow_human)

    start_point_human = (0.5, 0.6)
    end_point_human = (
        0.5 + 5 * float(norm) * np.cos(float(angle)), 
        0.6 + 5 * float(norm) * np.sin(float(angle))
    )

    arrow_human = FancyArrowPatch(start_point_human, end_point_human,
                                mutation_scale=10, arrowstyle='->', color='green', linewidth=2, transform=ax4.transAxes)
    ax4.add_patch(arrow_human)
    
    ax4.plot(*start_point_human, 'go')  # Dot at the start of the velocity vector
    

    # Update the velocity text
    # velocity_text = f'V = [{Vx:.2f}  {Vy:.2f}]'  # Horizontally oriented
    velocity_text = f'V = [{float(Vx):.2f}  {float(Vy):.2f}]'
    velocity_text_human.set_text(velocity_text)


    # Update Velocity Texts
    Vx = robot_action_value_x
    Vy = robot_action_value_y
    norm = np.sqrt(Vx**2 + Vy**2)/2
    angle = np.arctan2(Vy, Vx)

    # Clear previous arrows and dots

    for artist in ax5.patches:
        artist.remove()

    ax5.plot([], [], 'bo')  # Reset the dots in the Robot's Action box



    # Draw velocity vector for robot
    start_point_robot = (0.5, 0.6)
    end_point_robot = (0.5 + 5 * norm * np.cos(angle), 0.6 + 5* norm * np.sin(angle))
    arrow_robot = FancyArrowPatch(start_point_robot, end_point_robot,
                                  mutation_scale=10, arrowstyle='->', color='blue', linewidth=2, transform=ax5.transAxes)
    ax5.add_patch(arrow_robot)
    ax5.plot(*start_point_robot, 'bo')  # Dot at the start of the velocity vector

    # Update the velocity text
    velocity_text = f'V = [{Vx:.2f}  {Vy:.2f}]'  # Horizontally oriented
    velocity_text_robot.set_text(velocity_text)

    # Check if Signal is "on" and if the line hasn't been added yet

            
    dot1.set_data([x_H[0,i ]],[x_H[1,i ]])  # Example: sine wave for dot 1
    dot2.set_data([x_R[0 ,i]],[x_R[1 ,i]]) 
    ax0.relim()  # Recalculate limits for the moving dots subplot
    ax0.autoscale_view()  # Rescale the view limits for the moving dots subplot
    ax1.relim()  # Recalculate limits for the first subplot
    ax1.autoscale_view()  # Rescale the view limits for the first subplot
    ax2.relim()  # Recalculate limits for the second subplot
    ax2.autoscale_view()  # Rescale the view limits for the second subplot
    ax3.relim()  # Recalculate limits for the second subplot
    ax3.autoscale_view()  # Rescale the view limits for the second subplot
    plt.draw()  # Update the figure
    plt.pause(0.53)  # Pause to allow the plot to update
    if i>=1:
        print(np.linalg.norm(x_R[:, i] - x_H[:, i]) )
plt.ioff()  # Turn off interactive mode
plt.show()