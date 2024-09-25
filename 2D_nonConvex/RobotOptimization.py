
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch




#------------------------------------------
# Robot Model
n = 20
Prediction_Horizon = 2
Prediction_Horizon_H=2

Safe_Distance=2
deltaT=0.5

Signal="off" # Signal could be "on" or "off"
Human="Concerned"  # Human could be "Concerned" or "Unconcerned"



A_R =  np.array([[1.0, 0.],[0.,1.]])
B_R = np.array([[deltaT,0.0],[0.0,deltaT]])
C_R = np.eye(2,2)
D_R = np.zeros((2, 2))

NoI_R=B_R.shape[1]
NoS_R=A_R.shape[0]
NoO_R=C_R.shape[0]

# Human Model
A_H = np.array([[1.0, 0.],
                [0.,1.]])
B_H = np.array([[deltaT,0.],
               [0.,deltaT]])
C_H = np.eye(2,2)
D_H = np.zeros((2, 2))

NoI_H=B_H.shape[1]
NoS_H=A_H.shape[0]
NoO_H=C_H.shape[0]

#------------------------------------------------------------------------------------
## Robot Predictive model
Abar = A_R
if A_R.shape[0]==1:
    for i in range(2, Prediction_Horizon + 1):
        Abar = np.vstack((Abar, A_R**i))

    Bbar = np.zeros((NoS_R * Prediction_Horizon, NoI_R * Prediction_Horizon))
# Loop to fill Bbar with the appropriate blocks
    for i in range(1, Prediction_Horizon + 1):
      for j in range(1, i + 1):
        # Compute A_R^(i-j)
        A_power = A_R ** (i - j)
        
        # Compute the block (A_power * B_R), since B_R is scalar we multiply directly
        block = A_power * B_R

        # Calculate the indices for insertion
        row_indices = slice((i - 1) * NoS_R, i * NoS_R)
        col_indices = slice((j - 1) * NoI_R, j * NoI_R)

        # Insert the block into the appropriate position in Bbar
        Bbar[row_indices, col_indices] = block
else:
    Abar = np.vstack([np.linalg.matrix_power(A_R, i) for i in range(1, Prediction_Horizon+1)])
    Bbar = np.zeros((NoS_R * Prediction_Horizon, NoI_R * Prediction_Horizon))

    for i in range(1, Prediction_Horizon + 1):
        for j in range(1, i + 1):
            Bbar[(i-1)*NoS_R:i*NoS_R, (j-1)*NoI_R:j*NoI_R] = np.linalg.matrix_power(A_R, i-j) @ B_R
#------------------------------------------------------------------------------------
## Human Predictive model
Abar_H = A_H
if A_H.shape[0]==1:
    for i in range(2, Prediction_Horizon_H + 1):
        Abar_H = np.vstack((Abar_H, A_H**i))

    Bbar_H = np.zeros((NoS_H * Prediction_Horizon_H, NoI_H * Prediction_Horizon_H))
# Loop to fill Bbar_H with the appropriate blocks
    for i in range(1, Prediction_Horizon_H + 1):
      for j in range(1, i + 1):
        # Compute A_H^(i-j)
        A_power = A_H ** (i - j)
        
        # Compute the block (A_power * B_H), since B_H is scalar we multiply directly
        block = A_power * B_H

        # Calculate the indices for insertion
        row_indices = slice((i - 1) * NoS_H, i * NoS_H)
        col_indices = slice((j - 1) * NoI_H, j * NoI_H)

        # Insert the block into the appropriate position in Bbar_H
        Bbar_H[row_indices, col_indices] = block
else:
    Abar_H = np.vstack([np.linalg.matrix_power(A_H, i) for i in range(1, Prediction_Horizon_H+1)])
    Bbar_H = np.zeros((NoS_H * Prediction_Horizon_H, NoI_H * Prediction_Horizon_H))

    for i in range(1, Prediction_Horizon_H + 1):
        for j in range(1, i + 1):
            Bbar_H[(i-1)*NoS_H:i*NoS_H, (j-1)*NoI_H:j*NoI_H] = np.linalg.matrix_power(A_H, i-j) @ B_H
#------------------------------------------------------------------------------------



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

Nc=np.array([-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
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

g_H = np.array([[5.],[0.0]])
g_H_pr = np.tile(g_H, (Prediction_Horizon_H, 1))
g_R = np.array([[0.],[6.0]]).reshape(-1,1) 
g_R_pr = np.tile(g_R, (Prediction_Horizon, 1))
v_R =2.0
v_h = .5
w_H = np.array([0.2]).reshape(-1,1)  

u_H_values = np.array([-2*v_h, -1*v_h, 0, 1*v_h, 2*v_h])
u_H_value = np.array([-2*v_h, -1*v_h, 0, 1*v_h, 2*v_h])
X, Y = np.meshgrid(u_H_values, u_H_values)
coordinates_matrix = np.empty((u_H_values.shape[0], u_H_values.shape[0]), dtype=object)
for i in range(u_H_values.shape[0]):
    for j in range(u_H_values.shape[0]):
        coordinates_matrix[i, j] = np.array([[X[i, j]], [Y[i, j]]])
u_H_values=coordinates_matrix        

# u_R_values = np.array([0, .5*v_R, v_R])
u_R_values = np.array([- v_R, -.5*v_R, 0, .5*v_R, v_R])
X, Y = np.meshgrid(u_R_values, u_R_values)

# Combine X and Y into a 2D coordinate matrix
# Flatten X and Y to create 2D matrices
coordinates_matrix = np.empty((u_R_values.shape[0], u_R_values.shape[0]), dtype=object)

for i in range(u_R_values.shape[0]):
    for j in range(u_R_values.shape[0]):
        coordinates_matrix[i, j] = np.array([[X[i, j]], [Y[i, j]]])
u_R_values=coordinates_matrix        


P_th = np.array([0.1]).reshape(-1,1)  
T_R = np.array([5.0]).reshape(-1,1)  

gamma = 1
eta_1 = 1.0
eta_2 = 1.
theta_1 = np.array([4]).reshape(-1,1)   
theta_2 = np.array([.5]).reshape(-1,1)   
theta_3 = np.array([2.5]).reshape(-1,1)   
theta_4 = np.array([8.0]).reshape(-1,1)   
theta_5 = np.array([300]).reshape(-1,1) 
theta_5 = np.array([100]).reshape(-1,1) 
theta_6 = np.array([.06]).reshape(-1,1) 

 

U_H_constraint=np.array([[1], [1]]) 
initial_u_H=np.array([[0.],[0.]])
initial_u_R=np.array([[2.],[2.]])

U_H_constraints=np.tile(U_H_constraint, (Prediction_Horizon, 1))
initial_u_H=np.tile(initial_u_H, (Prediction_Horizon, 1))
initial_u_R=np.tile(initial_u_R, (Prediction_Horizon, 1))


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
# Define problem parameters
g_R = np.array([[0.], [10.0]]).reshape(-1, 1)
g_R_pr = np.tile(g_R, (Prediction_Horizon, 1))

theta_1 = np.array([1.0]).reshape(-1, 1)
theta_2 = np.array([0.5]).reshape(-1, 1)
P_th = np.array([0.1]).reshape(-1, 1)
P_xH=np.load('P_xH_all.npy')
P_xH=P_xH[3,:,:,:]
x_R0 = np.array([[0.], [-10.0]])  # Initial state
P_Col=[]
#-----------------------------------------------------------------------------------------------
 # Robot’s goal objective function
def objective(u_R):
    u_R = u_R.reshape((NoI_R * Prediction_Horizon, 1))
    x_pr = Abar @ x_R0 + Bbar @ u_R
    norm_u_R = np.sum(np.square(u_R))
    norm_x_R_g_R = np.sum(np.square(x_pr - g_R_pr))
    QR_g = theta_1 * norm_x_R_g_R + theta_2 * norm_u_R
    return QR_g[0]

# Define constraints
def constraint1(u_R):
    u_R = u_R.reshape((NoI_R * Prediction_Horizon, 1))
    return np.min(u_R) +2. # u_R >= 0

def constraint2(u_R):
    u_R = u_R.reshape((NoI_R * Prediction_Horizon, 1))
    return .50-np.max(u_R)   # u_R <= 2

def custom_constraints(u_R):
    u_R = u_R.reshape((NoI_R * Prediction_Horizon, 1))
    constraints = []


    for t in range(P_xH.shape[0]):
    # Get the current 2D slice
        matrix = P_xH[t, :, :]

    # Check if any value exceeds the threshold
        if np.any(matrix > 0.0):
        # Find indices where the condition is true
            indices = np.where(matrix > 0.0)
    
        # Use the first pair of indices for demonstration purposes
            # m, b = indices[0][0], indices[1][0]
            
            indices=np.array(indices)
            for tt in range(indices.shape[1]):# Check the constraint on x_pr

                # if np.linalg.norm(Nc[indices[0,tt],indices[1,tt]]-x_R[:,i])>1. and matrix[indices[0][tt],indices[1][tt]]>P_th:
                if matrix[indices[0][tt],indices[1][tt]]>P_th: 
                                
                                            
                    
                    def constraint_fun(u_R):
                        u_R_reshaped = u_R.reshape((NoI_R * Prediction_Horizon, 1))
                        x_pr_t = Abar @ x_R0 + Bbar @ u_R_reshaped
                        # Cons=np.linalg.norm(Nc[indices[0,tt],indices[1,tt]] - x_pr_t[NoI_R * (t+1)-NoI_R:NoI_R * (t+1) - 1]) - Safe_Distance
                        sc=Nc[indices[0,tt],indices[1,tt]] 
                        scssc=NoI_R * t
                        dvdvdvb=NoI_R * (t + 1)
                        scsv=x_pr_t[NoI_R * t:NoI_R * (t + 1) ]
                        svs=sc-scsv
                        Cons=1-np.linalg.norm(u_R)
                        return Cons
                    constraints.append({'type': 'ineq', 'fun': constraint_fun})


                    # P_Col.append(np.array(0.0))
                    P_Col.append(np.array(0.0))

                # elif np.linalg.norm(Nc[indices[0,tt],indices[1,tt]]-x_R[:,i])<=1. and matrix[indices[0][tt],indices[1][tt]]<=P_th and t==0 :
                elif matrix[indices[0][tt],indices[1][tt]]<=P_th and t==0 :
            # Find the maximum value smaller than the threshold
                    dvd=P_xH[0, indices[0][tt],indices[1][tt]]

                    P_Col.append(dvd)
            
            #print(f"Max value smaller than threshold: {P_Coll}")
                else:
                    P_Col.append(np.array(0.0))
    
    return constraints

# Initial guess for the optimization variables

initial_u_R=np.tile(np.array([[0],[2]]), (Prediction_Horizon, 1))

# Setup constraints for `minimize`
constraints = [{'type': 'ineq', 'fun': constraint1},
                {'type': 'ineq', 'fun': constraint2}]
constraints.extend(custom_constraints(initial_u_R))

# Perform the optimization
result = minimize(objective, initial_u_R.flatten(), constraints=constraints, method='SLSQP')

# Get the optimized values
# print(result.fun)
optimized_u_R = result.x

# rounded_u_R = min(u_R_values.flatten(), key=lambda x: np.linalg.norm(np.array([[x]]) - optimized_u_R[:NoI_R]))
rounded_u_R=optimized_u_R [:NoI_R]
s=1