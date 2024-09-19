"""
Title: [2D  Safe and Efficient Human Robot Interaction via Behavior-Driven Danger Signaling]

Description:
    This script is designed to provide an efficient and probabilistically safe plan for the Human Robot Interaction.

Author:
    [Mohsen Amiri]
    
Date:
    [8/16/2024]

Version: 2D Model Predictive
    1.0.0
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch


## Modelling


#------------------------------------------
# Robot Model
n = 20
Prediction_Horizon = 1
Prediction_Horizon_H=1
deltaT=0.5

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

Signal="off" # Signal could be "on" or "off"
Human="Concerned"  # Human could be "Concerned" or "Unconcerned"

if Human=="Concerned":
    beta=1
elif Human=="Unconcerned":
    beta=0

Safe_Distance=1.5


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
g_R = np.array([[0.],[10.0]]).reshape(-1,1) 
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
theta_1 = np.array([1.0]).reshape(-1,1)   
theta_2 = np.array([0.5]).reshape(-1,1)   
theta_3 = np.array([2.5]).reshape(-1,1)   
theta_4 = np.array([8.0]).reshape(-1,1)   
theta_5 = np.array([300]).reshape(-1,1) 
theta_5 = np.array([100]).reshape(-1,1) 
theta_6 = np.array([.06]).reshape(-1,1) 

x_H = np.array([[-5.],[0.0]])*np.ones((NoS_H,n+1))
x_R = np.array([[0.],[-10.0]])*np.ones((NoS_R,n+1))  

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

def human_s_action(NoI_H, u_H_value, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R, eta_1, eta_2, beta, u_H0,Prediction_Horizon_H,Abar_H,Bbar_H,U_H_constraints):
    # Objective function to minimize
    def objective(u_H_flattened):
        u_H = u_H_flattened.reshape(NoI_R * Prediction_Horizon_H, 1)  # Reshape back to 2D for computation
        x_pr_H = Abar_H @ x_H0 + Bbar_H @ u_H
        
        # QH_g term
        norm_x_H_g_H = np.linalg.norm(x_pr_H - g_H_pr) ** 2
        norm_u_H = np.linalg.norm(u_H) ** 2
        QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H

        # QH_s term
        norm_expr = np.linalg.norm(x_pr_H - hat_x_R)
        QH_s = theta_5 * np.exp(-theta_6 * norm_expr ** 2)

        # Total sigma_H        
        sigma_H = eta_1 * QH_g + beta * eta_2 * QH_s

        return float(sigma_H)  # Ensure the returned value is a scalar

    # Constraints
    def constraint1(u_H_flattened):
        u_H = u_H_flattened.reshape(NoI_H * Prediction_Horizon_H, 1)
        return (u_H + U_H_constraints).flatten()  # Flatten for compatibility

    def constraint2(u_H_flattened):
        u_H = u_H_flattened.reshape(NoI_H * Prediction_Horizon_H, 1)
        return (U_H_constraints - u_H).flatten()  # Flatten for compatibility

    # Define constraints as a dictionary
    constraints = [{'type': 'ineq', 'fun': constraint1},
                   {'type': 'ineq', 'fun': constraint2}]
    
    # Flatten the initial guess just for the solver
    # u_H0_flattened = u_H0.flatten()
  
 
        # initial_u_R=u_app_R[:, i-1]
    u_H0_flattened=np.tile(u_H0, (Prediction_Horizon, 1)).flatten()
    # Optimize using scipy's minimize function
    solution = minimize(objective, u_H0_flattened, method='trust-constr', constraints=constraints)
    
    # Extract the optimal value of u_H and reshape it back to 2D
    optimal_u_H = solution.x.reshape(NoI_H * Prediction_Horizon_H, 1)

    # Function to compute the closest value in u_H_values to a given value
    def find_nearest(value, u_H_value):
        return u_H_value[np.argmin(np.abs(u_H_value - value))]

    # Vectorize the find_nearest function to apply it to each element in optimal_u_H
    vectorized_find_nearest = np.vectorize(lambda x: find_nearest(x, u_H_value))

    # Apply the vectorized function to each element in optimal_u_H
    rounded_optimal_u_H = vectorized_find_nearest(optimal_u_H)

    return rounded_optimal_u_H


# Human Action Prediction
def Human_Action_Prediction(NoI_H, u_H_value,u_H_values, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, betas, u_H0,Prediction_Horizon_H,Abar_H,Bbar_H,U_H_constraints):
  
    # P_di=[]
    P_d=np.zeros((u_H_values.shape[0],u_H_values.shape[1],betas.shape[0]))
    u_H_optimized=np.zeros((NoI_H,betas.shape[0]))



    for i in range(betas.shape[0]):
        
 
        u_H_optimized_all=human_s_action(NoI_H, u_H_value, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, betas[i], u_H0[:NoI_H],Prediction_Horizon_H,Abar_H,Bbar_H,U_H_constraints)
        u_H_optimized[:,i]=u_H_optimized_all[:NoI_H].reshape(NoI_H)

    
    # For each i, search for u_H_optimized(NoI_H, i) in u_H_values
    for i in range(betas.shape[0]):
        # Get the optimized value
        optimized_value = u_H_optimized[:, i].reshape(-1,1)
        # print(optimized_value)
        # Search for the matching pair in u_H_values
        for row in range(u_H_values.shape[0]):
            for col in range(u_H_values.shape[1]):
                if np.array_equal(u_H_values[row, col], optimized_value):
                    # Set corresponding element in P_d to 1
                    P_d[row, col, i] = 1

    

    
    # Human’s Random Behavior:
    U_H=u_H_values.shape[0]*u_H_values.shape[1]
    P_r=1/U_H
     
    P_u_H=(1-w_H) * P_d+w_H * P_r
    # P_u_Hi=(1-w_H) * P_di+w_H * P_r
    return P_u_H
   
def human_s_goal(u_H,x_H0,g_H,theta_3,theta_4):
        norm_x_H_g_H = np.linalg.norm(x_H0+u_H - g_H)**2
        norm_u_H = np.linalg.norm(u_H)**2
        QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H
        return QH_g
    
def human_s_safety(u_H,x_H0,hat_x_R,theta_5,theta_6):

        a=x_H0+u_H
        b=hat_x_R
        QH_s=theta_5*np.exp(-theta_6*np.linalg.norm(a-b)**2)
        return QH_s

# Robot’s Belief About the Human’s Danger Awareness
def Robot_s_Belief_About_HDA(u_H,u_H_values, betas,P_t,P_u_H):
    sum_P_P_t=0.0
    P_ti=np.zeros((betas.shape[0],1))

    # Find the index where u_H matches one of the arrays in u_H_values
    def find_index(u_H_values, u_H):
        for i in range(u_H_values.shape[0]):
            for j in range(u_H_values.shape[1]):
                if np.array_equal(u_H_values[i, j], u_H):
                    return (i, j)
        return None

    index = find_index(u_H_values, u_H)

    for i in range(betas.shape[0]):
        sum_P_P_t+=P_u_H[index[0],index[1],i]*P_t[i]
        P_ti[i]=P_u_H[index[0],index[1],i]*P_t[i]

    P_t=P_ti/sum_P_P_t

    return P_t

# Probability distribution of the human’s states
def Probability_distribution_of_human_s_states(u_H,u_app_Robot,w_H,gamma,beta,betas,P_t,g_H_pr,u_H_value,u_H_values,Prediction_Horizon,x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,hat_x_R_pr,Nc,Abar,Bbar,A_H,B_H,NoI_H,u_H0,Abar_H,Bbar_H,eta_1, eta_2):
                                            
    x_pr = Abar @ x_R0 + Bbar @ u_app_Robot
    x_pr=np.vstack((x_pr,1.0))
    P=np.zeros((Prediction_Horizon,Nc.shape[0],Nc.shape[1]))
    P_x_H=np.zeros((Nc.shape[0],Nc.shape[1]))
    x_H_next=np.zeros((u_H_values.shape[0],u_H_values.shape[1]))
    new_cell = []
    new_cells=[1]
    new_cell.append(1)
    P_x_H_iks=[]
    x_H_next=x_H0
    u_H=u_H.reshape(Prediction_Horizon,NoI_H,1)
    for j in range(Prediction_Horizon):

        sum_x_H=0.0
        if j>=1:
            tuple_list = [tuple(arr) for arr in new_cell]
            # Find unique tuples using a set
            unique_tuples = set(tuple_list)
            unique_tuples=list(unique_tuples)
            x_H0_new=[Nc[f] for f in unique_tuples]   
            new_cells=(unique_tuples)
            new_cell = []
        for n in range(len(np.array(new_cells))):
        
            if j>=1:
                x_H0_prob=x_H0_new[n]
            
            for mx in range(Nc.shape[0]):
              for my in range(Nc.shape[1]):
            
                for kx in range(u_H_values.shape[0]):
                  for ky in range(u_H_values.shape[1]):
                    if j==0:
                        x_H0_prob=x_H0
                    sds=np.array(u_H_values[kx,ky])
                    x_H_next=A_H@x_H0_prob+B_H@u_H_values[kx,ky]

                    if np.allclose(x_H_next, Nc[mx, my], atol=tolerance):
                            
                        if j==0:
                            P_x_H_k=np.array([1.0])
                        else:
                            P_x_H_k=np.array([P[j-1,np.array(new_cells[n])[0],np.array(new_cells[n])[1]]])
                       
                        
                        

                    
                        P_u_H=Human_Action_Prediction(NoI_H, u_H_value,u_H_values, x_H0_prob, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, betas, u_H0,Prediction_Horizon_H,Abar_H,Bbar_H,U_H_constraints)

                            




                        for i in range(betas.shape[0]):
                        
                            P_x_H_iks.append(P_x_H_k*P_u_H[kx,ky,i]*P_t[i])
                            if P_x_H_k*P_u_H[kx,ky,i]*P_t[i]!=0:
                                new_cell.append(np.array([mx,my]))         
                            sum_x_H+=P_x_H_k*P_u_H[kx,ky,i]*P_t[i]

                sssss=np.array(P_x_H_iks).reshape(-1,1)
                sssssst=np.sum(sssss)
                P_x_H_iks=[]
                P_x_H[mx,my]=sssssst

            if j==0:
                P_x_Hn=np.zeros((1,Nc.shape[0],Nc.shape[1]))
                P_x_Hn[:,:,:]=P_x_H 
            else:
                P_x_Hn[n,:,:]=P_x_H             
        
        if j==0:
            P[j,:,:]= P_x_H/(sum_x_H) 

            new_cell=new_cell[1:]
            P_x_Hn=np.zeros((len(np.array(new_cell)),Nc.shape[0],Nc.shape[1]))

        else:              
            PPPPP=np.sum(P_x_Hn, axis=0)   
            P[j,:,:]=PPPPP/(sum_x_H) 
            P_x_Hn=np.zeros((len(np.array(new_cell)),Nc.shape[0],Nc.shape[1]))

        epsilon = np.random.normal(mean, std_deviation, num_samples)
        hat_x_R=x_pr[j+1]+epsilon  

    np.save('P.npy', P)
    return P

#------------------------------------------------------------------------------------------
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
P_Col=[]
P_Coll=[]
P_t_app=[]
u_app_H = np.zeros((NoI_H, n))
u_app_R = np.zeros((NoI_R, n))
P_xH_all=np.zeros((n,Prediction_Horizon,Nc.shape[0],Nc.shape[1]))
P_t_all = np.zeros((n, 1))
P_Coll = np.zeros((n, 1))

for i in range(n):
    x_H0=x_H[:, i][:, np.newaxis]  
    x_R0=x_R[:, i][:, np.newaxis]  
    
    # Generate zero-mean Gaussian noise
    epsilon = np.random.normal(mean, std_deviation, num_samples)

    if i==0:
        u_app_Robot=initial_u_R        
    else:
        u_app_Robot=np.tile(u_app_R[:, i-1], Prediction_Horizon).reshape(-1,1)
        
    x_pr = Abar @ x_R0 + Bbar @ u_app_Robot    
    hat_x_R_pr=x_pr+epsilon  
    hat_x_R=x_R0+epsilon  
     
    

    if i==0:
        
        P_xH=Probability_distribution_of_human_s_states(initial_u_H,u_app_Robot,w_H,gamma,beta,betas,P_t,g_H_pr,u_H_value,u_H_values,Prediction_Horizon, x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,hat_x_R_pr,Nc,Abar,Bbar,A_H,B_H,NoI_H,initial_u_H [:NoI_H],Abar_H,Bbar_H,eta_1, eta_2)
        # np.save('P_xH_MP.npy',P_xH)
        # P_xH=np.load('P_xH_MP.npy')
        P_u_H=Human_Action_Prediction(NoI_H, u_H_value,u_H_values, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, betas, initial_u_H,Prediction_Horizon_H,Abar_H,Bbar_H,U_H_constraints)
                                    
    else:
        P_xH=Probability_distribution_of_human_s_states(u_H,u_app_Robot,w_H,gamma,beta,betas,P_t,g_H_pr,u_H_value,u_H_values,Prediction_Horizon, x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,hat_x_R_pr,Nc,Abar,Bbar,A_H,B_H,NoI_H,u_app_H[:, i-1],Abar_H,Bbar_H,eta_1, eta_2)
        P_u_H=Human_Action_Prediction(NoI_H, u_H_value,u_H_values, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, betas, u_app_H[:, i-1],Prediction_Horizon_H,Abar_H,Bbar_H,U_H_constraints)
 
    P_xH_all[i,:,:,:]=P_xH
    #Updates
    # Human’s action objective function 
    if i==0:
 
       
        u_H=human_s_action(NoI_H,u_H_value,x_H0,g_H_pr,theta_3,theta_4,theta_5,theta_6,hat_x_R_pr,eta_1,eta_2,beta,initial_u_H [:NoI_H],Prediction_Horizon,Abar_H,Bbar_H,U_H_constraints)
    else:
        u_H=human_s_action(NoI_H,u_H_value,x_H0,g_H_pr,theta_3,theta_4,theta_5,theta_6,hat_x_R_pr,eta_1,eta_2,beta,u_app_H[:, i-1],Prediction_Horizon,Abar_H,Bbar_H,U_H_constraints)
    # u_H=1.0
    u_app_H[:, i]=u_H[:NoI_H].flatten()
    x_H[:, i+1] = A_H @ x_H[:, i] + B_H @ u_app_H[:, i]
    
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
        return np.min(u_R)  # u_R >= 0

    def constraint2(u_R):
        u_R = u_R.reshape((NoI_R * Prediction_Horizon, 1))
        return 2.0-np.max(u_R)   # u_R <= 2

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

                            Cons=np.linalg.norm(Nc[indices[0,tt],indices[1,tt]] - x_pr_t[NoI_R * t:NoI_R * (t + 1) ]) - Safe_Distance
                            return Cons
                        constraints.append({'type': 'ineq', 'fun': constraint_fun})


                        # P_Col.append(np.array(0.0))
                        P_Coll[i]=np.array(0.0)

                    # elif np.linalg.norm(Nc[indices[0,tt],indices[1,tt]]-x_R[:,i])<=1. and matrix[indices[0][tt],indices[1][tt]]<=P_th and t==0 :
                    elif matrix[indices[0][tt],indices[1][tt]]<=P_th and t==0 :
                # Find the maximum value smaller than the threshold
                    
                        # P_Col.append(P_xH[0, indices[0][tt]])
                        dvd=P_xH[0, indices[0][tt],indices[1][tt]]
                        P_Coll[i]=dvd
                #print(f"Max value smaller than threshold: {P_Coll}")
                    else:
                        # P_Col.append(np.array(0.0))
                        P_Coll[i]=np.array(0.0)
        
        return constraints

    # Initial guess for the optimization variables
    if i>=1:
 
        # initial_u_R=u_app_R[:, i-1]
        initial_u_R=np.tile(u_app_R[:, i-1], (Prediction_Horizon, 1))

    # Setup constraints for `minimize`
    constraints = [{'type': 'ineq', 'fun': constraint1},
                   {'type': 'ineq', 'fun': constraint2}]
    constraints.extend(custom_constraints(initial_u_R))

    # Perform the optimization
    result = minimize(objective, initial_u_R.flatten(), constraints=constraints, method='SLSQP')

    # Get the optimized values
    optimized_u_R = result.x.reshape((NoI_R * Prediction_Horizon, 1))

    rounded_u_R = min(u_R_values.flatten(), key=lambda x: np.linalg.norm(np.array([[x]]) - optimized_u_R[:NoI_R]))
    u_app_R[:, i] = rounded_u_R[:NoI_R, 0]

    x_R[:, i+1] = A_R@ x_R[:, i] + B_R @ u_app_R[:, i]

    # print(u_app_R[:, i],u_app_H[:, i])
    P_t=Robot_s_Belief_About_HDA(u_app_H[:, i].reshape(-1,1),u_H_values ,betas,P_t,P_u_H)               
    P_t_all[i]=P_t[1]
    
    # Signal System
    if P_t_all[i]<=0.08:
        if Signal == "on":
            if Human=="Unconcerned":
                beta=1
            elif Human=="Concerned":
                beta=1

    #-------------------------------------------------------------------------------------------------------------
    #Plot
    scalar_value = P_t_all[i]
    line1.set_data(time[:i+1], P_t_all[:i+1])
    # Update the line data for the second subplot
    line2.set_data(time[:i+1], P_Coll[:i+1])
  
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
    start_point_human = (0.5, 0.6)
    end_point_human = (0.5 + 0.4 * norm * np.cos(angle), 0.6 + 0.4 * norm * np.sin(angle))
    arrow_human = FancyArrowPatch(start_point_human, end_point_human,
                                  mutation_scale=10, arrowstyle='->', color='green', linewidth=2, transform=ax4.transAxes)
    ax4.add_patch(arrow_human)
    ax4.plot(*start_point_human, 'go')  # Dot at the start of the velocity vector
    

    # Update the velocity text
    velocity_text = f'V = [{Vx:.2f}  {Vy:.2f}]'  # Horizontally oriented
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
    end_point_robot = (0.5 + 0.4 * norm * np.cos(angle), 0.6 + 0.4 * norm * np.sin(angle))
    arrow_robot = FancyArrowPatch(start_point_robot, end_point_robot,
                                  mutation_scale=10, arrowstyle='->', color='blue', linewidth=2, transform=ax5.transAxes)
    ax5.add_patch(arrow_robot)
    ax5.plot(*start_point_robot, 'bo')  # Dot at the start of the velocity vector

    # Update the velocity text
    velocity_text = f'V = [{Vx:.2f}  {Vy:.2f}]'  # Horizontally oriented
    velocity_text_robot.set_text(velocity_text)

    # Check if Signal is "on" and if the line hasn't been added yet
    if Signal == "on" and not line_added and P_t_all[i]<=0.08:
        # Plot the dashed vertical line in ax1
        ax1.axvline(x=time[i], color='g', linestyle='--', linewidth=2)
        # Add the text annotation d_R=1 with the same color as the line
        ax1.text(time[i], 0.8, '$d_R=1$', color='g', fontsize=12, ha='right')
        # Set the flag to True so the line and text are not added again
        line_added = True
            
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
    plt.pause(0.1)  # Pause to allow the plot to update
plt.ioff()  # Turn off interactive mode
plt.show()
np.save('u_app_H.npy', u_app_H)
np.save('P_t_all.npy', P_t_all)
np.save('time.npy', time)
np.save('P_xH_all.npy', P_xH_all)
np.save('P_Coll.npy', P_Coll)

