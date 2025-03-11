#!/usr/bin/env python3
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
deltaT=7

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

# Nc=np.array([-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
Nc=np.array([-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0])
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

g_H = np.array([[-3],[0.0]])
g_H_pr = np.tile(g_H, (Prediction_Horizon_H, 1))
g_R = np.array([[0.],[-3]]).reshape(-1,1) 
g_R_pr = np.tile(g_R, (Prediction_Horizon, 1))
v_R =.3
v_h = .5/deltaT
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

 

U_H_constraint=np.array([[1/7], [1/7]]) 
initial_u_H=np.array([[0.],[0.]])
initial_u_R=np.array([[0],[0]])

U_H_constraints=np.tile(U_H_constraint, (Prediction_Horizon, 1))
initial_u_H=np.tile(initial_u_H, (Prediction_Horizon, 1))
initial_u_R=np.tile(initial_u_R, (Prediction_Horizon, 1))



theta_1 = np.array([1.0]).reshape(-1, 1)
theta_2 = np.array([0.5]).reshape(-1, 1)
P_th = np.array([0.1]).reshape(-1, 1)

P_xH=np.load('P_xH_all.npy')

P_xH=P_xH[6,:,:,:]
x_R0 = np.array([[-1.], [2.0]])  # Initial state
P_Col=[]
#-----------------------------------------------------------------------------------------------
 # Robot’s goal objective function
def objective(u_R):
    u_R = u_R.reshape((NoI_R * Prediction_Horizon, 1))
    x_pr = Abar @ x_R0 + Bbar @ u_R
    norm_u_R = np.sum(np.square(u_R))
    norm_x_R_g_R = np.sum(np.square(x_pr - g_R_pr))
    QR_g = theta_1 * norm_x_R_g_R + theta_2 * norm_u_R
    return QR_g

# Constraints
def constraint1(u_R):
    return np.min(u_R) + .30

def constraint2(u_R):
    return .30 - np.max(u_R)

def constraint3(u_R):
    x_pr = Abar @ x_R0 + Bbar @ u_R.reshape((NoI_R * Prediction_Horizon, 1))
    return x_pr[0]

# Custom constraints based on P_xH
def custom_constraints(u_R):
    constraints = []  # Initialize the constraints list
    for t in range(P_xH.shape[0]):  # Iterate over time steps
        matrix = P_xH[t, :, :]  # Human existence probability at time t
        if np.any(matrix > 0.0):
            indices = np.where(matrix > 0.0)  # Indices where the probability is non-zero
            indices = np.array(indices)
            for tt in range(indices.shape[1]):
                if matrix[indices[0][tt], indices[1][tt]] > P_th:
                    # Define a constraint function for each grid point
                    def constraint_fun(u_R, t=t, tt=tt):
                        u_R = u_R.reshape((NoI_R * Prediction_Horizon, 1))
                        x_pr_t = Abar @ x_R0 + Bbar @ u_R  # Predicted robot position at time t
                        # Calculate the distance to the grid point
                        grid_point = Nc[indices[0][tt], indices[1][tt]]
                        # Calculate distance to this grid point and add constraint
                        Cons = np.linalg.norm(grid_point - x_pr_t[NoI_R * t:NoI_R * (t + 1)]) - Safe_Distance
                        return Cons

                    # Append this constraint to the constraints list
                    constraints.append({'type': 'ineq', 'fun': constraint_fun})
                else:
                    P_Col.append(np.array(0.0))  # Store 0.0 if constraint isn't applied
    return constraints


# Initial guess for optimization
initial_u_R = initial_u_R
constraints = [{'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2}]

if x_R0[1] <= -50:
    constraints.append({'type': 'eq', 'fun': constraint3})

# Add the custom constraints for human existence probability
constraints += custom_constraints(initial_u_R)
# Perform the optimization
result = minimize(objective, initial_u_R.flatten(), constraints=constraints)

# Store the optimized values
optimized_u_R = result.x
rounded_u_R = optimized_u_R[:NoI_R]
s=1