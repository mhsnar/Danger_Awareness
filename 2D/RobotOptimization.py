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
Prediction_Horizon = 1
deltaT=0.5

A_R =  np.array([[1.0, 0.],[0.,1.]])
B_R = np.array([[deltaT,0.0],[0.0,deltaT]])
C_R = np.eye(2,2)
D_R = np.zeros((2, 2))

NoI_R=B_R.shape[1]
NoS_R=A_R.shape[0]
NoO_R=C_R.shape[0]

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

#------------------------------------------
## Contorller
#INPUTS=x_H ,X_R
#Initials=
betas=np.array([0,
                1])


Signal="on" # Signal could be "on" or "off"
Human="Unconcerned"  # Human could be "Concerned" or "Unconcerned"




if Human=="Concerned":
    beta=1
elif Human=="Unconcerned":
    beta=0
P_t=np.array([.5,
                .5])


Nc=np.array([-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
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
g_R = np.array([[0.],[10.0]]).reshape(-1,1) 
g_R_pr = np.tile(g_R, (Prediction_Horizon, 1))
v_R =2.0
v_h = .5
w_H = np.array([0.2]).reshape(-1,1)  

u_H_values = np.array([-2*v_h, -1*v_h, 0, 1*v_h, 2*v_h])
X, Y = np.meshgrid(u_H_values, u_H_values)
coordinates_matrix = np.empty((u_H_values.shape[0], u_H_values.shape[0]), dtype=object)
for i in range(u_H_values.shape[0]):
    for j in range(u_H_values.shape[0]):
        coordinates_matrix[i, j] = np.array([[X[i, j]], [Y[i, j]]])
u_H_values=coordinates_matrix        


u_R_values = np.array([0, .5*v_R, v_R])
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

# # Generate the estimation and noise samples
mean = 0  # Zero mean for the Gaussian noise
covariance = 2  # Example covariance (which is the variance in 1D)
std_deviation = np.sqrt(covariance)  # Standard deviation
num_samples = 1  # Number of samples
tolerance=1e-5

flag=0
P_Col=[]
P_Coll=[]
P_t_app=[]
u_app_H = np.zeros((NoI_H, n))
u_app_R = np.zeros((NoI_R, n))

P_t_all = np.zeros((n, 1))
P_Coll = np.zeros((n, 1))

x_H0=x_H[:, 0][:, np.newaxis]  
x_R0=x_R[:, 0][:, np.newaxis]  
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
        return np.min(u_R) -2.5 # u_R >= 0

def constraint2(u_R):
        u_R = u_R.reshape((NoI_R * Prediction_Horizon, 1))
        return 3-np.max(u_R)   # u_R <= 2
P_xH=np.load('P_xH.npy')
def custom_constraints(u_R):
        u_R = u_R.reshape((NoI_R * Prediction_Horizon, 1))
        constraints = []


        for t in range(1):
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
                            return 2.7- np.linalg.norm(u_R_reshaped) 
                        constraints.append({'type': 'ineq', 'fun': constraint_fun})


                        P_Col.append(np.array(0.0))

                    elif np.linalg.norm(Nc[indices[0,tt],indices[1,tt]]-x_R[:,i])<=1. and matrix[indices[0][tt],indices[1][tt]]<=P_th and t==0 :
                # Find the maximum value smaller than the threshold
                    
                        P_Col.append(P_xH[0, indices[0][tt]])
                #print(f"Max value smaller than threshold: {P_Coll}")
                    else:
                        P_Col.append(np.array(0.0))
        
        return constraints

    # Initial guess for the optimization variable
initial_u_R = np.array([[0.],[2.]])


initial_u_R = np.tile(initial_u_R, (Prediction_Horizon, 1)).reshape(-1,)

    # Setup constraints for `minimize`
constraints = [{'type': 'ineq', 'fun': constraint1},
                   {'type': 'ineq', 'fun': constraint2}]
constraints.extend(custom_constraints(initial_u_R))

    # Perform the optimization
result = minimize(objective, initial_u_R, constraints=constraints, method='SLSQP')

    # Get the optimized values
optimized_u_R = result.x.reshape((NoI_R * Prediction_Horizon, 1))
scs=1