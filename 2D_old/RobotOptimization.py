import numpy as np
import cvxpy as cp

# Robot Model
n = 20
Prediction_Horizon = 1
deltaT = 0.5

A_R = np.array([[1.0, 0.], [0., 1.]])
B_R = np.array([[deltaT, 0.0], [0.0, deltaT]])
C_R = np.eye(2, 2)
D_R = np.zeros((2, 2))

NoI_R = B_R.shape[1]
NoS_R = A_R.shape[0]
NoO_R = C_R.shape[0]

# Robot Predictive model
Abar = np.vstack([np.linalg.matrix_power(A_R, i) for i in range(1, Prediction_Horizon + 1)])
Bbar = np.zeros((NoS_R * Prediction_Horizon, NoI_R * Prediction_Horizon))

for i in range(1, Prediction_Horizon + 1):
    for j in range(1, i + 1):
        Bbar[(i-1)*NoS_R:i*NoS_R, (j-1)*NoI_R:j*NoI_R] = np.linalg.matrix_power(A_R, i-j) @ B_R

# Define problem parameters
g_R = np.array([[0.], [10.0]]).reshape(-1, 1)
g_R_pr = np.tile(g_R, (Prediction_Horizon, 1))

theta_1 = np.array([1.0]).reshape(-1, 1)
theta_2 = np.array([0.5]).reshape(-1, 1)
P_th = np.array([0.1]).reshape(-1, 1)

x_R0 = np.array([[0.], [-10.0]])  # Initial state

# Define CVXPY variable
u_R = cp.Variable((NoI_R * Prediction_Horizon, 1))
tvarialbe = cp.Variable()

# Define objective function (same as before)
x_pr = Abar @ x_R0 + Bbar @ u_R
norm_x_R_g_R = cp.sum_squares(x_pr - g_R_pr)
norm_u_R = cp.sum_squares(u_R)
QR_g = theta_1 * norm_x_R_g_R + theta_2 * norm_u_R

# Objective: minimize cost function
objective = cp.Minimize(QR_g)

# Define the constraints
constraints = []

# Constraint 1: u_R >= -2.5
constraints.append(u_R >= 2.5)
constraints.append(tvarialbe >= 2.7)    
# Constraint 2: u_R <= 3
constraints.append(u_R <= 3)

# Load P_xH
P_xH = np.load('P_xH.npy')  # Assuming this file is available

for t in range(1):
    matrix = P_xH[t, :, :]

    # Find indices where the value exceeds P_th
    indices = np.where(matrix > P_th)

    for tt in range(indices[0].shape[0]):
        # Add constraint based on those indices
        m, b = indices[0][tt], indices[1][tt]
        # Constraint: Distance constraint (example based on the original logic)
        # constraints.append(cp.norm(u_R) <= tvarialbe)

# Solve the problem using CVXPY
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.ECOS)

# Get the optimized u_R values
optimized_u_R = u_R.value
print("Optimized u_R:", optimized_u_R)
print(np.linalg.norm(optimized_u_R))
ssc=1
