import numpy as np
import cvxpy as cp
import mosek
env = mosek.Env()
print("MOSEK license is correctly installed.")
import os
os.environ['MOSEKLM_LICENSE_FILE'] = r'C:\Users\mohsen.amiri\mosek\mosek.lic'

# Human Model

n = 100
Prediction_Horizon = 10
deltaT=0.2
A_H = np.array([1.0])
B_H = np.array([deltaT]).reshape(-1,1)
C_H = np.eye(1)
D_H = np.zeros((1, 1))

NoI_H=B_H.shape[1]
NoS_H=A_H.shape[0]
NoO_H=C_H.shape[0]

x_H0 = np.ones((NoS_H,n))  
g_H = np.array([5.0])  
beta=np.array([1])
theta_1 = np.array([1.0])   
theta_2 = np.array([0.5])   
theta_3 = np.array([2.5])   
theta_4 = np.array([8.0*10**-3])   
theta_5 = np.array([100.0]) 
theta_6 = np.array([6.0*10**-3]) 
eta_1 = np.array([1.0])  
eta_2 = np.array([1.0]) 
x_R0 = np.array([1])  

# # Generate the estimation and noise samples
mean = 0  # Zero mean for the Gaussian noise
covariance = 2  # Example covariance (which is the variance in 1D)
std_deviation = np.sqrt(covariance)  # Standard deviation
num_samples = 1  # Number of samples


epsilon = np.random.normal(mean, std_deviation, num_samples)
hat_x_R=x_R0+epsilon  


    # Human’s action objective function
    # u_H = cp.Variable((NoI_H , 1))
u_H_values = np.array([-2, -1, 0, 1, 2])  # Possible values for u_H
    # Define binary variables
binary_vars = cp.Variable((NoI_H, len(u_H_values)), boolean=True)
    # Define u_H using matrix multiplication
u_H = binary_vars @ u_H_values




norm_x_H_g_H = cp.norm(x_H0 - g_H,'fro')**2
norm_u_H = cp.norm(u_H,'fro')**2
QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H
QH_s=theta_5*cp.exp(-theta_6*cp.norm(x_H0-hat_x_R,'fro')**2)
sigma_H = eta_1*QH_g+beta*eta_2*QH_s

 
 
objective = cp.Minimize(sigma_H)  # Minimize the sum of u_H values

    # Constraints (ensure each row selects exactly one value from u_H_values)
constraints = [  cp.sum(binary_vars, axis=1) == 1]

    # Define the problem
problem = cp.Problem(objective, constraints)

problem.solve(solver=cp.GLPK)