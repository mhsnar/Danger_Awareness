import numpy as np
from scipy.optimize import minimize

# Parameters
NoI_H = 1
beta = 1.0
eta_1 = 1.0
eta_2 = 1000000.0
theta_1 = np.array([1.0]).reshape(-1, 1)
theta_2 = np.array([0.5]).reshape(-1, 1)
theta_3 = 1.0  # as set in the last part
theta_4 = 8.0
theta_5 = 300.0
theta_6 = 0.006
g_H = 10.0
g_R = 80.0

v_R = 2.0
v_h = 0.5
u_H_values = np.array([-2 * v_h, -1 * v_h, 0, 1 * v_h, 2 * v_h]).reshape(-1, 1)
P_th = 0.1
T_R = 5.0
x_H0 = -5.0
g_H = 10.0

hat_x_R = np.array([-5.0]).reshape(-1,1)

# Objective function to minimize
def objective(u_H):
    u_H = np.array(u_H).reshape(-1, 1)
    
    # QH_g term
    norm_x_H_g_H = np.linalg.norm(x_H0 + u_H - g_H) ** 2
    norm_u_H = np.linalg.norm(u_H) ** 2
    QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H
    
    # QH_s term
    a = np.hstack([x_H0 + u_H, np.zeros((NoI_H, 1))])  # [x_H0 + u_H, 0]
    b = np.hstack([np.zeros((NoI_H, 1)), hat_x_R])  # [0.0, hat_x_R]
    norm_expr = np.linalg.norm(a - b)
    QH_s = theta_5 * np.exp(-theta_6 * norm_expr ** 2)
    
    # Total sigma_H
    sigma_H = eta_1 * QH_g + beta * eta_2 * QH_s
    return sigma_H.item()

# Constraints
def constraint1(u_H):
    return u_H - np.min(u_H_values)

def constraint2(u_H):
    return np.max(u_H_values) - u_H

# Initial guess
u_H0 = np.array([0.0])

# Define constraints as a dictionary
constraints = [{'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2}]

# Optimize using scipy's minimize function
solution = minimize(objective, u_H0, method='SLSQP', constraints=constraints)

# Extract the optimal value of u_H
optimal_u_H = solution.x

print("Optimal u_H:", optimal_u_H)
