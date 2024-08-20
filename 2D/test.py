import numpy as np
from scipy.optimize import minimize

# Example setup
v_h = 0.5
u_H_values = np.array([-2*v_h, -1*v_h, 0, 1*v_h, 2*v_h])
X, Y = np.meshgrid(u_H_values, u_H_values)
coordinates_matrix = np.empty((u_H_values.shape[0], u_H_values.shape[0]), dtype=object)

for i in range(u_H_values.shape[0]):
    for j in range(u_H_values.shape[0]):
        coordinates_matrix[i, j] = np.array([[X[i, j]], [Y[i, j]]])
u_H_values = coordinates_matrix

# Objective function example
def objective(u_H):
    u_H = u_H.reshape(2,1)
    # Your objective function implementation here
    return np.sum(u_H)  # Dummy example

# Constraints example
def constraint1(u_H):
    u_H = u_H.reshape(2,1)
    u_H_flat = u_H.flatten()
    u_H_values_flat = np.array([item.flatten() for sublist in u_H_values for item in sublist])
    min_value = np.min(u_H_values_flat)
    return np.min(u_H_flat) - min_value  # Example constraint function

# Initial guess: flattened version of a 5x5 grid
initial_guess = np.zeros((2,1)).flatten()

# Run the optimization
constraints = {'type': 'ineq', 'fun': constraint1}
solution = minimize(objective, initial_guess, method='SLSQP', constraints=constraints)

# Reshape the result back to the original shape
u_H_optimized = solution.x.reshape(u_H_values.shape[0], u_H_values.shape[0], 2)
