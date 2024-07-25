import cvxpy as cp
import numpy as np

# Problem dimensions
NoI_H = 5
u_H_values = np.array([-2, -1, 0, 1, 2])  # Possible values for u_H


# Define binary variables
binary_vars = cp.Variable((NoI_H, len(u_H_values)), boolean=True)

# Define u_H using matrix multiplication
u_H = binary_vars @ u_H_values

# Objective function (simple example)
objective = cp.Minimize(cp.sum(u_H))  # Minimize the sum of u_H values

# Constraints (ensure each row selects exactly one value from u_H_values)
constraints = [
    cp.sum(binary_vars, axis=1) == 1,
]

# Define the problem
problem = cp.Problem(objective, constraints)

# Solve using CBC
try:
    problem.solve(solver=cp.CBC)
except cp.error.SolverError as e:
    print("SolverError:", e)
except cp.error.DCPError as e:
    print("DCPError:", e)

# Print results
print("Optimal u_H values:", u_H.value)

