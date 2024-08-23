import numpy as np

# Define parameters
v_h = 0.5
w_H = np.array([0.2]).reshape(-1, 1)

# Define u_H_values
u_H_values = np.array([-2*v_h, -1*v_h, 0, 1*v_h, 2*v_h])

# Define optimal_u_H with shape (6, 1) - replace with actual values
optimal_u_H = np.random.randn(6, 1)  # Example shape (6, 1)

# Function to compute the closest value in u_H_values to a given value
def find_nearest(value, u_H_values):
    return u_H_values[np.argmin(np.abs(u_H_values - value))]

# Vectorize the find_nearest function to apply it to each element in optimal_u_H
vectorized_find_nearest = np.vectorize(lambda x: find_nearest(x, u_H_values))

# Apply the vectorized function to each element in optimal_u_H
rounded_optimal_u_H = vectorized_find_nearest(optimal_u_H)

print("Original optimal_u_H:\n", optimal_u_H)
print("Rounded optimal_u_H:\n", rounded_optimal_u_H)
