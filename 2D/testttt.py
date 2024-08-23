import numpy as np

# Example values for u_H and u_H_val
u_H = np.random.randn(6, 1).reshape(3, 2, 1)  # Shape (3, 2, 1), containing 3 (2, 1) arrays
u_H_val = np.random.randn(2, 1)  # Example (2, 1) array

# Function to check if two (2, 1) arrays are equal
def arrays_equal(array1, array2):
    return np.array_equal(array1, array2)
for i in range(u_H.shape[0]):  # Iterate over the 3 (2, 1) arrays
    if arrays_equal(u_H_val, u_H[i]):
# Check if u_H_val is equal to any of the (2, 1) arrays in u_H
match_found = False
for i in range(u_H.shape[0]):  # Iterate over the 3 (2, 1) arrays
    if arrays_equal(u_H_val, u_H[i]):
        match_found = True
        break

print("Match found:", match_found)
