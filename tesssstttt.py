import numpy as np

# Assuming P_x_H_k, P_u_H, and P_ts are defined
P_x_H_k = 1  # example value
P_u_H = 2     # example value
P_ts = np.random.rand(10)  # example array with 10 elements

# Initialize the 2D array with the correct dimensions
m = 0  # or whatever index you want to use
P_x_H_ik = np.zeros((1, len(P_ts)))  # Adjust dimensions as needed

# Calculate and populate the array
P_x_H_iks = []
for i in range(len(P_ts)):
    P_x_H_iks.append(P_x_H_k * P_u_H * P_ts[i])

P_x_H_ik[m, :] = np.array(P_x_H_iks)

print(P_x_H_ik)
