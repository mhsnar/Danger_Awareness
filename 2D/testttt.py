
import numpy as np
v_h=.5
NoI_H=1
betas=np.array([0,
                1])

u_H_values = np.array([-2*v_h, -1*v_h, 0, 1*v_h, 2*v_h])
u_H_value = np.array([-2*v_h, -1*v_h, 0, 1*v_h, 2*v_h])
X, Y = np.meshgrid(u_H_values, u_H_values)

coordinates_matrix = np.empty((u_H_values.shape[0], u_H_values.shape[0]), dtype=object)
for i in range(u_H_values.shape[0]):
    for j in range(u_H_values.shape[0]):
        coordinates_matrix[i, j] = np.array([[X[i, j]], [Y[i, j]]])
u_H_values=coordinates_matrix    
P_d=np.zeros((u_H_values.shape[0],u_H_values.shape[1],betas.shape[0]))
u_H_optimized=np.array([[-1,-1],[0,1]])

# Assume u_H_optimized is already defined with shape (NoI_H, 2)

    
    # For each i, search for u_H_optimized(NoI_H, i) in u_H_values
for i in range(betas.shape[0]):
        # Get the optimized value
        optimized_value = u_H_optimized[:, i].reshape(-1,1)
        print(optimized_value)
        # Search for the matching pair in u_H_values
        for row in range(u_H_values.shape[0]):
            for col in range(u_H_values.shape[1]):
                ddv=u_H_values[row, col]
                if np.array_equal(u_H_values[row, col], optimized_value):
                    # Set corresponding element in P_d to 1
                    P_d[row, col, i] = 1

sc=P_d
print(sc)
