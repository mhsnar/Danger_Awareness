import numpy as np

def generate_estimation(x_R, mean, covariance, num_samples):
    # Generate zero-mean Gaussian noise
    epsilon = np.random.multivariate_normal(mean, covariance, num_samples)
    
    # Perform the estimation
    x_hat_R = x_R + epsilon
    return x_hat_R, epsilon

# Parameters
n_R = 1  # Dimensionality of the random variable
mean = np.zeros(n_R)  # Zero mean
covariance = np.array([[1]])  # Example covariance matrix
num_samples = 1000  # Number of samples
x_R = np.array([1])  # Example state vector

# Generate the estimation and noise samples
x_hat_R, epsilon = generate_estimation(x_R, mean, covariance, num_samples)

# Print the first 10 estimations and noise samples
print("First 10 estimations (x_hat_R):")
print(x_hat_R[:10])
print("First 10 noise samples (epsilon):")
print(epsilon[:10])

# Optionally, you can plot the samples if you want to visualize them (for a specific dimension)
import matplotlib.pyplot as plt

plt.scatter(epsilon[:, 0], epsilon[:, 1], alpha=0.6, color='g')
plt.title("Scatter plot of Gaussian noise samples (ε(t))")
plt.xlabel("ε(t)[0]")
plt.ylabel("ε(t)[1]")
plt.axis('equal')
plt.show()


#INPUTS=x_H ,X_R

# #------------------------------------------
# ## Human Model
# x_H = np.array([1, 2, 3])  
# g_H = np.array([0, 0, 0])  
# gamma=100
# beta=1

# # Human’s goal objective function
# theta_3 = 1  
# theta_4 = 2  
# u_H = np.array([1, 1, 1])  
# norm_x_H_g_H = np.linalg.norm(x_H - g_H)**2
# norm_u_H = np.linalg.norm(u_H)**2
# QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H

# # Safety objective function
# hat_x_R=1
# theta_5=1
# theta_6=1
# QH_s=theta_5*np.exp(theta_6*np.linalg.norm(x_H-hat_x_R)**2)

# # Human’s Deliberate Behavior

# def Human_s_Deliberate_Behavior(gamma,beta,QH_g,QH_s):

#   P_d=np.exp(-gamma*(QH_g+beta*QH_s))
#   return P_d
#   print(P_d)
# # Human’s Random Behavior:
# u_H=np.array([1, 2])
# def Human_s_Random_Behavior(u_H):
#  U_H=len(u_H)
#  P_r=1/U_H
#  return P_r


# w_H=1


# # Human Action Prediction
# def Human_Action_Prediction(w_H,P_d,P_r,gamma,QH_g,QH_s,beta):
   
#    # Human’s Deliberate Behavior
#    P_d=np.exp(-gamma*(QH_g+beta*QH_s))
   
#    # Human’s Random Behavior:
#    U_H=len(u_H)
#    P_r=1/U_H

#    P=(1-w_H) * P_d+w_H * P_r
#    return P
#    print(P_r)

# #--------------------------------------------------
# # Robot’s Belief About the Human’s Danger Awareness
# P_beta_1=P(beta=1)
# P_t_beta_1=P_t(beta=1)

# P_t=(P_beta_1*P_t_beta_1)/(np.sum(P*P_t))
# # Probability of Collision
# P_Coll=