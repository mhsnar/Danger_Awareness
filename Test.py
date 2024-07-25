import numpy as np
import cvxpy as cp


## Modelling


#------------------------------------------
# Robot Model
n = 100
Prediction_Horizon = 10

A_R = np.array([1.0])
B_R = np.array([1.0]).reshape(-1,1)
C_R = np.eye(1)
D_R = np.zeros((1, 1))

NoI_R=B_R.shape[1]
NoS_R=A_R.shape[0]
NoO_R=C_R.shape[0]
#------------------------------------------
# Human Model
A_H = np.array([1.0])
B_H = np.array([1.0]).reshape(-1,1)
C_H = np.eye(1)
D_H = np.zeros((1, 1))

NoI_H=B_H.shape[1]
NoS_H=A_H.shape[0]
NoO_H=C_H.shape[0]
#------------------------------------------
## Contorller
#INPUTS=x_H ,X_R
#Initials=
betas=np.array([0,
                1])
beta=1
P_t=np.array([0.5])
P_ts=np.array([0,
                1])
P_x_H=1
print(betas)

x_H = np.array([1.0])  
g_H = np.array([5.0])  
g_R = np.array([80.0])  
v_R = np.array([2.0])  
v_h = np.array([0.5])  
w_H = np.array([0.5])  

P_th = np.array([0.1])  
T_R = np.array([5.0])  

gamma = np.array([100.0])  
eta_1 = np.array([1.0])  
eta_2 = np.array([1.0]) 
theta_1 = np.array([1.0])   
theta_2 = np.array([0.5])   
theta_3 = np.array([2.5])   
theta_4 = np.array([8.0*10**-3])   
theta_5 = np.array([100.0]) 
theta_6 = np.array([6.0*10**-3]) 
x_R = np.array([1])  

# # Generate the estimation and noise samples
mean = 0  # Zero mean for the Gaussian noise
covariance = 2  # Example covariance (which is the variance in 1D)
std_deviation = np.sqrt(covariance)  # Standard deviation
num_samples = 1  # Number of samples


# # Safety objective function
# Human Action Prediction
def Human_Action_Prediction(w_H,gamma,QH_g,QH_s,beta,betas):
   
   for i in range(betas.shape[0]):
      sum_P_d+=np.exp(-gamma * (QH_g + betas[i] * QH_s))

   # Human’s Deliberate Behavior
   P_d=np.exp(-gamma*(QH_g+beta*QH_s))/(sum_P_d)
   
   # Human’s Random Behavior:
   U_H=len(u_H)
   P_r=1/U_H
  
   P_u_H=(1-w_H) * P_d+w_H * P_r
   return P_u_H
   print(P_r)

#--------------------------------------------------
# Robot’s Belief About the Human’s Danger Awareness
def Robot_s_Belief_About_HDA(w_H,gamma,QH_g,QH_s,beta,betas,P_t,P_ts):
     sum_P_P_t=0.0
     for i in range(betas.shape[0]):
         sum_P_P_t+=Human_Action_Prediction(w_H,gamma,QH_g,QH_s,betas[i],betas)*P_ts[i]

     P_t=Human_Action_Prediction(w_H,gamma,QH_g,QH_s,beta,betas)*P_t/sum_P_P_t
     P_ts=np.zeros_like(P_ts)
     for i in range(betas.shape[0]):
         P_ts[i]= Human_Action_Prediction(w_H,gamma,QH_g,QH_s,beta[i],betas)*P_ts[i]/sum_P_P_t
     return P_t, P_ts



# Probability distribution of the human’s states
model=1
def Probability_distribution_of_human_s_states(w_H,gamma,QH_g,QH_s,beta,betas,P_t,P_ts,P_x_H):
    for i in range(betas.shape[0]):

        if model==1:
            P_x_H_k=1
        else:
            P_x_H_k=0
        sum_x_H+=P_x_H_k*Human_Action_Prediction(w_H,gamma,QH_g,QH_s,beta[i],betas)*P_ts[i]

    P_x_H=(P_x_H*Human_Action_Prediction(w_H,gamma,QH_g,QH_s,beta,betas)*P_t)/(sum_x_H)
    
    return P_x_H

# Probability of Collision

def Probability_of_Collision():
    P_Coll=1
    return P_Coll

flag=0
u_app_H=[]
u_app_R=[]
for i in range(n):

    #Updates
    # Generate zero-mean Gaussian noise
    epsilon = np.random.normal(mean, std_deviation, num_samples)
    hat_x_R=x_R+epsilon   

    # Human’s action objective function
    u_H = cp.Variable((NoI_H , 1))
    norm_x_H_g_H = np.linalg.norm(x_H - g_H)**2
    norm_u_H = np.linalg.norm(u_H)**2
    QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H
    QH_s=theta_5*np.exp(theta_6*np.linalg.norm(x_H-hat_x_R)**2)
    sigma_H = QH_s
    problem = cp.Problem(cp.Minimize(sigma_H))
    problem.solve(solver=cp.OSQP)
    if problem.status != cp.OPTIMAL:
        flag += 1
    u_app_H[:, i] = u_R.value[:NoI_R, 0]
    x_H[:, i+1] = A_H @ x_H[:, i] + B_H @ u_app_H[:, i]
    
    # Robot’s goal objective function
    u_R = cp.Variable((NoI_R , 1))
    norm_x_R_g_R = np.linalg.norm(x_R - g_R)**2
    norm_u_R = np.linalg.norm(u_R)**2
    QR_g = theta_1 * norm_x_R_g_R + theta_4 * norm_u_R
    sigma_R = QR_g
    constraints_R=np.zeros_like(Probability_of_Collision())
    for i in range(Probability_of_Collision().shape[0]):
        constraints_R[i] = [ P_Coll[i] <= P_th] 
    problem = cp.Problem(cp.Minimize(sigma_R), constraints_R)
    problem.solve(solver=cp.OSQP)
    if problem.status != cp.OPTIMAL:
        flag += 1
    u_app_R[:, i] = u_R.value[:NoI_R, 0]
    x_R[:, i+1] = A_R@ x_R[:, i] + B_R @ u_app_R[:, i]
