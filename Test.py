import numpy as np
import cvxpy as cp


## Modelling


#------------------------------------------
# Robot Model
n = 100
Prediction_Horizon = 5
deltaT=0.5

A_R = np.array([1.0])
B_R = np.array([deltaT]).reshape(-1,1)
C_R = np.eye(1)
D_R = np.zeros((1, 1))

NoI_R=B_R.shape[1]
NoS_R=A_R.shape[0]
NoO_R=C_R.shape[0]

#------------------------------------------------------------------------------------
## Robot Predictive model
Abar = A_R
if A_R.shape[0]==1:
    for i in range(2, Prediction_Horizon + 1):
        Abar = np.vstack((Abar, A_R**i))

    Bbar = np.zeros((NoS_R * Prediction_Horizon, NoI_R * Prediction_Horizon))
# Loop to fill Bbar with the appropriate blocks
    for i in range(1, Prediction_Horizon + 1):
      for j in range(1, i + 1):
        # Compute A_R^(i-j)
        A_power = A_R ** (i - j)
        
        # Compute the block (A_power * B_R), since B_R is scalar we multiply directly
        block = A_power * B_R

        # Calculate the indices for insertion
        row_indices = slice((i - 1) * NoS_R, i * NoS_R)
        col_indices = slice((j - 1) * NoI_R, j * NoI_R)

        # Insert the block into the appropriate position in Bbar
        Bbar[row_indices, col_indices] = block
else:
    Abar = np.vstack([np.linalg.matrix_power(A_R, i) for i in range(1, Prediction_Horizon+1)])
    Bbar = np.zeros((NoS_R * Prediction_Horizon, NoI_R * Prediction_Horizon))

    for i in range(1, Prediction_Horizon + 1):
        for j in range(1, i + 1):
            Bbar[(i-1)*NoS_R:i*NoS_R, (j-1)*NoI_R:j*NoI_R] = np.linalg.matrix_power(A_R, i-j) @ B_R
#------------------------------------------------------------------------------------

# print(Bbar)
# Human Model
A_H = np.array([1.0])
B_H = np.array([deltaT]).reshape(-1,1)
C_H = np.eye(1)
D_H = np.zeros((1, 1))

NoI_H=B_H.shape[1]
NoS_H=A_H.shape[0]
NoO_H=C_H.shape[0]

## Human Predictive model
Abar_H = A_H
if A_H.shape[0]==1:
    for i in range(2, Prediction_Horizon + 1):
        Abar_H = np.vstack((Abar_H, A_H**i))

    Bbar_H = np.zeros((NoS_H * Prediction_Horizon, NoI_H * Prediction_Horizon))
# Loop to fill Bbar with the appropriate blocks
    for i in range(1, Prediction_Horizon + 1):
      for j in range(1, i + 1):
        # Compute A_H^(i-j)
        A_power = A_H ** (i - j)
        
        # Compute the block (A_power * B_H), since B_H is scalar we multiply directly
        block = A_power * B_H

        # Calculate the indices for insertion
        row_indices = slice((i - 1) * NoS_H, i * NoS_H)
        col_indices = slice((j - 1) * NoI_H, j * NoI_H)

        # Insert the block into the appropriate position in Bbar
        Bbar_H[row_indices, col_indices] = block
else:
    Abar_H = np.vstack([np.linalg.matrix_power(A_H, i) for i in range(1, Prediction_Horizon+1)])
    Bbar_H = np.zeros((NoS_H * Prediction_Horizon, NoI_H * Prediction_Horizon))

    for i in range(1, Prediction_Horizon + 1):
        for j in range(1, i + 1):
            Bbar_H[(i-1)*NoS_H:i*NoS_H, (j-1)*NoI_H:j*NoI_H] = np.linalg.matrix_power(A_H, i-j) @ B_H
#------------------------------------------------------------------------------------
#------------------------------------------
## Contorller
#INPUTS=x_H ,X_R
#Initials=
betas=np.array([0,
                1])
beta=1
P_t=np.array([0.5]).reshape(-1,1)
P_ts=np.array([.5,
                .5])
P_x_H=-5
# print(betas)

# x_H = -5.0*np.ones((NoS_H,n))
Nc=np.array([-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5]).reshape(-1,1)   
x_H = 0.0*np.ones((NoS_H,n))
# Nc=np.array([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5]).reshape(-1,1)   

g_H = np.array([5.0]).reshape(-1,1)  
g_R = np.array([80.0]).reshape(-1,1) 
g_R_pr=np.tile(g_R, Prediction_Horizon) 
v_R = np.array([2.0]).reshape(-1,1)  
v_h = np.array([0.5]).reshape(-1,1)  
w_H = np.array([0.5]).reshape(-1,1)  
u_H_values = np.array([-2, -1, 0, 1, 2]).reshape(-1,1)

P_th = np.array([0.1]).reshape(-1,1)  
T_R = np.array([5.0]).reshape(-1,1)  

gamma = .001
eta_1 = 1.0
eta_2 = 1.0
theta_1 = np.array([1.0]).reshape(-1,1)   
theta_2 = np.array([0.5]).reshape(-1,1)   
theta_3 = np.array([2.5]).reshape(-1,1)   
theta_4 = np.array([8.0*10**-3]).reshape(-1,1)   
theta_5 = np.array([3.0]).reshape(-1,1) 
theta_6 = np.array([6.0*10**-3]).reshape(-1,1) 
x_R = np.array([1]).reshape(-1,1)  

# # Generate the estimation and noise samples
mean = 0  # Zero mean for the Gaussian noise
covariance = 2  # Example covariance (which is the variance in 1D)
std_deviation = np.sqrt(covariance)  # Standard deviation
num_samples = 1  # Number of samples


# # Safety objective function
# Human Action Prediction
def Human_Action_Prediction(u_H,u_H_values,w_H,gamma,beta,betas,x_H0,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6):
   QH_g=human_s_goal(u_H,x_H0,g_H,theta_3,theta_4)

   QH_s=human_s_safety(x_H0,hat_x_R,theta_5,theta_6)
   sum_P_d=0.0
   for i in range(betas.shape[0]):
      sum_P_d+=np.exp(-gamma * (QH_g + betas[i] * QH_s))

   # Human’s Deliberate Behavior
   P_d=np.exp(-gamma*(QH_g+beta*QH_s))/(sum_P_d)
   
   # Human’s Random Behavior:
   U_H=len(u_H_values)
   P_r=1/U_H
  
   P_u_H=(1-w_H) * P_d+w_H * P_r
#    print(P_u_H)
   return P_u_H
#    print(P_r)

#--------------------------------------------------

def  human_s_action(NoI_H,u_H_values,x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,eta_1,eta_2,beta):
        
        
        
        
        
        u_H = cp.Variable((NoI_H , 1), nonneg=True)
        # binary_vars = cp.Variable((NoI_H, len(u_H_values)), boolean=True)
        # u_H = binary_vars @ u_H_values

        norm_x_H_g_H = cp.norm(x_H0 - g_H,'fro')**2
        norm_u_H = cp.norm(u_H,'fro')**2
        QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H
                
        QH_s=human_s_safety(x_H0,hat_x_R,theta_5,theta_6)
        sigma_H = eta_1*QH_g+beta*eta_2*QH_s
        objective = cp.Minimize(sigma_H)  
        # Constraints (ensure each row selects exactly one value from u_H_values)
        # constraints = [  cp.sum(binary_vars, axis=1) == 1]
        constraints = [  u_H>= -2,
                       u_H<= 2]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)
        if problem.status != cp.OPTIMAL:
            flag += 1
        sss=u_H.value
        # print(sss)
        return sss
# Robot’s Belief About the Human’s Danger Awareness
def Robot_s_Belief_About_HDA(u_H,u_H_values,w_H,gamma,beta,betas,P_t,P_ts):
     sum_P_P_t=0.0
     for i in range(betas.shape[0]):
         sum_P_P_t+=Human_Action_Prediction(u_H,u_H_values,w_H,gamma,betas[i],betas,x_H0,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6)*P_ts[i]

     P_t=Human_Action_Prediction(u_H,u_H_values,w_H,gamma,beta,betas,x_H0,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6)*P_t/sum_P_P_t
    #  P_ts=np.zeros_like(P_ts)
     for i in range(betas.shape[0]):
         P_ts[i]= Human_Action_Prediction(u_H,u_H_values,w_H,gamma,betas[i],betas,x_H0,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6)*P_ts[i]/sum_P_P_t
     return P_t, P_ts

def human_s_goal(u_H,x_H0,g_H,theta_3,theta_4):



        norm_x_H_g_H = np.linalg.norm(x_H0 - g_H)**2
        norm_u_H = np.linalg.norm(u_H)**2
        QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H
        return QH_g
    
def human_s_safety(x_H0,hat_x_R,theta_5,theta_6):
        QH_s=theta_5*np.exp(-theta_6*np.linalg.norm(x_H0-hat_x_R)**2)
        return QH_s

# Probability distribution of the human’s states

tolerance=1e-5
def Probability_distribution_of_human_s_states(u_H,w_H,gamma,beta,betas,P_t,P_ts,P_x_H,u_H_values,Prediction_Horizon,x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,Nc):
    P_u_H=np.zeros((u_H_values.shape[0]*betas.shape[0])).reshape(-1,1)
    # fff=u_H_values.shape[0]*betas.shape[0]
    sum_x_H=0.0
    P=np.zeros((Prediction_Horizon,Nc.shape[0],1))
    # P[0,:,:]=1.0
    P_x_H=np.zeros((Nc.shape[0],1))
    P_x_H_ik=np.zeros((Nc.shape[0],u_H_values.shape[0]*betas.shape[0]))
    x_H_next_p=np.zeros((Prediction_Horizon,1))
    P_P=np.zeros((Prediction_Horizon,Nc.shape[0]))
    # u_H_values_flat = u_H_values.flatten()
    # u_H_values = np.tile(u_H_values_flat, Prediction_Horizon)
    # for tt in range(Prediction_Horizon):
    #     u_H_values_P=

    for j in range(Prediction_Horizon-1):

        for m in range(Nc.shape[0]):

            for k in range(u_H_values.shape[0]):
                x_H_next=A_H*x_H0+B_H*u_H_values[k]

                # x_H_next_p = Abar_H @ x_H0 + Bbar_H @ (u_H_values[k]*np.ones((Prediction_Horizon,1)))
                
                if np.allclose(x_H_next, Nc[m, 0], atol=tolerance):
                    P_x_H_k=1.0
                else:
                    P_x_H_k=0.0

                for i in range(betas.shape[0]):                
                    P_u_H=Human_Action_Prediction(u_H_values[k,0],u_H_values,w_H,gamma,betas[i],betas,x_H0,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6)
                    P_t, P_ts=Robot_s_Belief_About_HDA(u_H,u_H_values,w_H,gamma,beta,betas,P_t,P_ts)               
                    sum_x_H+=P_x_H_k*P_u_H*P_ts[i]

            for k in range(u_H_values.shape[0]): 

                x_H_next=A_H*x_H0+B_H*u_H_values[k]

                # x_H_next_p = Abar_H @ x_H0 + Bbar_H @ (u_H_values[k]*np.ones((Prediction_Horizon,1)))
                
                if np.allclose(x_H_next, Nc[m, 0], atol=tolerance):
                    P_x_H_k=1.0
                else:
                    P_x_H_k=0.0         

                for i in range(betas.shape[0]):

                    # print(u_H_values[k,0])
                    P_u_H=Human_Action_Prediction(u_H_values[k,0],u_H_values,w_H,gamma,betas[i],betas,x_H0,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6)
                    P_t, P_ts=Robot_s_Belief_About_HDA(u_H,u_H_values,w_H,gamma,beta,betas,P_t,P_ts)   
                    P_x_H_ik[m,i+k]=(P_x_H_k*P_u_H*P_t) 
                 
         
          
            P_x_H[m,:]=np.sum(P_x_H_ik[m,:]) 
            # print(P_x_H)   

        if j==0:
            P[j,:,:]= P_x_H/(sum_x_H)  
            print(P[0,:,:])
        else:   


            P[j+1,m,0]=P[j,m,0]* P_x_H[m,0]/(sum_x_H)  
    # P_P[j,m]= P[m,0]
    return P

# Probability of Collision

def Probability_of_Collision():
    P_Coll=1
    return P_Coll

flag=0
u_app_H = np.zeros((NoI_H, n))
u_app_R = np.zeros((NoI_R, n))
for i in range(n):
     
    #Updates
    x_H0=x_H[:, i].reshape(-1,1)
    x_R0=x_R[:, i].reshape(-1,1)
    # Generate zero-mean Gaussian noise
    epsilon = np.random.normal(mean, std_deviation, num_samples)
    hat_x_R=x_R0+epsilon   
    
    # Human’s action objective function
    



    


    
    u_H=human_s_action(NoI_H,u_H_values,x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,eta_1,eta_2,beta)
    u_app_H[:, i]=u_H
    x_H[:, i+1] = A_H @ x_H[:, i] + B_H @ u_app_H[:, i]
    
    # Robot’s goal objective function

    
    u_R = cp.Variable((NoI_R * Prediction_Horizon,1))
    x_pr = Abar @ x_R0 + Bbar @ u_R
    norm_u_R = cp.sum(cp.square(u_R))
    norm_x_R_g_R = cp.sum(cp.square(x_pr - g_R_pr))      
    QR_g = theta_1 * norm_x_R_g_R + theta_2 * norm_u_R
    sigma_R = QR_g
    P_xH=Probability_distribution_of_human_s_states(u_H,w_H,gamma,beta,betas,P_t,P_ts,P_x_H,u_H_values,Prediction_Horizon, x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,Nc)
    x_pr
    P_neighbour=x_H
    P_Coll=np.max(P_neighbour)
    print(np.sum(P_Coll))
    constraints_R=np.zeros_like(P_Coll)
    for i in range(P_Coll.shape[0]):
        constraints_R[i] = [ P_Coll[i] <= P_th] 
    problem = cp.Problem(cp.Minimize(sigma_R), constraints_R)
    problem.solve(solver=cp.OSQP)
    if problem.status != cp.OPTIMAL:
        flag += 1
    u_app_R[:, i] = u_R.value[:NoI_R, 0]
    x_R[:, i+1] = A_R@ x_R[:, i] + B_R @ u_app_R[:, i]
