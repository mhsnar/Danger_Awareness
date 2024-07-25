import numpy as np
import cvxpy as cp


# Model
n = 100
Prediction_Horizon = 10


A = np.array([1.0])

B = np.array([1.0]).reshape(-1,1)

C = np.eye(1)
D = np.zeros((1, 1))

NoI=B.shape[1]
NoS=A.shape[0]
NoO=C.shape[0]

# Contorller
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

x_H = np.array([1, 2, 3])  

g_H = np.array([5.0])  
g_R = np.array([80.0])  
v_R = np.array([2.0])  
v_h = np.array([0.5])  
w_H = np.array([0.5])  

P_th = np.array([5.0])  
T_R = np.array([80.0])  

gamma = np.array([100.0])  
eta_1 = np.array([1.0])  
eta_2 = np.array([1.0]) 
theta_1 = np.array([1.0])   
theta_2 = np.array([0.5])   
theta_3 = np.array([2.5])   
theta_4 = np.array([8.0*10**-3])   
theta_5 = np.array([100.0]) 
theta_6 = np.array([6.0*10**-3]) 



beta=1
u_H=np.array([1, 2])

#------------------------------------------
## Human Model
# Human’s goal objective function

u_H = np.array([1, 1, 1])  
norm_x_H_g_H = np.linalg.norm(x_H - g_H)**2
norm_u_H = np.linalg.norm(u_H)**2
QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H

# Safety objective function
hat_x_R=1

QH_s=theta_5*np.exp(theta_6*np.linalg.norm(x_H-hat_x_R)**2)



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
u_app=0


for i in range(n):

    
    u = cp.Variable((NoI , 1))

    x_R = np.array([1, 2, 3])  
    # w_H=1
    g_R = np.array([0, 0, 0])  



    # Robot’s goal objective function

    u_R = np.array([1, 1, 1])  

   
    norm_x_R_g_R = np.linalg.norm(x_R - g_R)**2
    norm_u_R = np.linalg.norm(u_R)**2
    QR_g = theta_1 * norm_x_R_g_R + theta_4 * norm_u_R

    sigma = QR_g

    constraints=np.zeros_like(Probability_of_Collision())
    for i in range(Probability_of_Collision().shape[0]):
        constraints[i] = [ P_Coll[i] <= P_th] 

    problem = cp.Problem(cp.Minimize(sigma), constraints)
    problem.solve(solver=cp.OSQP)

    if problem.status != cp.OPTIMAL:
        flag += 1

    u_app[:, i] = u.value[:NoI, 0]
    x_R[:, i+1] = A@ x_R[:, i] + B @ u_app[:, i]
