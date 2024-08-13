import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


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

P_t=np.array([.5,
                .5])
P_x_H=-5
# print(betas)

# x_H = -5.0*np.ones((NoS_H,n))
Nc=np.array([-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]).reshape(-1,1)   
x_H = -0.0*np.ones((NoS_H,n))
# Nc=np.array([0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5]).reshape(-1,1)   

g_H = np.array([5.0]).reshape(-1,1)  
g_R = np.array([80.0]).reshape(-1,1) 
g_R_pr=np.tile(g_R, Prediction_Horizon) 
v_R = np.array([2.0]).reshape(-1,1)  
v_h = np.array([0.5]).reshape(-1,1)  
w_H = np.array([0.05]).reshape(-1,1)  
u_H_values = np.array([-2, -1, 0, 1, 2]).reshape(-1,1)

P_th = np.array([0.1]).reshape(-1,1)  
T_R = np.array([5.0]).reshape(-1,1)  

gamma = 1
eta_1 = 1.0
eta_2 = 1.0
theta_1 = np.array([1.0]).reshape(-1,1)   
theta_2 = np.array([0.5]).reshape(-1,1)   
theta_3 = np.array([2.5]).reshape(-1,1)   
theta_4 = np.array([8.0]).reshape(-1,1)   
theta_5 = np.array([300]).reshape(-1,1) 
theta_6 = np.array([.006]).reshape(-1,1) 
x_R = -5.0*np.ones((NoS_R,n))  

# # Generate the estimation and noise samples
mean = 0  # Zero mean for the Gaussian noise
covariance = 2  # Example covariance (which is the variance in 1D)
std_deviation = np.sqrt(covariance)  # Standard deviation
num_samples = 1  # Number of samples


def  human_s_action(NoI_H,u_H_values,x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,eta_1,eta_2,beta):
        
        
        
        
        
        u_H = cp.Variable((NoI_H , 1), nonneg=True)
        # binary_vars = cp.Variable((NoI_H, len(u_H_values)), boolean=True)
        # u_H = binary_vars @ u_H_values

        norm_x_H_g_H = cp.norm(x_H0+u_H - g_H,'fro')**2
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
        closest_value = min(u_H_values, key=lambda x: abs(x - sss))
        # print(sss)
        return closest_value



# # Safety objective function
# Human Action Prediction
def Human_Action_Prediction(u_H,u_H_values,w_H,gamma,betas,x_H0,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6):
    sum_P_d=0.0
    # P_di=[]
    P_d=np.zeros((u_H_values.shape[0],betas.shape[0]))
    for k in range(u_H_values.shape[0]):
        for i in range(betas.shape[0]):
            QH_g=human_s_goal(u_H_values[k,0],x_H0,g_H,theta_3,theta_4)
            QH_s=human_s_safety(x_H0,hat_x_R,theta_5,theta_6)
   
            sum_P_d+=np.exp(-gamma * (QH_g + betas[i] * QH_s))
             
            # Human’s Deliberate Behavior
            P_d[k,i]=np.exp(-gamma*(QH_g+ betas[i]*QH_s))
            # P_di.append(np.exp(-gamma*(QH_g+ betas[i]*QH_s)))\\\


    QH_g=human_s_goal(u_H,x_H0,g_H,theta_3,theta_4)
    QH_s=human_s_safety(x_H0,hat_x_R,theta_5,theta_6)        
    P_di=np.exp(-gamma*(QH_g+ beta*QH_s))
    # P_d=np.array(P_di).reshape(-1,1)
    P_d=P_d/sum_P_d
    # Initialize a result matrix with the same shape as P_d
    result = np.zeros_like(P_d)

    # For each column (dimension), set the maximum value to 1 and others to 0
    for j in range(P_d.shape[1]):
    # Find the index of the maximum value in the j-th column
        max_index = np.argmax(P_d[:, j])
    # Set the corresponding element in the result matrix to 1
        result[max_index, j] = 1
 


    P_d=result
    
    P_di=P_di/sum_P_d
    # Human’s Random Behavior:
    U_H=len(u_H_values)
    P_r=1/U_H
     
    P_u_H=(1-w_H) * P_d+w_H * P_r
    P_u_Hi=(1-w_H) * P_di+w_H * P_r
    return P_u_H, P_u_Hi
   

def human_s_goal(u_H,x_H0,g_H,theta_3,theta_4):
        norm_x_H_g_H = np.linalg.norm(x_H0+u_H - g_H)**2
        norm_u_H = np.linalg.norm(u_H)**2
        QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H
        return QH_g
    
def human_s_safety(x_H0,hat_x_R,theta_5,theta_6):
        a=np.vstack((x_H0,0))
        b=np.vstack((0.0,hat_x_R))
        QH_s=theta_5*np.exp(-theta_6*np.linalg.norm(a-b)**2)
        return QH_s

#--------------------------------------------------
# Robot’s Belief About the Human’s Danger Awareness
def Robot_s_Belief_About_HDA(u_H,u_H_values,w_H,gamma,betas,P_t,x_H0,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6):
    sum_P_P_t=0.0
    P_ti=np.zeros((betas.shape[0],1))
    _, P_u_Hi=Human_Action_Prediction(u_H,u_H_values,w_H,gamma,betas,x_H0,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6)

    for i in range(betas.shape[0]):
        sum_P_P_t+=P_u_Hi*P_t[i]
        P_ti[i]=P_u_Hi*P_t[i]

    P_t=P_ti/sum_P_P_t

    return P_t

# Probability distribution of the human’s states

tolerance=1e-5
def Probability_distribution_of_human_s_states(u_H,u_app_Robot,w_H,gamma,beta,betas,P_t,P_x_H,u_H_values,Prediction_Horizon,x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,Nc,Abar,Bbar,A_H,B_H):
    # P_u_H=np.zeros((u_H_values.shape[0]*betas.shape[0])).reshape(-1,1)

    x_pr = Abar @ x_R0 + Bbar @ u_app_Robot
    x_pr=np.vstack((x_pr,1.0))
    # fff=u_H_values.shape[0]*betas.shape[0]
    # sum_x_H=0.0
    P=np.zeros((Prediction_Horizon,Nc.shape[0],1))
    # P[0,:,:]=1.0
    P_x_H=np.zeros((Nc.shape[0],1))
    # P_x_Hn=np.zeros(1,(Nc.shape[0],1))
    P_x_H_ik=np.zeros((Nc.shape[0],u_H_values.shape[0]*betas.shape[0]))
    x_H_next_p=np.zeros((Prediction_Horizon,1))
    x_H_next=np.zeros((u_H_values.shape[0],1))
    P_P=np.zeros((Prediction_Horizon,Nc.shape[0]))
    # u_H_values_flat = u_H_values.flatten()
    # u_H_values = np.tile(u_H_values_flat, Prediction_Horizon)
    # for tt in range(Prediction_Horizon):
    #     u_H_values_P=
    new_cell = []
    new_cells=[1]
    new_cell.append(1)
    P_x_H_iks=[]
    x_H_next=x_H0
    for j in range(Prediction_Horizon):
    #   print(len(np.array(new_cell)))
        sum_x_H=0.0


        if j>=1:
            unique_numbers = set(new_cell)
            new_cell = list(unique_numbers)
            x_H0_new=[Nc[f] for f in np.array(new_cell)]
            
            new_cells=(new_cell)
            new_cell = []
        for n in range(len(np.array(new_cells))):
        
            if j>=1:
                # x_H0_new=[Nc[f] for f in np.array(new_cell)]
                # new_cell = []
                x_H0_prob=x_H0_new[n]
            
            for m in range(Nc.shape[0]):
            
                for k in range(u_H_values.shape[0]):
                    if j==0:
                        x_H0_prob=x_H0

                    x_H_next=A_H*x_H0_prob+B_H*u_H_values[k]

                    if np.allclose(x_H_next, Nc[m, 0], atol=tolerance):
                        # new_cell=np.concatenate([new_cell, m], axis=0)
                        # new_cell.append(m)                               
                        P_x_H_k=1.0

                    else:
                        P_x_H_k=0.0

                    for i in range(betas.shape[0]):
                        # if j>=1:

                        #     u_H=human_s_action(NoI_H,u_H_values,x_H0_prob,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,eta_1,eta_2,beta)
                        #     x_H0_prob = A_H @ x_H0_prob + B_H @ u_H

                        #     # Robot’s goal objective function
                        #     x_pr = Abar @ x_R0 + Bbar @ u_app_Robot
                        #     epsilon = np.random.normal(mean, std_deviation, num_samples)
                        #     hat_x_R=x_pr[j]+epsilon 




                        P_u_H, P_u_Hi=Human_Action_Prediction(u_H,u_H_values,w_H,gamma,betas,x_H0_prob,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6)
                        P_t=Robot_s_Belief_About_HDA(u_H,u_H_values,w_H,gamma,betas,P_t,x_H0_prob,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6)                
                        P_x_H_iks.append(P_x_H_k*P_u_H[k,i]*P_t[i])
                        if P_x_H_k*P_u_H[k,i]*P_t[i]!=0:
                            new_cell.append(m)         
                        sum_x_H+=P_x_H_k*P_u_H[k,i]*P_t[i]

                sssss=np.array(P_x_H_iks).reshape(-1,1)
                sssssst=np.sum(sssss)
                P_x_H_iks=[]
                
                P_x_H[m,:]=sssssst

            if j==0:
                P_x_Hn=np.zeros((1,Nc.shape[0],1))
                P_x_Hn[:,:,:]=P_x_H 
            else:
                P_x_Hn[n,:,:]=P_x_H      

        
           
        
        if j==0:
            P[j,:,:]= P_x_H/(sum_x_H) 
            print(P[0,:,:])
            new_cell=new_cell[1:]
            P_x_Hn=np.zeros((len(np.array(new_cell)),Nc.shape[0],1))

        else:              
            PPPPP=np.sum(P_x_Hn, axis=0)   
            P[j,:,:]=PPPPP/(sum_x_H) 
            print(P[j,:,:])
            P_x_Hn=np.zeros((len(np.array(new_cell)),Nc.shape[0],1))

        # u_H=human_s_action(NoI_H,u_H_values,x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,eta_1,eta_2,beta)
        # x_H0 = A_H @ x_H0 + B_H @ u_H
        # Robot’s goal objective function

        epsilon = np.random.normal(mean, std_deviation, num_samples)
        hat_x_R=x_pr[j+1]+epsilon  


        # result = np.zeros_like(P_d)

        # # For each column (dimension), set the maximum value to 1 and others to 0
        # for j in range(P_d.shape[1]):
        #     # Find the index of the maximum value in the j-th column
        #     max_index = np.argmax(P_d[:, j])
        #     # Set the corresponding element in the result matrix to 1
        #     result[max_index, j] = 1


    #-----------------------------------------------------------------------------------
    # #print(P[:,:,0])
    assert P.shape == (Prediction_Horizon, len(Nc), 1), "P should have shape (5, 21, 1)"
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    plt.figure(figsize=(10, 6))
    for i in range(P.shape[0]):
        plt.plot(Nc,P[i], label=f'$P(x_H[ {i+1}])$')
    plt.xlabel('$N_c$')
    plt.ylabel('Prob. Dist. $P(x_H)$')
    plt.title('Probability Distributions for Different Prediction Horizons')
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(-5, 6, 1))
    vertical_lines = x_H[0,0]  # Specify the x-values for the vertical dashed lines
    plt.axvline(vertical_lines, color='black', linestyle=(0, (5, 5)), linewidth=2)
    plt.show()
    #-----------------------------------------------------------------------------------
    
    return P

# Probability of Collision

def Probability_of_Collision():
    P_Coll=0.0
    return P_Coll

flag=0
P_Col=[]
P_Coll=[]
P_t_app=[]
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
    u_H=1.0
    u_app_H[:, i]=u_H
    x_H[:, i+1] = A_H @ x_H[:, i] + B_H @ u_app_H[:, i]
    
    # Robot’s goal objective function

    
    u_R = cp.Variable((NoI_R * Prediction_Horizon,1))
    x_pr = Abar @ x_R0 + Bbar @ u_R
    norm_u_R = cp.sum(cp.square(u_R))
    norm_x_R_g_R = cp.sum(cp.square(x_pr - g_R_pr))      
    QR_g = theta_1 * norm_x_R_g_R + theta_2 * norm_u_R
    sigma_R = QR_g
    u_app_Robot=2.0*np.ones((NoI_R * Prediction_Horizon,1))
    if i>=1: 
        u_app_Robot=np.tile(u_app_R[:, i], Prediction_Horizon).reshape(-1,1)
        # u_up = np.tile(Uconstraint_flat, Prediction_Horizon)
    P_xH=Probability_distribution_of_human_s_states(u_H,u_app_Robot,w_H,gamma,beta,betas,P_t,P_x_H,u_H_values,Prediction_Horizon, x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,Nc,Abar,Bbar,A_H,B_H)

    P_t_ap=(u_H,u_H_values,w_H,gamma,betas,P_t,x_H0,hat_x_R,g_H,theta_3,theta_4,theta_5,theta_6)
    P_t_app.append(P_t_ap)
    for t in range(P_xH.shape[0]):
        # Get the current 2D slice
        matrix = P_xH[t, :, :]

        # Check if any value exceeds the threshold
        if np.any(matrix > 0.0):
            # Find indices where the condition is true
            indices = np.where(matrix > 0.0)
        
            # Use the first pair of indices for demonstration purposes
            j, k = indices[0][0], indices[1][0]
            constraints = []
            for tt in range(len(indices)):# Check the constraint on x_pr
                print(matrix[indices[tt][0],0])
                if indices[tt][0]>=8 and indices[tt][0]<=12 and matrix[indices[tt][0],0]>P_th:                 #-- Here we define a shortcut in a way that we know the exact equivalent position of the index 9,10,11.
                                             #-- So we limit the robot to croos that position.
                    constraints.append([x_pr[t] <= -1.0,
                                             x_pr[t] >= 1.0])
                    P_Col.append(np.array(0.0))

                elif indices[tt][0]>=9 and indices[tt][0]<=11 and matrix[indices[tt][0]]<=P_th and (x_pr[ttt] >= -1.0 or x_pr[ttt] <= 1.0 for ttt in range(x_pr.shape[0])):
                # Find the maximum value smaller than the threshold
                    # valid_values = matrix[matrix < P_th]
                    
                    P_Col.append(matrix[indices[tt][0]])
                #print(f"Max value smaller than threshold: {P_Coll}")
                else:
                    P_Col.append(np.array(0.0))


            
    # P_Coll.append(np.max(np.array(P_Col)))
    max_values = [np.max(p) for p in P_Col]

    # Append the maximum of these values to P_Coll
    P_Coll.append(np.max(max_values))   
    problem = cp.Problem(cp.Minimize(sigma_R), constraints)
    problem.solve(solver=cp.OSQP)
    if problem.status != cp.OPTIMAL:
        flag += 1
    u_app_R[:, i] = u_R.value[:NoI_R, 0]
    sss=A_R@ x_R[:, i] 
    sdsd=B_R @ u_app_R[:, i]
    x_R[:, i+1] = A_R@ x_R[:, i] + B_R @ u_app_R[:, i]


    print(i)

    # time = np.linspace(0, (i+1)*deltaT, i+1)
    # plt.figure(figsize=(10, 6))
    # for i in range(len(P_Coll)):
    #     plt.plot(time,P_Coll, label=f'$P(x_H[ {i+1}])$')
    # plt.xlabel('Time[s]')
    # plt.ylabel('Prob. Dist. $P(x_H)$')
    # plt.title('Probability Distributions for Different Prediction Horizons')
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(np.arange(0, 6, 1))
    # plt.show()









