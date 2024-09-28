#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from leg_tracker.msg import PersonArray
import math
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch

class RobotMPCTrackingController:
    def __init__(self):
        rospy.init_node('robot_mpc_tracking_controller')

        #Constatnt Contoller parameters----------------------------------------------------------------------------------------------------------------
        # Robot Model
        # Robot Model
        self.n = 20
        self.Prediction_Horizon = 5
        self.Prediction_Horizon_H = self.Prediction_Horizon
        self.Signal = "off"  # Signal could be "on" or "off"
        self.Human = "Concerned"  # Human could be "Concerned" or "Unconcerned"

        self.deltaT = 0.5
        self.Safe_Distance = 4
        self.A_R = np.array([[1.0, 0.], [0., 1.]])
        self.B_R = np.array([[self.deltaT, 0.0], [0.0, self.deltaT]])
        self.C_R = np.eye(2, 2)
        self.D_R = np.zeros((2, 2))

        self.NoI_R = self.B_R.shape[1]
        self.NoS_R = self.A_R.shape[0]
        self.NoO_R = self.C_R.shape[0]

        # Human Model
        self.A_H = np.array([[1.0, 0.], [0., 1.]])
        self.B_H = np.array([[self.deltaT, 0.], [0., self.deltaT]])
        self.C_H = np.eye(2, 2)
        self.D_H = np.zeros((2, 2))

        self.NoI_H = self.B_H.shape[1]
        self.NoS_H = self.A_H.shape[0]
        self.NoO_H = self.C_H.shape[0]

        # ------------------------------------------------------------------------------------
        ## Robot Predictive model
        self.Abar = self.A_R
        if self.A_R.shape[0] == 1:
            for i in range(2, self.Prediction_Horizon + 1):
                self.Abar = np.vstack((self.Abar, self.A_R ** i))

            self.Bbar = np.zeros((self.NoS_R * self.Prediction_Horizon, self.NoI_R * self.Prediction_Horizon))

            # Loop to fill Bbar with the appropriate blocks
            for i in range(1, self.Prediction_Horizon + 1):
                for j in range(1, i + 1):
                    A_power = self.A_R ** (i - j)
                    block = A_power * self.B_R
                    row_indices = slice((i - 1) * self.NoS_R, i * self.NoS_R)
                    col_indices = slice((j - 1) * self.NoI_R, j * self.NoI_R)
                    self.Bbar[row_indices, col_indices] = block
        else:
            self.Abar = np.vstack([np.linalg.matrix_power(self.A_R, i) for i in range(1, self.Prediction_Horizon + 1)])
            self.Bbar = np.zeros((self.NoS_R * self.Prediction_Horizon, self.NoI_R * self.Prediction_Horizon))

            for i in range(1, self.Prediction_Horizon + 1):
                for j in range(1, i + 1):
                    self.Bbar[(i - 1) * self.NoS_R:i * self.NoS_R, (j - 1) * self.NoI_R:j * self.NoI_R] = \
                        np.linalg.matrix_power(self.A_R, i - j) @ self.B_R

        # ------------------------------------------------------------------------------------
        ## Human Predictive model
        self.Abar_H = self.A_H
        if self.A_H.shape[0] == 1:
            for i in range(2, self.Prediction_Horizon_H + 1):
                self.Abar_H = np.vstack((self.Abar_H, self.A_H ** i))

            self.Bbar_H = np.zeros((self.NoS_H * self.Prediction_Horizon_H, self.NoI_H * self.Prediction_Horizon_H))

            # Loop to fill Bbar_H with the appropriate blocks
            for i in range(1, self.Prediction_Horizon_H + 1):
                for j in range(1, i + 1):
                    A_power = self.A_H ** (i - j)
                    block = A_power * self.B_H
                    row_indices = slice((i - 1) * self.NoS_H, i * self.NoS_H)
                    col_indices = slice((j - 1) * self.NoI_H, j * self.NoI_H)
                    self.Bbar_H[row_indices, col_indices] = block
        else:
            self.Abar_H = np.vstack([np.linalg.matrix_power(self.A_H, i) for i in range(1, self.Prediction_Horizon_H + 1)])
            self.Bbar_H = np.zeros((self.NoS_H * self.Prediction_Horizon_H, self.NoI_H * self.Prediction_Horizon_H))

            for i in range(1, self.Prediction_Horizon_H + 1):
                for j in range(1, i + 1):
                    self.Bbar_H[(i - 1) * self.NoS_H:i * self.NoS_H, (j - 1) * self.NoI_H:j * self.NoI_H] = \
                        np.linalg.matrix_power(self.A_H, i - j) @ self.B_H

        # ------------------------------------------------------------------------------------

        ## Controller
        self.betas = np.array([0, 1])

        if self.Human == "Concerned":
            self.beta = 1
        elif self.Human == "Unconcerned":
            self.beta = 0
        self.P_t = np.array([.5, .5])

        self.Nc = np.array([-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

        self.X, self.Y = np.meshgrid(self.Nc, self.Nc)

        self.coordinates_matrix = np.empty((self.Nc.shape[0], self.Nc.shape[0]), dtype=object)

        for i in range(self.Nc.shape[0]):
            for j in range(self.Nc.shape[0]):
                self.coordinates_matrix[i, j] = np.array([[self.X[i, j]], [self.Y[i, j]]])
        self.Nc = self.coordinates_matrix

        self.g_H = np.array([[5], [0.0]])
        self.g_H_pr = np.tile(self.g_H, (self.Prediction_Horizon_H, 1))
        self.g_R = np.array([[0.], [10.0]]).reshape(-1, 1)
        self.g_R_pr = np.tile(self.g_R, (self.Prediction_Horizon, 1))
        self.v_R = 2.0
        self.v_h = .5
        self.w_H = np.array([0.2]).reshape(-1, 1)

        self.u_H_values = np.array([-2 * self.v_h, -1 * self.v_h, 0, 1 * self.v_h, 2 * self.v_h])
        self.u_H_value = np.array([-2 * self.v_h, -1 * self.v_h, 0, 1 * self.v_h, 2 * self.v_h])
        self.X, self.Y = np.meshgrid(self.u_H_values, self.u_H_values)
        self.coordinates_matrix = np.empty((self.u_H_values.shape[0], self.u_H_values.shape[0]), dtype=object)

        for i in range(self.u_H_values.shape[0]):
            for j in range(self.u_H_values.shape[0]):
                self.coordinates_matrix[i, j] = np.array([[self.X[i, j]], [self.Y[i, j]]])
        self.u_H_values = self.coordinates_matrix

        self.u_R_values = np.array([0, .5 * self.v_R, self.v_R])
        self.X, self.Y = np.meshgrid(self.u_R_values, self.u_R_values)

        self.coordinates_matrix = np.empty((self.u_R_values.shape[0], self.u_R_values.shape[0]), dtype=object)

        for i in range(self.u_R_values.shape[0]):
            for j in range(self.u_R_values.shape[0]):
                self.coordinates_matrix[i, j] = np.array([[self.X[i, j]], [self.Y[i, j]]])
        self.u_R_values = self.coordinates_matrix

        self.P_th = np.array([0.05]).reshape(-1, 1)
        self.T_R = np.array([5.0]).reshape(-1, 1)

        self.gamma = 1
        self.eta_1 = 1.0
        self.eta_2 = 1.
        self.theta_1 = np.array([1.0]).reshape(-1, 1)
        self.theta_2 = np.array([1]).reshape(-1, 1)
        self.theta_3 = np.array([2.5]).reshape(-1, 1)
        self.theta_4 = np.array([8.0]).reshape(-1, 1)
        self.theta_5 = np.array([100]).reshape(-1, 1)
        
        self.theta_6 = np.array([1.]).reshape(-1, 1)
        U_H_constraint=np.array([[1], [1]]) 
        initial_u_H=np.array([[0.],[0.]])
        initial_u_R=np.array([[0],[2.]])
        
        self.U_H_constraints=np.tile(U_H_constraint, (self.Prediction_Horizon, 1))
        self.initial_u_H=np.tile(initial_u_H, (self.Prediction_Horizon, 1))
        self.initial_u_R=np.tile(initial_u_R, (self.Prediction_Horizon, 1))

        self.mean = 0  # Zero mean for the Gaussian noise
        self.covariance = 2  # Example covariance (which is the variance in 1D)
        self.std_deviation = np.sqrt(self.covariance)  # Standard deviation
        self.num_samples = 1  # Number of samples
        self.tolerance=1e-5


        
        self.P_Col=[]
        self.P_Coll=[]
        self.optimized_u_R=np.zeros((self.NoI_R * self.Prediction_Horizon, self.n))
        self.P_t_app=[]
        self.u_app_H = np.zeros((self.NoI_H, self.n))
        self.u_app_R = np.zeros((self.NoI_R, self.n))
        self.P_xH_all=np.zeros((self.n,self.Prediction_Horizon,self.self.Nc.shape[0],self.Nc.shape[1]))
        self.P_t_all = np.zeros((self.n, 1))
        self.P_Coll = np.zeros((self.n, 1))



        #-----------------------------------------------------------------------------------------------


        # self.P_Col=[]
        # self.P_Coll=[]
        # self.P_t_app=[]
        # self.u_app_H = np.zeros((self.NoI_H, self.n))
        # self.u_app_R = np.zeros((self.NoI_R, self.n))

        # self.P_t_all = np.zeros((self.n, 1))
        # self.P_Coll = np.zeros((self.n, 1))
        
        #----------------------------------------------------------------------------------------------------------------------------------------------

        

        # Robot state variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.human_position = None

        # Publisher and subscriber
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/mobile_base/odom', Odometry, self.odom_callback)
        self.people_sub = rospy.Subscriber('/people_tracked', PersonArray, self.people_callback)

        # Set the control loop rate (5 Hz -> 0.2 sec)
        self.rate = rospy.Rate(5)

        rospy.loginfo("Robot MPC Tracking Controller Initialized")

    def quaternion_to_euler(self, q):
        """
        Convert quaternion to Euler angles (yaw).
        """
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)  # Return only yaw

    def people_callback(self, data):
        # Check if there are people tracked
        if not data.people:
            rospy.loginfo("No people tracked")
            return
        # Select the person with the highest ID
        highest_id = -1
        selected_person = None

        for person in data.people:
            if person.id > highest_id:
                highest_id = person.id
                selected_person = person

        if selected_person:
            self.human_position = selected_person.pose.position
            rospy.loginfo(f"Tracking person with ID: {highest_id}, Position - x: {self.human_position.x}, y: {self.human_position.y}")

    def odom_callback(self, msg):
        # Extract robot's position and orientation from odometry
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_euler(orientation_q)

        if self.human_position is not None:
            # Call the MPC-based planner
            linear_vel = self.Human_robot_action_planner(self.human_position, (self.current_x, self.current_y))

            # Publish velocity commands
            cmd_msg = Twist()
            cmd_msg.linear.x = linear_vel
            self.cmd_vel_pub.publish(cmd_msg)

            rospy.loginfo(f"Robot Position: ({self.current_x}, {self.current_y}), Human Position: ({self.human_position.x}, {self.human_position.y})")
            rospy.loginfo(f"Linear Velocity: {linear_vel}")


    

    def Human_robot_action_planner(self,i, initial_u_R, initial_u_H, P_t,  u_app_R, u_app_H, P_t_all):
    
        constraints=[]
        x_H0=[[self.human_position.x],[self.human_position.x]]
        x_R0=[[self.current_x],[self.current_y]]
        # Controller code begins here:
        
        # Generate zero-mean Gaussian noise
        epsilon = np.random.normal(self.mean, self.std_deviation, self.num_samples)

        if i == 0:
            u_app_Robot = initial_u_R        
        else:
            u_app_Robot = np.tile(u_app_R[:, i-1], self.Prediction_Horizon).reshape(-1, 1)
            
        x_pr = self.Abar @ x_R0 + self.Bbar @ u_app_Robot    
        hat_x_R_pr = x_pr + epsilon  
        hat_x_R = x_R0 + epsilon  

        # Probability distribution based on human's initial actions or updated actions


        P_xH = self.Probability_distribution_of_human_s_states(self, u_app_Robot, P_t, x_H0,hat_x_R)
      
        P_u_H = self.Human_Action_Prediction(self, x_H0, hat_x_R)

        # Human's action update
        if i == 0:
                      
            u_H = self.human_s_action(  self,x_H0, hat_x_R_pr, initial_u_H[:self.NoI_H])
        else:
            u_H = self.human_s_action(self, x_H0, hat_x_R_pr, u_app_H[:, i-1])

        u_app_H[:, i]=u_H[:self.NoI_H].flatten()
        # x_H[:, i+1] = self.A_H @ x_H[:, i] + self.B_H @ u_app_H[:, i]
        
        # Robot’s goal objective function
        def objective(self,u_R):
            u_R = u_R.reshape((self.NoI_R * self.Prediction_Horizon, 1))
            x_pr = self.Abar @ x_R0 + self.Bbar @ u_R
            norm_u_R = np.sum(np.square(u_R))
            norm_x_R_g_R = np.sum(np.square(x_pr - self.g_R_pr))
            QR_g = self.theta_1 * norm_x_R_g_R + self.theta_2 * norm_u_R
            return QR_g[0]

        # Define constraints
        def constraint1(u_R):
            u_R = u_R.reshape((self.NoI_R * self.Prediction_Horizon, 1))
            return np.min(u_R) +2. # u_R >= 0

        def constraint2(u_R):
            u_R = u_R.reshape((self.NoI_R * self.Prediction_Horizon, 1))
            return 2.0-np.max(u_R)   # u_R <= 2

        def constraint3(u_R):
            u_R = u_R.reshape((self.NoI_R * self.Prediction_Horizon, 1))
            x_pr = self.Abar @ x_R0 + self.Bbar @ u_R
            return x_pr[0]     # u_R <= 2

        def custom_constraints(u_R):
            u_R = u_R.reshape((self.NoI_R * self.Prediction_Horizon, 1))
            


            for t in range(P_xH.shape[0]):
            # Get the current 2D slice
                matrix = P_xH[t, :, :]

            # Check if any value exceeds the threshold
                if np.any(matrix > 0.0):
                # Find indices where the condition is true
                    indices = np.where(matrix > 0.0)
            
                # Use the first pair of indices for demonstration purposes
                    # m, b = indices[0][0], indices[1][0]
                
                    indices=np.array(indices)
                    for tt in range(indices.shape[1]):# Check the constraint on x_pr

                        # if np.linalg.norm(Nc[indices[0,tt],indices[1,tt]]-x_R[:,i])>1. and matrix[indices[0][tt],indices[1][tt]]>P_th:
                        if matrix[indices[0][tt],indices[1][tt]]>self.P_th: 
                                    
                                                
                            
                            def constraint_fun(u_R):
                                u_R_reshaped = u_R.reshape((self.NoI_R * self.Prediction_Horizon, 1))
                                x_pr_t = self.Abar @ x_R0 + self.Bbar @ u_R_reshaped
                                # Cons=np.linalg.norm(Nc[indices[0,tt],indices[1,tt]] - x_pr_t[NoI_R * (t+1)-NoI_R:NoI_R * (t+1) - 1]) - Safe_Distance

                                Cons=np.linalg.norm(self.Nc[indices[0,tt],indices[1,tt]] - x_pr_t[self.NoI_R * t:self.NoI_R * (t + 1) ]) - self.Safe_Distance
                                return Cons
                            
                            constraints.append({'type': 'ineq', 'fun': constraint_fun})


                            # P_Col.append(np.array(0.0))
                            self.P_Col.append(np.array(0.0))

                        # elif np.linalg.norm(Nc[indices[0,tt],indices[1,tt]]-x_R[:,i])<=1. and matrix[indices[0][tt],indices[1][tt]]<=P_th and t==0 :
                        elif matrix[indices[0][tt],indices[1][tt]]<=self.P_th and t==0 :
                    # Find the maximum value smaller than the threshold
                            dvd=P_xH[0, indices[0][tt],indices[1][tt]]

                            self.P_Col.append(dvd)
                    
                    #print(f"Max value smaller than threshold: {P_Coll}")
                        else:
                            self.P_Col.append(np.array(0.0))
            
            return constraints

        # Initial guess for the optimization variables
        if i>=1:
    
            # initial_u_R=u_app_R[:, i-1]
            # initial_u_R=np.tile(u_app_R[:, i-1], (Prediction_Horizon, 1))
            initial_u_R=np.vstack([self.optimized_u_R[self.NoI_R:, i-1].reshape(-1, 1), self.optimized_u_R[-self.NoI_R:, i-1].reshape(-1, 1)])
        else:
            initial_u_R=np.tile(np.array([[0],[2]]), (self.Prediction_Horizon, 1))

        # Setup constraints for `minimize`
        if x_R0[1]<=-5:

            constraints = [{'type': 'ineq', 'fun': constraint1},
                    {'type': 'ineq', 'fun': constraint2},
                    {'type': 'eq', 'fun': constraint3}]
        else:

            constraints = [{'type': 'ineq', 'fun': constraint1},
                    {'type': 'ineq', 'fun': constraint2}]
        (custom_constraints(initial_u_R))

        # Perform the optimization
        result = minimize(objective, initial_u_R.flatten(), constraints=constraints, method='SLSQP')

        # Get the optimized values
        # print(result.fun)
        self.optimized_u_R[:,i] = result.x

        # rounded_u_R = min(u_R_values.flatten(), key=lambda x: np.linalg.norm(np.array([[x]]) - optimized_u_R[:NoI_R]))
        rounded_u_R=self.optimized_u_R[:,i] [:self.NoI_R]
        u_app_R[:, i] = rounded_u_R[:self.NoI_R]

        # x_R[:, i+1] = self.A_R@ x_R[:, i] + self.B_R @ u_app_R[:, i]

        # print(u_app_R[:, i],u_app_H[:, i])
        
        P_t=self.Robot_s_Belief_About_HDA(self,u_app_H[:, i].reshape(-1,1) ,P_t,P_u_H)               
        P_t_all[i]=P_t[1]
        

        linear_vel = u_app_R[:, i]  # You will replace this with the actual MPC calculation
        return linear_vel
        
        
    def human_s_action(self,x_H0,hat_x_R, u_H0):
        # Objective function to minimize
        def objective(u_H_flattened):
            u_H = u_H_flattened.reshape(self.NoI_R * self.Prediction_Horizon_H, 1)  # Reshape back to 2D for computation
            x_pr_H = self.Abar_H @ x_H0 + self.Bbar_H @ u_H
            
            # QH_g term
            norm_x_H_g_H = np.linalg.norm(x_pr_H - self.g_H_pr) ** 2
            norm_u_H = np.linalg.norm(u_H) ** 2
            QH_g = self.theta_3 * norm_x_H_g_H + self.theta_4 * norm_u_H

            # QH_s term
            norm_expr = np.linalg.norm(x_pr_H - hat_x_R)
            QH_s = self.theta_5 * np.exp(-self.theta_6 * norm_expr ** 2)

            # Total sigma_H        
            sigma_H = self.eta_1 * QH_g + self.beta * self.eta_2 * QH_s

            return float(sigma_H)  # Ensure the returned value is a scalar

        # Constraints
        def constraint1(u_H_flattened):
            u_H = u_H_flattened.reshape(self.NoI_H * self.Prediction_Horizon_H, 1)
            return (u_H + self.U_H_constraints).flatten()  # Flatten for compatibility

        def constraint2(u_H_flattened):
            u_H = u_H_flattened.reshape(self.NoI_H * self.Prediction_Horizon_H, 1)
            return (self.U_H_constraints - u_H).flatten()  # Flatten for compatibility

        # Define constraints as a dictionary
        constraints = [{'type': 'ineq', 'fun': constraint1},
                    {'type': 'ineq', 'fun': constraint2}]
        
        # Flatten the initial guess just for the solver
        # u_H0_flattened = u_H0.flatten()
    
    
            # initial_u_R=u_app_R[:, i-1]
        u_H0_flattened=np.tile(u_H0, (self.Prediction_Horizon, 1)).flatten()
        # Optimize using scipy's minimize function
        solution = minimize(objective, u_H0_flattened, method='trust-constr', constraints=constraints)
        
        # Extract the optimal value of u_H and reshape it back to 2D
        optimal_u_H = solution.x.reshape(self.NoI_H * self.Prediction_Horizon_H, 1)

        # Function to compute the closest value in u_H_values to a given value
        def find_nearest(value, u_H_value):
            return u_H_value[np.argmin(np.abs(u_H_value - value))]

        # Vectorize the find_nearest function to apply it to each element in optimal_u_H
        vectorized_find_nearest = np.vectorize(lambda x: find_nearest(x, self.u_H_value))

        # Apply the vectorized function to each element in optimal_u_H
        rounded_optimal_u_H = vectorized_find_nearest(optimal_u_H)

        return rounded_optimal_u_H


    # Human Action Prediction
    def Human_Action_Prediction(self,x_H0,hat_x_R):
        sum_P_d=np.zeros((self.betas.shape[0],1))
        # P_di=[]
        P_d=np.zeros((self.u_H_values.shape[0],self.u_H_values.shape[1],self.betas.shape[0]))
        P_dk=[]
        for i in range(self.betas.shape[0]):
            # k=0
            for kx in range(self.u_H_values.shape[0]):
              for ky in range(self.u_H_values.shape[1]):
                QH_g=self.human_s_goal(self,self.u_H_values[kx,ky],x_H0)
                QH_s=self.human_s_safety(self,self.u_H_values[kx,ky],x_H0,hat_x_R)
                
                # Human’s Deliberate Behavior
                P_d[kx,ky,i]=np.exp(-self.gamma*(QH_g+ self.betas[i]*QH_s))
                # k+=1
            sum_P_d[i,0]+=np.exp(-self.gamma * (QH_g + self.betas[i] * QH_s))
        P_d[:,:, 0] = P_d[:,:, 0] / sum_P_d[0, 0]
        # Divide the second column of P_d by the second row of sum_P_d
        P_d[:,:, 1] = P_d[:,:, 1] / sum_P_d[1, 0]
        # Initialize a result matrix with the same shape as P_d
        result = np.zeros_like(P_d)
        # Iterate over each slice in the last dimension
        for k in range(P_d.shape[2]):
        # Find the index of the maximum value in the (5, 5) slice for each slice along the last dimension
            max_index = np.unravel_index(np.argmax(P_d[:, :, k]), P_d[:, :, k].shape)
        # Create a mask of zeros and ones for the maximum value
            result[max_index[0], max_index[1], k] = 1
        P_d=result
        # Human’s Random Behavior:
        U_H=self.u_H_values.shape[0]*self.u_H_values.shape[1]
        P_r=1/U_H
        P_u_H=(1-self.w_H) * P_d+self.w_H * P_r
        return P_u_H
    
    def human_s_goal(self,u_H,x_H0):
            norm_x_H_g_H = np.linalg.norm(x_H0+u_H - self.g_H)**2
            norm_u_H = np.linalg.norm(u_H)**2
            QH_g = self.theta_3 * norm_x_H_g_H + self.theta_4 * norm_u_H
            return QH_g
        
    def human_s_safety(self,u_H,x_H0,hat_x_R):

            a=x_H0+u_H
            b=hat_x_R
            QH_s=self.theta_5*np.exp(-self.theta_6*np.linalg.norm(a-b)**2)
            return QH_s

    # Robot’s Belief About the Human’s Danger Awareness
    def Robot_s_Belief_About_HDA(self,u_H,P_t,P_u_H):
        sum_P_P_t=0.0
        P_ti=np.zeros((self.betas.shape[0],1))

        # Find the index where u_H matches one of the arrays in u_H_values
        def find_index(u_H_values, u_H):
            for i in range(u_H_values.shape[0]):
                for j in range(u_H_values.shape[1]):
                    if np.array_equal(u_H_values[i, j], u_H):
                        return (i, j)
            return None

        index = find_index(self.u_H_values, u_H)

        for i in range(self.betas.shape[0]):
            sum_P_P_t+=P_u_H[index[0],index[1],i]*P_t[i]
            P_ti[i]=P_u_H[index[0],index[1],i]*P_t[i]

        P_t=P_ti/sum_P_P_t

        return P_t

    # Probability distribution of the human’s states
    def Probability_distribution_of_human_s_states(self,u_app_Robot,P_t,x_H0,hat_x_R):
        x_pr = self.Abar @ hat_x_R + self.Bbar @ u_app_Robot
        x_pr=np.vstack((x_pr,1.0))
        P=np.zeros((self.Prediction_Horizon,self.Nc.shape[0],self.Nc.shape[1]))
        P_x_H=np.zeros((self.Nc.shape[0],self.Nc.shape[1]))
        x_H_next=np.zeros((self.u_H_values.shape[0],self.u_H_values.shape[1]))
        new_cell = []
        new_cells=[1]
        new_cell.append(1)
        P_x_H_iks=[]
        x_H_next=x_H0
        for j in range(self.Prediction_Horizon):

            sum_x_H=0.0
            if j>=1:
                tuple_list = [tuple(arr) for arr in new_cell]
                # Find unique tuples using a set
                unique_tuples = set(tuple_list)
                unique_tuples=list(unique_tuples)
                x_H0_new=[self.Nc[f] for f in unique_tuples]   
                new_cells=(unique_tuples)
                new_cell = []
            for n in range(len(np.array(new_cells))):
            
                if j>=1:
                    x_H0_prob=x_H0_new[n]
                
                for mx in range(self.Nc.shape[0]):
                  for my in range(self.Nc.shape[1]):
                
                    for kx in range(self.u_H_values.shape[0]):
                      for ky in range(self.u_H_values.shape[1]):
                        if j==0:
                            x_H0_prob=x_H0
                        sds=np.array(self.u_H_values[kx,ky])
                        x_H_next=self.A_H@x_H0_prob+self.B_H@self.u_H_values[kx,ky]

                        if np.allclose(x_H_next, self.Nc[mx, my], atol=self.tolerance):
                                
                            if j==0:
                                P_x_H_k=np.array([1.0])
                            else:
                                P_x_H_k=np.array([P[j-1,np.array(new_cells[n])[0],np.array(new_cells[n])[1]]])
                        

                            P_u_H=self.Human_Action_Prediction(self,x_H0_prob,hat_x_R)
                            for i in range(self.betas.shape[0]):
                            
                                P_x_H_iks.append(P_x_H_k*P_u_H[kx,ky,i]*P_t[i])
                                if P_x_H_k*P_u_H[kx,ky,i]*P_t[i]!=0:
                                    new_cell.append(np.array([mx,my]))         
                                sum_x_H+=P_x_H_k*P_u_H[kx,ky,i]*P_t[i]

                    sssss=np.array(P_x_H_iks).reshape(-1,1)
                    sssssst=np.sum(sssss)
                    P_x_H_iks=[]
                    P_x_H[mx,my]=sssssst

                if j==0:
                    P_x_Hn=np.zeros((1,self.Nc.shape[0],self.Nc.shape[1]))
                    P_x_Hn[:,:,:]=P_x_H 
                else:
                    P_x_Hn[n,:,:]=P_x_H             
            
            if j==0:
                P[j,:,:]= P_x_H/(sum_x_H) 

                new_cell=new_cell[1:]
                P_x_Hn=np.zeros((len(np.array(new_cell)),self.Nc.shape[0],self.Nc.shape[1]))

            else:              
                PPPPP=np.sum(P_x_Hn, axis=0)   
                P[j,:,:]=PPPPP/(sum_x_H) 
                P_x_Hn=np.zeros((len(np.array(new_cell)),self.Nc.shape[0],self.Nc.shape[1]))

            epsilon = np.random.normal(self.mean, self.std_deviation, self.num_samples)
            hat_x_R=x_pr[j+1]+epsilon  

 
        return P

      

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()


if __name__ == '__main__':
    try:
        controller = RobotMPCTrackingController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
