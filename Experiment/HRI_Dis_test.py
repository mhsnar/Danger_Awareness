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
        n = 20
        Prediction_Horizon = 1
        Prediction_Horizon_H=1
        deltaT=0.5

        A_R =  np.array([[1.0, 0.],[0.,1.]])
        B_R = np.array([[deltaT,0.0],[0.0,deltaT]])
        C_R = np.eye(2,2)
        D_R = np.zeros((2, 2))

        NoI_R=B_R.shape[1]
        NoS_R=A_R.shape[0]
        NoO_R=C_R.shape[0]

        # Human Model
        A_H = np.array([[1.0, 0.],
                        [0.,1.]])
        B_H = np.array([[deltaT,0.],
                    [0.,deltaT]])
        C_H = np.eye(2,2)
        D_H = np.zeros((2, 2))

        NoI_H=B_H.shape[1]
        NoS_H=A_H.shape[0]
        NoO_H=C_H.shape[0]

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
        ## Human Predictive model
        Abar_H = A_H
        if A_H.shape[0]==1:
            for i in range(2, Prediction_Horizon_H + 1):
                Abar_H = np.vstack((Abar_H, A_H**i))

            Bbar_H = np.zeros((NoS_H * Prediction_Horizon_H, NoI_H * Prediction_Horizon_H))
        # Loop to fill Bbar_H with the appropriate blocks
            for i in range(1, Prediction_Horizon_H + 1):
                for j in range(1, i + 1):
                    # Compute A_H^(i-j)
                    A_power = A_H ** (i - j)
                    
                    # Compute the block (A_power * B_H), since B_H is scalar we multiply directly
                    block = A_power * B_H

                    # Calculate the indices for insertion
                    row_indices = slice((i - 1) * NoS_H, i * NoS_H)
                    col_indices = slice((j - 1) * NoI_H, j * NoI_H)

                    # Insert the block into the appropriate position in Bbar_H
                    Bbar_H[row_indices, col_indices] = block
        else:
            Abar_H = np.vstack([np.linalg.matrix_power(A_H, i) for i in range(1, Prediction_Horizon_H+1)])
            Bbar_H = np.zeros((NoS_H * Prediction_Horizon_H, NoI_H * Prediction_Horizon_H))

            for i in range(1, Prediction_Horizon_H + 1):
                for j in range(1, i + 1):
                    Bbar_H[(i-1)*NoS_H:i*NoS_H, (j-1)*NoI_H:j*NoI_H] = np.linalg.matrix_power(A_H, i-j) @ B_H
        #------------------------------------------------------------------------------------



        #------------------------------------------
        ## Contorller
        #INPUTS=x_H ,X_R
        #Initials=
        betas=np.array([0,
                        1])

        Signal="off" # Signal could be "on" or "off"
        Human="Concerned"  # Human could be "Concerned" or "Unconcerned"

        if Human=="Concerned":
            beta=1
        elif Human=="Unconcerned":
            beta=0

        Safe_Distance=1.5


        P_t=np.array([.5,
                        .5])

        Nc=np.array([-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0])
        # Nc=np.array([-3.0,-2.5,-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,2.5,3.0])
        # Create a meshgrid for X and Y coordinates
        # Create a meshgrid for X and Y coordinates
        X, Y = np.meshgrid(Nc, Nc)

        # Combine X and Y into a 2D coordinate matrix
        # Flatten X and Y to create 2D matrices
        coordinates_matrix = np.empty((Nc.shape[0], Nc.shape[0]), dtype=object)

        for i in range(Nc.shape[0]):
            for j in range(Nc.shape[0]):
                coordinates_matrix[i, j] = np.array([[X[i, j]], [Y[i, j]]])
        Nc=coordinates_matrix        

        g_H = np.array([[5.],[0.0]])
        g_H_pr = np.tile(g_H, (Prediction_Horizon_H, 1))
        g_R = np.array([[0.],[10.0]]).reshape(-1,1) 
        g_R_pr = np.tile(g_R, (Prediction_Horizon, 1))
        v_R =2.0
        v_h = .5
        w_H = np.array([0.2]).reshape(-1,1)  

        u_H_values = np.array([-2*v_h, -1*v_h, 0, 1*v_h, 2*v_h])
        u_H_value = np.array([-2*v_h, -1*v_h, 0, 1*v_h, 2*v_h])
        X, Y = np.meshgrid(u_H_values, u_H_values)
        coordinates_matrix = np.empty((u_H_values.shape[0], u_H_values.shape[0]), dtype=object)
        for i in range(u_H_values.shape[0]):
            for j in range(u_H_values.shape[0]):
                coordinates_matrix[i, j] = np.array([[X[i, j]], [Y[i, j]]])
        u_H_values=coordinates_matrix        

        # u_R_values = np.array([0, .5*v_R, v_R])
        u_R_values = np.array([- v_R, -.5*v_R, 0, .5*v_R, v_R])
        X, Y = np.meshgrid(u_R_values, u_R_values)

        # Combine X and Y into a 2D coordinate matrix
        # Flatten X and Y to create 2D matrices
        coordinates_matrix = np.empty((u_R_values.shape[0], u_R_values.shape[0]), dtype=object)

        for i in range(u_R_values.shape[0]):
            for j in range(u_R_values.shape[0]):
                coordinates_matrix[i, j] = np.array([[X[i, j]], [Y[i, j]]])
        u_R_values=coordinates_matrix        


        P_th = np.array([0.1]).reshape(-1,1)  
        T_R = np.array([5.0]).reshape(-1,1)  

        gamma = 1
        eta_1 = 1.0
        eta_2 = 1.
        theta_1 = np.array([1.0]).reshape(-1,1)   
        theta_2 = np.array([0.5]).reshape(-1,1)   
        theta_3 = np.array([2.5]).reshape(-1,1)   
        theta_4 = np.array([8.0]).reshape(-1,1)   
        theta_5 = np.array([300]).reshape(-1,1) 
        theta_5 = np.array([100]).reshape(-1,1) 
        theta_6 = np.array([.06]).reshape(-1,1) 

        x_H = np.array([[-5.],[0.0]])*np.ones((NoS_H,n+1))
        x_R = np.array([[0.],[-10.0]])*np.ones((NoS_R,n+1))  

        U_H_constraint=np.array([[1], [1]]) 
        initial_u_H=np.array([[0.],[0.]])
        initial_u_R=np.array([[2.],[2.]])

        U_H_constraints=np.tile(U_H_constraint, (Prediction_Horizon, 1))
        initial_u_H=np.tile(initial_u_H, (Prediction_Horizon, 1))
        initial_u_R=np.tile(initial_u_R, (Prediction_Horizon, 1))


        # # Generate the estimation and noise samples
        mean = 0  # Zero mean for the Gaussian noise
        covariance = 2  # Example covariance (which is the variance in 1D)
        std_deviation = np.sqrt(covariance)  # Standard deviation
        num_samples = 1  # Number of samples
        tolerance=1e-5

        time = np.linspace(0, n*deltaT, n) 
        

        # Variable to store if the dashed line and text were already added
        line_added = False

        #-----------------------------------------------------------------------------------------------

        flag=0
        P_Col=[]
        P_Coll=[]
        P_t_app=[]
        u_app_H = np.zeros((NoI_H, n))
        u_app_R = np.zeros((NoI_R, n))

        P_t_all = np.zeros((n, 1))
        P_Coll = np.zeros((n, 1))
        
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


    

    def Human_robot_action_planner(self,human_position, robot_position,i, x_H, x_R, initial_u_R, initial_u_H, Abar, Bbar, A_H, B_H, w_H, gamma, beta, betas, P_t, g_H_pr, u_H_value, u_H_values, Prediction_Horizon, g_H, theta_3, theta_4, theta_5, theta_6, hat_x_R, hat_x_R_pr, Nc, NoI_H, Abar_H, Bbar_H, eta_1, eta_2, u_app_R, u_app_H, x_H0, x_R0, P_th, Safe_Distance, U_H_constraints, A_R, B_R, P_xH, time, P_Coll, line1, line2, ax3, ax4, ax5, velocity_text_human, velocity_text_robot, P_t_all, Signal, Human):
        
        # Controller code begins here:
        
        # Generate zero-mean Gaussian noise
        epsilon = np.random.normal(mean, std_deviation, num_samples)

        if i == 0:
            u_app_Robot = initial_u_R        
        else:
            u_app_Robot = np.tile(u_app_R[:, i-1], Prediction_Horizon).reshape(-1, 1)
            
        x_pr = Abar @ x_R0 + Bbar @ u_app_Robot    
        hat_x_R_pr = x_pr + epsilon  
        hat_x_R = x_R0 + epsilon  

        # Probability distribution based on human's initial actions or updated actions
        if i == 0:
            P_xH = Probability_distribution_of_human_s_states(
                initial_u_H, u_app_Robot, w_H, gamma, beta, betas, P_t, g_H_pr, u_H_value, 
                u_H_values, Prediction_Horizon, x_H0, g_H, theta_3, theta_4, theta_5, theta_6,
                hat_x_R, hat_x_R_pr, Nc, Abar, Bbar, A_H, B_H, NoI_H, initial_u_H[:NoI_H], Abar_H, Bbar_H, eta_1, eta_2
            )
            P_u_H = Human_Action_Prediction(NoI_H, u_H_value, u_H_values, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, betas, initial_u_H, Prediction_Horizon, Abar_H, Bbar_H, U_H_constraints)
        else:
            P_xH = Probability_distribution_of_human_s_states(
                u_H, u_app_Robot, w_H, gamma, beta, betas, P_t, g_H_pr, u_H_value, 
                u_H_values, Prediction_Horizon, x_H0, g_H, theta_3, theta_4, theta_5, theta_6, 
                hat_x_R, hat_x_R_pr, Nc, Abar, Bbar, A_H, B_H, NoI_H, u_app_H[:, i-1], Abar_H, Bbar_H, eta_1, eta_2
            )
            P_u_H = Human_Action_Prediction(NoI_H, u_H_value, u_H_values, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, betas, u_app_H[:, i-1], Prediction_Horizon, Abar_H, Bbar_H, U_H_constraints)

        # Human's action update
        if i == 0:
            u_H = human_s_action(NoI_H, u_H_value, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, beta, initial_u_H[:NoI_H], Prediction_Horizon, Abar_H, Bbar_H, U_H_constraints)
        else:
            u_H = human_s_action(NoI_H, u_H_value, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, beta, u_app_H[:, i-1], Prediction_Horizon, Abar_H, Bbar_H, U_H_constraints)

        u_app_H[:, i] = u_H[:NoI_H].flatten()
        x_H[:, i+1] = A_H @ x_H[:, i] + B_H @ u_app_H[:, i]

        # Robot's action optimization
        def objective(u_R):
            u_R = u_R.reshape((NoI_R * Prediction_Horizon, 1))
            x_pr = Abar @ x_R0 + Bbar @ u_R
            norm_u_R = np.sum(np.square(u_R))
            norm_x_R_g_R = np.sum(np.square(x_pr - g_R_pr))
            QR_g = theta_1 * norm_x_R_g_R + theta_2 * norm_u_R
            return QR_g[0]

        def constraint1(u_R):
            return np.min(u_R)

        def constraint2(u_R):
            return 2 - np.max(u_R)

        # Custom constraints based on collision probability
        constraints = [{'type': 'ineq', 'fun': constraint1}, {'type': 'ineq', 'fun': constraint2}]
        constraints.extend(custom_constraints(initial_u_R, P_xH, Nc, x_R, Abar, Bbar, Safe_Distance))

        # Perform optimization for robot's actions
        result = minimize(objective, initial_u_R.flatten(), constraints=constraints, method='SLSQP')

        optimized_u_R = result.x.reshape((NoI_R * Prediction_Horizon, 1))
        rounded_u_R = min(u_R_values.flatten(), key=lambda x: np.linalg.norm(np.array([[x]]) - optimized_u_R[:NoI_R]))
        u_app_R[:, i] = rounded_u_R[:NoI_R, 0]

        x_R[:, i+1] = A_R @ x_R[:, i] + B_R @ u_app_R[:, i]

        # Update beliefs and signaling system
        P_t = Robot_s_Belief_About_HDA(u_app_H[:, i].reshape(-1, 1), u_H_values, betas, P_t, P_u_H)
        P_t_all[i] = P_t[1]

        if P_t_all[i] <= 0.08 and Signal == "on":
            if Human == "Unconcerned":
                beta = 1
            elif Human == "Concerned":
                beta = 1

        # Update the plot
        update_plot(P_t_all, P_Coll, P_xH, x_H, i, ax3, ax4, ax5, line1, line2, velocity_text_human, velocity_text_robot)

        linear_vel = u_app_R[:, i]  # You will replace this with the actual MPC calculation
        return linear_vel
        
        
    def human_s_action(NoI_H, u_H_value, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R, eta_1, eta_2, beta, u_H0,Prediction_Horizon_H,Abar_H,Bbar_H,U_H_constraints):
        # Objective function to minimize
        def objective(u_H_flattened):
            u_H = u_H_flattened.reshape(NoI_R * Prediction_Horizon_H, 1)  # Reshape back to 2D for computation
            x_pr_H = Abar_H @ x_H0 + Bbar_H @ u_H
            
            # QH_g term
            norm_x_H_g_H = np.linalg.norm(x_pr_H - g_H_pr) ** 2
            norm_u_H = np.linalg.norm(u_H) ** 2
            QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H

            # QH_s term
            norm_expr = np.linalg.norm(x_pr_H - hat_x_R)
            QH_s = theta_5 * np.exp(-theta_6 * norm_expr ** 2)

            # Total sigma_H        
            sigma_H = eta_1 * QH_g + beta * eta_2 * QH_s

            return float(sigma_H)  # Ensure the returned value is a scalar

        # Constraints
        def constraint1(u_H_flattened):
            u_H = u_H_flattened.reshape(NoI_H * Prediction_Horizon_H, 1)
            return (u_H + U_H_constraints).flatten()  # Flatten for compatibility

        def constraint2(u_H_flattened):
            u_H = u_H_flattened.reshape(NoI_H * Prediction_Horizon_H, 1)
            return (U_H_constraints - u_H).flatten()  # Flatten for compatibility

        # Define constraints as a dictionary
        constraints = [{'type': 'ineq', 'fun': constraint1},
                    {'type': 'ineq', 'fun': constraint2}]
        
        # Flatten the initial guess just for the solver
        # u_H0_flattened = u_H0.flatten()
    
    
            # initial_u_R=u_app_R[:, i-1]
        u_H0_flattened=np.tile(u_H0, (Prediction_Horizon, 1)).flatten()
        # Optimize using scipy's minimize function
        solution = minimize(objective, u_H0_flattened, method='trust-constr', constraints=constraints)
        
        # Extract the optimal value of u_H and reshape it back to 2D
        optimal_u_H = solution.x.reshape(NoI_H * Prediction_Horizon_H, 1)

        # Function to compute the closest value in u_H_values to a given value
        def find_nearest(value, u_H_value):
            return u_H_value[np.argmin(np.abs(u_H_value - value))]

        # Vectorize the find_nearest function to apply it to each element in optimal_u_H
        vectorized_find_nearest = np.vectorize(lambda x: find_nearest(x, u_H_value))

        # Apply the vectorized function to each element in optimal_u_H
        rounded_optimal_u_H = vectorized_find_nearest(optimal_u_H)

        return rounded_optimal_u_H


    # Human Action Prediction
    def Human_Action_Prediction(NoI_H, u_H_value,u_H_values, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, betas, u_H0,Prediction_Horizon_H,Abar_H,Bbar_H,U_H_constraints):
    
        # P_di=[]
        P_d=np.zeros((u_H_values.shape[0],u_H_values.shape[1],betas.shape[0]))
        u_H_optimized=np.zeros((NoI_H,betas.shape[0]))



        for i in range(betas.shape[0]):
            
    
            u_H_optimized_all=human_s_action(NoI_H, u_H_value, x_H0, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, betas[i], u_H0[:NoI_H],Prediction_Horizon_H,Abar_H,Bbar_H,U_H_constraints)
            u_H_optimized[:,i]=u_H_optimized_all[:NoI_H].reshape(NoI_H)

        
        # For each i, search for u_H_optimized(NoI_H, i) in u_H_values
        for i in range(betas.shape[0]):
            # Get the optimized value
            optimized_value = u_H_optimized[:, i].reshape(-1,1)
            # print(optimized_value)
            # Search for the matching pair in u_H_values
            for row in range(u_H_values.shape[0]):
                for col in range(u_H_values.shape[1]):
                    if np.array_equal(u_H_values[row, col], optimized_value):
                        # Set corresponding element in P_d to 1
                        P_d[row, col, i] = 1

        

        
        # Human’s Random Behavior:
        U_H=u_H_values.shape[0]*u_H_values.shape[1]
        P_r=1/U_H
        
        P_u_H=(1-w_H) * P_d+w_H * P_r
        # P_u_Hi=(1-w_H) * P_di+w_H * P_r
        return P_u_H
    
    def human_s_goal(u_H,x_H0,g_H,theta_3,theta_4):
            norm_x_H_g_H = np.linalg.norm(x_H0+u_H - g_H)**2
            norm_u_H = np.linalg.norm(u_H)**2
            QH_g = theta_3 * norm_x_H_g_H + theta_4 * norm_u_H
            return QH_g
        
    def human_s_safety(u_H,x_H0,hat_x_R,theta_5,theta_6):

            a=x_H0+u_H
            b=hat_x_R
            QH_s=theta_5*np.exp(-theta_6*np.linalg.norm(a-b)**2)
            return QH_s

    # Robot’s Belief About the Human’s Danger Awareness
    def Robot_s_Belief_About_HDA(u_H,u_H_values, betas,P_t,P_u_H):
        sum_P_P_t=0.0
        P_ti=np.zeros((betas.shape[0],1))

        # Find the index where u_H matches one of the arrays in u_H_values
        def find_index(u_H_values, u_H):
            for i in range(u_H_values.shape[0]):
                for j in range(u_H_values.shape[1]):
                    if np.array_equal(u_H_values[i, j], u_H):
                        return (i, j)
            return None

        index = find_index(u_H_values, u_H)

        for i in range(betas.shape[0]):
            sum_P_P_t+=P_u_H[index[0],index[1],i]*P_t[i]
            P_ti[i]=P_u_H[index[0],index[1],i]*P_t[i]

        P_t=P_ti/sum_P_P_t

        return P_t

    # Probability distribution of the human’s states
    def Probability_distribution_of_human_s_states(u_H,u_app_Robot,w_H,gamma,beta,betas,P_t,g_H_pr,u_H_value,u_H_values,Prediction_Horizon,x_H0,g_H,theta_3,theta_4,theta_5,theta_6,hat_x_R,hat_x_R_pr,Nc,Abar,Bbar,A_H,B_H,NoI_H,u_H0,Abar_H,Bbar_H,eta_1, eta_2):
                                                
        x_pr = Abar @ x_R0 + Bbar @ u_app_Robot
        x_pr=np.vstack((x_pr,1.0))
        P=np.zeros((Prediction_Horizon,Nc.shape[0],Nc.shape[1]))
        P_x_H=np.zeros((Nc.shape[0],Nc.shape[1]))
        x_H_next=np.zeros((u_H_values.shape[0],u_H_values.shape[1]))
        new_cell = []
        new_cells=[1]
        new_cell.append(1)
        P_x_H_iks=[]
        x_H_next=x_H0
        u_H=u_H.reshape(Prediction_Horizon,NoI_H,1)
        for j in range(Prediction_Horizon):

            sum_x_H=0.0
            if j>=1:
                tuple_list = [tuple(arr) for arr in new_cell]
                # Find unique tuples using a set
                unique_tuples = set(tuple_list)
                unique_tuples=list(unique_tuples)
                x_H0_new=[Nc[f] for f in unique_tuples]   
                new_cells=(unique_tuples)
                new_cell = []
            for n in range(len(np.array(new_cells))):
            
                if j>=1:
                    x_H0_prob=x_H0_new[n]
                
                for mx in range(Nc.shape[0]):
                    for my in range(Nc.shape[1]):
                
                        for kx in range(u_H_values.shape[0]):
                            for ky in range(u_H_values.shape[1]):
                                if j==0:
                                    x_H0_prob=x_H0
                                sds=np.array(u_H_values[kx,ky])
                                x_H_next=A_H@x_H0_prob+B_H@u_H_values[kx,ky]

                                if np.allclose(x_H_next, Nc[mx, my], atol=tolerance):
                                        
                                    if j==0:
                                        P_x_H_k=np.array([1.0])
                                    else:
                                        P_x_H_k=np.array([P[j-1,np.array(new_cells[n])[0],np.array(new_cells[n])[1]]])
                                
                                    
                                    

                                
                                    P_u_H=Human_Action_Prediction(NoI_H, u_H_value,u_H_values, x_H0_prob, g_H_pr, theta_3, theta_4, theta_5, theta_6, hat_x_R_pr, eta_1, eta_2, betas, u_H0,Prediction_Horizon_H,Abar_H,Bbar_H,U_H_constraints)

                                        




                                    for i in range(betas.shape[0]):
                                    
                                        P_x_H_iks.append(P_x_H_k*P_u_H[kx,ky,i]*P_t[i])
                                        if P_x_H_k*P_u_H[kx,ky,i]*P_t[i]!=0:
                                            new_cell.append(np.array([mx,my]))         
                                        sum_x_H+=P_x_H_k*P_u_H[kx,ky,i]*P_t[i]

                        sssss=np.array(P_x_H_iks).reshape(-1,1)
                        sssssst=np.sum(sssss)
                        P_x_H_iks=[]
                        P_x_H[mx,my]=sssssst

                if j==0:
                    P_x_Hn=np.zeros((1,Nc.shape[0],Nc.shape[1]))
                    P_x_Hn[:,:,:]=P_x_H 
                else:
                    P_x_Hn[n,:,:]=P_x_H             
            
            if j==0:
                P[j,:,:]= P_x_H/(sum_x_H) 

                new_cell=new_cell[1:]
                P_x_Hn=np.zeros((len(np.array(new_cell)),Nc.shape[0],Nc.shape[1]))

            else:              
                PPPPP=np.sum(P_x_Hn, axis=0)   
                P[j,:,:]=PPPPP/(sum_x_H) 
                P_x_Hn=np.zeros((len(np.array(new_cell)),Nc.shape[0],Nc.shape[1]))

            epsilon = np.random.normal(mean, std_deviation, num_samples)
            hat_x_R=x_pr[j+1]+epsilon  

        np.save('P.npy', P)
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
