#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from leg_tracker.msg import PersonArray
import math
import yaml

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
        self.inc = 0  
        #Constatnt Contoller parameters----------------------------------------------------------------------------------------------------------------
        # Robot Model
        self.experimental_data = {
            'u_app_H': [],
            'P_t_all': [],
            'time': [],
            'P_xH_all': [],
            'P_Coll': [],
            'x_H': [],
            'x_R': [],
            'u_app_R': [],
            'tictoc': []
        }

        # Robot Model
        self.n = 500
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

        self.g_H = np.array([[-5], [0.0]])
        self.g_H_pr = np.tile(self.g_H, (self.Prediction_Horizon_H, 1))
        self.g_R = np.array([[10], [0.0]]).reshape(-1, 1)
        self.g_R_pr = np.tile(self.g_R, (self.Prediction_Horizon, 1))
        self.v_R = 1.0
        self.v_h = .3
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
        self.tictoc= []
        
        self.theta_6 = np.array([1.]).reshape(-1, 1)
        U_H_constraint=np.array([[1], [1]]) 
        initial_u_H=np.array([[0.],[0.]])
        initial_u_R=np.array([[0],[0]])
        
        self.U_H_constraints=np.tile(U_H_constraint, (self.Prediction_Horizon, 1))
        self.initial_u_H=np.tile(initial_u_H, (self.Prediction_Horizon, 1))
        self.initial_u_RR=np.tile(initial_u_R, (self.Prediction_Horizon, 1))

        self.mean = 0  # Zero mean for the Gaussian noise
        self.covariance = 2  # Example covariance (which is the variance in 1D)
        self.std_deviation = np.sqrt(self.covariance)  # Standard deviation
        self.num_samples = 1  # Number of samples
        self.tolerance=1e-5

        # PD gains
        self.kp_linear = 1.0
        self.kd_linear = 0.1
        self.kp_angular = 4.0
        self.kd_angular = 0.1

        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0


        
        self.P_Col=[]
        self.P_Coll=[]
        self.optimized_u_R=np.zeros((self.NoI_R * self.Prediction_Horizon, self.n))
        self.P_t_app=[]
        self.u_app_H = np.zeros((self.NoI_H, self.n))
        self.u_app_R = np.zeros((self.NoI_R, self.n))
        self.P_xH_all=np.zeros((self.n,self.Prediction_Horizon,self.Nc.shape[0],self.Nc.shape[1]))
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
        self.prev_x = None
        self.prev_y = None
        self.prev_yaw = None


        # Human state variables
        self.current_x_human = 0.0
        self.current_y_human = 0.0
        self.current_yaw_human= 0.0
        self.human_position = None
        self.prev_x_human = None
        self.prev_y_human = None
        self.prev_yaw_human = None
        self.prev_time = rospy.Time.now().to_sec()

        # Publisher and subscriber
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/mobile_base/odom', Odometry, self.odom_callback)
        
        self.people_sub = rospy.Subscriber('/people_tracked', PersonArray, self.people_callback)

        # Set the control loop rate (5 Hz -> 0.2 sec)
        self.rate = rospy.Rate(5)

        # self.reset_odometry()

        rospy.loginfo("Robot MPC Tracking Controller Initialized")

        # self.save_experimental_data
        rospy.on_shutdown(self.save_experimental_data)

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

    # def odom_callback(self, msg):
    #     # Extract robot's position and orientation from odometry
    #     self.current_x = msg.pose.pose.position.x
    #     self.current_y = msg.pose.pose.position.y

    #     orientation_q = msg.pose.pose.orientation
    #     self.current_yaw = self.quaternion_to_euler(orientation_q)

    #     if self.human_position is not None:
    #         # Call the MPC-based planner
    #         linear_vel = self.Human_robot_action_planner(self.human_position, (self.current_x, self.current_y))

    #         # Publish velocity commands
    #         cmd_msg = Twist()
    #         cmd_msg.linear.x = linear_vel[0, 0]  # X velocity
    #         cmd_msg.linear.y = linear_vel[1, 0] 
    #         self.cmd_vel_pub.publish(cmd_msg)

    #         rospy.loginfo(f"Robot Position: ({self.current_x}, {self.current_y}), Human Position: ({self.human_position.x}, {self.human_position.y})")
    #         rospy.loginfo(f"Linear Velocity: {linear_vel}")
    #         self.inc += 1

    def odom_callback(self, msg):
        # Extract robot's position and orientation from odometry
        
        self.current_x = msg.pose.pose.position.x-4.84619-5.
        
        print("x",self.current_x )
        self.current_y = msg.pose.pose.position.y-4.89829
        print(self.current_y )
        orientation_q = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_euler(orientation_q)
        print(self.current_yaw)

        # Calculate the time difference for velocity estimation
        current_time = rospy.Time.now().to_sec()  # Get the current time

        # Initialize self.prev_time if not done previously
        if self.prev_time is None:
            self.prev_time = current_time
            return  # Exit callback until next iteration when time difference can be computed

        dt = current_time - self.prev_time  # Time difference in seconds

        # Proceed only if time difference (dt) exceeds the desired sampling time (0.2 seconds)
        if dt >= 0.2:
            if (self.human_position is not None and
                self.prev_x is not None and 
                self.prev_y is not None and 
                self.prev_yaw is not None):

                # Extract human position (assuming self.human_position is a tuple of (x, y))
                self.current_x_human = self.human_position.x*math.cos(self.current_yaw )+self.human_position.x*math.sin(self.current_yaw )+self.current_x 
                self.current_y_human = self.human_position.y*math.cos(self.current_yaw )-self.human_position.y*math.sin(self.current_yaw )+self.current_y

                           # Convert human relative position to global position using correct trigonometric transformation
                # self.current_x_human = self.human_position.x 
                                        
                # self.current_y_human = self.human_position.y
                print("x",self.current_x_human)
                print(self.current_y_human)
                # Update previous time
                self.prev_time = current_time

                dx = self.current_x - self.prev_x
                dy = self.current_y - self.prev_y
                dtheta = self.current_yaw - self.prev_yaw

                # Normalize angle difference to the range [-pi, pi]
                dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi

                # Estimated velocities
                linear_vel_robot = math.sqrt(dx ** 2 + dy ** 2) / dt
                angular_vel_robot = dtheta / dt

                # Compute position error (Euclidean distance to the goal)
                pos_error = math.sqrt((self.current_x_human - self.current_x) ** 2 + (self.current_y_human - self.current_y) ** 2)

                # Compute desired angle to the goal
                desired_theta = math.atan2(self.current_y_human- self.current_y, self.current_x_human - self.current_x)

                # Compute angle error
                angle_error = desired_theta - self.current_yaw
                angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

                # PD control for linear velocity
                linear_error_deriv = (linear_vel_robot - self.prev_linear_vel) / dt
                linear_vel = self.kp_linear * pos_error + self.kd_linear * linear_error_deriv

                # PD control for angular velocity
                angular_error_deriv = (angular_vel_robot - self.prev_angular_vel) / dt
                angular_vel = self.kp_angular * angle_error + self.kd_angular * angular_error_deriv

      
                #--------------------------------------------------------------
                # Human velocity estimation
                Vx_human = (self.current_x_human - self.prev_x_human) / dt if dt > 0 else 0
                Vy_human = (self.current_y_human - self.prev_y_human) / dt if dt > 0 else 0

                # Estimated human linear velocity (as a (2,1) array)
                linear_vel_human = np.array([[Vx_human], [Vy_human]])

                #--------------------------------------------------------------

                # Compute desired angle to the goal (robot's movement towards a goal)
                # desired_theta = math.atan2(self.g_R[1] - self.current_y, self.g_R[0] - self.current_x)

                # # Compute angle error for angular control
                # angle_error = desired_theta - self.current_yaw
                # angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

                # PD control for angular velocity
                # angular_error_deriv = (angular_vel_robot - self.prev_angular_vel) / dt if dt > 0 else 0
                # angular_vel = self.kp_angular * angle_error + self.kd_angular * angular_error_deriv
                tic=rospy.Time.now().to_sec()
                # Call the MPC-based planner to get the linear velocity
                # vel = self.Human_robot_action_planner(self.human_position, (self.current_x, self.current_y), linear_vel_human)
                # linear_vel = math.sqrt(vel[0]**2 + vel[1]**2)
                toc=rospy.Time.now().to_sec()
                self.tictoc=(toc-tic)
                
                vel=.1
                # print(vel)
                # Calculate the current angle of the velocity vector
                # theta = math.atan2(vel[1], vel[0])  # Angle in radians

                # Calculate the angular velocity (rate of change of theta over time)
                # angular_velocity = theta

                # Normalize the angular velocity to ensure it is in the range [-pi, pi]
                # angular_vel = (angular_velocity + np.pi) % (2 * np.pi) - np.pi


                MAX_LINEAR_VEL = 1.0  # Maximum linear velocity
                MIN_LINEAR_VEL = -1.0  # Minimum linear velocity
                MAX_ANGULAR_VEL = 1.0  # Maximum angular velocity
                MIN_ANGULAR_VEL = -1.0  # Minimum angular velocity

                # Saturate linear velocity
                linear_vel = max(MIN_LINEAR_VEL, min(MAX_LINEAR_VEL, linear_vel))

                # Saturate angular velocity
                angular_vel = max(MIN_ANGULAR_VEL, min(MAX_ANGULAR_VEL, angular_vel))

                self.save_iteration_data(linear_vel_human, vel, self.current_x, self.current_y, self.current_x_human, self.current_y_human,self.tictoc)
                # Publish velocity commands to the robot
                cmd_msg = Twist()
                cmd_msg.linear.x = linear_vel # Use the planned linear velocity
                cmd_msg.angular.z = angular_vel  # Use the calculated angular velocity

                self.cmd_vel_pub.publish(cmd_msg)

                # rospy.loginfo(f"Robot Position: ({self.current_x}, {self.current_y}), Human Position: ({self.human_position.x}, {self.human_position.y})")
                # rospy.loginfo(f"Linear Velocity: {linear_vel}, Angular Velocity: {angular_vel}")
                self.inc += 1

                


                # self.prev_yaw_human should be defined elsewhere if yaw tracking for the human is needed

                # Uncomment and update these if necessary for tracking velocities
                # self.prev_linear_vel = linear_vel_robot
                self.prev_angular_vel = angular_vel_robot
            # Update previous values for the robot and human
            self.prev_x_human = self.current_x_human
            self.prev_y_human = self.current_y_human
            self.prev_x = self.current_x
            self.prev_y = self.current_y
            self.prev_yaw = self.current_yaw
            # Update previous time if time has elapsed sufficiently
            self.prev_time = current_time


    # def reset_odometry(self):
    #         pub = rospy.Publisher('/mobile_base/odom', Odometry, queue_size=10)
            
    #         reset_odom = Odometry()
    #         reset_odom.pose.pose.position.x = 0.0
    #         reset_odom.pose.pose.position.y = 0.0
    #         reset_odom.pose.pose.position.z = 0.0
    #         reset_odom.pose.pose.orientation.x = 0.0
    #         reset_odom.pose.pose.orientation.y = 0.0
    #         reset_odom.pose.pose.orientation.z = 0.0
    #         reset_odom.pose.pose.orientation.w = 1.0
            
    #         rospy.sleep(1)  # Give some time for the publisher to initialize
    #         pub.publish(reset_odom)
    #         rospy.loginfo("Odometry reset to (0, 0, 0) with orientation (0, 0, 0, 1).")
    

    def Human_robot_action_planner(self, human_position, robot_position,linear_vel_human):
        constraints = []
        x_human = human_position.x
        y_human= human_position.y
        x_robot, y_robot = robot_position

        self.u_app_H[:,self.inc]=linear_vel_human.flatten()
 
        x_H0 = np.array([[x_human], [y_human]])
        x_R0 = np.array([[x_robot], [y_robot]])

        # Generate zero-mean Gaussian noise
        epsilon = np.random.normal(self.mean, self.std_deviation, self.num_samples)

        if self.inc  == 0:
            u_app_Robot = self.initial_u_RR
        else:
            u_app_Robot = np.tile(self.u_app_R[:, self.inc -1], self.Prediction_Horizon).reshape(-1, 1)

        x_pr = self.Abar @ x_R0 + self.Bbar @ u_app_Robot
        hat_x_R_pr = x_pr + epsilon
        hat_x_R = x_R0 + epsilon

        # Probability distribution based on human's initial actions or updated actions
        P_xH = self.Probability_distribution_of_human_s_states(u_app_Robot, self.P_t, x_H0, hat_x_R)
        P_u_H = self.Human_Action_Prediction(x_H0, hat_x_R)
        self.P_xH_all[self.inc,:,:,:]=P_xH

        # Human's action update
        if self.inc  == 0:
            u_H = self.human_s_action(x_H0, hat_x_R_pr, self.initial_u_H[:self.NoI_H])
        else:
            u_H = self.human_s_action(x_H0, hat_x_R_pr, self.u_app_H[:, self.inc -1])

        self.u_app_H[:, self.inc ] = u_H[:self.NoI_H].flatten()

        # Objective function for robot's goal
        def objective(u_R):
            u_R = u_R.reshape((self.NoI_R * self.Prediction_Horizon, 1))
            x_pr = self.Abar @ x_R0 + self.Bbar @ u_R
            norm_u_R = np.sum(np.square(u_R))
            norm_x_R_g_R = np.sum(np.square(x_pr - self.g_R_pr))
            QR_g = self.theta_1 * norm_x_R_g_R + self.theta_2 * norm_u_R
            return QR_g

        # Constraints
        def constraint1(u_R):
            return np.min(u_R) + 1.0

        def constraint2(u_R):
            return 1.0 - np.max(u_R)

        def constraint3(u_R):
            x_pr = self.Abar @ x_R0 + self.Bbar @ u_R.reshape((self.NoI_R * self.Prediction_Horizon, 1))
            return x_pr[0]

        # Custom constraints based on P_xH
        def custom_constraints(u_R):
            for t in range(P_xH.shape[0]):
                matrix = P_xH[t, :, :]
                if np.any(matrix > 0.0):
                    indices = np.where(matrix > 0.0)
                    indices = np.array(indices)
                    for tt in range(indices.shape[1]):
                        if matrix[indices[0][tt], indices[1][tt]] > self.P_th:
                            def constraint_fun(u_R):
                                u_R = u_R.reshape((self.NoI_R * self.Prediction_Horizon, 1))
                                x_pr_t = self.Abar @ x_R0 + self.Bbar @ u_R
                                Cons = np.linalg.norm(self.Nc[indices[0, tt], indices[1, tt]] - x_pr_t[self.NoI_R * t:self.NoI_R * (t + 1)]) - self.Safe_Distance
                                return Cons
                            constraints.append({'type': 'ineq', 'fun': constraint_fun})
                        else:
                            self.P_Col.append(np.array(0.0))
            return constraints

        # Initial guess for optimization
        if self.inc  >= 1:
            initial_u_R = np.vstack([self.optimized_u_R[self.NoI_R:, self.inc -1].reshape(-1, 1), self.optimized_u_R[-self.NoI_R:, self.inc -1].reshape(-1, 1)])
        else:
            initial_u_R = self.initial_u_RR

        # Setup constraints for minimize
        if x_R0[1] <= -50:
            constraints = [{'type': 'ineq', 'fun': constraint1},
                        {'type': 'ineq', 'fun': constraint2},
                        {'type': 'eq', 'fun': constraint3}]
        else:
            constraints = [{'type': 'ineq', 'fun': constraint1},
                        {'type': 'ineq', 'fun': constraint2}]
        custom_constraints(initial_u_R)

        # Perform the optimization
        result = minimize(objective, initial_u_R.flatten(), constraints=constraints, method='SLSQP')

        # Store the optimized values
        self.optimized_u_R[:, self.inc ] = result.x
        rounded_u_R = self.optimized_u_R[:, self.inc ][:self.NoI_R]
        self.u_app_R[:, self.inc ] = rounded_u_R

        # Update belief
        self.P_t = self.Robot_s_Belief_About_HDA(self.u_app_H[:, self.inc ].reshape(-1, 1), self.P_t, P_u_H)
        self.P_t_all[self.inc ] = self.P_t[1]

        # Return the linear velocity
        linear_vel = self.u_app_R[:, self.inc ]
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
                QH_g=self.human_s_goal(self.u_H_values[kx,ky],x_H0)
                QH_s=self.human_s_safety(self.u_H_values[kx,ky],x_H0,hat_x_R)
                
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
    def save_experimental_data(self):
        # Convert lists to numpy arrays before saving
        for key in self.experimental_data:
            self.experimental_data[key] = np.array(self.experimental_data[key])

        # Save the experimental data to a .npz file
        np.savez('experiment_data7a.npz', **self.experimental_data)
        rospy.loginfo("Experimental data saved successfully.")

    def save_iteration_data(self, linear_vel_human, vel, current_x, current_y, current_x_human, current_y_human,tictoc):
        # Save data from the current iteration into the experimental data dictionary
        self.experimental_data['u_app_H'].append(linear_vel_human)
        self.experimental_data['u_app_R'].append(vel)
        self.experimental_data['x_H'].append([self.current_x_human, self.current_y_human])
        self.experimental_data['x_R'].append([self.current_x, self.current_y])
        self.experimental_data['P_xH_all'].append(self.P_xH_all)
        self.experimental_data['P_t_all'].append(self.P_t_all)
        self.experimental_data['tictoc'].append(tictoc)
        
        
        # Add other variables like P_t_all, P_Coll, etc. if needed
        # You can also save timestamps or iteration counts
        self.experimental_data['time'].append(rospy.Time.now().to_sec())

    def run(self):
        while not rospy.is_shutdown():
            self.rate.sleep()


if __name__ == '__main__':
    try:
        # rospy.init_node('robot_mpc_tracking_controller')
        controller = RobotMPCTrackingController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
