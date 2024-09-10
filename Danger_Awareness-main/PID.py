#!/usr/bin/env python

import rospy
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from irobot_create_msgs.msg import WheelVels  # For wheel velocities

class RobotPDTrackingController:
    def __init__(self):
        # PD controller gains for position (linear) and angle (angular)
        self.kp_linear = 2.0
        self.kd_linear = 0.5
        self.kp_angular = 1.5
        self.kd_angular = 0.1

        # Desired goal (x, y, theta)
        self.goal_x = 3.0  # meters
        self.goal_y = 3.0  # meters
        self.goal_theta = 0.0  # radians (desired orientation)

        # Initialize position and velocity errors
        self.prev_pos_error = 0.0
        self.prev_angle_error = 0.0
        self.prev_linear_vel = 0.0  # Use wheel velocity for derivative
        self.prev_time = None

        # Wheelbase (distance between the two wheels) in meters
        self.wheelbase = 0.3

        # Initialize the node
        rospy.init_node('robot_pd_tracking_controller', anonymous=True)

        # Subscribe to odometry topic to get the robot's position and orientation
        self.odom_sub = rospy.Subscriber('/mobile_base/odom', Odometry, self.odom_callback)

        # Subscribe to wheel velocities to get linear velocity directly
        self.wheel_vels_sub = rospy.Subscriber('/mobile_base/wheel_vels', WheelVels, self.wheel_vels_callback)

        # Publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/cmd_vel', Twist, queue_size=10)

        # Store wheel velocities
        self.left_wheel_vel = 0.0
        self.right_wheel_vel = 0.0

        rospy.loginfo("Robot PD Tracking Controller Initialized")

    def wheel_vels_callback(self, msg):
        # Store the current wheel velocities
        self.left_wheel_vel = msg.velocity_left
        self.right_wheel_vel = msg.velocity_right

    def odom_callback(self, msg):
        # Extract current position and orientation from odometry
        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y

        # Get robot's current orientation (yaw) from quaternion
        orientation_q = msg.pose.pose.orientation
        yaw = self.quaternion_to_euler(orientation_q)

        # Get current time for time delta calculation
        current_time = rospy.Time.now().to_sec()
        if self.prev_time is None:
            self.prev_time = current_time

        dt = current_time - self.prev_time

        # Compute the position error (Euclidean distance to the goal)
        pos_error = math.sqrt((self.goal_x - current_x) ** 2 + (self.goal_y - current_y) ** 2)

        # Compute the desired angle to the goal
        desired_theta = math.atan2(self.goal_y - current_y, self.goal_x - current_x)

        # Compute the angle error (difference between desired and current orientation)
        angle_error = desired_theta - yaw

        # Normalize angle error to the range [-pi, pi]
        angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

        # Calculate current linear velocity using wheel velocities
        linear_vel_robot = (self.left_wheel_vel + self.right_wheel_vel) / 2.0

        # PD control for linear velocity (proportional to the position error)
        linear_error_deriv = (linear_vel_robot - self.prev_linear_vel) / dt
        linear_vel = self.kp_linear * pos_error + self.kd_linear * linear_error_deriv

        # PD control for angular velocity (proportional to the angle error)
        angular_error_deriv = (angle_error - self.prev_angle_error) / dt
        angular_vel = self.kp_angular * angle_error + self.kd_angular * angular_error_deriv

        # Publish the velocity commands to the robot
        cmd_msg = Twist()
        cmd_msg.linear.x = linear_vel
        cmd_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(cmd_msg)

        rospy.loginfo(f"Current Position: ({current_x}, {current_y}), Yaw: {yaw}")
        rospy.loginfo(f"Linear Velocity: {linear_vel_robot}, Angular Velocity: {angular_vel}")

        # Update previous errors and time
        self.prev_pos_error = pos_error
        self.prev_angle_error = angle_error
        self.prev_linear_vel = linear_vel_robot
        self.prev_time = current_time

    def quaternion_to_euler(self, orientation_q):
        # Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw)
        x = orientation_q.x
        y = orientation_q.y
        z = orientation_q.z
        w = orientation_q.w

        # Perform conversion to Euler yaw (around Z-axis)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return yaw

if __name__ == '__main__':
    try:
        controller = RobotPDTrackingController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
