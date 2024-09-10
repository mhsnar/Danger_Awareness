#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math

class RobotPDTrackingController:
    def __init__(self):
        rospy.init_node('robot_pd_tracking_controller')

        # Goal position (set these to your desired target)
        self.goal_x = 0.0
        self.goal_y = 0.0

        # PD gains
        self.kp_linear = 1.0
        self.kd_linear = 0.1
        self.kp_angular = 4.0
        self.kd_angular = 0.1

        # Robot state variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.prev_x = None
        self.prev_y = None
        self.prev_yaw = None
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0
        self.prev_time = rospy.Time.now().to_sec()

        # Publisher and subscriber
        self.cmd_vel_pub = rospy.Publisher('/mobile_base/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/mobile_base/odom', Odometry, self.odom_callback)

        # Set the control loop rate (5 Hz -> 0.2 sec)
        self.rate = rospy.Rate(5)

        rospy.loginfo("Robot PD Tracking Controller Initialized")

    def quaternion_to_euler(self, q):
        """
        Convert quaternion to Euler angles (yaw).
        """
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)  # Return only yaw

    def odom_callback(self, msg):
        # Extract position and orientation from odometry
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        self.current_yaw = self.quaternion_to_euler(orientation_q)

        # Get current time for time delta calculation
        current_time = rospy.Time.now().to_sec()
        dt = current_time - self.prev_time

        # Proceed only if time difference (dt) exceeds the desired sampling time (0.2 seconds)
        if dt >= 0.2:
            if self.prev_x is not None and self.prev_y is not None and self.prev_yaw is not None:
                # Compute the estimated linear and angular velocities
                dx = self.current_x - self.prev_x
                dy = self.current_y - self.prev_y
                dtheta = self.current_yaw - self.prev_yaw

                # Normalize angle difference to the range [-pi, pi]
                dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi

                # Estimated velocities
                linear_vel_robot = math.sqrt(dx ** 2 + dy ** 2) / dt
                angular_vel_robot = dtheta / dt

                # Compute position error (Euclidean distance to the goal)
                pos_error = math.sqrt((self.goal_x - self.current_x) ** 2 + (self.goal_y - self.current_y) ** 2)

                # Compute desired angle to the goal
                desired_theta = math.atan2(self.goal_y - self.current_y, self.goal_x - self.current_x)

                # Compute angle error
                angle_error = desired_theta - self.current_yaw
                angle_error = (angle_error + math.pi) % (2 * math.pi) - math.pi

                # PD control for linear velocity
                linear_error_deriv = (linear_vel_robot - self.prev_linear_vel) / dt
                linear_vel = self.kp_linear * pos_error + self.kd_linear * linear_error_deriv

                # PD control for angular velocity
                angular_error_deriv = (angular_vel_robot - self.prev_angular_vel) / dt
                angular_vel = self.kp_angular * angle_error + self.kd_angular * angular_error_deriv

                # Publish velocity commands
                cmd_msg = Twist()
                cmd_msg.linear.x = linear_vel
                cmd_msg.angular.z = angular_vel
                self.cmd_vel_pub.publish(cmd_msg)

                rospy.loginfo(f"Current Position: ({self.current_x}, {self.current_y}), Yaw: {self.current_yaw}")
                rospy.loginfo(f"Linear Velocity: {linear_vel_robot}, Angular Velocity: {angular_vel_robot}")

                # Update previous values
                self.prev_linear_vel = linear_vel_robot
                self.prev_angular_vel = angular_vel_robot

            # Update previous odometry values
            self.prev_x = self.current_x
            self.prev_y = self.current_y
            self.prev_yaw = self.current_yaw
            self.prev_time = current_time

    def run(self):
        while not rospy.is_shutdown():
            # The control logic is handled in the odom_callback
            self.rate.sleep()


if __name__ == '__main__':
    try:
        controller = RobotPDTrackingController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
