#!/usr/bin/env python

import rospy
from irobot_create_msgs.msg import WheelVels  # Import the correct message type

class WheelVelocitySubscriber:
    def __init__(self):
        # Initialize the node
        rospy.init_node('wheel_velocity_subscriber', anonymous=True)
        
        # Subscribe to the wheel velocities topic
        self.wheel_vel_sub = rospy.Subscriber('/mobile_base/wheel_vels', WheelVels, self.wheel_vel_callback)
        
        rospy.loginfo("Subscribed to /mobile_base/wheel_vels")

    def wheel_vel_callback(self, msg):
        # Access the left and right wheel velocities from the WheelVels message
        left_wheel_vel = msg.velocity_left
        right_wheel_vel = msg.velocity_right
        
        rospy.loginfo(f"Left Wheel Velocity: {left_wheel_vel}, Right Wheel Velocity: {right_wheel_vel}")
        
        # Convert wheel velocities to linear and angular velocities
        linear_vel = (left_wheel_vel + right_wheel_vel) / 2.0
        angular_vel = (right_wheel_vel - left_wheel_vel) / 0.5  # Adjust 0.5 according to the wheelbase
        
        rospy.loginfo(f"Linear Velocity: {linear_vel}, Angular Velocity: {angular_vel}")
        
if __name__ == '__main__':
    try:
        WheelVelocitySubscriber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
