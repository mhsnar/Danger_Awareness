#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist

def publish_velocity():
    # Initialize the ROS node
    rospy.init_node('velocity_publisher', anonymous=True)
    
    # Create a publisher object
    pub = rospy.Publisher('/mobile_base/cmd_vel', Twist, queue_size=10)
    
    # Wait for the publisher to be connected
    rospy.loginfo("Waiting for publisher to connect...")
    while pub.get_num_connections() == 0:
        rospy.sleep(0.1)
    
    # Create a Twist message instance
    velocity_msg = Twist()
    velocity_msg.linear.x = 0.0
    velocity_msg.linear.y = 0.0
    velocity_msg.linear.z = 0.0
    velocity_msg.angular.x = 0.0
    velocity_msg.angular.y = 0.0
    velocity_msg.angular.z = -0.4
    
    # Publish the message
    rospy.loginfo("Publishing velocity command")
    pub.publish(velocity_msg)
    
    # Keep the message published for 3 seconds
    rospy.sleep(3)
    
    # Stop the robot by publishing zero velocities
    rospy.loginfo("Stopping the robot")
    stop_msg = Twist()  # All velocities are zero by default
    pub.publish(stop_msg)
    
    # Wait for a bit before shutting down
    rospy.sleep(1)

if __name__ == '__main__':
    try:
        publish_velocity()
    except rospy.ROSInterruptException:
        pass
