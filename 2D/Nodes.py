#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from leg_tracker.msg import PersonArray

# Function to move robot based on person's position
def move_robot_based_on_person(data):
    if len(data.people) > 0:
        person = data.people[0]  # Taking the first tracked person
        person_position = person.pose.position
        
        # Initialize velocity command
        cmd_vel = Twist()
        
        # Simple logic to move the robot towards the person
        # Assuming you want the robot to follow in the X-Y plane
        if abs(person_position.x) > 0.1:  # Tolerance in X
            cmd_vel.linear.x = 0.5 if person_position.x > 0 else -0.5
        
        if abs(person_position.y) > 0.1:  # Tolerance in Y
            cmd_vel.angular.z = 0.5 if person_position.y > 0 else -0.5
        
        # Publish the velocity command
        vel_pub.publish(cmd_vel)
    else:
        # Stop the robot if no one is tracked
        cmd_vel = Twist()
        vel_pub.publish(cmd_vel)

if __name__ == '__main__':
    rospy.init_node('person_follower')

    # Publisher to command velocity
    vel_pub = rospy.Publisher('/mobile_base/cmd_vel', Twist, queue_size=10)

    # Subscriber to the people_tracked topic
    rospy.Subscriber('/people_tracked', PersonArray, move_robot_based_on_person)

    # Keep the node running
    rospy.spin()
