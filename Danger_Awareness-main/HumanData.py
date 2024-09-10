#!/usr/bin/env python

import rospy
from leg_tracker.msg import PersonArray
from geometry_msgs.msg import Pose

def callback(data):
    # Print received PersonArray message
    print("Received PersonArray message")

    # Check if there are any people in the message
    if not data.people:
        print("No people tracked")
        return
    
    # Iterate over all tracked people in the message
    for person in data.people:
        position = person.pose.position
        # Print the person's ID and position
        print(f"Person ID: {person.id}")
        print(f"Position - x: {position.x:.2f}, y: {position.y:.2f}, z: {position.z:.2f}")

def listener():
    # Initialize the ROS node
    rospy.init_node('person_position_listener', anonymous=True)
    
    # Subscribe to the /people_tracked topic
    rospy.Subscriber('/people_tracked', PersonArray, callback)
    
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    listener()
