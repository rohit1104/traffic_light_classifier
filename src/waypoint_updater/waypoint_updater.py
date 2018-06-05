#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        self.pose = None
        self.waypoints = None
        self.waypoints_xy = None
        self.waypoints_tree = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        rate = rospy.Rate(40)
        while not rospy.is_shutdown():
            if self.pose and self.waypoints_tree:
                curr_x = self.pose.position.x
                curr_y = self.pose.position.y
                curr_point = [curr_x, curr_y]
                
                closest_waypoint_index = self.waypoints_tree.query([curr_x, curr_y], 1)[1]
                closest_waypoint = self.waypoints_xy[closest_waypoint_index]

                previous_waypoint_index = (closest_waypoint_index-1) % len(self.waypoints_xy)
                previous_waypoint = self.waypoints_xy[previous_waypoint_index]

		closest_waypoint_vector = np.array(closest_waypoint)
		previous_waypoint_vector = np.array(previous_waypoint)
		curr_point_vector = np.array(curr_point)

                vector1 = closest_waypoint_vector - previous_waypoint_vector
                vector2 = curr_point_vector - closest_waypoint_vector
                val = np.dot(vector1, vector2)
                if val > 0:
                    closest_waypoint_index = (closest_waypoint_index+1) % len(self.waypoints_xy)

                lane = Lane()
                lane.waypoints = self.waypoints[closest_waypoint_index: closest_waypoint_index+LOOKAHEAD_WPS]
                self.final_waypoints_pub.publish(lane)
            
            rate.sleep()


    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg.pose

    def waypoints_cb(self, msg):
        # TODO: Implement
        if not self.waypoints:
            self.waypoints = msg.waypoints
            self.waypoints_xy = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in self.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_xy)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
