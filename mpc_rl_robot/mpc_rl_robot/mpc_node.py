#!/usr/bin/env python3
"""
MPC ROS 2 Node — Low-Level Controller
Subscribes to /odom, publishes to /cmd_vel
PhD Preliminary Results — MPC-RL Hierarchical Navigation
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np
import math

from mpc_rl_robot.mpc_controller import MPCController

class MPCNode(Node):
    def __init__(self):
        super().__init__('mpc_node')

        # --- MPC Controller ---
        self.mpc = MPCController()

        # --- Goal Position (we will later get this from RL agent) ---
        self.goal = np.array([0.0, 7.0, 0.0])   # x, y, theta

        # --- Robot State ---
        self.current_state = np.array([0.0, 0.0, 0.0])
        self.goal_reached  = False

        # --- ROS 2 Publisher & Subscriber ---
        self.cmd_pub = self.create_publisher(
            Twist, '/cmd_vel', 10)

        self.odom_sub = self.create_subscription(
            Odometry, '/odom',
            self.odom_callback, 10)

        # --- Timer: runs MPC at 10 Hz ---
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('[MPC Node] Started. Navigating to goal...')

    def odom_callback(self, msg):
        """Extract robot state from odometry."""
        x   = msg.pose.pose.position.x
        y   = msg.pose.pose.position.y

        # Convert quaternion to yaw angle
        q   = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)

        self.current_state = np.array([x, y, theta])

    def control_loop(self):
        """Main MPC control loop — runs at 10 Hz."""

        if self.goal_reached:
            self.stop_robot()
            return

        # Check if goal is reached
        dist = np.linalg.norm(self.current_state[:2] - self.goal[:2])
        if dist < 0.1:
            self.get_logger().info('[MPC Node] Goal reached!')
            self.goal_reached = True
            self.stop_robot()
            return

        # Compute MPC control
        v, omega = self.mpc.compute_control(
            self.current_state, self.goal)

        # Publish velocity command
        cmd = Twist()
        cmd.linear.x  = v
        cmd.angular.z = omega
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'[MPC] State: ({self.current_state[0]:.2f}, '
            f'{self.current_state[1]:.2f}) | '
            f'dist: {dist:.2f}m | '
            f'v: {v:.3f} | w: {omega:.3f}')

    def stop_robot(self):
        """Publish zero velocity to stop the robot."""
        cmd = Twist()
        cmd.linear.x  = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = MPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
