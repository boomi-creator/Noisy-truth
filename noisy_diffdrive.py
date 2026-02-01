import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import math
import random

def yaw_to_quaternion(yaw):
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q

class DiffDriveSimulator(Node):
    def __init__(self):
        super().__init__('diffdrive_simulator')

        self.dt = 0.05  # 20 Hz
        self.timer = self.create_timer(self.dt, self.step)

        self.odom_pub = self.create_publisher(Odometry, '/odom/noisy', 10)
        self.gt_pub   = self.create_publisher(Odometry, '/ground_truth', 10)

        # Ground truth state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Constant control input
        self.v = 0.5      # m/s
        self.omega = 0.3  # rad/s

        # Noise parameters
        self.sigma_x = 0.1
        self.sigma_y = 0.1
        self.sigma_theta = 0.05

        self.get_logger().info("DiffDrive noisy simulator started")

    def step(self):
        # Ground truth propagation
        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt
        self.theta += self.omega * self.dt

        # Noisy measurement
        x_n = self.x + random.gauss(0, self.sigma_x)
        y_n = self.y + random.gauss(0, self.sigma_y)
        theta_n = self.theta + random.gauss(0, self.sigma_theta)

        self.gt_pub.publish(self.make_odom(self.x, self.y, self.theta))
        self.odom_pub.publish(self.make_odom(x_n, y_n, theta_n))

    def make_odom(self, x, y, theta):
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation = yaw_to_quaternion(theta)

        return msg

def main():
    rclpy.init()
    node = DiffDriveSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
