import rclpy
from rclpy.node import Node

import numpy as np
import math

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

from ukf_localization.ukf import UKF
from ukf_localization.dd2d import iterate


def quat_to_yaw(q):
    return math.atan2(
        2.0 * q.z * q.w,
        1.0 - 2.0 * q.z * q.z
    )


def yaw_to_quat(yaw):
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class UKFNode(Node):

    def __init__(self):
        super().__init__('ukf_node')

        # -------------------------------
        # Filter parameters
        # -------------------------------
        self.dt = 0.05  # must match simulator

        Q = np.diag([
            0.05**2,
            0.05**2,
            0.02**2
        ])

        self.R = np.diag([
            0.02**2,
            0.02**2,
            (5 * math.pi / 180)**2
        ])

        x0 = np.zeros(3)

        P0 = np.diag([
            0.5**2,
            0.5**2,
            (5 * math.pi / 180)**2
        ])

        # -------------------------------
        # Create UKF
        # -------------------------------
        self.ukf = UKF(
            num_states=3,
            process_noise=Q,
            initial_state=x0,
            initial_covar=P0,
            alpha=1e-3,
            k=0.0,
            beta=2.0,
            iterate_function=iterate
        )

        # Known control input (same as simulator)
        self.u = [0.5, 0.3]

        # -------------------------------
        # ROS interfaces
        # -------------------------------
        self.sub = self.create_subscription(
            Odometry,
            '/odom/noisy',
            self.odom_callback,
            10
        )

        self.pub = self.create_publisher(
            Odometry,
            '/ukf/estimate',
            10
        )

        self.get_logger().info("UKF node started")

    # -------------------------------
    # Measurement callback
    # -------------------------------
    def odom_callback(self, msg):

        # 1. PREDICT
        self.ukf.predict(self.dt, self.u)

        # 2. FORM MEASUREMENT VECTOR
        z = np.zeros(3)
        z[0] = msg.pose.pose.position.x
        z[1] = msg.pose.pose.position.y
        z[2] = quat_to_yaw(msg.pose.pose.orientation)

        # 3. UPDATE
        self.ukf.update(
            states=[0, 1, 2],
            data=z,
            r_matrix=self.R
        )

        # 4. PUBLISH ESTIMATE
        x_est = self.ukf.get_state()

        out = Odometry()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "odom"
        out.child_frame_id = "base_link"

        out.pose.pose.position.x = x_est[0]
        out.pose.pose.position.y = x_est[1]
        out.pose.pose.orientation = yaw_to_quat(x_est[2])

        self.pub.publish(out)


def main():
    rclpy.init()
    node = UKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()  
