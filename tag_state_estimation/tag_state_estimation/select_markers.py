import numpy as np
import cv2
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory


class MarkerSelector(Node):
    def __init__(self):
        super().__init__("marker_selector")
        self.declare_parameter("output_path", "")
        self.sub = self.create_subscription(
            Image, "tag_camera/image", self.select_marker, 1
        )
        self.br = CvBridge()
        cv2.namedWindow("Select markers")
        cv2.setMouseCallback("Select markers", self.draw_x)

        self.markers = []
        self.clicked = True
        self.number_str = ["lower-left", "lower-right", "upper-right", "upper-left"]
        self.marker_str = [
            "FIXED frame reference",
            "MOVING board/map",
        ]

        self.get_logger().info(
            "Make sure the labyrinth playing board is approximately even (Inclination angles close to 0)"
        )

    def select_marker(self, data):
        frame = self.br.imgmsg_to_cv2(data)
        for i, c in enumerate(self.markers):
            color = (255, 255, 255) if i < 4 else (255, 255, 0)
            cv2.drawMarker(frame, c, color, cv2.MARKER_CROSS, 15, 2)
            cv2.putText(
                frame,
                ("F" if i < 4 else "M") + str(i % 4 + 1),
                (c[0] + 10, c[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        if self.clicked and len(self.markers) != 8:
            self.get_logger().info(
                "Please select the {} marker of the {} markers".format(
                    self.number_str[len(self.markers) % 4],
                    self.marker_str[len(self.markers) // 4],
                )
            )
            self.clicked = False
        if len(self.markers) == 8:
            shared = get_package_share_directory("tag_state_estimation")
            configured = str(self.get_parameter("output_path").value)
            output = configured or os.path.join(shared, "markers.csv")
            np.savetxt(
                output,
                np.asarray(self.markers),
                delimiter=",",
            )
            self.get_logger().info(
                f"Successfully selected and saved all markers to {output}. "
                "Press any key to exit."
            )
            cv2.imshow("Select markers", frame)
            cv2.waitKey(0)
            rclpy.shutdown()
            return

        cv2.putText(
            frame,
            "F1-F4=fixed reference | M1-M4=moving board | u=undo | r=restart",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Select markers", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("u"), 8, 127) and self.markers:
            self.markers.pop()
            self.clicked = True
        elif key == ord("r"):
            self.markers.clear()
            self.clicked = True
        elif key in (ord("q"), 27):
            rclpy.shutdown()

    def draw_x(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.markers.append([x, y])
            self.clicked = True


def main(args=None):
    rclpy.init(args=args)
    ms = MarkerSelector()
    try:
        rclpy.spin(ms)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    if rclpy.ok():
        rclpy.shutdown()
