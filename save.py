import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import os

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/rgb2',  # Replace with your image topic
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.br = CvBridge()

        # Track last save time
        self.last_save_time = 0

        # Make sure save directory exists
        self.save_dir = "saved_images"
        os.makedirs(self.save_dir, exist_ok=True)

    def listener_callback(self, data):
        self.get_logger().info('Receiving video frame')
        current_frame = self.br.imgmsg_to_cv2(data, "bgr8")
        
        # Display live feed
        cv2.imshow("Camera Feed", current_frame)
        cv2.waitKey(1)

        # Save every 5 seconds
        now = time.time()
        if now - self.last_save_time >= 1:
            filename = os.path.join(self.save_dir, f"frame_{int(now)}.jpg")
            cv2.imwrite(filename, current_frame)
            self.get_logger().info(f"Saved image: {filename}")
            self.last_save_time = now

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

