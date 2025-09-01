import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Bool
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
# //from mv01 import detect_movement_options
import torch
import random

screenX = 1280;
screenY = 720;

img_width=screenX
img_height=screenY

#detection box constants
robot_box = [0,img_height,img_width,0];#//scan across entire screen
rx_min, ry_min, rx_max, ry_max = robot_box
robot_center_x = (rx_min + rx_max) / 2
robot_center_y = (ry_min + ry_max) / 2

#detection zone constants
fxMin = robot_center_x - 190;
fxMax = robot_center_x + 190;
fyMin = ry_min - 300;
fyMax = ry_min

lxMin = 0;
lxMax = robot_center_x - 200;
lyMin = ry_min - 350;
lyMax = ry_min;

rxMin = robot_center_x + 200;
rxMax = img_width;
ryMin = ry_min - 350;
ryMax = ry_min;




def detect_movement_options(boxes):
    
    # Define detection zones
    forward_zone = [fxMin, fyMin,
                    fxMax, fyMax];
    left_zone = [lxMin, lyMin, lxMax, lyMax]
    right_zone = [rxMin, ryMin, rxMax, ryMax]

    def overlaps(zone, box):
        zx1, zy1, zx2, zy2 = zone
        bx1, by1, bx2, by2 = box
        return not (bx2 < zx1 or bx1 > zx2 or by2 < zy1 or by1 > zy2)

    can_forward = True
    can_left = True
    can_right = True

    for box in boxes:
        if overlaps(forward_zone, box):
            can_forward = False
            
        if overlaps(left_zone, box):
            can_left = False
            
        if overlaps(right_zone, box):
            can_right = False
            
            
        

    # Decide turn angle
    if can_forward:
        angle = 0
    elif can_left and not can_right:
        angle = -45
    elif can_right and not can_left:
        angle = 45
    elif can_left and can_right:
        angle = -45  # prefer left
    else:
        angle = 180

    return {
        "forward": can_forward,
        "left": can_left,
        "right": can_right
        #"turn_angle": angle
    }






def draw_zone(img, x1, y1, x2, y2, color, bold=False):
    #y is reversed
    cv2.rectangle(
        img,
        (int(min(x1, x2)), int(max(y1, y2))),
        (int(max(x1, x2)), int(min(y1, y2))),
        color=color,
        thickness = 8 if bold else 3
    )
    
    if bold:
        # Compute center of the box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Draw "Triggered!" in bold text
        cv2.putText(
            img,
            "Triggered!",
            (center_x - 60, center_y),   # shift left so text is centered
            cv2.FONT_HERSHEY_SIMPLEX,
            1,        # font scale
            color,    # same color as box
            3,        # thickness (bold)
            cv2.LINE_AA
        )









class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        #flag for dead end
        self.dead_end = False;

        # Image subscription
        self.subscription = self.create_subscription(
            Image,
            '/rgb2',  # Your image topic
            self.listener_callback,
            10)

        # Publisher for bounding boxes
        self.box_pub = self.create_publisher(Float32MultiArray, 'yolo_boxes', 10)

        # Publishers for movement topics
        self.pub_w = self.create_publisher(Bool, '/w', 10)
        self.pub_s = self.create_publisher(Bool, '/s', 10)
        self.pub_a = self.create_publisher(Bool, '/a', 10)
        self.pub_d = self.create_publisher(Bool, '/d', 10)

        self.br = CvBridge()
        self.model = YOLO("/home/j1/exec/yolo/obstacleENVModel.pt")  # Load YOLO model

    def listener_callback(self, data):
        fBoxTriggered = False;
        lBoxTriggered = False;
        rBoxTriggered = False;
        
        self.get_logger().info('Receiving video frame')

        # Convert ROS Image to OpenCV format
        frame = self.br.imgmsg_to_cv2(data, "bgr8")

        # Run YOLO detection
        results = self.model.predict(source=frame, conf=0.6, verbose=False)

        # Annotate detections
        annotated_frame = results[0].plot()

        # Extract bounding boxes
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Prepare for movement decision
        boxes2 = [[round(float(v), 1) for v in row] for row in boxes]
        
        dir = detect_movement_options(boxes2)
        print("Direction:", dir)

        # --- Decide movement commands ---
        can_forward = dir.get("forward", False)
        can_left = dir.get("left", False)
        can_right = dir.get("right", False)
        
        
        
        #update box trigger vars
        fBoxTriggered = not can_forward;
        lBoxTriggered = not can_left;
        rBoxTriggered = not can_right;

        # --- Movement decision with dead-end handling ---
        if not self.dead_end:
            if can_forward:
                w, s, a, d = True, False, False, False  # forward
            elif can_right:
                w, s, a, d = False, False, False, True  # right
            elif can_left:
                w, s, a, d = False, False, True, False  # left
            else:
                # dead end reached -> start backing up
                self.dead_end = True
                w, s, a, d = False, True, False, False  # backward
        else:
            # Currently in dead-end mode: keep backing up until a turn is possible
            if can_right:
                self.dead_end = False
                w, s, a, d = False, False, False, True  # exit dead end to right
            elif can_left:
                self.dead_end = False
                w, s, a, d = False, False, True, False  # exit dead end to left
            else:
                w, s, a, d = False, True, False, False  # keep backing up

        # Publish all movement topics
        self.pub_w.publish(Bool(data=w))
        self.pub_s.publish(Bool(data=s))
        self.pub_a.publish(Bool(data=a))
        self.pub_d.publish(Bool(data=d))

        # Add custom text overlay
        cv2.putText(
            annotated_frame,
            str(dir),
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA
        )

        
        # Draw forward rectangle (box)
        draw_zone(annotated_frame, fxMin, fyMin, fxMax, fyMax, (0,255,0), bold=fBoxTriggered)#green
        
        # Draw left rectangle (box)
        draw_zone(annotated_frame, lxMin, lyMin, lxMax, lyMax, (0,0,255),bold=lBoxTriggered)#red
        
        # Draw right rectangle (box)
        draw_zone(annotated_frame, rxMin, ryMin, rxMax, ryMax, (255,0,0),bold=rBoxTriggered)#blue


        # Show image
        cv2.imshow("YOLOv11 Detection", annotated_frame)
        cv2.waitKey(1)

        # Publish bounding boxes
        msg = Float32MultiArray()
        msg.data = boxes.flatten().astype(float).tolist()
        self.box_pub.publish(msg)
        self.get_logger().info(f'Published {len(boxes)} bounding boxes')


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

