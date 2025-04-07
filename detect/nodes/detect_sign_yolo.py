#!/usr/bin/env python3

import rospy
import requests
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

class DetectSignYOLO:
    def __init__(self):
        rospy.init_node('detect_sign_yolo', anonymous=True)

        # Flask Server URL (Update with Windows laptop IP)
        self.flask_url = "http://192.168.200.210:5001/detect"  # UPDATE with your Windows IP

        # ROS Topics
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.callback, queue_size=1, buff_size=2**24)
        self.sign_type_pub = rospy.Publisher("detected_sign", String, queue_size=10)
        self.roi_pub = rospy.Publisher("/detect_sign_yolo", Image, queue_size=10)

        rospy.loginfo("ROS Node: Sending images to Windows Flask server for YOLO detection")

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.process_image(cv_image)
        except CvBridgeError as e:
            rospy.logerr(f"Error converting image: {e}")

    def process_image(self, image):
        try:
            # Encode image as JPEG
            _, img_encoded = cv2.imencode(".jpg", image)
            response = requests.post(self.flask_url, files={"file": img_encoded.tobytes()})

            if response.status_code == 200:
                detection_results = response.json()

                if "detections" in detection_results and detection_results["detections"]:
                    for detection in detection_results["detections"]:
                        # ✅ Check for bounding box keys
                        sign_type = detection.get("sign_type", "Unknown")
                        confidence = round(float(detection.get("confidence", 0)), 2)

                        rospy.loginfo(f"Detected: {sign_type}, Confidence: {confidence}%")
                        self.sign_type_pub.publish(f"{sign_type}:{confidence}")

                        # If bounding box is missing, overlay text without ROI cropping
                        if all(k in detection for k in ["x1", "y1", "x2", "y2"]):
                            x1, y1, x2, y2 = map(int, [detection["x1"], detection["y1"], detection["x2"], detection["y2"]])
                            roi = image[y1:y2, x1:x2].copy()
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        else:
                            rospy.logwarn(f"No bounding box for {sign_type}, displaying text only.")
                            roi = image  # Use the full image

                        # Overlay sign name and confidence
                        label = f"{sign_type} ({confidence}%)"
                        cv2.putText(image, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Convert and publish image
                        roi_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
                        self.roi_pub.publish(roi_msg)

                else:
                    rospy.loginfo("No detections found")

        except Exception as e:
            rospy.logerr(f"Error in process_image: {e}")

def main():
    DetectSignYOLO()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down YOLO ROS node")

if __name__ == '__main__':
    main()
