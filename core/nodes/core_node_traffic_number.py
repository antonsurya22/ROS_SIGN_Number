#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import requests
import sys
import os
import cv2
import tempfile

# Add the detect package path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../detect/nodes'))

from detect_sign_detector import DetectSign
from detect_number_detector import DetectNumber

class CoreNodeTrafficNumber:
    def __init__(self):
        rospy.init_node('core_node_traffic_number')
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.image_callback, queue_size=1, buff_size=2**24)
        
        # Publisher for detected sign
        self.detected_sign_pub = rospy.Publisher("detected_sign", String, queue_size=10)
        
        # Initialize detectors
        self.sign_detector = DetectSign()
        self.number_detector = DetectNumber()
        self.current_detection = None  # Initialize current_detection as None

        # Flask server URL
        self.flask_url = "http://192.168.200.210:50005/check_logo"  # Update with your Flask server URL

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            
            # Try to detect sign first
            if self.contains_sign(cv_image):
                rospy.loginfo("Detected traffic sign, switching to sign detection")
                detected_sign = self.process_image_with_flask(cv_image)
                if detected_sign:
                    self.publish_detected_sign(detected_sign)
            # If no sign detected, try to detect number
            elif self.contains_number(cv_image):
                rospy.loginfo("No traffic sign detected. Detected number, switching to number detection")
                self.number_detector.process_image(cv_image)
            else:
                rospy.logwarn("No valid detection type set")
                
        except CvBridgeError as e:
            rospy.logerr(f"CvBridgeError: {e}")

    def contains_sign(self, image):
        # Simple placeholder logic; replace with actual traffic sign detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours) > 0  # Placeholder: detect signs based on contours

    def contains_number(self, image):
        # Simple placeholder logic; replace with actual number detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours) > 0  # Placeholder: detect numbers based on contours

    def process_image_with_flask(self, image):
        try:
            # Save the image temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            temp_file_path = temp_file.name
            cv2.imwrite(temp_file_path, image)

            with open(temp_file_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(self.flask_url, files=files)
            
            os.remove(temp_file_path)
            
            if response.status_code == 200:
                response_data = response.json()
                if response_data:
                    detected_sign = response_data[0]['sign_type']
                    rospy.loginfo(f"Detected sign: {detected_sign}")
                    return detected_sign
                else:
                    rospy.logwarn("No similar traffic sign found.")
            else:
                rospy.logwarn(f"Failed to process image with Flask: {response.content}")
        except Exception as e:
            rospy.logerr(f"Error in process_image_with_flask: {e}")
        return None

    def publish_detected_sign(self, detected_sign):
        rospy.loginfo(f"Publishing detected sign: {detected_sign}")
        self.detected_sign_pub.publish(detected_sign)

if __name__ == '__main__':
    try:
        node = CoreNodeTrafficNumber()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
