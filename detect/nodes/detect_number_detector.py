#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from tensorflow.keras.models import load_model

class DetectNumber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.callback, queue_size=1, buff_size=2**24)
        self.roi_pub = rospy.Publisher("/number_roi", Image, queue_size=10)
        self.threshold_pub = rospy.Publisher("/threshold_image", Image, queue_size=10)
        self.status_pub = rospy.Publisher("/detection_status", String, queue_size=10)
        
        self.model_path = '/mnt/hgfs/Model/mnist_model.h5'
        self.model = load_model(self.model_path)
        rospy.loginfo(f"MNIST model loaded from {self.model_path}")
        rospy.loginfo("Number Detection Node Initialized")

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            rospy.loginfo("Image received")
            self.process_image(cv_image)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridgeError: {e}")

    def preprocess_image(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
        return binary_image

    def recognize_number(self, roi):
        roi_resized = cv2.resize(roi, (28, 28))
        roi_reshaped = roi_resized.reshape(1, 28, 28, 1).astype('float32') / 255.0
        prediction = self.model.predict(roi_reshaped)
        return np.argmax(prediction)

    def process_image(self, image):
        try:
            preprocessed_image = self.preprocess_image(image)
            rospy.loginfo("Image preprocessed")

            # Publish the threshold image
            threshold_image_msg = self.bridge.cv2_to_imgmsg(preprocessed_image, "mono8")
            self.threshold_pub.publish(threshold_image_msg)
            rospy.loginfo("Threshold image published")

            # Find contours
            contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                rospy.loginfo(f"{len(contours)} contours found")

                detected_numbers = []
                cv_image_with_contour = image.copy()

                # Sort contours based on area and take the largest three
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = preprocessed_image[y:y+h, x:x+w]

                    # Recognize the number
                    number = self.recognize_number(roi)

                    # Draw a rectangle around the ROI on the original image for visualization
                    cv2.rectangle(cv_image_with_contour, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(cv_image_with_contour, str(number), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    detected_numbers.append(str(number))

                # Publish the original image with ROIs and numbers highlighted
                roi_image_msg = self.bridge.cv2_to_imgmsg(cv_image_with_contour, "bgr8")
                self.roi_pub.publish(roi_image_msg)
                rospy.loginfo("ROI image published")

                # Print result to terminal with timestamp
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                detected_numbers_str = ', '.join(detected_numbers)
                rospy.loginfo(f"[{current_time}] Numbers detected: {detected_numbers_str}")
                self.status_pub.publish(f"Numbers detected: {detected_numbers_str}")
            else:
                rospy.loginfo("No contours found.")
                self.status_pub.publish("No number detected")

        except Exception as e:
            rospy.logerr(f"Error in process_image: {e}")

def main():
    rospy.init_node('detect_number', anonymous=True)
    dn = DetectNumber()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
