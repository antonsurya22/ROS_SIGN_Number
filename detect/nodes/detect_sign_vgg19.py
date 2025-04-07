#!/usr/bin/env python3

import rospy
import tensorflow as tf
import numpy as np
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

class DetectSignVGG19:
    def __init__(self):
        rospy.init_node('detect_sign_vgg19_ros', anonymous=True)

        # Load TensorFlow `.h5` model
        self.model_path = "/home/autorace/Desktop/TrafficSignVGG19.h5"  # Update path
        self.model = tf.keras.models.load_model(self.model_path)
        rospy.loginfo("Loaded VGG19 model")

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.callback, queue_size=1, buff_size=2**24)
        self.sign_type_pub = rospy.Publisher("detected_sign", String, queue_size=10)
        self.roi_pub = rospy.Publisher("/vgg19_sign_roi", Image, queue_size=10)  
        self.threshold_pub = rospy.Publisher("/vgg19_threshold", Image, queue_size=10)  

        # Define traffic sign names based on dataset
        self.sign_names = ["Careful", "Construction", "DoNotEnter", "SpeedLimit", 
                           "Stop", "Straight", "TrafficLight", "TurnLeft", "TurnRight"]

        rospy.loginfo("ROS Node: Using VGG19 `.h5` model for traffic sign detection")

    def callback(self, data):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.process_image(cv_image)
        except CvBridgeError as e:
            rospy.logerr(f"Error converting image: {e}")

    def process_image(self, image):
        try:
            # Convert to Grayscale for Thresholding
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Thresholding (Binary)
            _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

            # Publish thresholded image
            threshold_image_msg = self.bridge.cv2_to_imgmsg(thresholded_image, "mono8")
            self.threshold_pub.publish(threshold_image_msg)

            # Preprocess image for VGG19
            img_resized = cv2.resize(image, (128, 128))  
            img_array = np.expand_dims(img_resized, axis=0) / 255.0  

            # Run inference
            predictions = self.model.predict(img_array)

            # Ensure predictions are valid
            if predictions.shape[1] == 0:
                rospy.logerr("Error: Model returned empty predictions")
                return

            sign_index = np.argmax(predictions)
            confidence = np.max(predictions) * 100

            # Ensure the sign index is valid
            if sign_index >= len(self.sign_names):
                rospy.logerr(f"Error: Invalid index {sign_index}, check class mapping")
                return

            sign_type = self.sign_names[sign_index]

            # Publish detection result
            detection_msg = f"{sign_type}:{confidence:.2f}"
            rospy.loginfo(f"Detected: {sign_type}, Confidence: {confidence:.2f}%")
            self.sign_type_pub.publish(detection_msg)

            # Resize ROI for better visibility
            enlarged_size = (512, 512)  # Bigger output size
            roi = cv2.resize(image, enlarged_size)  

            # Overlay detection text
            cv2.putText(roi, f"{sign_type} ({confidence:.2f}%)", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

            # Publish to `/vgg19_sign_roi`
            roi_image_msg = self.bridge.cv2_to_imgmsg(roi, "bgr8")
            self.roi_pub.publish(roi_image_msg)

        except Exception as e:
            rospy.logerr(f"Error in process_image: {e}")

def main():
    DetectSignVGG19()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down VGG19 ROS node")

if __name__ == '__main__':
    main()
