#!/usr/bin/env python3

import rospy
import os
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import mysql.connector
import requests
import joblib
import time
import tempfile
from datetime import datetime

class DetectSign:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/image", Image, self.callback, queue_size=1, buff_size=2**24)
        self.roi_pub = rospy.Publisher("/traffic_sign_roi", Image, queue_size=10)
        self.status_pub = rospy.Publisher("/detection_status", String, queue_size=10)
        self.threshold_pub = rospy.Publisher("/threshold_sign", Image, queue_size=10)
        self.sign_type_pub = rospy.Publisher("detected_sign", String, queue_size=10)
        self.flask_url = "http://192.168.200.210:50005/check_logo"  # Update with your Flask server URL

        # Attempt to connect to the database
        self.connect_to_database()

        # Initialize SIFT detector with fewer features
        self.sift = cv2.SIFT_create(nfeatures=1000)

        # Load the new BoW model
        self.bow_model_path = '/mnt/hgfs/traffic_signn/bow.pkl'  # Path to the newly generated BoW model
        if os.path.exists(self.bow_model_path):
            with open(self.bow_model_path, 'rb') as file:
                self.bow_model = joblib.load(file)
            rospy.loginfo(f"BOW model loaded from {self.bow_model_path}")
        else:
            rospy.logerr(f"BOW model not found at {self.bow_model_path}")
            self.bow_model = None

        # Initialize the BoW extractor
        if self.bow_model is not None:
            self.bow_extractor = cv2.BOWImgDescriptorExtractor(
                self.sift, 
                cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            )
            self.bow_extractor.setVocabulary(self.bow_model.cluster_centers_)
        else:
            self.bow_extractor = None

    def connect_to_database(self):
        while True:
            try:
                self.mydb = mysql.connector.connect(
                    host="192.168.200.210",  # Update with your MySQL server IP
                    user="root",
                    password="",  # Update with your MySQL user password if any
                    database="traffic_sign_db"
                )
                self.cursor = self.mydb.cursor()
                rospy.loginfo("Connected to MySQL server successfully")
                break
            except mysql.connector.Error as err:
                self.mydb = None
                self.cursor = None
                rospy.logerr(f"Error connecting to MySQL server: {err}")
                rospy.loginfo("Retrying to connect to the database in 5 seconds...")
                time.sleep(5)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.process_image(cv_image)
        except CvBridgeError as e:
            rospy.logerr(e)

    def process_image(self, image):
        try:
            if self.bow_extractor is None:
                rospy.logerr("BOW extractor is not initialized.")
                return

            # Resize the image to speed up processing
            scale_percent = 50  # Percent of original size
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            
            # Thresholding to create a black and white image
            _, thresholded_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

            # Publish the thresholded image
            threshold_image_msg = self.bridge.cv2_to_imgmsg(thresholded_image, "mono8")
            self.threshold_pub.publish(threshold_image_msg)
            
            # Find contours
            contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour which will be the region of interest
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Filter the contour by aspect ratio and area
                aspect_ratio = w / float(h)
                area = cv2.contourArea(largest_contour)
                min_aspect_ratio = 0.8
                max_aspect_ratio = 1.2
                min_area = 1000
                max_area = 50000
                
                if min_aspect_ratio < aspect_ratio < max_aspect_ratio and min_area < area < max_area:
                    # Extract the region of interest
                    roi = resized_image[y:y+h, x:x+w]
                    
                    # Save the ROI temporarily
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    temp_file_path = temp_file.name
                    cv2.imwrite(temp_file_path, roi)

                    # Send the ROI to the server for matching
                    sign_type, similarity = self.send_to_server(temp_file_path)

                    # Print result to terminal with timestamp
                    if sign_type:
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{current_time}] Traffic sign detected: Type: {sign_type}, Similarity: {similarity:.2f}%")
                        # Publish the sign_type and similarity
                        message = f"{sign_type}:{similarity:.2f}"
                        self.sign_type_pub.publish(message)
                        # Overlay sign type text on the ROI image
                        cv2.putText(roi, f"{sign_type} ({similarity:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    # Publish the ROI
                    roi_image_msg = self.bridge.cv2_to_imgmsg(roi, "bgr8")
                    self.roi_pub.publish(roi_image_msg)

        except Exception as e:
            rospy.logerr(f"Error in process_image: {e}")

    def send_to_server(self, image_path):
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(self.flask_url, files=files)
            if response.status_code == 200:
                response_data = response.json()
                if response_data:
                    image_name = response_data[0]['image_name']
                    sign_type = response_data[0]['sign_type']
                    similarity = response_data[0]['similarity']
                    self.status_pub.publish(f"Traffic sign detected: {image_name}, Type: {sign_type}, Similarity: {similarity:.2f}%")
                    return sign_type, similarity
                else:
                    self.status_pub.publish("No similar traffic sign found.")
                    return None, None
            else:
                self.status_pub.publish(f"Failed to send image: {response.content}")
                return None, None
        except Exception as e:
            self.status_pub.publish(f"Error in send_to_server: {e}")
            return None, None
        finally:
            os.remove(image_path)

def main():
    rospy.init_node('detect_sign_detector', anonymous=True)
    ds = DetectSign()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
