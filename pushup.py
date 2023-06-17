# Jake 
# 6-16-2023
# The purpose of this program is to count the number of pushups as well as correct user form.

# Inport libraries
import cv2
from ultralytics import YOLO
import numpy as np

# Load Model
pose_model = YOLO("yolov8s-pose.pt")

# Keypoint names
keypoint_names = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

####################
# Functions

def linear_regression(keypoint_name_list):
    ############################
    # This function takes the list of L/R body part dictionaries and finds the linear regression for the data
    # Input: list of dictionaries
    # Output: slope and r_squared value from linear regression
    ############################
    
    # Initialize Lists
    x_data = []
    y_data = []

    # Extract x,y coordinates from keypoint dictionaries 
    for keypoint in keypoint_name_list:
        x_data.append(keypoint["x"])
        y_data.append(keypoint["y"])

    # Linear Regression
    N = len(x_data)

    sum_x = sum(x_data)
    sum_y = sum(y_data)
    sum_xy = sum(x*y for x, y in zip(x_data, y_data))
    sum_xx = sum(x*x for x in x_data)
    sum_yy = sum(y*y for y in y_data)

    # Calculate slope
    slope = (N * sum_xy - sum_x * sum_y) / (N * sum_xx - sum_x**2)

    # Calculate R-squared value
    numerator = (N * sum_xy - sum_x * sum_y)**2
    denominator = (N * sum_xx - sum_x**2) * (N * sum_yy - sum_y**2)
    r_squared = numerator / denominator

    return slope, r_squared



# Open the webcam
cap = cv2.VideoCapture(1)

rep_counter = 0
is_rep = False

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Pose detection
        pose_results = pose_model(frame, verbose=False, conf=0.5)

        # Print each body coordinate as a dictionary
        for person in pose_results:
            keypoints = person.keypoints.data[0]
            keypoint_dict = {}
            for keypoint, name in zip(keypoints, keypoint_names):
                x, y, probability = keypoint
                keypoint_dict[name] = {
                    "x": x.item(),
                    "y": y.item(),
                    "probability": probability.item(),
                }

            # Check if shoulder, hip, knee, and ankle keypoints exist
            if "right_shoulder" in keypoint_dict and "right_hip" in keypoint_dict and "right_knee" in keypoint_dict and "right_ankle" in keypoint_dict:
                
                keypoint_name_list = [keypoint_dict["right_shoulder"], keypoint_dict["right_hip"], keypoint_dict["right_knee"], keypoint_dict["right_ankle"]]

            
            
                



            elif "left_shoulder" in keypoint_dict and "left_hip" in keypoint_dict and "left_knee" in keypoint_dict and "left_ankle" in keypoint_dict:


            # # Check if right_wrist and right_elbow keypoints exist
            # if "right_wrist" in keypoint_dict and "right_elbow" in keypoint_dict:
            #     # If wrist is above the elbow and to its left
            #     if (
            #         keypoint_dict["right_wrist"]["y"]
            #         < keypoint_dict["right_elbow"]["y"]
            #         and keypoint_dict["right_wrist"]["x"]
            #         < keypoint_dict["right_elbow"]["x"]
            #     ):
            #         is_rep = True
            #     # Rep ends when wrist goes down past the elbow
            #     elif (
            #         is_rep
            #         and keypoint_dict["right_wrist"]["y"]
            #         > keypoint_dict["right_elbow"]["y"]
            #     ):
            #         is_rep = False
            #         rep_counter += 1

            # Annotate the frame with the rep count
            pose_annotated_frame = person.plot()
            cv2.putText(
                pose_annotated_frame,
                "Reps: {}".format(rep_counter),
                (300, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.imshow("Pose Detection", pose_annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
