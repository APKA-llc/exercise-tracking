# Jake and Krish
# 6-16-2023
# The purpose of this program is to count the number of pushups as well as correct user form.

#DEV NOTES
# Fixed the line of best fit m and r_sq display crashing issue

#############################
# Feature : Need to find the start/mid/end ( find a robust way to solve this problem will save us a lot of time )

# Possilbe solution 1: Angle measures for the arms
#
# Limitations: 
# need to keep track of extra limbs


# Possilbe solution 2: Derivative of the linear regression line slope (m)
# 
# Research: numerical forward difference method of taking the derivative
#
# Limitations: 
# need to keep track of time
# issue with frame rate sampleing
# computationally expensive


#############################
# Feature : Improvement sugesstions
#
# Find residual for each point against the linear regression line, check if outlier, print advice 


#############################
# Feature : Rep counting
# 
# A 'score' that tells you how many good reps vs total reps


#############################
# Import libraries
import cv2
from ultralytics import YOLO
import numpy as np

# Load Model
pose_model = YOLO("yolov8s-pose.pt")


####################
# Initialize
m = None
b = None
r_sq = None

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
        x_data.append(keypoint_dict[keypoint]["x"])
        y_data.append(keypoint_dict[keypoint]["y"])

    N = len(x_data)
    sum_x = sum(x_data)
    sum_y = sum(y_data)
    sum_xy = sum(x*y for x, y in zip(x_data, y_data))
    sum_xx = sum(x*x for x in x_data)
    sum_yy = sum(y*y for y in y_data)

    # Calculate slope
    slope = (N * sum_xy - sum_x * sum_y) / (N * sum_xx - sum_x**2)

    # Calculate the intercept
    intercept = (sum_y - slope * sum_x) / N

    # Calculate R-squared value
    numerator = (N * sum_xy - sum_x * sum_y)**2
    denominator = (N * sum_xx - sum_x**2) * (N * sum_yy - sum_y**2)
    
    # Check for division by zero
    if denominator == 0:
        print("R-squared cannot be calculated because the variance of y is 0")
        r_squared = None

    else:
        r_squared = numerator / denominator

    return slope, intercept , r_squared


# Open the webcam
cap = cv2.VideoCapture(0)


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
           
        # Check if all of the shoulder, hip, knee, and ankle keypoints can be seen and have a high probability
        probability_threshold = 0.3
        required_keypoints = [ "right_hip", "right_shoulder", "right_knee", "right_ankle"]
        missing_keypoints = []

        # Dev Notes: the search for all the valid keypoints "right side" seems to be working correctly
        for keypoint in required_keypoints:
            # Check to make sure the keypoint exists and has a high probability
            if keypoint not in keypoint_dict or keypoint_dict[keypoint]["probability"] < probability_threshold:
                missing_keypoints.append(keypoint)

        if missing_keypoints != []:
            print(f"Not all keypoints are visible: {missing_keypoints}") #Debug statement
        
        #Dev Notes: Program crashes when it tries to run the lin reg 
        else:
            m, b, r_sq =linear_regression(required_keypoints)
            
            sig_fig = 3
            m = round(m, sig_fig)
            b = round(b, sig_fig)
            r_sq = round(r_sq, sig_fig)

            print(f"m: {m}, b: {b}, r_sq: {r_sq}")


        ############################
        # Display Settings

        # Calculate the center point of the frame
        height, width, _ = frame.shape
        center_height = height // 2
        center_width = width // 2

        pose_annotated_frame = person.plot()
           
        # Display the adherance to the line (r_sq)
        text = "Back Straightness: {}".format(r_sq) 
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = center_width - text_size[0] // 2
        text_y = center_height - text_size[1] // 2
        cv2.putText(
            pose_annotated_frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Display the slop of the back line (m)
        text = "Back Angle: {}".format(m)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (center_width - text_size[0] // 2)
        text_y = (center_height - text_size[1] // 2) - 30
        cv2.putText(
            pose_annotated_frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Display the line of best fit over the image of the body and keypoints (y = mx + b)
        if m is not None and b is not None:
            if r_sq > 0.7:
                cv2.line(
                    pose_annotated_frame,
                    (0, int(b)),
                    (width, int(m * width + b)),
                    (0, 192, 0),
                    2,
                )
            else:
                cv2.line(
                    pose_annotated_frame,
                    (0, int(b)),
                    (width, int(m * width + b)),
                    (0, 0, 255),
                    2,
                )

        # Expand the image to fit the screen
        pose_annotated_frame = cv2.resize(pose_annotated_frame, (1000, 800))

        # Display the frame
        cv2.imshow("Pose Detection", pose_annotated_frame)

        # Press "q" to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
