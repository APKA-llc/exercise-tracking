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

                m, b, r_sq = linear_regression(keypoint_name_list)


            elif "left_shoulder" in keypoint_dict and "left_hip" in keypoint_dict and "left_knee" in keypoint_dict and "left_ankle" in keypoint_dict:

                keypoint_name_list = [keypoint_dict["left_shoulder"], keypoint_dict["left_hip"], keypoint_dict["left_knee"], keypoint_dict["left_ankle"]]

                m, b, r_sq = linear_regression(keypoint_name_list)
            
            else:
                print("Keypoints not found")
                m = 0
                b = 0
                r_sq = None

           
            # Annotate the frame with the rep count
            pose_annotated_frame = person.plot()
            
            # Calculate the center point of the frame
            height, width, _ = frame.shape
            center_height = height // 2
            center_width = width // 2

            # Calculate the start and end points for the line
            start_point = (center_width - 100, int(center_height - 100 * m))
            end_point = (center_width + 100, int(center_height + 100 * m))

            # Display the line
            cv2.line(pose_annotated_frame, start_point, end_point, (0, 0, 255), 5)

            # Display the text
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
            # Expand the image to fit the screen
            pose_annotated_frame = cv2.resize(pose_annotated_frame, (1000, 800))

            cv2.imshow("Pose Detection", pose_annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
