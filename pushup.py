# Jake and Krish
# 6-16-2023
# The purpose of this program is to count the number of pushups as well as correct user form.


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
#  needs a slope 'filter' to only look at a range of slopes for when you are standing , vs when you are doing a pushup
# A 'score' that tells you how many good reps vs total reps


#############################
# Import libraries
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

# Load Model
pose_model = YOLO("yolov8s-pose.pt")


####################
# Initialize
m = None
b = None
r_sq = None

rep_count = 0
frame_count = 0

slope_dict = {}
slope_prime_dict = {}

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

def backwards_difference(current_frame, dictionary_name):
    ############################
    # This function uses the current slope and the last available slope from n th frame back to calculate the derivative at the current frame
    # Input: frame count, y value
    # Output: write to slope_prime_dict with the derivative ate current frame count
    
    # Step size between current frame and last frame with a slope
    past_frame = current_frame -1
    while dictionary_name[past_frame] == np.nan:
        past_frame -= 1

    h = current_frame - past_frame

    # Backwards difference method formula
    m_prime = (dictionary_name[current_frame] - dictionary_name[past_frame] ) / h

    # Write m_prime to slope_prime_dict
    slope_prime_dict[current_frame] = m_prime


# Open the webcam
cap = cv2.VideoCapture(0)


while cap.isOpened():
    frame_count += 1
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
        probability_threshold = 0.5
        required_keypoints_right = [ "right_hip", "right_shoulder", "right_knee", "right_ankle"] #Included here for distinction R vs L
        missing_keypoints = [] # Initialize / clear the list

        
        # Dev Notes -> Need to add in 'Left' side of points
        for keypoint in required_keypoints_right:
            # Check to make sure the keypoint exists and has a high probability
            if keypoint not in keypoint_dict or keypoint_dict[keypoint]["probability"] < probability_threshold:
                missing_keypoints.append(keypoint)

        if missing_keypoints != []:
            print(f"Not all keypoints are visible: {missing_keypoints}") #Debug statement
            
            # No m value found for this frame
            null = np.nan
            slope_dict[frame_count] = null 
            slope_prime_dict[frame_count] = null
        
        else:
            m, b, r_sq =linear_regression(required_keypoints_right)

            sig_fig = 5 # Set a sig fig value for the print statements to reduce size
            m = round(m, sig_fig)
            b = round(b, sig_fig)
            r_sq = round(r_sq, sig_fig)

            # Add slope value at fame count to dictionary
            slope_dict[frame_count] = m

            # Calculate the derivative at the current frame
            backwards_difference(frame_count, slope_dict)

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

        # Display the slop of the back line (m)
        text = "Rep Counter: {}".format(rep_count)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (center_width - text_size[0] // 2)
        text_y = (center_height - text_size[1] // 2) - 60
        cv2.putText(
            pose_annotated_frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 225),
            3,
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

    

# Release the webcam
cap.release()
# Close all windows
cv2.destroyAllWindows()



# After the loop, you have a dictionary of {frame: value} pairs that you can plot
frame, slope = zip(*slope_dict.items())
frame, slope_prime = zip(*slope_prime_dict.items())

# Plot both the values of the slope and its derivative in differnet colors
plt.plot(frame, slope, color="blue")
plt.plot(frame, slope_prime, color="red")

plt.xlabel("Frame (n)")
plt.title("Slope = Blue , Slope Derivative = Red")

# Set x,y axis limits
plt.xlim(0, frame_count)
# plt.ylim(-5, 5)

plt.show()




