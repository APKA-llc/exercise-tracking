# Description: This script will detect the pose of a person and plot the 3D graph of the pose
import cv2
from ultralytics import YOLO
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Open the webcam (0 for front camera, 1 for back)
cap = cv2.VideoCapture(0)

# Initialize a dictionary to store keypoints
keypoints_data = {name: {'x': [], 'y': [], 't': []} for name in keypoint_names}

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Pose detection
        pose_results = pose_model(frame, verbose=False, conf=0.5)

        # Get the current time
        current_time = time.time()

        # Print each body coordinate as a dictionary
        for person in pose_results:
            keypoints = person.keypoints.data[0]
            for keypoint, name in zip(keypoints, keypoint_names):
                x, y, probability = keypoint
                keypoints_data[name]['x'].append(x.item())
                keypoints_data[name]['y'].append(y.item())
                keypoints_data[name]['t'].append(current_time)

            pose_annotated_frame = person.plot()
            cv2.imshow("Pose Detection", pose_annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

# Now, plot the 3D graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Choose keypoints to plot
# change this to the keypoints you want to plot
keypoints_to_plot = ["left_shoulder", "left_elbow", "left_wrist",]
colors = ['r', 'g', 'b']  # colors for each keypoint

for keypoint, color in zip(keypoints_to_plot, colors):
    ax.scatter(keypoints_data[keypoint]['x'], keypoints_data[keypoint]
               ['y'], keypoints_data[keypoint]['t'], c=color)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Time')
plt.show()
