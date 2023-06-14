# Same bicep counting method but attempting to use uploaded footage instead of webcam

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

# Open the webcam
#cap = cv2.VideoCapture(0)

# Select Excersize Video
# video_name = "pushup_good.mp4"
# video_name = "deadlift_good.mp4"
# video_name = "squat_good.mp4"
video_name = "curls_good.mp4"

# Open the video file
cap = cv2.VideoCapture(f"videos\{video_name}")



rep_counter = 0
is_rep = False

while cap.isOpened():
    success, frame = cap.read()

    if success:
        frame = cv2.resize(frame, (300, 300))
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

            # Check if right_wrist and right_elbow keypoints exist
            if "right_wrist" in keypoint_dict and "right_elbow" in keypoint_dict:
                # If wrist is above the elbow and to its left
                if (
                    keypoint_dict["right_wrist"]["y"]
                    < keypoint_dict["right_elbow"]["y"]
                    and keypoint_dict["right_wrist"]["x"]
                    < keypoint_dict["right_elbow"]["x"]
                ):
                    is_rep = True
                # Rep ends when wrist goes down past the elbow
                elif (
                    is_rep
                    and keypoint_dict["right_wrist"]["y"]
                    > keypoint_dict["right_elbow"]["y"]
                ):
                    is_rep = False
                    rep_counter += 1

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
