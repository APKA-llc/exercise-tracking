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
cap = cv2.VideoCapture(1)

# Initiate variable to track spacebar status
spacebar_on = False

# Prepare a file to write the coordinates
output_file = open("coordinates.txt", "w")

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Pose detection
        pose_results = pose_model(frame, verbose=False, conf=0.5)

        # Print each body coordinate as a dictionary
        for person in pose_results:
            keypoints = person.keypoints.data[0]
            for keypoint, name in zip(keypoints, keypoint_names):
                x, y, probability = keypoint

                # Only print and write right_wrist coordinates when spacebar_on is True
                if name == "nose" and spacebar_on:
                    output_file.write(
                        f"x={x.item()}, y={y.item()}, probability={probability.item()}\n"
                    )
                    print(
                        {
                            "keypoint": name,
                            "x": x.item(),
                            "y": y.item(),
                            "probability": probability.item(),
                        }
                    )

            pose_annotated_frame = person.plot()
            cv2.imshow("Pose Detection", pose_annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):  # spacebar press event
            spacebar_on = not spacebar_on  # toggle status
    else:
        break

output_file.close()
cap.release()
cv2.destroyAllWindows()
