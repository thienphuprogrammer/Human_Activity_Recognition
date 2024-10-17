import time

import cv2
import mediapipe as mp
import numpy as np


def extract_and_zip_landmarks(landmark_list, num_landmarks, prefix) -> dict:
    landmarks = {}
    if landmark_list:
        for i, landmark in enumerate(landmark_list.landmark):
            landmarks[f"{prefix}_{i}"] = (landmark.x, landmark.y, landmark.z)
    else:
        for i in range(num_landmarks):
            landmarks[f"{prefix}_{i}"] = (np.nan, np.nan, np.nan)  # If no landmark, fill with NaN
    return landmarks


class PoseTracker:
    def __init__(self):
        # Initialize the MediaPipe Face Detection and Face Mesh modules
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                                                  enable_segmentation=False, refine_face_landmarks=False)
        # FPS calculation
        self.prev_time = 0
        self.current_time = 0

    def process_frame(self, frame, num_landmarks_pose=33):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame
        row_data = {}
        results = self.holistic.process(rgb_frame)
        row_data.update(extract_and_zip_landmarks(results.pose_landmarks, num_landmarks_pose, "pose"))
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

        return row_data

    def get_fps(self):
        self.current_time = time.time()
        fps = 1 / (self.current_time - self.prev_time)
        self.prev_time = self.current_time
        return fps
