import mediapipe as mp
import numpy as np
import cv2


class PoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def extract_pose(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            # Extract the coordinates of the landmarks
            landmarks = results.pose_landmarks.landmark
            return [(lm.x, lm.y, lm.z) for lm in landmarks]
        return None