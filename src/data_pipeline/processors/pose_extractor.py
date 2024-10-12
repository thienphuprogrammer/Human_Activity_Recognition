import mediapipe as mp
import numpy as np


def extract_and_zip_landmarks(landmarks_list, num_landmarks, prefix) -> dict:
    landmarks = {}

    if landmarks_list:
        for i, landmark in enumerate(landmarks_list):
            landmarks[f"{prefix}_{i}"] = (landmark.x, landmark.y, landmark.z)

    else:
        for i in range(num_landmarks):
            landmarks[f"{prefix}_{i}"] = (np.nan, np.nan, np.nan)

    return landmarks


class PoseExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    def extract_pose(self, frame_rgb, timestamp, frame_num):
        with self.mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                                       refine_face_landmarks=False) as holistic:
            # Process the frame
            results = holistic.process(frame_rgb)

            # Initialize list to store landmarks of the frame
            row_data = {
                'timestamp': timestamp,
                'frame_num': frame_num,
            }
            # Extract pose landmarks and zip (x, y, z)
            pose_landmarks = extract_and_zip_landmarks(results.pose_landmarks, 33, 'pose')
            row_data.update(pose_landmarks)

            # Extract left hand landmarks and zip (x, y, z)
            left_hand_landmarks = extract_and_zip_landmarks(results.left_hand_landmarks, 21, 'left_hand')
            row_data.update(left_hand_landmarks)

            # Extract right hand landmarks and zip (x, y, z)
            right_hand_landmarks = extract_and_zip_landmarks(results.right_hand_landmarks, 21, 'right_hand')
            row_data.update(right_hand_landmarks)

        return row_data
