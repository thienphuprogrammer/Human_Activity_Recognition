import mediapipe as mp
import numpy as np


def extract_and_zip_landmarks(landmark_list, num_landmarks, prefix):
    landmarks = {}
    if landmark_list:
        for i, landmark in enumerate(landmark_list.landmark):
            landmarks[f"{prefix}_{i}"] = (landmark.x, landmark.y, landmark.z)
    else:
        for i in range(num_landmarks):
            landmarks[f"{prefix}_{i}"] = (np.nan, np.nan, np.nan)  # If no landmark, fill with NaN
    return landmarks


def get_pose_image(image,
                   num_landmarks_pose=33,
                   num_landmarks_left_hand=21,
                   num_landmarks_right_hand=21) -> dict:
    # Initialize MediaPipe Holistic models and OpenCV
    mp_holistic = mp.solutions.holistic
    row_data = {}

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                              enable_segmentation=False, refine_face_landmarks=False) as holistic:
        results = holistic.process(image)

        row_data.update(extract_and_zip_landmarks(results.pose_landmarks, num_landmarks_pose, "pose"))
        row_data.update(extract_and_zip_landmarks(results.left_hand_landmarks, num_landmarks_left_hand, "left_hand"))
        row_data.update(extract_and_zip_landmarks(results.right_hand_landmarks, num_landmarks_right_hand, "right_hand"))

    return row_data
