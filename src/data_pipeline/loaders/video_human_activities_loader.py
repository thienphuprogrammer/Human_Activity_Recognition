import cv2
import os
from src.data_pipeline.processors.pose_extractor import PoseExtractor


class VideoHumanActivitiesLoader:
    def __init__(self, video_dir, output_csv_dir):
        self.video_dir = video_dir
        self.output_csv_dir = output_csv_dir
        self.pose_extractor = PoseExtractor()

    def load_videos(self):
        for video_file in os.listdir(self.video_dir):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(self.video_dir, video_file)
                self.extract_poses(video_path)

    def extract_poses(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.pose_extractor.extract_pose(frame)
            if landmarks:
                self.save_landmarks(landmarks, frame_count)
            frame_count += 1
        cap.release()