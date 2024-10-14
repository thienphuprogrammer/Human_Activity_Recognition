import cv2
import os
import pandas as pd
from moviepy.editor import VideoFileClip
from src.data_pipeline.processors.pose_extractor import PoseExtractor
from tqdm import tqdm

folder_mapping = {
    "Đấm": "Punch",
    "Đá": "Kick",
    "Nhảy": "Jump",
    "Sang trái": "Left",
    "Sang phải": "Right",
    "Đứng yên": "Stand"
    # Thêm các thư mục khác nếu cần
}


class VideoHumanActivitiesLoader:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.pose_extractor = PoseExtractor()
        self.mapping = folder_mapping

    def process_videos(self):
        for class_folder in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_folder)
            if not os.path.isdir(class_path):
                print(f"Skipping {class_path} as it is not a directory")
                continue

            for video_file in os.listdir(class_path):
                video_path = os.path.join(class_path, video_file)
                if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                    print(f"Skipping {video_path} as it is not a video file")
                    continue

                print(f"Processing video {video_path}")
                df = self.extract_poses_to_csv(video_path)
                # self.save_to_csv(df, class_folder, video_file)


    def extract_poses_to_csv(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize list to store landmarks
        landmarks_data = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_num / fps
            # Convert the BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Extract pose landmarks
            landmarks_data.append(
                self.pose_extractor.extract_pose(frame_rgb, timestamp, frame_num)
            )
            frame_num += 1

        cap.release()
        cv2.destroyAllWindows()

        # headers = ["frame_num", "timestamp"]

        # pose_headers = [f"pose_{i}" for i in range(33)]
        # left_hand_headers = [f"left_hand_{i}" for i in range(21)]
        # right_hand_headers = [f"right_hand_{i}" for i in range(21)]
        #
        # headers += pose_headers + left_hand_headers + right_hand_headers
        #
        # df = pd.DataFrame(landmarks_data, columns=headers)
        # return df


def process_videos_to_csv(source_root: str, dest_root: str):
    """
    Process the videos in the source directory and save the extracted poses to CSV files in the destination directory.

    :param source_root: The source directory containing the videos
    :param dest_root: The destination directory to save the CSV files
    """
    for dir_path, dir_names, filenames in os.walk(source_root):
        relative_dir = os.path.relpath(dir_path, source_root)
        dest_dir = os.path.join(dest_root, relative_dir)

        for filename in tqdm(filenames, desc=f"Processing {relative_dir}"):
            if filename.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(dir_path, filename)
                video_name = os.path.splitext(filename)[0]
                csv_filename = f"{video_name}.csv"
                csv_path = os.path.join(dest_dir, csv_filename)

                # Extract poses from the video
                loader = VideoHumanActivitiesLoader(source_root, dest_root)
                try:
                    df = loader.extract_poses_to_csv(video_path)

                    # Save the extracted poses to a CSV file
                    df.to_csv(csv_path, index=False)
                    print(f"Saved poses to {csv_path}")
                except Exception as e:
                    print(f"Error processing video {video_path}: {e}")
