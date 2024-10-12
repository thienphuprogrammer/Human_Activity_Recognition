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
    def __init__(self, dataset_path: str, output_csv_dir: str):
        self.dataset_path = dataset_path

        self.output_csv_dir = output_csv_dir
        self.pose_extractor = PoseExtractor()
        self.mapping = folder_mapping

    def rename_files(self):
        """
        Rename the files in the dataset directory.
        """
        for split in ['Train', 'Test']:
            split_path = os.path.join(self.dataset_path, split)

            if not os.path.exists(split_path):
                # raise an exception if the path does not exist
                raise Exception(f'The path {split_path} does not exist.')

            for old_name, new_name in self.mapping.items():
                old_path = os.path.join(split_path, old_name)
                new_path = os.path.join(split_path, new_name)
                if not os.path.exists(old_path):
                    if not os.path.exists(new_path):
                        os.rename(old_path, new_path)
                        print(f'Renamed {old_path} to {new_path}')
                    else:
                        print(f'The path {new_path} already exists.')

            for class_folder in os.listdir(split_path):
                class_path = os.path.join(split_path, class_folder)

                if not os.path.isdir(class_path):
                    print(f'The path {class_path} is not a directory.')
                    continue

                videos = sorted(os.listdir(class_path))
                video_count = 1

                for video in videos:
                    video_ext = os.path.splitext(video)[1]  # Get the file extension (e.g., .mp4)
                    old_video_path = os.path.join(class_path, video)
                    new_video_name = f'{class_folder}_{video_count:03}{video_ext}'  # e.g., 'Class_001.mp4'
                    new_video_path = os.path.join(class_path, new_video_name)

                    if os.path.isfile(old_video_path):
                        os.rename(old_video_path, new_video_path)
                        print(f'Renamed {old_video_path} to {new_video_path}')
                        video_count += 1
                    else:
                        print(f'The path {old_video_path} does not exist.')

    def copy_directory_structure(self):
        """
        Copy the directory structure from the source directory to the destination directory.
        """
        for dir_path, dir_names, file_names in os.walk(self.dataset_path):
            structure = os.path.join(self.output_csv_dir, os.path.relpath(dir_path, self.dataset_path))
            if not os.path.isdir(structure):
                os.makedirs(structure)
                print(f'Created directory: {structure}')

    def resize_videos_to_new_folder(self, new_folder, target_size):
        """
        Resize the videos in the dataset to a new folder.

        :param new_folder: The path to the new folder
        :param target_size: The target size (e.g., (1280, 720))
        """
        os.makedirs(new_folder, exist_ok=True)

        for split in ['Train', 'Test']:
            split_path = os.path.join(self.dataset_path, split)
            if not os.path.exists(split_path):
                raise Exception(f'The path {split_path} does not exist.')

            # Create the new folder structure
            new_split_path = os.path.join(new_folder, split)
            os.makedirs(new_split_path, exist_ok=True)

            # Resize the videos
            for class_folder in os.listdir(split_path):
                class_path = os.path.join(split_path, class_folder)

                if not os.path.isdir(class_path):
                    print(f'The path {class_path} is not a directory.')
                    continue

                new_class_path = os.path.join(new_split_path, class_folder)
                os.makedirs(new_class_path, exist_ok=True)

                for video in os.listdir(class_path):
                    video_path = os.path.join(class_path, video)
                    video_ext = os.path.splitext(video)[1]

                    if (not os.path.isfile(video_path)
                            or video_ext not in ['.mp4', '.avi', '.mov']):
                        print(f'The path {video_path} is not a video file.')
                        continue
                    try:
                        # Resize the video
                        clip = VideoFileClip(video_path)
                        resized_clip = clip.resize(target_size)

                        new_video_name = f"resized_{video}"
                        new_video_path = os.path.join(new_class_path, new_video_name)

                        resized_clip.write_videofile(new_video_path, codec='libx264')

                        clip.close()
                        resized_clip.close()
                        print(f'Resized {video_path} to {new_video_path}')
                    except Exception as e:
                        raise Exception(f'Error resizing {video_path}: {e}')

    def extract_poses_to_csv(self, video_path):
        # Load the video
        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Initialize list to store landmarks
        landmarks_data = []
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timestamp of the frame
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

        headers = ["frame_num", "timestamp"]

        pose_headers = [f"pose_{i}" for i in range(33)]
        left_hand_headers = [f"left_hand_{i}" for i in range(21)]
        right_hand_headers = [f"right_hand_{i}" for i in range(21)]

        headers += pose_headers + left_hand_headers + right_hand_headers

        df = pd.DataFrame(landmarks_data, columns=headers)
        return df


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
