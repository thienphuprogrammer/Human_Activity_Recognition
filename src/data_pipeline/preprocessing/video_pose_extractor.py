from moviepy.editor import VideoFileClip
import cv2
import pandas as pd
import os
from tqdm import tqdm
from pose_detect import *


def resize_videos_and_save(dataset_path, output_path, target_size=(640, 640)):
    os.makedirs(output_path, exist_ok=True)

    for split in ['Train', 'Test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            raise ValueError(f"Not found {split} folder in {dataset_path}")

        output_split_path = os.path.join(output_path, split)
        os.makedirs(output_split_path, exist_ok=True)

        for class_folder in os.listdir(split_path):
            class_path = os.path.join(split_path, class_folder)

            # Check if it is a directory
            if os.path.isdir(class_path):
                # Create class folder in output path
                output_class_path = os.path.join(output_split_path, class_folder)
                os.makedirs(output_class_path, exist_ok=True)

                for video in os.listdir(class_path):
                    video_path = os.path.join(class_path, video)
                    video_ext = os.path.splitext(video)[1]

                    if os.path.isfile(video_path) and video_ext in ['.mp4', '.avi', '.mov']:
                        try:
                            clip = VideoFileClip(video_path)
                            resized_clip = clip.resize(target_size)
                            new_video_name = f"resized_{video}"
                            new_video_path = os.path.join(output_class_path, new_video_name)
                            resized_clip.write_videofile(new_video_path, codec="libx264")
                            clip.close()
                            resized_clip.close()
                            print(f"Handled video {video_path}")
                        except Exception as e:
                            raise ValueError(f"Error when handling video {video_path}: {e}")


def detect_pose_video(video_path,
                      num_landmarks_pose=33,
                      num_landmarks_left_hand=21,
                      num_landmarks_right_hand=21):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize list to store results for each frame
    landmark_data = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        # Calculate timestamp for current frame
        timestamp = frame_num / fps
        # Convert the image to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image with MediaPipe Holistic model (without face detection)
        row_data = get_pose_image(image, frame_num, timestamp)
        # Append the row data to the list
        landmark_data.append(row_data)
        # Move to the next frame
        frame_num += 1

    # Define the headers for the CSV file
    headers = []

    # Add headers for each landmark (Pose, Face, Left Hand, Right Hand)
    pose_headers = [f"pose_{i}" for i in range(num_landmarks_pose)]
    left_hand_headers = [f"left_hand_{i}" for i in range(num_landmarks_left_hand)]
    right_hand_headers = [f"right_hand_{i}" for i in range(num_landmarks_right_hand)]

    # Combine all headers
    headers += pose_headers + left_hand_headers + right_hand_headers

    df = pd.DataFrame(landmark_data, columns=headers)
    return df


def process_videos(source_root: str, destination_root: str):
    X, y = [], []
    for dir_path, dir_names, filenames in os.walk(source_root):
        relative_dir: str = os.path.relpath(dir_path, source_root)
        dest_dir = os.path.join(destination_root, relative_dir)

        for filename in tqdm(filenames, desc=f"Processing {relative_dir}"):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(dir_path, filename)

                csv_filename = os.path.splitext(filename)[0] + '.csv'  # Change extension to .csv
                csv_path = os.path.join(dest_dir, csv_filename)  # Create the path for the CSV file

                try:
                    df = detect_pose_video(video_path)  # Process the video
                    X.append(df)  # Append the DataFrame to the list
                    y.append(relative_dir)  # Append the label to the list

                    df.to_csv(csv_path, index=False)
                except Exception as e:
                    raise ValueError(f"Error when processing video {video_path}: {e}")

    return X, y
