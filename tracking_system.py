import time

import cv2
import numpy as np

from src.core_model import CoreModel
from src.data_pipeline.preprocessing.pose_tracker import *


class TrackingSystem:
    def __init__(self, model_path, num_frames=35, max_stored_clips=3, save_interval=0.5):
        self.pose_tracker = PoseTracker()
        self.clip_storage = []
        self.num_frames = num_frames
        self.max_stored_clips = max_stored_clips
        self.save_interval = save_interval
        self.count_frames = 0
        self.last_save_time = time.time()
        self.core_model = CoreModel(model_path=model_path)

    def update_and_predict_clip(self, clip):
        results = None
        if len(self.clip_storage) >= self.max_stored_clips:
            # flatten the array result
            flatten_result = np.array(self.clip_storage)
            flatten_result = flatten_result.flatten()

            results = self.core_model.predict(flatten_result)
            self.count_frames -= len(self.clip_storage[0])
            self.clip_storage.pop(0)
        self.clip_storage.append(clip)
        self.count_frames += len(clip)

        return results

    def start_camera(self):
        cap = cv2.VideoCapture("./../data/raw/7752246914108791071.mp4")
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        print("Press 'q' to quit")
        current_clip = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror the frame
            row_data = self.pose_tracker.process_frame(frame)
            current_clip.append(row_data)

            if time.time() - self.last_save_time >= self.save_interval:
                # self.core_model.predict(current_clip)
                self.update_and_predict_clip(current_clip)
                current_clip = []
                self.last_save_time = time.time()

            # Display the FPS
            fps = self.pose_tracker.get_fps()
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Clips stored: {self.count_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2)

            # Display the frame
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pose_tracker = TrackingSystem()
    pose_tracker.start_camera()
    del pose_tracker
