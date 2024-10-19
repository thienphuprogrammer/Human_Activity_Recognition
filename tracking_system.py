from src.core_model import CoreModel
from src.data_pipeline.preprocessing.pose_tracker import *

label_list = {"Jump": 0, "Kick": 1, "Punch": 2, "Left": 3, "Right": 4, "Stand": 5}


class TrackingSystem:
    def __init__(self, model_path, max_stored_clips=25, save_interval=0.1):
        self.pose_tracker = PoseTracker()
        self.clip_storage = []
        self.max_stored_clips = max_stored_clips
        self.save_interval = save_interval
        self.count_frames = 0
        self.last_save_time = time.time()
        self.core_model = CoreModel(model_path=model_path)

    def update_and_predict_clip(self, clip):
        results = None
        if len(self.clip_storage) >= self.max_stored_clips:
            # flatten the array result
            flatten_result = []
            for clip in self.clip_storage:
                for frame in clip:
                    flatten_result.append(frame)
            flatten_result = np.array(flatten_result)
            results = self.core_model.predict(flatten_result)
            print(f"Prediction: {results}")
            self.count_frames -= len(self.clip_storage[0])
            self.clip_storage.pop(0)
        clip = np.array(clip)
        self.clip_storage.append(clip)
        self.count_frames += len(clip)

        return results

    def start_camera(self):
        # cap = cv2.VideoCapture("./data/raw/HAR/Train/Right/Right_1.mp4")
        cap = cv2.VideoCapture("./data/7752246914108791071.mp4")
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        print("Press 'q' to quit")

        current_clip = []
        results = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror the frame
            row_data = self.pose_tracker.process_frame(frame)
            current_clip.append(row_data)

            if time.time() - self.last_save_time >= self.save_interval:
                # self.core_model.predict(current_clip)
                results = self.update_and_predict_clip(current_clip)
                current_clip = []
                self.last_save_time = time.time()

            # Display the FPS
            fps = self.pose_tracker.get_fps()
            if results is not None:
                cv2.putText(
                    frame,
                    f"Prediction: {list(label_list.keys())[results[0]]}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Clips stored: {self.count_frames}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Display the frame
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pose_tracker = TrackingSystem(model_path="./results/models/har_model.pth")
    pose_tracker.start_camera()
    del pose_tracker
