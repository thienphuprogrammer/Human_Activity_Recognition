import cv2
import mediapipe as mp


class PossTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

    def track_pose(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            return [(lm.x, lm.y, lm.z) for lm in landmarks]
        return None


def process_frame(video_frame):
    cap = cv2.VideoCapture(0)
    tracker = PossTracker()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = tracker.track_pose(frame)
        if landmarks:
            print("Landmarks detected: ", landmarks)
    cap.release()
    cv2.destroyAllWindows()