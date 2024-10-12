import numpy as np


class PoseProcessor:
    def __init__(self):
        self.pose = None

    def process_pose(self, landmarks):
        if not landmarks:
            return None

        # Extract the coordinates of the landmarks
        coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])

        # Normalize the coordinates
        coords = self.normalize(coords)

        return coords

    def normalize(self, coords):
        # Normalize the coordinates
        return coords / np.linalg.norm(coords, axis=0)


def process_pose(landmarks):
    processor = PoseProcessor()
    return [processor.process_pose(frame_lm) for frame_lm in landmarks]