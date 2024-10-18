import numpy as np
import torch

from src.data_pipeline.preprocessing.handle_dim_sequence import pad_and_truncate
from src.data_pipeline.preprocessing.handle_missing_value import get_variance_of_metrics, handle_fill_nan_by_variance
from src.models import Model


class CoreModel:
    def __init__(self, model_path, device='cuda'):
        self.model = Model(device=device)
        self.model.load_model(model_path=model_path)

    def predict(self, clip_frame, max_dim=35):
        # Resize the size of clip
        X = []
        for i, ele in enumerate(clip_frame):
            padded_element = pad_and_truncate(ele, max_dim)
            X = X + padded_element
        X = np.array(X)

        # Fill nan by variance
        variance_metric = get_variance_of_metrics(X)
        for i in range(X.shape[1]):
            X[:, i] = handle_fill_nan_by_variance(
                X[:, i], variance_metric
            )
        X = torch.from_numpy(X).float()
        self.model.predict(X)
