import tensorflow as tf
from src.model.architectures.lstm import build_lstm_model
from src.data_pipeline.loaders.dataset_classes.sequential_data_loader import load_data


def train_model(file_path: str):
    data = load_data(file_path)
    model = build_lstm_model((32, 32), 10)
    model.fit(data, epochs=10)
    model.save('results/models/lstm_model.h5')