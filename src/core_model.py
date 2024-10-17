from src.models.lstm import *


class CoreModel:
    def __init__(self):
        self.lstm = LSTM(output_size=6, patch_size=42, lstm_layers=1, hidden_size=64, number_block=1)
        self.model.load_model()

    def predict(self, clip):
        self.model.predict(clip)
