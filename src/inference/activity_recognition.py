import torch
import numpy as np
from src.model.utils.pose_processing import process_pose
from src.model.lstm import LSTMPoseModel


def load_model(model_path):
    model = LSTMPoseModel(seq_len=30, input_dim=33, hidden_dim=128, num_layers=2, dropout=0.5, output_dim=1, batch_size=1, device='cpu')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def recognize_activity(pose_seq, model):
    pose_seq = process_pose(pose_seq)
    pose_seq = np.array(pose_seq)
    pose_seq = torch.tensor(pose_seq, dtype=torch.float32).unsqueeze(0)
    output = model(pose_seq)
    return output.item()