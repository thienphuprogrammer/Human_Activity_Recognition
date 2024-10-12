import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, seq_len):
        self.data = pd.read_csv(data_path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx+self.seq_len, 1:].values
        y = self.data.iloc[idx+self.seq_len, 1:].values
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_data(data_path, seq_len, batch_size):
    dataset = TimeSeriesDataset(data_path, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
