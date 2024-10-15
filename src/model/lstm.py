import torch
from torch.autograd import Variable
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, num_layers, dropout,
                 output_dim=1, batch_size=1, device='gpu'):
        super(LSTM, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.device = device
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden(batch_size, device)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.hidden_dim).to(device),
                torch.zeros(1, batch_size, self.hidden_dim).to(device))

    def forward(self, x):
        lstm_out, _ = self.lstm(x, self.hidden)
        out = self.fc(lstm_out[:, -1, :])
        return out

    def reset_hidden(self):
        self.hidden = self.init_hidden(self.batch_size, self.device)