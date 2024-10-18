import torch
import torch.nn as nn


class ResBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlockMLP, self).__init__()
        # Define the layers for the Residual Block
        # Define layer normalization for input size
        self.morm1 = nn.LayerNorm(input_size)
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, input_size // 2)

        # Layer normalization for input size//2
        self.morm2 = nn.LayerNorm(input_size // 2)
        # Second fully connected layer
        self.fc2 = nn.Linear(input_size // 2, output_size)

        # Final fully connected layer
        self.fc3 = nn.Linear(input_size, output_size)

        # Define the activation function
        self.activation = nn.ELU()

    def forward(self, x):
        out = self.activation(self.morm1(x))
        # Calculate the residual
        skip = self.fc3(x)

        # Apply layer normalization, fully connected layer, and activation function
        out = self.activation(self.morm2(self.fc1(out)))
        out = self.fc2(out)

        # Add the residual to the output
        return out + skip


class DeepLSTMModel(nn.Module):
    def __init__(self, input_size, output_size, patch_size, lstm_layers=2, hidden_size=1024, number_block=1,
                 dropout=0.5):
        super(DeepLSTMModel, self).__init__()

        # Input layer to match patch size to hidden size
        self.activation = nn.ELU()

        # Deep LSTM: Bidirectional with many layers
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(input_size=hidden_size * 2,
                             hidden_size=hidden_size // 2, batch_first=True,
                             bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)

        self.lstm3 = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size // 8, batch_first=True,
                             bidirectional=True)
        self.dropout3 = nn.Dropout(dropout)

        blocks = [ResBlockMLP(hidden_size, hidden_size // 8) for _ in range(number_block)]
        self.blocks = nn.Sequential(*blocks)
        self.fc_out = nn.Linear(hidden_size // 4, output_size)

    def initialize_hidden(self, batch_size, hidden_size):
        # Initialize hidden and cell states (2 for bidirectional, * layers)
        num_directions = 2
        hidden = torch.zeros(self.lstm.num_layers * num_directions, batch_size, hidden_size)
        cell = torch.zeros(self.lstm.num_layers * num_directions, batch_size, hidden_size)
        return hidden, cell

    def forward(self, input_data):
        bs, seq, col, row = input_data.size()
        input_data = input_data.view(bs, seq, -1)  # Flatten the input data

        # LSTM layer 1
        out, _ = self.lstm1(input_data)
        out = self.dropout1(out)

        # LSTM layer 2
        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        # LSTM layer 3
        out, _ = self.lstm3(out)
        out = self.dropout3(out)

        # # Residual blocks
        # out = self.blocks(out)

        # Fully connected layer
        out = self.fc_out(out)
        return out


__all__ = ['ResBlockMLP', 'DeepLSTMModel']
