import torch
from torch.autograd import Variable
import torch.nn as nn


class  ResBlockMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResBlockMLP, self).__init__()
        # Define the layers for the Residual Block
        # Define layer normalization for input size
        self.morm1 = nn.LayerNorm(input_size)
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, input_size//2)

        # Layer normalization for input size//2
        self.morm2 = nn.LayerNorm(input_size//2)
        # Second fully connected layer
        self.fc2 = nn.Linear(input_size//2, output_size)

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


class LSTM(nn.Module):
    def __init__(self, output_size, patch_size, lstm_layers, hidden_size=1, number_block=1):
        super(LSTM, self).__init__()

        # Define the layers for the LSTM models
        self.fc_in = nn.Linear(patch_size, hidden_size)

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=lstm_layers, batch_first=True)

        # Define the Residual Blocks
        blocks = [ResBlockMLP(hidden_size, hidden_size) for _ in range(number_block)]
        self.blocks = nn.Sequential(*blocks)

        # Define the output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        # Activation function
        self.activation = nn.ELU()
        self.patch_size = patch_size

    def forward(self, input_data, hidden_in, mem_in):
        bs, seq, col, row = input_data.size()

        # Reshape the input data
        input_data = input_data.view(bs, seq, -1)

        out = self.fc_in(input_data)

        # Apply the input fully connected layer
        out = self.activation(out)

        # The LSTM layer
        out, (hidden_out, mem_out) = self.lstm(out, (hidden_in, mem_in))

        # Apply the Residual Blocks
        out = self.blocks(out)

        # Apply activation function
        out = self.activation(out)

        # Apply the output fully connected layer
        out = self.fc_out(out)
        out = self.softmax(out)

        return out, hidden_out, mem_out
