import torch
import torch.nn as nn
import torch.optim as optim
from src.model.lstm import LSTMPoseModel
from src.data_pipeline.preprocessing.data_loader import load_data

# Hyperparameters
input_size = 10  # Adjust based on data
hidden_size = 128
num_layers = 2
output_size = 1
learning_rate = 0.001
batch_size = 64
sequence_length = 30
num_epochs = 100

#  load data
train_loader = load_data('data/train.csv', sequence_length, batch_size)
test_loader = load_data('data/test.csv', sequence_length, batch_size)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMPoseModel(sequence_length, input_size, hidden_size, num_layers, 0.5, output_size, batch_size, device).to(device)
criteria = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criteria(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')



