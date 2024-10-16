import torch
import torch.nn as nn
import torch.optim as optim
from src.model.lstm import LSTM
from tqdm.notebook import trange, tqdm


def train_model_lstm(lstm_model: LSTM, train_loader, val_loader, lstm_layer=35, hidden_size=256, learning_rate=0.001, momentum=0.9, epochs=100, device="cuda"):
    # Initialize the optimizer with the model parameters
    optimizer = optim.Adam(lstm_model.parameters(), lr=learning_rate, betas=(momentum, 0.999))

    # Define the loss function
    loss_function = nn.CrossEntropyLoss()

    # Initialize training and validation loss and accuracy
    training_loss_logger = []
    validation_loss_logger = []
    training_accuracy_logger = []
    validation_accuracy_logger = []

    num_models_params = 0
    for param in lstm_model.parameters():
        num_models_params += param.flatten().shape[0]

    print(f"Number of parameters in the model: {num_models_params}")

    # Train the model
    train_acc = 0
    val_acc = 0

    # Initialize a progress bar to track epochs and display training and validation accuracies
    pbar = trange(0, epochs, leave=False, desc="Epoch")
    for epoch in pbar:
        # Update the progress bar description with the current epoch
        pbar.set_description(f"Accuracy: Train {train_acc:.2f}%, Validation {val_acc:.2f}%")

        # Set the model to training mode
        lstm_model.train()
        correct = 0

        for data, target in tqdm(train_loader, desc="Training", leave=False):
            # Move the data and target to the device (GPU if available)
            data = data.to(device)
            target = target.to(device)

            # Initialize the hidden state and memory buffer for the LSTM
            hidden = torch.zeros(lstm_layer, data.shape[0], hidden_size, device=device)
            memory = torch.zeros(lstm_layer, data.shape[0], hidden_size, device=device)

            # Forward pass through the model
            data_pred, hidden, memory = lstm_model(data, hidden, memory)

            # Select the last output from the last time step for calculating the loss
            last_output = data_pred[:, -1, :]

            # Calculate the loss
            loss = loss_function(last_output, target)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the training loss and calculate the number of correct predictions
            training_loss_logger.append(loss.item())
            correct += (last_output.argmax(dim=1) == target).sum()

        # Calculate the training accuracy
        train_acc = (correct / len(train_loader)).item()
        training_loss_logger.append(train_acc)

        # Set the model to evaluation mode
        lstm_model.eval()
        correct = 0

        # Validation loop
        while torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation", leave=False):
                # Move data and labels to the available device
                data = data.to(device)
                target = target.to(device)

                # Initialize the hidden state and memory buffer for the LSTM
                hidden = torch.zeros(lstm_layer, data.shape[0], hidden_size, device=device)
                memory = torch.zeros(lstm_layer, data.shape[0], hidden_size, device=device)

                # Forward pass through the model
                data_pred, hidden, memory = lstm_model(data, hidden, memory)

                # Select the last output from the last time step for calculating the loss
                last_output = data_pred[:, -1, :]

                # Calculate the loss
                loss = loss_function(last_output, target)

                # Log the validation loss and calculate the number of correct predictions
                validation_loss_logger.append(loss.item())
                correct += (last_output.argmax(dim=1) == target).sum()

        # Calculate the validation accuracy
        val_acc = (correct / len(val_loader)).item()
