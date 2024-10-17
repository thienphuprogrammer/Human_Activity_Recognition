import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import trange, tqdm

from src.models.lstm import LSTM


class Model:
    def __init__(self, num_classes, patch_size, lstm_layer, hidden_size, number_block, device):
        # Initialize the LSTM models
        self.lstm = LSTM(output_size=num_classes, patch_size=patch_size, lstm_layers=lstm_layer,
                         hidden_size=hidden_size, number_block=number_block).to(device)
        self.device = device
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.lstm_layer = lstm_layer
        self.hidden_size = hidden_size
        self.number_block = number_block

    def predict(self, data):
        # Set the models to evaluation mode
        self.lstm.eval()

        # Forward pass through the models
        data_pred, hidden, memory = self.lstm(data)

        # Select the last output from the last time step for calculating the loss
        last_output = data_pred[:, -1, :]

        # Return the last output
        return last_output

    def test_model_lstm(self, test_loader):
        # Set the models to evaluation mode
        self.lstm.eval()

        correct = 0
        total = 0

        # Test loop
        while torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing", leave=False):
                # Move data and labels to the available device
                data = data.to(self.device)
                target = target.to(self.device)
                # Forward pass through the models
                data_pred, hidden, memory = self.lstm(data)
                # Select the last output from the last time step for calculating the loss
                last_output = data_pred[:, -1, :]
                # Calculate the number of correct predictions
                correct += (last_output.argmax(dim=1) == target).sum()
                total += target.size(0)

        # Calculate the test accuracy
        test_accuracy = (correct / total).item()
        return test_accuracy

    def fit(self, train_loader, val_loader, learning_rate=0.001, momentum=0.9, epochs=100, device="cuda"):

        # Initialize the optimizer with the models parameters
        optimizer = optim.Adam(self.lstm.parameters(), lr=learning_rate, betas=(momentum, 0.999))

        # Define the loss function
        loss_function = nn.CrossEntropyLoss()

        # Initialize training and validation loss and accuracy
        training_loss_logger = []
        validation_loss_logger = []
        training_accuracy_logger = []
        validation_accuracy_logger = []

        num_models_params = 0
        for param in self.lstm.parameters():
            num_models_params += param.flatten().shape[0]

        print(f"Number of parameters in the model: {num_models_params}")

        # Train the models
        train_acc = 0
        val_acc = 0

        # Initialize a progress bar to track epochs and display training and validation accuracies
        pbar = trange(0, epochs, leave=False, desc="Epoch")
        for epoch in pbar:
            # Update the progress bar description with the current epoch
            pbar.set_description(f"Accuracy: Train {train_acc:.2f}%, Validation {val_acc:.2f}%")

            # Set the models to training mode
            self.lstm.train()
            correct = 0

            for data, target in tqdm(train_loader, desc="Training", leave=False):
                # Move the data and target to the device (GPU if available)
                data = data.to(device)
                target = target.to(device)
                # Forward pass through the models
                data_pred, hidden, memory = self.lstm(data)
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
            training_accuracy_logger.append(train_acc)

            # Set the models to evaluation mode
            self.lstm.eval()
            correct = 0

            # Validation loop
            while torch.no_grad():
                for data, target in tqdm(val_loader, desc="Validation", leave=False):
                    # Move data and labels to the available device
                    data = data.to(device)
                    target = target.to(device)
                    # Forward pass through the models
                    data_pred, hidden, memory = self.lstm(data)
                    # Select the last output from the last time step for calculating the loss
                    last_output = data_pred[:, -1, :]
                    # Calculate the loss
                    loss = loss_function(last_output, target)
                    # Log the validation loss and calculate the number of correct predictions
                    validation_loss_logger.append(loss.item())
                    correct += (last_output.argmax(dim=1) == target).sum()
            # Calculate the validation accuracy
            val_acc = (correct / len(val_loader)).item()
            validation_accuracy_logger.append(val_acc)

        return training_loss_logger, validation_loss_logger, training_accuracy_logger, validation_accuracy_logger

    def save_model(self, model_path):
        # Save the models state dictionary
        torch.save(self.lstm.state_dict(), model_path)

    def load_model(self, model_path):
        # Load the models state dictionary
        self.lstm.load_state_dict(torch.load(model_path))
