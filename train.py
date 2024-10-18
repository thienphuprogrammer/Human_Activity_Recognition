import torch
import yaml

from src.data_pipeline.loaders.loaders import load_har_dataset
from src.models.model import Model
from src.utils.loggers import visualize_loss, visualize_accuracy

# Load the configuration file
with open("config/model_config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

    dataset_path = cfg["PATHS"]["dataset_path"]
    resize_dataset_video_path = cfg["PATHS"]["resize_dataset_video_path"]
    output_csv_path = cfg["PATHS"]["output_csv_path"]

    hidden_size = cfg["HYPERPARAMETERS"]["hidden_size"]
    lstm_layer = cfg["HYPERPARAMETERS"]["lstm_layer"]
    patch_size = cfg["HYPERPARAMETERS"]["patch_size"]
    num_classes = cfg["HYPERPARAMETERS"]["num_classes"]
    num_epochs = cfg["HYPERPARAMETERS"]["num_epochs"]
    batch_size = cfg["HYPERPARAMETERS"]["batch_size"]
    learning_rate = cfg["HYPERPARAMETERS"]["learning_rate"]
    number_block = cfg["HYPERPARAMETERS"]["number_block"]
    device = cfg["HYPERPARAMETERS"]["device"]
    momentum = cfg["HYPERPARAMETERS"]["momentum"]

# Set the paths

x_train, y_train = load_har_dataset(dataset_path, resize_dataset_video_path,
                                    output_csv_path, max_dim=35, train_test='train')

x_val, y_val = load_har_dataset(dataset_path, resize_dataset_video_path,
                                output_csv_path, max_dim=35, train_test='test')

# Convert the numpy arrays to PyTorch tensors
x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Training data type: {x_train.dtype}, Training labels type: {y_train.dtype}")
print('-' * 100)

x_val, y_val = torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()
print(f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}")
print(f"Validation data type: {x_val.dtype}, Validation labels type: {y_val.dtype}")
print('-' * 100)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val),
                                         batch_size=batch_size, shuffle=False)

# Initialize the model
bs, seq, col, row = x_train.size()
model = Model(input_size=col * row,
              num_classes=num_classes, patch_size=patch_size, lstm_layer=lstm_layer,
              hidden_size=hidden_size, number_block=number_block, device=device)

# Train the model
training_loss_logger, validation_loss_logger, training_accuracy_logger, validation_accuracy_logger \
    = model.fit(train_loader, val_loader, learning_rate=learning_rate, epochs=num_epochs, device=device)

# Save the model
model_path = "./../../../results/models/har_model.pth"
model.save_model(model_path)

# Visualize the training and validation loss
loss_plot_path = "./../../../results/models/loss_plot.png"
visualize_loss(training_loss_logger, validation_loss_logger, loss_plot_path)

# Visualize the training and validation accuracy
accuracy_plot_path = "./../../../results/models/accuracy_plot.png"
visualize_accuracy(training_accuracy_logger, validation_accuracy_logger, accuracy_plot_path)
