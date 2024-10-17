import configparser

import numpy as np

from src.data_pipeline.loaders.loaders import load_har_dataset
from src.models.model import *
from src.training.scripts.utils import visualize_loss, visualize_accuracy

# Read hyperparameters from the config file .yml
config = configparser.ConfigParser()
config.read('./config/config.yml')

# Set the paths
dataset_path = config['PATHS']['dataset_path']
resize_dataset_video_path = config['PATHS']['resize_dataset_video_path']
output_csv_path = config['PATHS']['output_csv_path']

hidden_size = int(config['HYPERPARAMETERS']['hidden_size'])
lstm_layer = int(config['HYPERPARAMETERS']['lstm_layer'])
patch_size = int(config['HYPERPARAMETERS']['patch_size'])
num_classes = int(config['HYPERPARAMETERS']['num_classes'])
num_epochs = int(config['HYPERPARAMETERS']['num_epochs'])
batch_size = int(config['HYPERPARAMETERS']['batch_size'])
learning_rate = float(config['HYPERPARAMETERS']['learning_rate'])
number_block = int(config['HYPERPARAMETERS']['number_block'])
device = config['HYPERPARAMETERS']['device']

# dataset_path = "./../data/raw/HAR/"
# resize_dataset_video_path = "./../data/processed/HAR/"
# output_csv_path = "./../data/processed/HAR/UCF"
#
# # Set hyperparameters
# hidden_size = 128
# lstm_layer = 35
# patch_size = 225
# num_classes = 6
# num_epochs = 100
# batch_size = 64
# learning_rate = 0.001
# number_block = 1
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


x_train, y_train = load_har_dataset(dataset_path,
                                    resize_dataset_video_path,
                                    output_csv_path,
                                    max_dim=35, train_test='train')

# check nan values
print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
for i in range(x_train.shape[0]):
    if np.isnan(x_train[i]).any():
        print(f'X_train nan values in {i}th row')
        print(x_train[i])
        exit(0)

x_val, y_val = load_har_dataset(dataset_path,
                                resize_dataset_video_path,
                                output_csv_path,
                                max_dim=35, train_test='test')

# Convert the numpy arrays to PyTorch tensors
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Training data type: {x_train.dtype}, Training labels type: {y_train.dtype}")

x_val = torch.from_numpy(x_val).float()
y_val = torch.from_numpy(y_val).float()
print(f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}")
print(f"Validation data type: {x_val.dtype}, Validation labels type: {y_val.dtype}")

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)

val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val),
                                         batch_size=batch_size, shuffle=False)

model = Model(num_classes=num_classes, patch_size=patch_size, lstm_layer=lstm_layer,
              hidden_size=hidden_size, number_block=number_block, device=device)

training_loss_logger, validation_loss_logger, training_accuracy_logger, validation_accuracy_logger \
    = model.fit(train_loader, val_loader, learning_rate=learning_rate, epochs=num_epochs, device=device)

# Save the model
model_path = "./../models/har_model.pth"
model.save_model(model_path)

# Visualize the training and validation loss
loss_plot_path = "./../reports/figures/loss_plot.png"
visualize_loss(training_loss_logger, validation_loss_logger, loss_plot_path)

# Visualize the training and validation accuracy
accuracy_plot_path = "./../reports/figures/accuracy_plot.png"
visualize_accuracy(training_accuracy_logger, validation_accuracy_logger, accuracy_plot_path)
