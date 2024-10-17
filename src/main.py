from src.data_pipeline.loaders.loaders import load_har_dataset
from src.training.scripts.training import train_model_lstm
from src.models.lstm import LSTM
import torch
import torch.nn as nn
import numpy as np

dataset_path = "./../data/raw/HAR/"
resize_dataset_video_path = "./../data/processed/HAR/"
output_csv_path = "./../data/processed/HAR/UCF"


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
        break

x_val, y_val = load_har_dataset(dataset_path,
                                resize_dataset_video_path,
                                output_csv_path,
                                max_dim=35, train_test='test', update=True)

# # Convert the numpy arrays to PyTorch tensors
# x_train = torch.from_numpy(x_train).float()
# y_train = torch.from_numpy(y_train).float()
# print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
# print(f"Training data type: {x_train.dtype}, Training labels type: {y_train.dtype}")

#
# x_val = torch.from_numpy(x_val).float()
# y_val = torch.from_numpy(y_val).float()
# print(f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}")
# print(f"Validation data type: {x_val.dtype}, Validation labels type: {y_val.dtype}")
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
#
# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
#                                            batch_size=batch_size, shuffle=True)
#
# val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val),
#                                          batch_size=batch_size, shuffle=False)
#
#
# # Create the LSTM models
# lstm_model = LSTM(output_size=num_classes, patch_size=patch_size, lstm_layers=lstm_layer,
#                   hidden_size=hidden_size, number_block=number_block).to(device)
#
# train_model_lstm(lstm_model, train_loader, val_loader, lstm_layer=lstm_layer,
#                  hidden_size=hidden_size, learning_rate=learning_rate, epochs=num_epochs, device=device)
