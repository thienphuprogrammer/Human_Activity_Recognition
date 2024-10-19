import torch
import yaml
from sklearn.model_selection import train_test_split

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
    max_dim = cfg["HYPERPARAMETERS"]["max_dim"]

# Set the paths
x_train, y_train = load_har_dataset(
    dataset_path,
    max_dim=max_dim,
    resize_dataset_video_path=resize_dataset_video_path,
    dataset_csv_path=output_csv_path,
    train_test="train",
)

# Convert the numpy arrays to PyTorch tensors
x_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Training data type: {x_train.dtype}, Training labels type: {y_train.dtype}")
print("-" * 100)

X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# x_val, y_val = torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()
# print(f"Validation data shape: {x_val.shape}, Validation labels shape: {y_val.shape}")
# print(f"Validation data type: {x_val.dtype}, Validation labels type: {y_val.dtype}")
# print("-" * 100)

model = Model(
    input_size=X_train.shape[2],
    num_classes=num_classes,
    patch_size=patch_size,
    lstm_layer=lstm_layer,
    hidden_size=hidden_size,
    number_block=number_block,
    device=device,
)

# Train the model
(
    training_loss_logger,
    validation_loss_logger,
    training_accuracy_logger,
    validation_accuracy_logger,
) = model.fit(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    learning_rate=learning_rate,
    epochs=num_epochs,
    device=device,
)

# Save the model
model_path = "./results/models/har_model.pth"
model.save_model(model_path)

# Visualize the training and validation loss
loss_plot_path = "./results/models/loss_plot.png"
visualize_loss(training_loss_logger, validation_loss_logger, loss_plot_path)

# Visualize the training and validation accuracy
accuracy_plot_path = "./results/models/accuracy_plot.png"
visualize_accuracy(
    training_accuracy_logger, validation_accuracy_logger, accuracy_plot_path
)
