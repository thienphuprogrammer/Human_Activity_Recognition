from src.models.model import Model
from torchinfo import summary

hidden_size = [1024, 512, 256, 128]
lstm_layer = 1
patch_size = 99
num_classes = 6
num_epochs = 100
batch_size = 256
learning_rate = 0.001
momentum = 0.9
number_block = 1
max_dim = 50
device = 'cpu'

model = Model(
    batch_size=batch_size,
    input_size=99,
    num_classes=num_classes,
    patch_size=patch_size,
    lstm_layer=lstm_layer,
    hidden_size=hidden_size,
    number_block=number_block,
    device=device,
)
summary(model.lstm, input_size=(batch_size, 35, 99))
