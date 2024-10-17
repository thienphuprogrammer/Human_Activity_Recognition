import torch
from src.models.lstm import LSTM
from tqdm.notebook import trange, tqdm


def test_model(lstm_model: LSTM, test_loader, lstm_layers, hidden_size, device="cuda"):
    lstm_model.eval()
    test_acc = 0
    with torch.no_grad():
        for data, label in tqdm(test_loader, desc="Testing", leave=False):
            data = data.to(device)
            label = label.to(device)

            hidden = torch.zeros(lstm_layers, data.shape[0], hidden_size, device=device)
            memory = torch.zeros(lstm_layers, data.shape[0], hidden_size, device=device)

            data_pred, hidden, memory = lstm_model(data, hidden, memory)
            last_target = data_pred[:, -1, :]

            test_acc += (last_target.argmax(1) == label).sum()
    test_acc = (test_acc / len(test_loader)).item()

    print("Test Accuracy %.2f%%" % (test_acc * 100))