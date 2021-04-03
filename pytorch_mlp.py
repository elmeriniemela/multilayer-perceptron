import torch

class MLP(torch.nn.Module):
    "PyTorch MLP"
    def __init__(self):
        super().__init__()
        self.group = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=10),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=10, out_features=11),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=11, out_features=1),
        )

    def forward(self, x):
        return self.group(x)


def train(X, Y, epochs, learning_rate):
    X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    for i in range(epochs):
        optimizer.zero_grad() # Set all gradient values to zeros.
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
    return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        Y_pred = model(torch.FloatTensor(X))
    return Y_pred.numpy()
