
import numpy as np

class MSELoss:
    def forward(self, y, target):
        self.diff = diff = y - target
        return np.sum(np.square(diff)) / diff.size

    def backward(self):
        assert hasattr(self, 'diff'), "The values passed to forward were not saved"
        diff = self.diff
        return (2 * diff) / diff.size

class Linear:
    def __init__(self, in_features, out_features):
        # Initialization: TODO: Research the best way.
        bound = 3 / np.sqrt(in_features)
        self.W = np.random.uniform(-bound, bound, (out_features, in_features))
        bound = 1 / np.sqrt(in_features)
        self.b = np.random.uniform(-bound, bound, out_features)

        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.W.transpose()) + self.b

    def backward(self, dy):
        assert hasattr(self, 'x'), "The values passed to forward were not saved"
        assert dy.ndim == 2 and dy.shape[1] == self.W.shape[0], f'{dy.ndim} != 2 or {dy.shape[1]} != {self.W.shape[0]}'
        self.grad_W = np.matmul(dy.transpose(), self.x)
        self.grad_b = np.sum(dy, axis=0)
        dx = np.matmul(dy, self.W)
        return dx

class Tanh:
    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, dy):
        assert hasattr(self, 'x'), "The values passed to forward were not saved"
        return dy * (1.0 - np.square(np.tanh(self.x)))

class MLP:
    "NumPy MLP"
    def __init__(self):
        self.layer_input = Linear(in_features=1, out_features=10)
        self.tanh_1 = Tanh()
        self.layer_hidden_1 = Linear(in_features=10, out_features=11)
        self.tanh_2 = Tanh()
        self.layer_output = Linear(in_features=11, out_features=1)

    def forward(self, x):
        x = self.layer_input.forward(x)
        x = self.tanh_1.forward(x)
        x = self.layer_hidden_1.forward(x)
        x = self.tanh_2.forward(x)
        x = self.layer_output.forward(x)
        return x

    def backward(self, dy):
        dx = self.layer_output.backward(dy)
        dx = self.tanh_2.backward(dx)
        dx = self.layer_hidden_1.backward(dx)
        dx = self.tanh_1.backward(dx)
        dx = self.layer_input.backward(dx)
        return dx

class Adam:
    "TODO: Proper implementation"
    def __init__(self, lr):
        self.lr = lr

    def step(self, model):
        for module in model.__dict__.values():
            if isinstance(module, Linear):
                module.W = module.W - module.grad_W * self.lr
                module.b = module.b - module.grad_b * self.lr


def train(X, Y, epochs, learning_rate):
    model = MLP()
    optimizer = Adam(lr=learning_rate)
    criterion = MSELoss()
    for i in range(epochs):
        # Forward computations
        Y_pred = model.forward(X)
        loss = criterion.forward(Y_pred, Y)

        # Backward computations
        dy = criterion.backward()
        dx = model.backward(dy)

        # Gradient decsent
        optimizer.step(model)

    return model

def predict(model, X):
    return model.forward(X)

