
import unittest
import pytorch_mlp
import numpy_mlp
import numpy as np

def noisy_sin():
    X = np.random.randn(100, 1)
    X = np.sort(X, axis=0)
    Y = np.sin(X * 2 * np.pi / 3)
    Y = Y + 0.2 * np.random.randn(*Y.shape)
    return X, Y

def mse(Y, Y_pred):
    return ((Y - Y_pred)**2).mean()

def r_sqared(Y, Y_pred):
    return 1 - ((Y - Y_pred)**2).sum() / ((Y - np.average(Y))**2).sum()


EPOCHS = 200
LEARNING_RATE = 0.01

class TestMLP(unittest.TestCase):

    def test_pytorch_mlp_1(self):
        X, Y = noisy_sin()
        model = pytorch_mlp.train(X, Y, EPOCHS, LEARNING_RATE)
        Y_pred = pytorch_mlp.predict(model, X)
        self.assertEqual(
            Y.shape,
            Y_pred.shape,
        )
        self.assertLess(
            mse(Y, Y_pred),
            0.11,
        )
        self.assertGreater(
            r_sqared(Y, Y_pred),
            0.82,
        )

    def test_numpy_mlp_1(self):
        X, Y = noisy_sin()
        model = numpy_mlp.train(X, Y, EPOCHS, LEARNING_RATE)
        Y_pred = numpy_mlp.predict(model, X)
        self.assertEqual(
            Y.shape,
            Y_pred.shape,
        )
        self.assertLess(
            mse(Y, Y_pred),
            0.11,
        )
        self.assertGreater(
            r_sqared(Y, Y_pred),
            0.82,
        )

def plot_fit(compare, X, Y):
    import matplotlib.pyplot as plt
    assert len(compare) == 2, "Comparison requires 2 implementations"
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
    fig.canvas.set_window_title('Comparison of MLP implementations')
    for ax, module in zip(axes, compare):
        model = module.train(X, Y, EPOCHS, LEARNING_RATE)
        y_pred = module.predict(model, X)
        ax.plot(X, Y, '.')
        ax.plot(X, y_pred, 'r-')
        ax.set_title(model.__doc__)
        ax.grid(True)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

if __name__ == "__main__":
    X, Y = noisy_sin()
    plot_fit([pytorch_mlp, numpy_mlp], X, Y)