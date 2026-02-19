import numpy as np

class Neuron:
    def __init__(self, input_dim=2, learning_rate=0.1):
        self.w = np.random.randn(input_dim)
        self.b = np.random.randn()
        self.eta = learning_rate

    def net_input(self, x):
        # s = w^T x + b
        return np.dot(self.w, x) + self.b

    def predict(self, x, activation_fn):
        s = self.net_input(x)
        return activation_fn(s)

    def train(self, X, d, activation_fn, activation_derivative, epochs=100):
        """
        X : array of shape (n_samples, 2)
        d : true labels (0 or 1)
        """
        for _ in range(epochs):
            for x_j, d_j in zip(X, d):
                s = self.net_input(x_j)
                y = activation_fn(s)

                # Formula:
                # Δw = η (d − y) f'(s) x
                error = d_j - y
                delta_w = self.eta * error * activation_derivative(s) * x_j
                delta_b = self.eta * error * activation_derivative(s)

                self.w += delta_w
                self.b += delta_b
