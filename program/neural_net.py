import numpy as np
import json

def normalize(data):
    return np.array(data) / 255.0

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0, keepdims=True)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='sigmoid', learning_rate=0.1):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation_str = activation
        self.weights = [np.random.randn(n_out, n_in) * np.sqrt(2 / (n_in + n_out))
                        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((n_out, 1)) for n_out in layer_sizes[1:]]

    def activation(self, z, is_output=False):
        if is_output:
            return softmax(z)
        if self.activation_str == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_str == 'tanh':
            return np.tanh(z)
        elif self.activation_str == 'relu':
            return np.maximum(0, z)
        return z

    def activation_derivative(self, a, is_output=False):
        if is_output:
            return 1
        if self.activation_str == 'sigmoid':
            return a * (1 - a)
        elif self.activation_str == 'tanh':
            return 1 - a**2
        elif self.activation_str == 'relu':
            return (a > 0).astype(float)
        return 1

    def forward(self, x):
        a = x.reshape(-1, 1)
        activations = [a]
        zs = []
        for i, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            z = np.dot(w, a) + b
            zs.append(z)
            a = self.activation(z)
            activations.append(a)
        z = np.dot(self.weights[-1], a) + self.biases[-1]
        zs.append(z)
        a = self.activation(z, is_output=True)
        activations.append(a)
        return activations, zs

    def backward(self, x, y):
        x = x.reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        activations, zs = self.forward(x)
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        delta = activations[-1] - y
        grads_w[-1] = np.dot(delta, activations[-2].T)
        grads_b[-1] = delta
        for l in range(2, len(self.layer_sizes)):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activation_derivative(activations[-l])
            grads_w[-l] = np.dot(delta, activations[-l - 1].T)
            grads_b[-l] = delta
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def train(self, X, Y, epochs=10, batch_size=32):
        n_samples = len(X)
        indices = np.arange(n_samples)
        for epoch in range(epochs):
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                for x, y in zip(X_batch, Y_batch):
                    self.backward(x, y)
            print(f"Epoch {epoch+1}/{epochs} completed")

    def predict(self, x):
        activations, _ = self.forward(x)
        return activations[-1]

    def predict_class(self, x):
        probabilities = self.predict(x)
        return np.argmax(probabilities)

    def evaluate(self, X, Y):
        correct = 0
        total = len(X)
        for x, y in zip(X, Y):
            predicted_class = self.predict_class(x)
            actual_class = np.argmax(y)
            if predicted_class == actual_class:
                correct += 1
        return correct / total

    def save(self, filename):
        model = {
            'layer_sizes': self.layer_sizes,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'activation': self.activation_str,
            'learning_rate': self.learning_rate
        }
        with open(filename, 'w') as f:
            json.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            model = json.load(f)
        nn = NeuralNetwork(model['layer_sizes'], model['activation'], model['learning_rate'])
        nn.weights = [np.array(w) for w in model['weights']]
        nn.biases = [np.array(b) for b in model['biases']]
        return nn
