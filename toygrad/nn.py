import random

from toygrad.engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_weights, non_lin=True):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(n_weights)]
        self.bias = Value(0)
        self.non_lin = non_lin

    def __call__(self, x):
        output = sum([x_i * w_i for x_i, w_i in zip(x, self.weights)], self.bias)
        if self.non_lin:
            output = output.tanh()
        return output

    def parameters(self):
        return self.weights + [self.bias]


class Layer(Module):
    def __init__(self, input_size, hidden_size, non_lin):
        self.neurons = [Neuron(input_size, non_lin) for _ in range(hidden_size)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    def __init__(self, input_size, layer_sizes):
        sizes = [input_size] + layer_sizes
        self.layers = []
        for i in range(len(layer_sizes)):
            non_lin = i != len(layer_sizes) - 1
            layer = Layer(sizes[i], sizes[i + 1], non_lin=non_lin)
            self.layers.append(layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
