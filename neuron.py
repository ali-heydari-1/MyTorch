import random
from .tensor import Tensor

class Neuron:
    def __init__(self, input_size, activation_function=Tensor.sigmoid):
        self.weights = [Tensor(random.uniform(-8, 3)) for _ in range(input_size)]
        self.bias = Tensor(random.uniform(-5, 2))
        self.activation_function = activation_function

    def forward(self, x):
        res = sum([w_i * x_i for w_i, x_i in zip(self.weights, x)]) + self.bias
        return self.activation_function(res)

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self.weights + [self.bias]
