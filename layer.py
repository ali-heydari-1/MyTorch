from .neuron import Neuron

class Layer:
    def __init__(self, input_size, output_size, activation_function=Tensor.sigmoid):
        self.neurons = [Neuron(input_size, activation_function) for _ in range(output_size)]

    def forward(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
