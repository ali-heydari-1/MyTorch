from .layer import Layer

class MLP:
    def __init__(self, input_size, layer_sizes, activation_function=Tensor.sigmoid):
        layers_total = [input_size] + layer_sizes
        self.layers = [Layer(layers_total[i], layers_total[i+1], activation_function) for i in range(len(layer_sizes))]

    def set_labels(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].neurons)):
                for k in range(len(self.layers[i].neurons[j].weights)):
                    self.layers[i].neurons[j].weights[k].label = f"w[{i + 1}][{j + 1}][{k + 1}]"
                self.layers[i].neurons[j].bias.label = f"b[{i + 1}][{j + 1}]"

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        res = []
        for layer in self.layers:
            res.extend(layer.parameters())
        return res
