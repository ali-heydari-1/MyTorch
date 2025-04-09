class Optimizer:
    def __init__(self, parameters, lr):
        self.lr = lr
        self.parameters = parameters

    def zero_grad(self):
        for item in self.parameters:
            item.reset_grad()

    def step(self):
        for item in self.parameters:
            item.update_value(self.lr)
