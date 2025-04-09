from mlp_model.mlp import MLP
from mlp_model.optimizer import Optimizer
from mlp_model.tensor import Tensor
from utils.graph import draw_dot

data = [[2, 1], [-3, 0]]
target = [1, 0]
n_epochs = 2

model = MLP(input_size=len(data[0]), layer_sizes=[2, 1], activation_function=Tensor.sigmoid)
optim = Optimizer(model.parameters(), lr=0.01)

loss = 0
for epoch in range(n_epochs):
    y_hats = [model(x) for x in data]
    loss = sum((y_hat.value - t) ** 2 for y_hat, t in zip(y_hats, target)) / len(y_hats)
    optim.zero_grad()
    loss.backward()
    optim.step()

y_hats = [model(x) for x in data]
for i, y_hat in enumerate(y_hats):
    print(f"Prediction: {y_hat.value}, Target: {target[i]}")

draw_dot(loss)
