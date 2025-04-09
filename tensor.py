import math


class Tensor:
    __counter = 0

    def __init__(self, value=0, label='', children=(), operator=None):
        self.value = value
        self.children = children
        self.operator = operator
        self.grad = 0
        self._backward = lambda: None
        self.label = str(Tensor.__counter)
        self.id = Tensor.__counter
        Tensor.__counter += 1
        self.number_of_repetition = 0

    def __repr__(self):
        return f'Tensor(id:{self.id}, data:{self.value}, label:{self.label}, grad:{self.grad}, children:{list(map(lambda a: a.id, self.children))}, operator:{self.operator}, number_of_repetion:{self.number_of_repetition})'

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value * other.value, children=(self, other), operator='*')

        def backward_mul():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
            self.number_of_repetition += 1
            other.number_of_repetition += 1

        out._backward = backward_mul
        return out

    # Other methods like __add__, __sub__, __truediv__, etc.

    @staticmethod
    def sigmoid(input_value):
        output_value = 1 / (1 + math.e ** (-input_value.value))
        out = Tensor(output_value, children=(input_value,), operator="sigmoid")

        def backward_sigmoid():
            input_value.grad += output_value * (1 - output_value) * out.grad

        out._backward = backward_sigmoid
        return out
