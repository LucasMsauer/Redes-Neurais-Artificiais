"""Entidade Neurônio."""
from .activations import get_activation


class Neuron:
    def __init__(self, weights, bias, activation):
        self.weights = list(weights)
        self.bias = float(bias)
        self.activation_name, self._f, self._df = get_activation(activation)
        # valores intermediários usados no treinamento
        self.inputs = []
        self.z = 0.0
        self.output = 0.0
        self.delta = 0.0  # dE/dz

    def forward(self, inputs):
        self.inputs = inputs
        z = self.bias
        for w, x in zip(self.weights, inputs):
            z += w * x
        self.z = z
        self.output = self._f(z)
        return self.output

    def derivative(self):
        """f'(z) usando os valores da última propagação direta."""
        return self._df(self.z, self.output)
