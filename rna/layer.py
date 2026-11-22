"""Entidade Camada, com inicialização Xavier/Glorot ou He."""
import math

from .neuron import Neuron


def init_weights(rng, fan_in, fan_out, activation):
    """He (normal) para ReLU; Xavier/Glorot (uniforme) para os demais."""
    if activation == "relu":
        std = math.sqrt(2.0 / fan_in)
        return [rng.gauss(0.0, std) for _ in range(fan_in)]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return [rng.uniform(-limit, limit) for _ in range(fan_in)]


class Layer:
    def __init__(self, neurons):
        self.neurons = neurons

    @classmethod
    def create(cls, n_neurons, n_inputs, activation, rng, fan_out=None):
        fan_out = fan_out or n_neurons
        neurons = []
        for _ in range(n_neurons):
            w = init_weights(rng, n_inputs, fan_out, activation.lower())
            neurons.append(Neuron(w, 0.0, activation))  # viés inicial = 0
        return cls(neurons)

    @property
    def activation_name(self):
        return self.neurons[0].activation_name

    @property
    def n_inputs(self):
        return len(self.neurons[0].weights)

    def forward(self, inputs):
        return [n.forward(inputs) for n in self.neurons]
