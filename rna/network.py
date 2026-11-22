"""Entidade Rede Neural (MLP) com propagação direta, erro e retropropagação."""
import random

from .layer import Layer


class NeuralNetwork:
    def __init__(self, n_inputs, layers):
        self.n_inputs = n_inputs
        self.layers = layers

    @classmethod
    def build(cls, n_inputs, specs, seed=None):
        """specs: lista de (n_neurônios, nome_da_ativação), da 1ª oculta até a saída."""
        if n_inputs < 1 or not specs:
            raise ValueError("A rede precisa de ao menos 1 entrada e 1 camada.")
        rng = random.Random(seed)
        layers, prev = [], n_inputs
        for i, (n, act) in enumerate(specs):
            if n < 1:
                raise ValueError("Cada camada precisa de ao menos 1 neurônio.")
            fan_out = specs[i + 1][0] if i + 1 < len(specs) else n
            layers.append(Layer.create(n, prev, act, rng, fan_out))
            prev = n
        return cls(n_inputs, layers)

    def set_output_bias(self, values):
        """Define o viés inicial dos neurônios de saída (opcional, antes do treino)."""
        for n, v in zip(self.layers[-1].neurons, values):
            n.bias = float(v)

    @property
    def n_outputs(self):
        return len(self.layers[-1].neurons)

    def forward(self, x):
        if len(x) != self.n_inputs:
            raise ValueError(f"Esperadas {self.n_inputs} entradas, recebidas {len(x)}.")
        out = list(x)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, x):
        """Apenas propagação direta: não altera pesos nem vieses."""
        return self.forward(x)

    # ---------------------------------------------------------------- erro
    @staticmethod
    def example_error(output, target):
        """Erro quadrático médio de um exemplo (média sobre as saídas)."""
        m = len(target)
        return sum((o - t) ** 2 for o, t in zip(output, target)) / m

    def mse(self, X, Y):
        """MSE de um conjunto (média dos erros por exemplo)."""
        if not X:
            return float("nan")
        total = sum(self.example_error(self.forward(x), y) for x, y in zip(X, Y))
        return total / len(X)

    # ------------------------------------------------------ retropropagação
    def backward(self, x, target, lr):
        """Forward + cálculo de TODOS os deltas + atualização (SGD, 1 exemplo)."""
        output = self.forward(x)
        m = len(target)

        # 1) deltas da saída: dE/dz = dE/da * f'(z), com E = (1/m) * soma (a - t)^2
        last = self.layers[-1]
        for n, o, t in zip(last.neurons, output, target):
            n.delta = (2.0 / m) * (o - t) * n.derivative()

        # 2) deltas das ocultas (pesos ainda NÃO atualizados)
        for l in range(len(self.layers) - 2, -1, -1):
            nxt = self.layers[l + 1].neurons
            for j, n in enumerate(self.layers[l].neurons):
                soma = sum(k.weights[j] * k.delta for k in nxt)
                n.delta = soma * n.derivative()

        # 3) atualização dos pesos e vieses (descida do gradiente)
        for layer in self.layers:
            for n in layer.neurons:
                for i, xi in enumerate(n.inputs):
                    n.weights[i] -= lr * n.delta * xi
                n.bias -= lr * n.delta
        return self.example_error(output, target)

    # --------------------------------------------------------- serialização
    def to_dict(self):
        return {
            "n_inputs": self.n_inputs,
            "layers": [
                {
                    "activation": layer.activation_name,
                    "neurons": [{"weights": n.weights, "bias": n.bias} for n in layer.neurons],
                }
                for layer in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, d):
        from .neuron import Neuron

        layers = []
        for ld in d["layers"]:
            neurons = [Neuron(nd["weights"], nd["bias"], ld["activation"]) for nd in ld["neurons"]]
            layers.append(Layer(neurons))
        return cls(d["n_inputs"], layers)
