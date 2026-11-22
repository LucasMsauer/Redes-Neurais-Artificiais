import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rna import dataset as ds
from rna.network import NeuralNetwork
from rna.persistence import load_model, save_model
from rna.predictor import predict


class TestRNA(unittest.TestCase):
    def test_gradiente_numerico(self):
        """Backprop deve bater com a derivada numérica (diferenças finitas)."""
        net = NeuralNetwork.build(2, [(3, "tanh"), (3, "sigmoid"), (2, "relu"), (1, "linear")], seed=1)
        x, t = [0.3, -0.7], [0.5]
        net.forward(x)
        import copy
        ref = copy.deepcopy(net)
        net.backward(x, t, lr=1.0)  # com lr=1, w_novo = w - dE/dw
        eps = 1e-6
        for l, layer in enumerate(ref.layers):
            for j, n in enumerate(layer.neurons):
                for i in range(len(n.weights)):
                    n.weights[i] += eps; e1 = ref.example_error(ref.forward(x), t)
                    n.weights[i] -= 2 * eps; e2 = ref.example_error(ref.forward(x), t)
                    n.weights[i] += eps
                    grad = (e1 - e2) / (2 * eps)
                    analitico = n.weights[i] - net.layers[l].neurons[j].weights[i]
                    self.assertAlmostEqual(grad, analitico, places=5)

    def test_salvar_carregar(self):
        net = NeuralNetwork.build(2, [(4, "tanh"), (1, "linear")], seed=3)
        norm = ds.Normalizer.fit("zscore", [[1, 2], [3, 4], [5, 9]])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            save_model(p, net, norm)
            net2, norm2, _ = load_model(p)
        for v in ([1.5, 3.0], [10, -2]):
            self.assertEqual(predict(net, norm, v), predict(net2, norm2, v))

    def test_seed_reprodutivel(self):
        a = NeuralNetwork.build(2, [(3, "relu"), (1, "linear")], seed=9).to_dict()
        b = NeuralNetwork.build(2, [(3, "relu"), (1, "linear")], seed=9).to_dict()
        self.assertEqual(a, b)

    def test_divisao(self):
        X = [[i] for i in range(100)]; Y = [[i] for i in range(100)]
        a = ds.split(X, Y, 70, 15, 15, seed=1); b = ds.split(X, Y, 70, 15, 15, seed=1)
        self.assertEqual(a, b)
        self.assertEqual([len(p[0]) for p in a], [70, 15, 15])
        with self.assertRaises(ValueError):
            ds.split(X, Y, 50, 20, 20)

    def test_normalizacao_so_treino(self):
        n = ds.Normalizer.fit("minmax", [[0, 10], [10, 20]])
        self.assertEqual(n.transform_row([5, 15]), [0.5, 0.5])
        self.assertEqual(n.transform_row([20, 10]), [2.0, 0.0])  # fora da faixa do treino: sem clipping

    def test_validacao_entradas(self):
        net = NeuralNetwork.build(2, [(1, "linear")], seed=1)
        with self.assertRaises(ValueError):
            predict(net, None, [1.0])

    def test_predict_nao_altera_pesos(self):
        net = NeuralNetwork.build(2, [(3, "tanh"), (1, "linear")], seed=1)
        antes = net.to_dict(); predict(net, None, [1.0, 2.0])
        self.assertEqual(antes, net.to_dict())


if __name__ == "__main__":
    unittest.main()
