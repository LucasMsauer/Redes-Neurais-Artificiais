"""Previsão com uma rede já treinada."""


def predict(net, normalizer, values):
    if len(values) != net.n_inputs:
        raise ValueError(f"A rede espera {net.n_inputs} valores de entrada, mas foram informados {len(values)}.")
    x = normalizer.transform_row(values) if normalizer else list(values)
    return net.predict(x)
