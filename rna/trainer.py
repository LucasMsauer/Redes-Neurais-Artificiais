"""Treinamento com SGD (atualização a cada exemplo)."""
import random


def train(net, train_set, val_set, epochs, lr, seed=None, on_epoch=None, lr_decay=0.0):
    """Treina usando APENAS o conjunto de treino. Retorna o histórico por época.

    A validação só é usada para calcular o MSE (nunca atualiza pesos).
    lr_decay (opcional): taxa da época e = lr / (1 + lr_decay * (e - 1)). Com 0, a taxa é constante.
    """
    X, Y = train_set
    Xv, Yv = val_set
    if not X:
        raise ValueError("O conjunto de treinamento está vazio.")
    rng = random.Random(seed)
    order = list(range(len(X)))
    history = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(order)  # embaralha a ordem dos exemplos a cada época
        for i in order:
            net.backward(X[i], Y[i], lr / (1.0 + lr_decay * (epoch - 1)))  # SGD: atualiza após cada exemplo
        rec = {"epoch": epoch, "train_mse": net.mse(X, Y), "val_mse": net.mse(Xv, Yv)}
        history.append(rec)
        if on_epoch:
            on_epoch(rec)
    return history
