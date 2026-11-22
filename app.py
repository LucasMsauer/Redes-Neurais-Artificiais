"""Interface Streamlit do sistema de MLP.  Execute:  streamlit run app.py"""
import csv
import json
import os
import tempfile

import pandas as pd
import streamlit as st

from rna import dataset as ds
from rna.activations import ACTIVATIONS
from rna.network import NeuralNetwork
from rna.persistence import load_model, save_model
from rna.predictor import predict
from rna.trainer import train

st.set_page_config(page_title="MLP configurável", page_icon="🧠", layout="wide")
st.title("🧠 Sistema Configurável de Redes Neurais (MLP)")

ss = st.session_state
for k in ("net", "norm", "meta", "history", "test_mse", "test_df"):
    ss.setdefault(k, None)


def to_tempfile(uploaded, suffix):
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(uploaded.getvalue())
    f.close()
    return f.name


tab_train, tab_pred, tab_load = st.tabs(["⚙️ Configurar e treinar", "🔮 Prever", "💾 Carregar rede"])

# ----------------------------------------------------------------- TREINAR
with tab_train:
    up = st.file_uploader("Dataset CSV", type="csv")
    path = to_tempfile(up, ".csv") if up else ("dataset.csv" if os.path.exists("dataset.csv") else None)
    if path is None:
        st.info("Envie um CSV para começar.")
        st.stop()

    with open(path, newline="", encoding="utf-8-sig") as f:
        header = [h.strip() for h in next(csv.reader(f))]

    c1, c2 = st.columns(2)
    out_cols = c2.multiselect("Colunas de saída", header, default=header[-1:])
    in_cols = c1.multiselect("Colunas de entrada", [h for h in header if h not in out_cols],
                             default=[h for h in header if h not in out_cols])

    st.subheader("Arquitetura")
    n_hidden = st.number_input("Camadas ocultas", 0, 6, 1)
    acts = list(ACTIVATIONS)
    specs = []
    cols = st.columns(max(1, int(n_hidden)) + 1)
    for i in range(int(n_hidden)):
        n = cols[i].number_input(f"Neurônios (oculta {i + 1})", 1, 256, 10, key=f"n{i}")
        a = cols[i].selectbox(f"Ativação (oculta {i + 1})", acts, index=1, key=f"a{i}")
        specs.append((int(n), a))
    a_out = cols[-1].selectbox("Ativação da saída", acts, index=0, key="aout")
    cols[-1].caption(f"Neurônios de saída: {len(out_cols)}")
    specs.append((max(1, len(out_cols)), a_out))

    st.subheader("Treinamento")
    c1, c2, c3, c4 = st.columns(4)
    lr = c1.number_input("Taxa de aprendizado", 0.00001, 1.0, 0.0005, format="%.5f")
    epochs = c2.number_input("Épocas", 1, 100000, 1000)
    seed = c3.number_input("Seed", 0, 10**9, 1)
    norm_method = c4.selectbox("Normalização das entradas", ["none", "minmax", "zscore"], index=2)
    c1, c2, c3 = st.columns(3)
    p_train = c1.number_input("% treino", 0, 100, 70)
    p_val = c2.number_input("% validação", 0, 100, 15)
    p_test = c3.number_input("% teste", 0, 100, 15)
    c1, c2 = st.columns(2)
    decay = c1.number_input("Decaimento da taxa (opcional)", 0.0, 10.0, 0.0, format="%.4f")
    bias_t = c2.checkbox("Viés de saída inicial = média dos alvos de treino", value=True)

    if st.button("▶️ Treinar", type="primary"):
        try:
            if not in_cols or not out_cols:
                raise ValueError("Escolha ao menos uma coluna de entrada e uma de saída.")
            X, Y, in_names, out_names = ds.load_csv(path, in_cols, out_cols)
            net = NeuralNetwork.build(len(in_names), specs, seed=int(seed))
            ds.check_compatibility(net, X, Y)
            ranges = [[min(c), max(c)] for c in zip(*X)]  # faixa real de cada entrada (para a aba Prever)
            tr, va, te = ds.split(X, Y, p_train, p_val, p_test, seed=int(seed))
            normalizer = None
            if norm_method != "none":
                normalizer = ds.Normalizer.fit(norm_method, tr[0])  # só com o treino
                tr, va, te = [(normalizer.transform(s[0]), s[1]) for s in (tr, va, te)]
            if bias_t:
                net.set_output_bias([sum(c) / len(c) for c in zip(*tr[1])])

            st.write(f"Amostras: treino {len(tr[0])} · validação {len(va[0])} · teste {len(te[0])}")
            bar, chart, status = st.progress(0.0), st.empty(), st.empty()
            hist = []

            def on_epoch(rec):
                hist.append(rec)
                if rec["epoch"] % max(1, int(epochs) // 100) == 0 or rec["epoch"] == epochs:
                    bar.progress(rec["epoch"] / int(epochs))
                    status.write(f"Época {rec['epoch']} · MSE treino {rec['train_mse']:.2f} · "
                                 f"MSE validação {rec['val_mse']:.2f}")
                    chart.line_chart(pd.DataFrame(hist).set_index("epoch")[["train_mse", "val_mse"]])

            train(net, tr, va, int(epochs), float(lr), seed=int(seed), on_epoch=on_epoch, lr_decay=float(decay))
            ss.net, ss.norm, ss.history = net, normalizer, hist
            ss.test_mse = net.mse(*te)
            preds = [net.forward(x) for x in te[0]]  # só propagação direta
            ss.test_df = pd.DataFrame({"real": [y[0] for y in te[1]], "previsto": [p[0] for p in preds]})
            ss.meta = {"input_names": in_names, "output_names": out_names, "input_ranges": ranges, "lr": lr,
                       "lr_decay": decay, "epochs": int(epochs), "seed": int(seed)}
            st.success("Treinamento concluído.")
        except ValueError as e:
            st.error(f"Erro: {e}")

    if ss.net is not None and ss.test_mse is not None:
        st.metric("MSE de teste (avaliação final)", f"{ss.test_mse:.2f}")
        if ss.test_df is not None:
            st.subheader("Valor previsto × valor real (conjunto de teste)")
            st.caption("Se a rede acertasse tudo, os pontos formariam uma reta ascendente (previsto = real). Com dados ruidosos, alguma dispersão é esperada.")
            st.scatter_chart(ss.test_df, x="real", y="previsto")
        tmp = os.path.join(tempfile.gettempdir(), "modelo_streamlit.json")
        save_model(tmp, ss.net, ss.norm, ss.meta)
        with open(tmp, "rb") as f:
            st.download_button("💾 Baixar rede (JSON)", f.read(), file_name="modelo.json", mime="application/json")

# ------------------------------------------------------------------ PREVER
with tab_pred:
    if ss.net is None:
        st.info("Treine ou carregue uma rede primeiro.")
    else:
        names = (ss.meta or {}).get("input_names") or [f"entrada_{i + 1}" for i in range(ss.net.n_inputs)]
        st.write("Informe os valores de entrada (sem normalizar; a normalização é aplicada automaticamente):")
        ranges = (ss.meta or {}).get("input_ranges")
        vals = []
        for i, n in enumerate(names):
            if ranges:
                lo, hi = ranges[i]
                v = st.number_input(f"{n}  (faixa do dataset: {lo:g} a {hi:g})", value=float((lo + hi) / 2), key=f"p_{n}")
                if not lo <= v <= hi:
                    st.warning(f"{n} está fora da faixa vista no treino; a previsão pode ser pouco confiável.")
            else:
                v = st.number_input(n, value=0.0, key=f"p_{n}")
            vals.append(v)
        if st.button("Calcular previsão"):
            try:
                out = predict(ss.net, ss.norm, vals)
                onames = (ss.meta or {}).get("output_names") or [f"saida_{i + 1}" for i in range(len(out))]
                for n, v in zip(onames, out):
                    st.metric(n, f"{v:.2f}")
            except ValueError as e:
                st.error(f"Erro: {e}")

# ------------------------------------------------------------------ CARREGAR
with tab_load:
    up_m = st.file_uploader("Arquivo JSON da rede", type="json", key="model_up")
    if up_m and st.button("Carregar rede"):
        try:
            net, norm, meta = load_model(to_tempfile(up_m, ".json"))
            ss.net, ss.norm, ss.meta, ss.history, ss.test_mse = net, norm, meta, None, None
            st.success("Rede carregada.")
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            st.error(f"Arquivo inválido: {e}")
    if ss.net is not None:
        st.write(f"**Entradas:** {ss.net.n_inputs} · **Normalização:** {ss.norm.method if ss.norm else 'nenhuma'}")
        for i, l in enumerate(ss.net.layers, 1):
            st.write(f"Camada {i}: {len(l.neurons)} neurônios · {l.activation_name}")
        m = ss.meta or {}
        if m.get("epochs"):
            st.caption(f"Treinada por {m['epochs']} épocas, taxa {m.get('lr')}, seed {m.get('seed')}.")
