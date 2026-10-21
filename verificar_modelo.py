#!/usr/bin/env python3
"""Verifica que el modelo guardado de ARGus carga y predice en esta maquina.

Uso:
    python verificar_modelo.py --modelo data/models/argus_rf_k6_id95.joblib \
        --fasta data/interim/muestra_genes_completos.fasta \
        --etiquetas data/interim/muestra_genes_labels.tsv --n 40

Con --etiquetas compara la prediccion con la etiqueta real (primeros N genes, N/2 de cada clase si es posible).
"""
import argparse, sys
from pathlib import Path
import numpy as np
import joblib


def leer_fasta(path):
    d, cur = {}, None
    for l in open(path):
        l = l.strip()
        if l.startswith(">"):
            cur = l[1:].split()[0]; d[cur] = []
        elif cur:
            d[cur].append(l)
    return {k: "".join(v) for k, v in d.items()}


def fragmentar(seq, ventana=500):
    seq = seq.upper(); L = len(seq)
    if L < ventana:
        return [seq + "0" * (ventana - L)]
    n = L // ventana
    fr = [seq[i * ventana:(i + 1) * ventana] for i in range(n)]
    r = L - n * ventana
    if r > 0:
        fr.append(seq[n * ventana:] + "0" * (ventana - r))
    return fr


def vector_kmer(seq, K):
    lut = np.full(256, 4, dtype=np.int8)
    for i, b in enumerate(b"ACGT"):
        lut[b] = i
    pot = (4 ** np.arange(K - 1, -1, -1)).astype(np.int64)
    a = lut[np.frombuffer(seq.upper().encode(), dtype=np.uint8)]
    out = np.zeros(4 ** K, dtype=np.float32)
    if len(a) - K + 1 <= 0:
        return out
    win = np.lib.stride_tricks.sliding_window_view(a, K)
    ok = (win < 4).all(axis=1)
    if not ok.any():
        return out
    idx = (win[ok].astype(np.int64) * pot).sum(axis=1)
    out += np.bincount(idx, minlength=4 ** K).astype(np.float32)
    return out / out.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--etiquetas", default=None)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()

    import sklearn
    print("scikit-learn en esta maquina:", sklearn.__version__, "| numpy:", np.__version__)
    pk = joblib.load(a.modelo)
    rf, K, ventana, umbral = pk["modelo"], pk["K"], pk["ventana"], pk["umbral"]
    print(f"modelo cargado: k={K}, ventana={ventana}, umbral={umbral}, arboles={len(rf.estimators_)}")

    fa = leer_fasta(a.fasta)
    ids = list(fa)
    y = {}
    if a.etiquetas:
        for l in open(a.etiquetas):
            p = l.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1] in ("0", "1"):
                y[p[0]] = int(p[1])
        pos = [i for i in ids if y.get(i) == 1][: a.n // 2]
        neg = [i for i in ids if y.get(i) == 0][: a.n // 2]
        ids = pos + neg
    else:
        ids = ids[: a.n]

    aciertos = 0
    for gid in ids:
        X = np.array([vector_kmer(f, K) for f in fragmentar(fa[gid], ventana)])
        p = rf.predict_proba(X)[:, 1]
        pred = int((p >= umbral).any())
        etq = y.get(gid)
        marca = "" if etq is None else ("  OK" if pred == etq else "  ERROR")
        aciertos += (etq is not None and pred == etq)
        print(f"{gid[:60]:60s} prob_max={p.max():.3f} -> {'ARG' if pred else 'no ARG'}{marca}")
    if y:
        print(f"\nAciertos: {aciertos}/{len(ids)}")
    print("Verificacion terminada: el modelo carga y predice en esta maquina.")


if __name__ == "__main__":
    sys.exit(main())
