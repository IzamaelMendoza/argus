#!/usr/bin/env python3
"""Evalúa el 1k+1k usando EXACTAMENTE la función predecir_mag de la GUI (sin abrir ventana)."""
import argparse, importlib.util, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

ap = argparse.ArgumentParser()
ap.add_argument("--fasta", default="data/interim/muestra_genes_completos.fasta")
ap.add_argument("--etiquetas", default="data/interim/muestra_genes_labels.tsv")
ap.add_argument("--gui", default="app/main_gui.py")
ap.add_argument("--salida", default="results/evaluacion_gui_1k1k.csv")
a = ap.parse_args()

if "__main__" not in open(a.gui, encoding="utf-8").read():
    sys.exit("main_gui.py no tiene 'if __name__ == \"__main__\":'; importarlo abriría la ventana. Avísame.")
spec = importlib.util.spec_from_file_location("main_gui", a.gui)
gui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gui)

seqs, cur = {}, None
for l in open(a.fasta):
    l = l.strip()
    if l.startswith(">"):
        cur = l[1:].split()[0]; seqs[cur] = []
    elif cur:
        seqs[cur].append(l)
seqs = {k: "".join(v) for k, v in seqs.items()}

y = {}
for l in open(a.etiquetas):
    p = l.rstrip("\n").split("\t")
    if len(p) >= 2 and p[1] in ("0", "1"):
        y[p[0]] = int(p[1])

ids = [i for i in seqs if i in y]
print(f"genes con etiqueta: {len(ids)}  (ARG={sum(y[i] for i in ids)}, no ARG={sum(1 - y[i] for i in ids)})")

res = gui.predecir_mag([(i, seqs[i]) for i in ids])
assert len(res) == len(ids), "la GUI devolvió un número distinto de resultados"
print(f"K usado por la GUI: {gui.K}")

df = pd.DataFrame({"id": ids, "real": [y[i] for i in ids],
                   "pred": [int(r["presencia"] == "Sí") for r in res],
                   "prob_max": [r["probabilidad"] for r in res]})
os.makedirs(os.path.dirname(a.salida), exist_ok=True)
df.to_csv(a.salida, index=False)

p, r, f, _ = precision_recall_fscore_support(df.real, df.pred, average="binary")
tn, fp, fn, tp = confusion_matrix(df.real, df.pred).ravel()
auc = roc_auc_score(df.real, df.prob_max)
print(f"\nRecall {r:.3f} | Precisión {p:.3f} | F1 {f:.3f} | AUC {auc:.3f}")
print(f"Matriz de confusión [[TN FP],[FN TP]] = [[{tn},{fp}],[{fn},{tp}]]")
print("Referencia notebook (ARGus v2 id95): R 0.893 | P 0.752 | F1 0.817 | AUC 0.884 | [[706,294],[107,893]]")
print("Guardado en", a.salida)
