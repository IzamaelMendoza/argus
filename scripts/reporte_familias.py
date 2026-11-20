"""
reporte_familias.py -- reporte de clasificacion y matriz de confusion del modelo
de familias que YA esta entrenado.

NO reentrena nada. Carga dnabert6_familias_cabeza.pt (el mismo archivo que usa la
GUI), reconstruye el mismo corte de test y produce:

  * classification_report a nivel FRAGMENTO y a nivel GEN
  * matriz de confusion en texto y como figura
  * analisis de "imanes": que familias atraen predicciones que no les tocan

Se auto-verifica: la balanced accuracy que calcula debe coincidir con la que
quedo guardada en familias_dnabert_meta.json. Si no coincide, el corte no se
reprodujo bien y el script avisa en lugar de dar numeros falsos.

Uso:
    python scripts/reporte_familias.py
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--chunk", type=int, default=256)
args = ap.parse_args()

if args.raiz:
    RAIZ = Path(args.raiz).expanduser()
else:
    RAIZ = None
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "data").is_dir():
            RAIZ = cand
            break
    if RAIZ is None:
        sys.exit('No encontre el proyecto. Usa --raiz "ruta/al/Proyecto_Residencias"')

META = RAIZ / "results" / "familias" / "familias_dnabert_meta.json"
CKPT = RAIZ / "data" / "models" / "dnabert6_familias_cabeza.pt"
NORM = RAIZ / "results" / "familias" / "dnabert6_norm.npy"
ETQ = RAIZ / "results" / "familias" / "etiquetas_finales.tsv"
FASTA = RAIZ / "data" / "processed" / "argdb_POSITIVO_final.fasta"
for p in (META, CKPT, NORM, ETQ, FASTA):
    if not p.exists():
        sys.exit(f"Falta: {p}\nCorre antes scripts/familias_dnabert.py --completo")

CACHE = RAIZ / "data" / "models" / "dnabert6_emb_familias"
SAL = RAIZ / "results" / "familias"

with open(META, encoding="utf-8") as fh:
    meta = json.load(fh)
clases = list(meta["clases"])
K = int(meta.get("K", 6))
VENTANA = int(meta.get("ventana", 500))
MIN_PB = int(meta.get("min_pb", 150))
SEED = int(meta.get("semilla", 42))
MAX_LEN = int(meta.get("max_len", 512))
DIM = int(meta.get("hidden", 768))
MODELO_HF = meta.get("modelo_base", "zhihan1996/DNA_bert_6")
BAL_GUARDADA = meta.get("test_gen_balanced_acc")

print("=" * 96)
print("REPORTE Y MATRIZ DE CONFUSION -- modelo de familias (DNABERT-6)")
print(f"cabeza: {CKPT.name}")
print(f"{len(clases)} familias | K={K} | ventana={VENTANA} | min_pb={MIN_PB} | semilla={SEED}")
print(f"balanced accuracy guardada en el meta: {BAL_GUARDADA:.4f}")
print("=" * 96)

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("Falta PyTorch.")
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold

np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------- datos
etq = {}
with open(ETQ, encoding="utf-8") as fh:
    for f in csv.DictReader(fh, delimiter="\t"):
        if f.get("etiqueta"):
            etq[f["gen"]] = f["etiqueta"]

fa, h, s = {}, None, []
with open(FASTA, encoding="utf-8", errors="replace") as fh:
    for linea in fh:
        linea = linea.rstrip("\n")
        if linea.startswith(">"):
            if h:
                fa[h] = "".join(s)
            h, s = linea[1:].split()[0], []
        else:
            s.append(linea.strip())
if h:
    fa[h] = "".join(s)


def fragmentar(seq, ventana=VENTANA, min_pb=MIN_PB):
    seq = seq.upper(); L = len(seq)
    if L < ventana:
        return [seq] if L >= min_pb else []
    n = L // ventana
    fr = [seq[i * ventana:(i + 1) * ventana] for i in range(n)]
    if L - n * ventana >= min_pb:
        fr.append(seq[n * ventana:])
    return fr


def sequence_to_kmers(seq, k=K):
    seq = seq.upper().replace("0", "")
    seq = re.sub(r"[^ACGT]", "N", seq)
    if len(seq) < k:
        return "N" * k
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


seqs, ys, gs = [], [], []
for gen, lab in etq.items():
    sq = fa.get(gen)
    if not sq:
        continue
    for f in fragmentar(sq):
        seqs.append(f); ys.append(lab); gs.append(gen)
y = np.array(ys); g = np.array(gs)
assert sorted(set(y)) == clases, (
    "Las familias del TSV no coinciden con las del meta. Reentrena o revisa "
    "etiquetas_finales.tsv")
y_idx = np.array([clases.index(v) for v in y])

# mismo corte que familias_dnabert.py (mismo metodo, misma semilla -> mismo test)
genes_por_fam = Counter({gg: yy for gg, yy in zip(g, y)}.values())
n_splits = max(2, min(5, min(genes_por_fam.values())))
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
tr_full, te = next(sgkf.split(np.zeros(len(y)), y, g))
n_val = max(2, min(5, min(Counter({gg: yy for gg, yy in zip(g[tr_full], y[tr_full])}
                                  .values()).values())))
sgkf2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=SEED + 7)
rel_tr, rel_va = next(sgkf2.split(np.zeros(len(tr_full)), y[tr_full], g[tr_full]))
tr = tr_full[rel_tr]
print(f"\ntest reconstruido: {len(te):,} fragmentos de {len(set(g[te])):,} genes")

# ----------------------------------------------------------------- embeddings
from transformers import AutoTokenizer, AutoModel
print(f"\nCargando {MODELO_HF} ...")
tok = AutoTokenizer.from_pretrained(MODELO_HF)
bert = AutoModel.from_pretrained(MODELO_HF).eval()


@torch.inference_mode()
def codificar(lista):
    out = np.empty((len(lista), DIM), dtype=np.float32)
    for i in range(0, len(lista), args.batch):
        lote = [sequence_to_kmers(x) for x in lista[i:i + args.batch]]
        enc = tok(lote, padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")
        hs = bert(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out[i:i + len(lote)] = ((hs * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy()
    return out


def embeddings(idxs, nombre):
    """Misma convencion de cache que familias_dnabert.py: la reusa si ya esta."""
    firma = hashlib.md5(("|".join(seqs[i] for i in idxs)).encode()).hexdigest()[:10]
    d = CACHE / f"{nombre}_{len(idxs)}_{firma}"
    d.mkdir(parents=True, exist_ok=True)
    partes = []
    for b0 in range(0, len(idxs), args.chunk):
        f = d / f"{b0:07d}.npy"
        if f.exists():
            partes.append(np.load(f)); continue
        e = codificar([seqs[i] for i in idxs[b0:b0 + args.chunk]])
        np.save(f, e); partes.append(e)
        print(f"\r  {nombre}: {min(b0+args.chunk, len(idxs))}/{len(idxs)}  ",
              end="", flush=True)
    print(f"\r  {nombre}: listo ({len(idxs)} fragmentos)      ")
    return np.vstack(partes)


print("\nEmbeddings del test (se reusan de la cache si ya estaban):")
E_te = embeddings(te, "test")

norm = np.load(NORM)
mu, sd = norm[0:1].astype(np.float32), norm[1:2].astype(np.float32)
Xte = torch.tensor((E_te - mu) / sd)

cabeza = nn.Sequential(nn.Linear(DIM, 256), nn.ReLU(), nn.Dropout(0.3),
                       nn.Linear(256, len(clases)))
cabeza.load_state_dict(torch.load(CKPT, map_location="cpu"))
cabeza.eval()

with torch.no_grad():
    prob = torch.softmax(cabeza(Xte), dim=1).numpy()
pred_frag = prob.argmax(1)
real_frag = y_idx[te]

# nivel gen: promedio de probabilidades
por_gen = defaultdict(list)
for i, gg in enumerate(g[te]):
    por_gen[gg].append(i)
gr, gp = [], []
for gg, idxs in por_gen.items():
    gr.append(real_frag[idxs[0]])
    gp.append(int(np.argmax(prob[idxs].mean(axis=0))))
gr, gp = np.array(gr), np.array(gp)

bal_g = balanced_accuracy_score(gr, gp)
f1_g = f1_score(gr, gp, average="macro", zero_division=0)

# --- auto-verificacion ---
print(f"\n{'='*96}")
if BAL_GUARDADA is not None and abs(bal_g - BAL_GUARDADA) > 0.005:
    print(f"[AVISO] la balanced accuracy que calculo ({bal_g:.4f}) NO coincide con la")
    print(f"        guardada en el meta ({BAL_GUARDADA:.4f}). El corte de test no se")
    print(f"        reprodujo igual: los numeros de abajo no son los del modelo que")
    print(f"        corriste. Revisa que etiquetas_finales.tsv no haya cambiado.")
else:
    print(f"[OK] balanced accuracy {bal_g:.4f} coincide con el meta "
          f"({BAL_GUARDADA:.4f}): el corte de test es el mismo.")
print("=" * 96)

# =============================================================================
#  REPORTES
# =============================================================================
nombres = [str(c) for c in clases]

print(f"\n{'='*96}")
print("REPORTE A NIVEL FRAGMENTO")
print(f"{'='*96}")
print(f"balanced accuracy {balanced_accuracy_score(real_frag, pred_frag):.4f} | "
      f"F1 macro {f1_score(real_frag, pred_frag, average='macro', zero_division=0):.4f} | "
      f"exactitud {(pred_frag == real_frag).mean():.4f}  ({len(te):,} fragmentos)")
pres_f = sorted(set(real_frag.tolist()) | set(pred_frag.tolist()))
print(classification_report(real_frag, pred_frag, labels=pres_f,
                            target_names=[nombres[i][:34] for i in pres_f],
                            zero_division=0, digits=3))

print(f"{'='*96}")
print("REPORTE A NIVEL GEN  (es el que cuenta: un gen tiene UNA familia)")
print(f"{'='*96}")
print(f"balanced accuracy {bal_g:.4f} | F1 macro {f1_g:.4f} | "
      f"exactitud {(gp == gr).mean():.4f}  ({len(gr)} genes)")
pres = sorted(set(gr.tolist()) | set(gp.tolist()))
rep = classification_report(gr, gp, labels=pres,
                            target_names=[nombres[i][:34] for i in pres],
                            zero_division=0, digits=3)
print(rep)
with open(SAL / "reporte_familias.txt", "w", encoding="utf-8") as fh:
    fh.write(f"Modelo: {CKPT.name}\n{len(clases)} familias | {len(gr)} genes de test\n")
    fh.write(f"balanced accuracy {bal_g:.4f} | F1 macro {f1_g:.4f} | "
             f"exactitud {(gp==gr).mean():.4f}\n\n{rep}\n")

# =============================================================================
#  MATRIZ DE CONFUSION
# =============================================================================
cm = confusion_matrix(gr, gp, labels=range(len(clases)))


def abreviar(nombre):
    """3 letras para encabezar las columnas: OXA, MFS, RND, CEF..."""
    base = re.sub(r"\(.*?\)", " ", str(nombre))          # fuera lo de parentesis
    palabras = re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ][\wÁÉÍÓÚÑáéíóúñ-]*", base)
    for p in palabras:                                    # primero una sigla
        cab = p.split("-")[0]
        if len(cab) >= 2 and cab.isupper() and cab.isalpha():
            return cab[:3].upper()
    for p in palabras:                                    # si no, la 1a palabra
        cab = p.split("-")[0]
        if len(cab) >= 3:
            return cab[:3].upper()
    return (palabras[0][:3].upper() if palabras else "???")


abrev, vistos = [], {}
for n in nombres:
    a = abreviar(n)
    if a in vistos:                                       # desempatar colisiones
        vistos[a] += 1
        a = f"{a[:2]}{vistos[a]}"
    else:
        vistos[a] = 1
    abrev.append(a)

w = max(4, len(str(cm.max())) + 1)
ancho_fila = 38
print(f"{'='*96}")
print(f"MATRIZ DE CONFUSION -- familias del gen (nivel gen, {len(gr)} genes de test)")
print(f"{'='*96}\n")
print(" " * (ancho_fila + 7) + "predicha ->")
print("   real \\" + " " * (ancho_fila - 1) +
      "".join(f"{a:>{w}}" for a in abrev) + "  | total")
print("   " + "-" * (ancho_fila + 4 + w * len(clases) + 9))
for i in range(len(clases)):
    fila = "".join((f"{v:>{w}}" if v else f"{'.':>{w}}") for v in cm[i])
    etq = f"{abrev[i]}  {nombres[i]}"[:ancho_fila + 4]
    marca = ""
    if cm[i].sum() and cm[i, i] / cm[i].sum() < 0.5:
        marca = "  <- menos de la mitad bien"
    print(f"   {etq:<{ancho_fila + 4}}{fila}  |{cm[i].sum():>6}{marca}")
print("   " + "-" * (ancho_fila + 4 + w * len(clases) + 9))
print(f"   {'total predichas':<{ancho_fila + 4}}" +
      "".join(f"{v:>{w}}" for v in cm.sum(axis=0)) + f"  |{cm.sum():>6}")
print("\n   La diagonal son los aciertos. Todo lo que cae fuera es confusion:")
print("   la fila dice de que familia era el gen y la columna que familia le puso el modelo.")
print("\n   Abreviaturas:")
for i in range(0, len(clases), 2):
    izq = f"{abrev[i]} = {nombres[i]}"
    der = f"{abrev[i+1]} = {nombres[i+1]}" if i + 1 < len(clases) else ""
    print(f"     {izq:<46}{der}")

# --- por familia, en el mismo estilo que el reporte binario ---
p_, r_, f_, s_ = precision_recall_fscore_support(
    gr, gp, labels=range(len(clases)), zero_division=0)
print(f"\n{'-'*96}")
print("DESEMPENO POR FAMILIA")
print(f"{'-'*96}\n")
for i in np.argsort(-s_):
    print(f"   {nombres[i][:36]:<36} recall={r_[i]:.3f}  precision={p_[i]:.3f}  "
          f"f1={f_[i]:.3f}   ({s_[i]} genes)")
print(f"\n   {'-'*86}")
print(f"   balanced accuracy = {bal_g:.3f}   (el promedio de los recall de arriba)")
print(f"   F1 macro          = {f1_g:.3f}   (el promedio de los f1)")
print(f"   exactitud         = {(gp == gr).mean():.3f}   (aciertos / total, sin balancear)")
print(f"   genes bien clasificados: {int((gp == gr).sum())} de {len(gr)}")

# --- imanes: familias que reciben mas de lo que les toca ---
print(f"\n{'-'*96}")
print("FAMILIAS QUE ACTUAN DE 'IMAN' (reciben predicciones que no les tocan)")
print(f"{'-'*96}\n")
reales = cm.sum(axis=1)
predichas = cm.sum(axis=0)
aciertos = np.diag(cm)
ajenas = predichas - aciertos
filas = []
print(f"{'familia':<40}{'reales':>8}{'predichas':>11}{'ajenas':>8}{'razon':>8}")
for i in np.argsort(-ajenas):
    r = reales[i]
    razon = predichas[i] / r if r else float("inf")
    print(f"{nombres[i][:40]:<40}{r:>8}{predichas[i]:>11}{ajenas[i]:>8}"
          f"{razon:>8.2f}" if r else
          f"{nombres[i][:40]:<40}{r:>8}{predichas[i]:>11}{ajenas[i]:>8}{'-':>8}")
    filas.append(dict(familia=nombres[i], reales=int(r), predichas=int(predichas[i]),
                      ajenas=int(ajenas[i]), aciertos=int(aciertos[i]),
                      razon=round(float(razon), 3) if r else None))
peor = int(np.argmax(ajenas))
if ajenas[peor] > 0:
    print(f"\n  {nombres[peor]} es la que mas absorbe: recibe {ajenas[peor]} genes")
    print(f"  que son de otras familias.")
    orig = np.argsort(-cm[:, peor])
    print(f"  Vienen de:")
    for i in orig[:5]:
        if i != peor and cm[i, peor] > 0:
            print(f"      {cm[i, peor]:>3} de {nombres[i][:50]}")

import pandas as pd
pd.DataFrame(filas).to_csv(SAL / "familias_imanes.csv", index=False)
pd.DataFrame(cm, index=nombres, columns=nombres).to_csv(SAL / "matriz_confusion.csv")

# =============================================================================
#  FIGURA
# =============================================================================
cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)   # normalizada por fila
etq_corta = [n[:30] for n in nombres]

fig, ax = plt.subplots(figsize=(11.5, 9.5))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(clases))); ax.set_yticks(range(len(clases)))
ax.set_xticklabels(etq_corta, rotation=42, ha="right", fontsize=8.5)
ax.set_yticklabels([f"{n}  ({c})" for n, c in zip(etq_corta, reales)], fontsize=8.5)
ax.set_xlabel("familia predicha", fontsize=10)
ax.set_ylabel("familia real  (genes en el test)", fontsize=10)
ax.set_title(f"Matriz de confusión — familias del gen, nivel gen\n"
             f"DNABERT-6 + class_weight · balanced accuracy {bal_g:.3f} · "
             f"{len(gr)} genes de test", fontsize=11.5)
for i in range(len(clases)):
    for j in range(len(clases)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8.5,
                    color="white" if cmn[i, j] > 0.55 else "#222222",
                    fontweight="bold" if i == j else "normal")
cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03)
cb.set_label("proporción de la fila (recall)", fontsize=9)
ax.set_xticks(np.arange(-.5, len(clases), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(clases), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="minor", length=0)
plt.tight_layout(); plt.savefig(SAL / "matriz_confusion.png", dpi=200); plt.close()

# --- figura 2: precision / recall / F1 por familia ---
p, r, f1c, sup = precision_recall_fscore_support(
    gr, gp, labels=range(len(clases)), zero_division=0)
orden = np.argsort(-sup)
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(clases)); w = 0.27
ax.bar(x - w, p[orden], w, label="precisión", color="#4C72B0")
ax.bar(x, r[orden], w, label="recall", color="#DD8452")
ax.bar(x + w, f1c[orden], w, label="F1", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels([f"{etq_corta[i]}\n({sup[i]} genes)" for i in orden],
                   rotation=38, ha="right", fontsize=8)
ax.set_ylim(0, 1.05); ax.set_ylabel("valor")
ax.set_title(f"Desempeño por familia (ordenadas de mayor a menor número de genes)\n"
             f"balanced accuracy {bal_g:.3f} · F1 macro {f1_g:.3f}", fontsize=11.5)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=.25, linewidth=.7)
ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(SAL / "desempeno_por_familia.png", dpi=200); plt.close()

print(f"\n{'='*96}")
print("LISTO. En results/familias/:")
for n in ("reporte_familias.txt", "matriz_confusion.png", "matriz_confusion.csv",
          "desempeno_por_familia.png", "familias_imanes.csv"):
    print(f"   {n}")
print("=" * 96)