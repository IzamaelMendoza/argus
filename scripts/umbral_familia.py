"""
umbral_familia.py -- elige UMBRAL_FAMILIA con datos, no a ojo.

El problema: sin la clase "Otras familias" el modelo no puede decir "no sé", así que
se usa la probabilidad como filtro: si la clase ganadora no llega al umbral, la GUI
muestra "Sin determinar". Pero un umbral muy alto rechaza casi todo y uno muy bajo
deja pasar basura.

Este script reconstruye EL MISMO conjunto de prueba que usó el entrenamiento (misma
semilla, mismo corte por gen) y para cada umbral reporta:

  - cuántos genes quedarían como "Sin determinar" (cobertura)
  - de los que SÍ se clasifican, qué porcentaje acierta (precisión efectiva)

Así se ve el intercambio y se elige con número. No reentrena nada: carga el modelo
que ya está guardado.

Uso:
    python umbral_familia.py
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--ventana", type=int, default=500)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
K, VENTANA, SEED = args.k, args.ventana, args.seed

if args.raiz:
    RAIZ = Path(args.raiz).expanduser()
else:
    RAIZ = None
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "data").is_dir():
            RAIZ = cand
            break
    if RAIZ is None:
        sys.exit('No encontré el proyecto. Usa --raiz "ruta\\al\\Proyecto_Residencias"')

MODELO = RAIZ / "app" / "modelo_familias.joblib"
ETQ = RAIZ / "results" / "familias" / "etiquetas_finales.tsv"
FASTA = RAIZ / "data" / "processed" / "argdb_POSITIVO_final.fasta"
for p, q in [(MODELO, "app/modelo_familias.joblib"), (ETQ, "etiquetas_finales.tsv"),
             (FASTA, "argdb_POSITIVO_final.fasta")]:
    if not p.exists():
        sys.exit(f"Falta {q}: {p}")

rf = joblib.load(MODELO)
print("=" * 88)
print("ELECCIÓN DE UMBRAL_FAMILIA")
print(f"modelo: {MODELO.name} | {len(rf.classes_)} clases | azar = "
      f"{1/len(rf.classes_):.3f}")
print("=" * 88)

# ---------------------------------------------------------------- k-meros
LUT = np.full(256, 4, dtype=np.int8)
for _i, _b in enumerate(b"ACGT"):
    LUT[_b] = _i
POT = (4 ** np.arange(K - 1, -1, -1)).astype(np.int64)


def vector_kmer(seq):
    a = LUT[np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)]
    out = np.zeros(4 ** K, dtype=np.float32)
    if len(a) - K + 1 <= 0:
        return out
    win = np.lib.stride_tricks.sliding_window_view(a, K)
    ok = (win < 4).all(axis=1)
    if not ok.any():
        return out
    idx = (win[ok].astype(np.int64) * POT).sum(axis=1)
    out += np.bincount(idx, minlength=4 ** K).astype(np.float32)
    return out / out.sum()


def fragmentar(seq, ventana=VENTANA):
    seq = seq.upper()
    L = len(seq)
    if L < ventana:
        return [seq + "0" * (ventana - L)]
    n = L // ventana
    fr = [seq[i * ventana:(i + 1) * ventana] for i in range(n)]
    if L % ventana:
        fr.append(seq[-ventana:])
    return fr


# ---------------------------------------------------------------- datos
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

Xs, ys, gs = [], [], []
for gen, lab in etq.items():
    seq = fa.get(gen)
    if not seq:
        continue
    for fr in fragmentar(seq):
        Xs.append(vector_kmer(fr))
        ys.append(lab)
        gs.append(gen)
X, y, g = np.array(Xs, dtype=np.float32), np.array(ys), np.array(gs)
print(f"\n{len(set(g)):,} genes -> {len(X):,} fragmentos")

# mismo corte que el entrenamiento
genes_por_clase = Counter({gg: yy for gg, yy in zip(g, y)}.values())
n_splits = max(2, min(5, min(genes_por_clase.values())))
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
tr, te = next(sgkf.split(X, y, g))
assert not set(g[tr]) & set(g[te]), "FUGA"
print(f"test: {len(te):,} fragmentos ({len(set(g[te])):,} genes) "
      f"-- el mismo que usó el entrenamiento (semilla {SEED})")

# ---------------------------------------------------------------- por gen
proba = rf.predict_proba(X[te])
por_gen = defaultdict(list)
for i, gg in enumerate(g[te]):
    por_gen[gg].append(i)

reales, ganadoras, pmax = [], [], []
for gg, idxs in por_gen.items():
    pr = proba[idxs].mean(axis=0)          # misma regla que la GUI corregida
    j = int(np.argmax(pr))
    reales.append(y[te][idxs[0]])
    ganadoras.append(rf.classes_[j])
    pmax.append(float(pr[j]))
reales, ganadoras, pmax = np.array(reales), np.array(ganadoras), np.array(pmax)
aciertos = reales == ganadoras

print(f"\nProbabilidad de la clase ganadora, en los {len(pmax)} genes de prueba:")
qs = [5, 10, 25, 50, 75, 90, 95]
print("  percentiles: " + "  ".join(f"p{q}={np.percentile(pmax,q):.2f}" for q in qs))
print(f"  en los que ACIERTA : mediana {np.median(pmax[aciertos]):.2f}")
print(f"  en los que FALLA   : mediana {np.median(pmax[~aciertos]):.2f}")

# ---------------------------------------------------------------- barrido
print(f"\n{'='*88}")
print("  umbral   clasificados      'Sin determinar'   acierto en los clasificados")
print(f"{'-'*88}")
mejor = None
for t in [0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
    sel = pmax >= t
    n_cl = int(sel.sum())
    n_sd = len(pmax) - n_cl
    acc = aciertos[sel].mean() if n_cl else float("nan")
    marca = ""
    if n_cl and n_cl / len(pmax) >= 0.70 and (mejor is None or acc > mejor[1]):
        mejor = (t, acc, n_cl)
        marca = ""
    print(f"  {t:>5.2f}   {n_cl:>4} ({100*n_cl/len(pmax):>5.1f} %)   "
          f"{n_sd:>4} ({100*n_sd/len(pmax):>5.1f} %)        "
          + (f"{acc:.3f}" if n_cl else "-"))
print(f"{'-'*88}")
if mejor:
    print(f"\n  Sugerencia: UMBRAL_FAMILIA = {mejor[0]:.2f}")
    print(f"    clasifica {mejor[2]} de {len(pmax)} genes "
          f"({100*mejor[2]/len(pmax):.0f} %) con {mejor[1]:.1%} de acierto.")
    print("    Criterio: el mayor acierto entre los umbrales que aún clasifican")
    print("    al menos el 70 % de los genes. Si prefieres cubrir más y equivocarte")
    print("    más, baja el umbral; si prefieres lo contrario, súbelo.")
else:
    print("\n  Ningún umbral clasifica el 70 % de los genes: usa 0.00 o 0.10 y")
    print("  acepta que el modelo se equivoque, o vuelve a entrenar con menos clases.")

print(f"\n  Referencia sin umbral (todo clasificado):")
print(f"    exactitud {aciertos.mean():.3f} | balanced accuracy "
      f"{balanced_accuracy_score(reales, ganadoras):.3f} | F1 macro "
      f"{f1_score(reales, ganadoras, average='macro', zero_division=0):.3f}")

# ---------------------------------------------------------------- por clase
print(f"\n{'='*88}\nPOR CLASE (con el umbral sugerido)\n{'='*88}")
t = mejor[0] if mejor else 0.0
print(f"  {'familia':<36} {'n':>4} {'clasif.':>8} {'acierto':>8}")
for c in sorted(set(reales)):
    m = reales == c
    sel = m & (pmax >= t)
    acc = aciertos[sel].mean() if sel.sum() else float("nan")
    print(f"  {c[:36]:<36} {int(m.sum()):>4} {int(sel.sum()):>8} "
          + (f"{acc:>8.3f}" if sel.sum() else f"{'-':>8}"))
print("\n  Una clase con 'clasif.' muy por debajo de 'n' es una familia que el modelo")
print("  reconoce mal: casi siempre va a salir 'Sin determinar' en la GUI.")
