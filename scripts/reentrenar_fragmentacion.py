"""
reentrenar_fragmentacion.py -- regenera el dataset fragmentado con la regla de la
asesora y reentrena el modelo binario.

REGLA NUEVA (opción D): se descarta todo fragmento con menos de MIN_PB bases reales
y NO se usa padding. Quedan solo fragmentos de entre MIN_PB y 500 pb, con su largo
real. Sin ceros, sin vectores degenerados y sin solapamiento.

Por qué importa reentrenar y no solo cambiar la predicción: el modelo actual aprendió
con fragmentos rellenados, y en el entrenamiento cerca del 71 % de los fragmentos
degenerados eran positivos. O sea que absorbió "vector raro = resistente" como si
fuera señal. Con datos limpios no gasta capacidad en eso.

Dato fino que vale conocer: para el vector de k-meros, rellenar con ceros era
IRRELEVANTE. La función ignora toda ventana que toque un carácter que no sea ACGT,
así que 200 bases reales + 300 ceros da el mismo vector que 200 bases reales solas
(verificado). El problema nunca fueron los ceros sino tener MUY POCAS bases reales:
con menos de 6 no hay ninguna ventana válida y el vector queda en cero. De ahí que
lo que mide sea el DESCARTE a 150 pb, no el quitar el padding.

Uso:
    python reentrenar_fragmentacion.py --identidad 95
    python reentrenar_fragmentacion.py --identidad 90
    python reentrenar_fragmentacion.py --identidad 95 --min-pb 150 --arboles 200
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--identidad", default="95", choices=["95", "90", "85", "99"])
ap.add_argument("--filtro", default=None,
                help="FASTA que restringe el pool de positivos; sustituye al que se deduce de --identidad")
ap.add_argument("--min-pb", type=int, default=150, help="bases reales mínimas (asesora: 150)")
ap.add_argument("--ventana", type=int, default=500)
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--arboles", type=int, default=200)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--frac-test", type=float, default=0.10)
ap.add_argument("--guardar-tsv", action="store_true",
                help="además escribe el dataset fragmentado nuevo en data/interim")
args = ap.parse_args()
K, VENTANA, MIN_PB, SEED = args.k, args.ventana, args.min_pb, args.seed

# ------------------------------------------------------------------ rutas
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

PROC, INTER = RAIZ / "data" / "processed", RAIZ / "data" / "interim"
POS = PROC / "argdb_POSITIVO_final.fasta"
NEG = PROC / "argdb_NEGATIVO_final.fasta"
SET_1K1K = INTER / "muestra_genes_labels.tsv"
FASTA_1K1K = INTER / "muestra_genes_completos.fasta"
# El filtro solo puede QUITAR positivos del pool, nunca agregar. 90 y 85 son
# subconjuntos del 95 %, asi que ahi un filtro alcanza. Para 95 y 99 el FASTA de
# POS ya contiene exactamente el conjunto de ese nivel, asi que no se filtra.
if args.filtro:
    FILTRO = Path(args.filtro).expanduser()
elif args.identidad in ("95", "99"):
    FILTRO = None
else:
    FILTRO = INTER / f"positivo_id{args.identidad}_limpio.fasta"

for p, q in [(POS, POS.name), (NEG, NEG.name), (SET_1K1K, SET_1K1K.name),
             (FASTA_1K1K, FASTA_1K1K.name)]:
    if not p.exists():
        sys.exit(f"Falta {q}: {p}")
if FILTRO is not None and not FILTRO.exists():
    sys.exit(f"Falta el FASTA filtrado: {FILTRO}")

print("=" * 94)
print(f"REENTRENAMIENTO con la fragmentación de la asesora (descartar < {MIN_PB} pb, sin padding)")
print(f"identidad {args.identidad} % | K={K} | ventana={VENTANA} | semilla {SEED}")
print("=" * 94)


def leer_fasta(p):
    d, h, s = {}, None, []
    with open(p, encoding="utf-8", errors="replace") as fh:
        for l in fh:
            l = l.rstrip("\n")
            if l.startswith(">"):
                if h:
                    d[h] = "".join(s)
                h, s = l[1:].split()[0], []
            else:
                s.append(l.strip())
    if h:
        d[h] = "".join(s)
    return d


# ------------------------------------------------------------------ fragmentación NUEVA
def fragmentar(seq, ventana=VENTANA, min_pb=MIN_PB):
    """Opción D: bloques de 500; el resto final se conserva SIN rellenar si
    alcanza min_pb bases reales, y se descarta si no. Nunca devuelve ceros."""
    seq = seq.upper()
    L = len(seq)
    if L < ventana:
        return [seq] if L >= min_pb else []
    n = L // ventana
    fr = [seq[i * ventana:(i + 1) * ventana] for i in range(n)]
    r = L - n * ventana
    if r >= min_pb:
        fr.append(seq[n * ventana:])
    return fr


# ------------------------------------------------------------------ k-meros
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


def matriz(seqs):
    X = np.empty((len(seqs), 4 ** K), dtype=np.float32)
    for i, s in enumerate(seqs):
        X[i] = vector_kmer(s)
    return X


# ------------------------------------------------------------------ genes
t0 = time.time()
pos, neg = leer_fasta(POS), leer_fasta(NEG)
print(f"\npositivos {len(pos):,} | negativos {len(neg):,}  ({time.time()-t0:.0f}s)")

lab1k = pd.read_csv(SET_1K1K, sep="\t")
ids_1k = set(lab1k.iloc[:, 0].astype(str))
y_1k = dict(zip(lab1k.iloc[:, 0].astype(str), lab1k.iloc[:, 1].astype(int)))

genes = {}
for gid, s in pos.items():
    genes[gid] = (s, 1)
for gid, s in neg.items():
    genes[gid] = (s, 0)

hallados = ids_1k & set(genes)
print(f"genes del 1k+1k encontrados: {len(hallados):,} de {len(ids_1k):,}")
if len(hallados) < len(ids_1k):
    print(f"  AVISO: {len(ids_1k)-len(hallados)} no aparecen en los FASTA; se omiten")

pool_ids = [g for g in genes if g not in hallados]

if FILTRO is not None:
    hdrs = [l[1:].strip() for l in open(FILTRO, encoding="utf-8", errors="replace")
            if l.startswith(">")]
    ok = set(hdrs) | {h.split()[0] for h in hdrs}
    antes = len(pool_ids)
    pool_ids = [g for g in pool_ids if genes[g][1] == 0 or g in ok]
    print(f"filtro de identidad {args.identidad} %: {antes-len(pool_ids):,} positivos "
          f"fuera del set reducido")

print(f"pool de entrenamiento: {len(pool_ids):,} genes "
      f"(pos {sum(genes[g][1] for g in pool_ids):,})")

# ------------------------------------------------------------------ fragmentar
t0 = time.time()
seqs, ys, gs = [], [], []
sin_frag = []
for g in pool_ids:
    s, y = genes[g]
    fr = fragmentar(s)
    if not fr:
        sin_frag.append(g)
        continue
    for f in fr:
        seqs.append(f)
        ys.append(y)
        gs.append(g)
y_all = np.array(ys)
g_all = np.array(gs)
print(f"\nfragmentos: {len(seqs):,} de {len(set(g_all)):,} genes  ({time.time()-t0:.0f}s)")
print(f"  pos {int(y_all.sum()):,} | neg {int((y_all==0).sum()):,}")
print(f"  genes descartados por medir < {MIN_PB} pb: {len(sin_frag):,}"
      + (f"  (pos {sum(genes[g][1] for g in sin_frag):,})" if sin_frag else ""))
largos = np.array([len(s) for s in seqs])
print(f"  largo de fragmento: min {largos.min()} | mediana {int(np.median(largos))} | "
      f"max {largos.max()} | de 500 pb: {int((largos==VENTANA).sum()):,} "
      f"({100*(largos==VENTANA).mean():.1f} %)")
assert not any("0" in s for s in seqs[:5000]), "quedó relleno con ceros"

if args.guardar_tsv:
    out = INTER / f"dataset_fragmentado_v2_min{MIN_PB}.tsv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "lectura", "etiqueta"])
        cont = Counter()
        for s, yy, gg in zip(seqs, ys, gs):
            w.writerow([f"{gg}__frag{cont[gg]}", s, yy])
            cont[gg] += 1
    print(f"  dataset escrito: {out.name}")

# ------------------------------------------------------------------ corte por gen
sgkf = StratifiedGroupKFold(n_splits=int(round(1 / args.frac_test)), shuffle=True,
                            random_state=SEED)
tr_full, te = next(sgkf.split(np.zeros(len(y_all)), y_all, g_all))
assert not set(g_all[tr_full]) & set(g_all[te]), "FUGA: un gen en train y test"
print(f"\ntrain {len(tr_full):,} frag ({len(set(g_all[tr_full])):,} genes) | "
      f"test {len(te):,} ({len(set(g_all[te])):,}) con proporción natural")

rng = np.random.default_rng(SEED)
p_idx = tr_full[y_all[tr_full] == 1]
n_idx = tr_full[y_all[tr_full] == 0]
sel = np.concatenate([p_idx, rng.choice(n_idx, size=min(len(p_idx), len(n_idx)),
                                        replace=False)])
rng.shuffle(sel)
print(f"submuestreo 1:1 -> {len(sel):,} fragmentos "
      f"(razón original neg:pos {len(n_idx)/max(1,len(p_idx)):.1f})")

# ------------------------------------------------------------------ entrenar
t0 = time.time()
X_tr = matriz([seqs[i] for i in sel])
print(f"\nX_train {X_tr.shape} {X_tr.nbytes/1e6:.0f} MB ({time.time()-t0:.0f}s)")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=args.arboles, n_jobs=-1,
                            random_state=SEED).fit(X_tr, y_all[sel])
print(f"RandomForest entrenado en {time.time()-t0:.0f}s")
del X_tr

# ------------------------------------------------------------------ test interno
def met(yv, pv, pr=None):
    d = dict(recall=recall_score(yv, pv, zero_division=0),
             precision=precision_score(yv, pv, zero_division=0),
             f1=f1_score(yv, pv, zero_division=0))
    if pr is not None:
        d["auc"] = roc_auc_score(yv, pr)
    return d


X_te = matriz([seqs[i] for i in te])
p_te = rf.predict_proba(X_te)[:, 1]
UMBRAL = 0.5
m_frag = met(y_all[te], (p_te >= UMBRAL).astype(int), p_te)
d = pd.DataFrame({"gen": g_all[te], "y": y_all[te], "p": p_te}).groupby("gen").agg(
    y=("y", "max"), p=("p", "max"))
m_gen = met(d.y.values, (d.p.values >= UMBRAL).astype(int), d.p.values)
del X_te
print(f"\nTEST interno, fragmento: " + "  ".join(f"{k}={v:.3f}" for k, v in m_frag.items()))
print(f"TEST interno, gen       : " + "  ".join(f"{k}={v:.3f}" for k, v in m_gen.items()))

# ------------------------------------------------------------------ 1k+1k
fa1 = leer_fasta(FASTA_1K1K)
ids = [i for i in fa1 if i in y_1k]
filas, no_eval = [], 0
for gid in ids:
    fr = fragmentar(fa1[gid])
    if not fr:
        no_eval += 1
        continue
    p = rf.predict_proba(matriz(fr))[:, 1]
    filas.append((gid, y_1k[gid], int((p >= UMBRAL).any()), float(p.max())))
r1k = pd.DataFrame(filas, columns=["gen", "y", "pred", "p"])
m1k = met(r1k.y.values, r1k.pred.values, r1k.p.values)
cm = confusion_matrix(r1k.y, r1k.pred)

print(f"\n{'='*94}\n1k+1k (nivel gen, regla: algún fragmento >= {UMBRAL})\n{'='*94}")
print("  " + "  ".join(f"{k}={v:.3f}" for k, v in m1k.items()))
print(f"\n                  Predijo NO ARG   Predijo ARG")
print(f"  Real NO ARG (0)      {cm[0,0]:>8}      {cm[0,1]:>8}   <- falsos positivos")
print(f"  Real ARG (1)         {cm[1,0]:>8}      {cm[1,1]:>8}")
print(f"  marcados resistentes: {cm[0,1]+cm[1,1]}")
if no_eval:
    print(f"\n  {no_eval} gen(es) sin ningún fragmento (miden < {MIN_PB} pb): NO EVALUABLES.")
    print("  Quedan fuera de estas métricas. En la GUI hay que marcarlos aparte,")
    print("  no contarlos como 'No resistente'.")

# ------------------------------------------------------------------ comparación
REF = {  # medido con el modelo entrenado con fragmentos rellenados
    "95": {"A": (0.893, 0.752, 0.817, 0.884, 294), "D_pred": (0.857, 0.790, 0.822, 0.899, 228)},
    "90": {"A": (0.896, 0.730, 0.804, 0.878, 332), "D_pred": (0.865, 0.771, 0.815, 0.889, 257)},
}
if args.identidad in REF:
    r = REF[args.identidad]
    print(f"\n{'='*94}\nCOMPARACIÓN (1k+1k, nivel gen)\n{'='*94}")
    print(f"  {'':<46} {'recall':>7} {'prec.':>7} {'F1':>7} {'AUC':>7} {'FP':>5}")
    print(f"  {'A  original (entrenado con relleno)':<46} "
          + "".join(f"{v:>7.3f}" for v in r['A'][:4]) + f" {r['A'][4]:>5}")
    print(f"  {'D  solo en predicción (mismo modelo viejo)':<46} "
          + "".join(f"{v:>7.3f}" for v in r['D_pred'][:4]) + f" {r['D_pred'][4]:>5}")
    print(f"  {'D  REENTRENADO (este script)':<46} "
          f"{m1k['recall']:>7.3f}{m1k['precision']:>7.3f}{m1k['f1']:>7.3f}"
          f"{m1k['auc']:>7.3f} {int(cm[0,1]):>5}")
    dauc = m1k["auc"] - r["D_pred"][3]
    print(f"\n  ganancia del reentrenamiento sobre corregir solo la predicción: "
          f"AUC {dauc:+.3f}")
    if dauc > 0.002:
        print("  -> reentrenar sí aportó: el modelo ya no gasta capacidad en el ruido")
        print("     de los fragmentos degenerados.")
    elif dauc < -0.002:
        print("  -> reentrenar NO aportó. Puede ser que los fragmentos de largo variable")
        print("     introduzcan su propia dificultad. Conviene reportarlo tal cual.")
    else:
        print("  -> sin diferencia apreciable: el beneficio ya estaba en la predicción.")

# ------------------------------------------------------------------ guardar
MOD = RAIZ / "data" / "models"
MOD.mkdir(parents=True, exist_ok=True)
ruta = MOD / f"argus_rf_k{K}_id{args.identidad}_v2.joblib"
rf.set_params(n_jobs=1)
joblib.dump({
    "modelo": rf, "K": K, "ventana": VENTANA, "umbral": UMBRAL,
    "fragmentacion": f"opcion D: descartar < {MIN_PB} pb reales, sin padding",
    "min_pb": MIN_PB, "relleno": "ninguno", "regla_gen": "any", "semilla": SEED,
    "identidad": args.identidad, "n_arboles": args.arboles,
    "metricas_test_fragmento": m_frag, "metricas_test_gen": m_gen,
    "metricas_1k1k_gen": m1k, "fp_1k1k": int(cm[0, 1]), "fn_1k1k": int(cm[1, 0]),
    "genes_no_evaluables_1k1k": int(no_eval),
}, ruta, compress=3)
print(f"\nGuardado: {ruta}  ({ruta.stat().st_size/1e6:.0f} MB)")
r1k.to_csv(RAIZ / "results" / f"r1k_id{args.identidad}_v2.csv", index=False)

print(f"\nPara usarlo en la GUI, cambia en app/argus_gui.py:")
print(f'    RUTA_MODELO = os.path.join(RAIZ, "data", "models", '
      f'"argus_rf_k{K}_id{args.identidad}_v2.joblib")')
print(f"y pon MIN_PB_FRAGMENTO = {MIN_PB} con la fragmentación nueva (ver el parche aparte).")
