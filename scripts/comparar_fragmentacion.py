"""
comparar_fragmentacion.py -- compara las cuatro formas de fragmentar, sobre el 1k+1k,
con el modelo YA entrenado (no reentrena nada).

Propuesta de la asesora (opción D): descartar todo fragmento con menos de MIN_PB bases
reales y NO usar padding; solo se quedan los fragmentos de entre MIN_PB y 500 pb, con
su largo real. Su argumento contra la opción C es correcto: al correr el último bloque
hacia atrás, las bases del solapamiento se cuentan dos veces y esa región pesa doble.

Las cuatro opciones:

  A  relleno con ceros            el original. Un gen de 504 pb deja un fragmento de
                                  4 bases reales + 496 ceros -> vector todo en cero
                                  -> el bosque responde 0.714 -> positivo automático.
  B  descartar cortos + padding   se tira el fragmento si tiene < MIN_PB reales; los
                                  demás se rellenan a 500.
  C  correr hacia atrás           el último bloque toma las últimas 500 bases reales.
                                  Sin ceros, pero con solapamiento.
  D  descartar cortos, SIN padding  la propuesta de la asesora: fragmentos de largo
                                  variable entre MIN_PB y 500, sin ceros y sin solape.

Nota sobre D: el vector de k-meros es una FRECUENCIA normalizada por el número de
ventanas válidas, así que un fragmento de 150 pb da un vector perfectamente válido
(145 ventanas) sin necesidad de rellenar. Es más grueso que uno de 495 ventanas, pero
no está distorsionado por ceros.

Uso:
    python comparar_fragmentacion.py
    python comparar_fragmentacion.py --min-pb 150 --identidad 95
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--identidad", default="95")
ap.add_argument("--min-pb", type=int, default=150, help="mínimo de bases reales (asesora: 150)")
ap.add_argument("--ventana", type=int, default=500)
ap.add_argument("--umbral", type=float, default=None)
args = ap.parse_args()
VENTANA, MIN_PB = args.ventana, args.min_pb

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

MODELO = RAIZ / "data" / "models" / f"argus_rf_k6_id{args.identidad}.joblib"
FASTA = RAIZ / "data" / "interim" / "muestra_genes_completos.fasta"
LABELS = RAIZ / "data" / "interim" / "muestra_genes_labels.tsv"
for p, q in [(MODELO, MODELO.name), (FASTA, FASTA.name), (LABELS, LABELS.name)]:
    if not p.exists():
        sys.exit(f"Falta {q}: {p}")

paq = joblib.load(MODELO)
rf = paq["modelo"] if isinstance(paq, dict) else paq
UMBRAL = args.umbral if args.umbral is not None else (
    float(paq.get("umbral", 0.5)) if isinstance(paq, dict) else 0.5)
K = 1
while 4 ** K < rf.n_features_in_:
    K += 1
try:
    rf.set_params(n_jobs=-1)
except Exception:
    pass

print("=" * 94)
print("COMPARACIÓN DE LAS CUATRO FRAGMENTACIONES (modelo ya entrenado, no se reentrena)")
print(f"modelo: {MODELO.name} | K={K} | ventana={VENTANA} | umbral={UMBRAL} | "
      f"mínimo de bases reales = {MIN_PB}")
print("=" * 94)

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


# ---------------------------------------------------------------- las cuatro opciones
def frag_A(seq):
    """relleno con ceros (el original)."""
    seq = seq.upper(); L = len(seq)
    if L < VENTANA:
        return [seq + "0" * (VENTANA - L)]
    n = L // VENTANA
    fr = [seq[i*VENTANA:(i+1)*VENTANA] for i in range(n)]
    r = L - n * VENTANA
    if r > 0:
        fr.append(seq[n*VENTANA:] + "0" * (VENTANA - r))
    return fr


def frag_B(seq):
    """descarta los de < MIN_PB bases reales; los demás se rellenan a 500."""
    seq = seq.upper(); L = len(seq)
    if L < VENTANA:
        return [seq + "0" * (VENTANA - L)] if L >= MIN_PB else []
    n = L // VENTANA
    fr = [seq[i*VENTANA:(i+1)*VENTANA] for i in range(n)]
    r = L - n * VENTANA
    if r >= MIN_PB:
        fr.append(seq[n*VENTANA:] + "0" * (VENTANA - r))
    return fr


def frag_C(seq):
    """último bloque corrido hacia atrás (la que está puesta ahora)."""
    seq = seq.upper(); L = len(seq)
    if L < VENTANA:
        return [seq + "0" * (VENTANA - L)]
    n = L // VENTANA
    fr = [seq[i*VENTANA:(i+1)*VENTANA] for i in range(n)]
    if L % VENTANA:
        fr.append(seq[-VENTANA:])
    return fr


def frag_D(seq):
    """PROPUESTA DE LA ASESORA: descarta < MIN_PB y NO rellena. Largo variable."""
    seq = seq.upper(); L = len(seq)
    if L < VENTANA:
        return [seq] if L >= MIN_PB else []
    n = L // VENTANA
    fr = [seq[i*VENTANA:(i+1)*VENTANA] for i in range(n)]
    r = L - n * VENTANA
    if r >= MIN_PB:
        fr.append(seq[n*VENTANA:])          # tal cual, sin rellenar
    return fr


OPCIONES = [
    ("A  relleno con ceros (original)", frag_A),
    (f"B  descarta <{MIN_PB} pb + padding", frag_B),
    ("C  corrido hacia atrás (actual)", frag_C),
    (f"D  descarta <{MIN_PB} pb SIN padding (asesora)", frag_D),
]

# ---------------------------------------------------------------- datos
def leer_fasta(p):
    d, cur = {}, None
    for l in open(p, encoding="utf-8", errors="replace"):
        l = l.strip()
        if l.startswith(">"):
            cur = l[1:].split()[0]; d[cur] = []
        elif cur:
            d[cur].append(l)
    return {k: "".join(v) for k, v in d.items()}


fa = leer_fasta(FASTA)
lab = pd.read_csv(LABELS, sep="\t")
y_1k = dict(zip(lab.iloc[:, 0].astype(str), lab.iloc[:, 1].astype(int)))
ids = [i for i in fa if i in y_1k]
print(f"\n{len(ids):,} genes del 1k+1k | positivos {sum(y_1k[i] for i in ids):,}")

largos = sorted(len(fa[i]) for i in ids)
print(f"largo de gen: min {largos[0]} | mediana {largos[len(largos)//2]} | max {largos[-1]}")
cortos = sum(1 for i in ids if len(fa[i]) < MIN_PB)
print(f"genes más cortos que {MIN_PB} pb (se perderían con B y D): {cortos}")

# ---------------------------------------------------------------- evaluar
filas = []
for nombre, fn in OPCIONES:
    reales, preds, probs = [], [], []
    sin_frag, n_frag, pb_perdidas = 0, 0, 0
    largos_frag = Counter()
    for gid in ids:
        seq = fa[gid]
        fr = fn(seq)
        n_frag += len(fr)
        pb_perdidas += max(0, len(seq) - sum(min(len(f.replace("0", "")), VENTANA) for f in fr))
        for f in fr:
            largos_frag[len(f.replace("0", "")) // 50 * 50] += 1
        if not fr:
            sin_frag += 1
            continue                      # gen no evaluable
        X = np.array([vector_kmer(f) for f in fr], dtype=np.float32)
        p = rf.predict_proba(X)[:, 1]
        reales.append(y_1k[gid])
        preds.append(int((p >= UMBRAL).any()))     # regla de la etapa 1: algún fragmento
        probs.append(float(p.max()))
    reales, preds, probs = np.array(reales), np.array(preds), np.array(probs)
    cm = confusion_matrix(reales, preds)
    filas.append(dict(
        opcion=nombre,
        recall=recall_score(reales, preds, zero_division=0),
        precision=precision_score(reales, preds, zero_division=0),
        f1=f1_score(reales, preds, zero_division=0),
        auc=roc_auc_score(reales, probs),
        FP=int(cm[0, 1]), FN=int(cm[1, 0]),
        no_eval=sin_frag, frags=n_frag,
    ))

t = pd.DataFrame(filas)
print(f"\n{'='*94}")
print("RESULTADO (nivel gen, regla: algún fragmento >= umbral)")
print(f"{'='*94}")
print(f"  {'opción':<44} {'recall':>7} {'prec.':>7} {'F1':>7} {'AUC':>7} {'FP':>5} {'FN':>5} "
      f"{'no eval':>8} {'frags':>7}")
for _, r in t.iterrows():
    print(f"  {r.opcion:<44} {r.recall:>7.3f} {r.precision:>7.3f} {r.f1:>7.3f} "
          f"{r.auc:>7.3f} {r.FP:>5} {r.FN:>5} {r.no_eval:>8} {r.frags:>7}")

mejor_f1 = t.loc[t.f1.idxmax()]
mejor_auc = t.loc[t.auc.idxmax()]
menos_fp = t.loc[t.FP.idxmin()]
print(f"\n  mejor F1 : {mejor_f1.opcion}  ({mejor_f1.f1:.3f})")
print(f"  mejor AUC: {mejor_auc.opcion}  ({mejor_auc.auc:.3f})")
print(f"  menos FP : {menos_fp.opcion}  ({menos_fp.FP})")
print("\n  El AUC es la medida más limpia para comparar: no depende del umbral.")
if mejor_auc.no_eval:
    print(f"\n  OJO: la opción del mejor AUC deja {int(mejor_auc.no_eval)} gen(es) sin ningún")
    print("  fragmento, o sea no evaluables. En la GUI habría que marcarlos aparte en")
    print("  lugar de contarlos como negativos.")

SAL = RAIZ / "results"
SAL.mkdir(exist_ok=True)
t.to_csv(SAL / f"comparacion_fragmentacion_id{args.identidad}.csv", index=False)
print(f"\nGuardado: {SAL / f'comparacion_fragmentacion_id{args.identidad}.csv'}")
print("\nADVERTENCIA: esto corrige solo la PREDICCIÓN. El modelo sigue entrenado con")
print("fragmentos rellenados, así que con las opciones B y D ve en inferencia una")
print("distribución de fragmentos que no vio al entrenar. El arreglo completo es")
print("regenerar dataset_fragmentado_500.tsv con la opción ganadora y reentrenar.")
