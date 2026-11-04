"""
verificar_fragmento_degenerado.py  --  comprueba un defecto en la fragmentación y mide el arreglo.

EL PROBLEMA
Dos genes DISTINTOS de 504 pb dieron exactamente p = 0.714. Eso no es casualidad.
Un gen de 504 pb se parte en: fragmento 1 = 500 pb reales, fragmento 2 = 4 pb reales + 496 ceros.
Con K = 6, un fragmento de 4 bases reales NO tiene NINGUNA ventana de 6 letras ACGT,
así que su vector de 4096 frecuencias queda TODO EN CERO.

El Random Forest le asigna a ese vector vacío una probabilidad FIJA. Si esa probabilidad
es mayor que el umbral, entonces TODO gen cuyo último fragmento tenga menos de 6 bases
reales se marca como ARG automáticamente, sin importar su secuencia.

Y el problema es más amplio: un fragmento con 50 bases reales tiene 45 ventanas, así que
cada frecuencia vale 1/45 = 0.022, diez veces más que las ~0.002 de un fragmento completo.
El modelo nunca vio vectores así durante el entrenamiento.

EL ARREGLO
En vez de rellenar con ceros, el último fragmento se corre hacia atrás para que termine
donde termina el gen: seq[-500:]. Así todos los fragmentos tienen 500 bases reales, no se
pierde información y no hay vectores degenerados. Solo los genes de menos de 500 pb
necesitan relleno.

Este script mide las tres opciones con el MISMO modelo ya entrenado (sin reentrenar nada),
así que el resultado se obtiene en minutos.

USO (desde la raíz del proyecto):
    python verificar_fragmento_degenerado.py
    python verificar_fragmento_degenerado.py --identidad 95
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=".")
ap.add_argument("--identidad", default="90")
ap.add_argument("--minreal", type=int, default=100,
                help="mínimo de bases reales para que un fragmento cuente (opción B)")
args = ap.parse_args()

RAIZ = Path(args.raiz).expanduser().resolve()
INTERIM = RAIZ / "data" / "interim"
RES = RAIZ / "results"
RES.mkdir(parents=True, exist_ok=True)
FASTA = INTERIM / "muestra_genes_completos.fasta"
LABELS = INTERIM / "muestra_genes_labels.tsv"
MODELO = RAIZ / "data" / "models" / f"argus_rf_k6_id{args.identidad}.joblib"
for p, q in [(FASTA, "muestra_genes_completos.fasta"), (LABELS, "muestra_genes_labels.tsv"),
             (MODELO, MODELO.name)]:
    if not p.exists():
        sys.exit(f"ERROR: no encontré {q}:\n  {p}")

paq = joblib.load(MODELO)
rf = paq["modelo"] if isinstance(paq, dict) else paq
umbral = float(paq.get("umbral", 0.5)) if isinstance(paq, dict) else 0.5
VENTANA = int(paq.get("ventana", 500)) if isinstance(paq, dict) else 500
K = 1
while 4 ** K < rf.n_features_in_:
    K += 1
try:
    rf.set_params(n_jobs=-1)
except Exception:
    pass

_COD = np.full(256, 4, dtype=np.int8)
for _i, _b in enumerate(b"ACGT"):
    _COD[_b] = _i
_POT = (4 ** np.arange(K - 1, -1, -1)).astype(np.int64)


def vector_kmer(seq):
    a = _COD[np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)]
    out = np.zeros(4 ** K, dtype=np.float32)
    if len(a) - K + 1 <= 0:
        return out
    w = np.lib.stride_tricks.sliding_window_view(a, K)
    ok = (w < 4).all(axis=1)
    if not ok.any():
        return out
    idx = (w[ok].astype(np.int64) * _POT).sum(axis=1)
    c = np.bincount(idx, minlength=4 ** K).astype(np.float32)
    return c / c.sum()


# ===================================================== 1) la prueba directa
print("=" * 74)
print("1) ¿QUÉ PROBABILIDAD LE DA EL MODELO A UN VECTOR VACÍO?")
print("=" * 74)
p_cero = float(rf.predict_proba(np.zeros((1, 4 ** K), dtype=np.float32))[0, 1])
print(f"  K = {K} | umbral = {umbral}")
print(f"  probabilidad del vector TODO EN CERO: {p_cero:.4f}")
if p_cero >= umbral:
    print(f"  *** {p_cero:.3f} >= {umbral}: CONFIRMADO. Todo gen cuyo último fragmento tenga")
    print(f"      menos de {K} bases reales se marca como ARG automáticamente. ***")
else:
    print(f"  {p_cero:.3f} < {umbral}: el vector vacío NO dispara un positivo por sí solo,")
    print("      pero sigue siendo un vector que el modelo nunca vio en entrenamiento.")

print("\n  Probabilidad según cuántas ventanas válidas trae el fragmento:")
rng = np.random.default_rng(0)
for nv in (0, 1, 5, 15, 45, 145, 295, 495):
    v = np.zeros((1, 4 ** K), dtype=np.float32)
    if nv:
        v[0, rng.choice(4 ** K, size=nv, replace=False)] = 1.0 / nv
    print(f"    {nv:>4} ventanas (≈{nv+K-1:>4} pb reales): cada frecuencia {1/nv if nv else 0:.4f}  "
          f"p = {float(rf.predict_proba(v)[0,1]):.3f}")
print(f"\n  Un fragmento completo reparte ~{VENTANA-K+1} ventanas: cada frecuencia ~{1/(VENTANA-K+1):.4f}.")

# ===================================================== 2) cuántos genes se ven afectados
def leer_fasta(path):
    d, cur, ch = {}, None, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for l in f:
            l = l.rstrip("\n")
            if l.startswith(">"):
                if cur is not None:
                    d[cur] = "".join(ch)
                cur = l[1:].strip().split()[0]
                ch = []
            elif cur is not None:
                ch.append(l.strip())
    if cur is not None:
        d[cur] = "".join(ch)
    return d


fa = leer_fasta(FASTA)
lab = pd.read_csv(LABELS, sep="\t")
c_id, c_y = lab.columns[0], next(c for c in lab.columns[1:]
                                 if pd.to_numeric(lab[c], errors="coerce").dropna().isin([0, 1]).all())
lab = lab[[c_id, c_y]].rename(columns={c_id: "gen", c_y: "y"})
lab["gen"] = lab["gen"].astype(str).str.split().str[0]
lab["y"] = lab["y"].astype(int)
ids = [g for g in lab["gen"] if g in fa]
y_true = dict(zip(lab["gen"], lab["y"]))

print("\n" + "=" * 74)
print("2) CUÁNTOS GENES DEL 1k+1k TIENEN UN FRAGMENTO DEGENERADO")
print("=" * 74)
resto = np.array([len(fa[g]) % VENTANA if len(fa[g]) >= VENTANA else len(fa[g]) for g in ids])
largos = np.array([len(fa[g]) for g in ids])
multi = largos >= VENTANA
print(f"  genes de {VENTANA} pb o más (tienen último fragmento con relleno): {int(multi.sum())}")
for a, b in [(1, K - 1), (K, 49), (50, 99), (100, 199), (200, VENTANA - 1)]:
    m = multi & (resto >= a) & (resto <= b)
    et = f"{a}-{b}"
    extra = "  <- VECTOR VACÍO" if b < K else ""
    print(f"    último fragmento con {et:>8} pb reales: {int(m.sum()):>4} genes{extra}")
print(f"    sin relleno (múltiplo exacto de {VENTANA}): {int((multi & (resto == 0)).sum()):>4} genes")

# ===================================================== 3) comparar las tres fragmentaciones
def frag_relleno(seq):
    """la actual: rellena con ceros a la derecha"""
    seq = seq.upper()
    L = len(seq)
    if L < VENTANA:
        return [seq + "0" * (VENTANA - L)]
    n = L // VENTANA
    fr = [seq[i * VENTANA:(i + 1) * VENTANA] for i in range(n)]
    r = L - n * VENTANA
    if r > 0:
        fr.append(seq[n * VENTANA:] + "0" * (VENTANA - r))
    return fr


def frag_descarta(seq):
    """B: igual, pero ignora los fragmentos con poco contenido real"""
    seq = seq.upper()
    L = len(seq)
    if L < VENTANA:
        return [seq + "0" * (VENTANA - L)]
    n = L // VENTANA
    fr = [seq[i * VENTANA:(i + 1) * VENTANA] for i in range(n)]
    r = L - n * VENTANA
    if r >= args.minreal:
        fr.append(seq[n * VENTANA:] + "0" * (VENTANA - r))
    return fr


def frag_corrido(seq):
    """C: el último fragmento se corre hacia atrás; todos con 500 bases reales"""
    seq = seq.upper()
    L = len(seq)
    if L < VENTANA:
        return [seq + "0" * (VENTANA - L)]
    n = L // VENTANA
    fr = [seq[i * VENTANA:(i + 1) * VENTANA] for i in range(n)]
    if L % VENTANA:
        fr.append(seq[-VENTANA:])          # se solapa con el anterior, sin perder el final
    return fr


print("\n" + "=" * 74)
print("3) EL MISMO MODELO CON TRES FORMAS DE FRAGMENTAR (sin reentrenar)")
print("=" * 74)
filas = {}
for nombre, fn in [("A. relleno de ceros (la actual)", frag_relleno),
                   (f"B. descarta fragmentos < {args.minreal} pb reales", frag_descarta),
                   ("C. último fragmento corrido hacia atrás", frag_corrido)]:
    pmax, yv = [], []
    for g in ids:
        fr = fn(fa[g])
        X = np.array([vector_kmer(s) for s in fr], dtype=np.float32)
        pmax.append(float(rf.predict_proba(X)[:, 1].max()))
        yv.append(y_true[g])
    pmax, yv = np.array(pmax), np.array(yv)
    pred = (pmax >= umbral).astype(int)
    VPx = int(((yv == 1) & (pred == 1)).sum())
    FPx = int(((yv == 0) & (pred == 1)).sum())
    FNx = int(((yv == 1) & (pred == 0)).sum())
    r = VPx / max(1, VPx + FNx)
    pr = VPx / max(1, VPx + FPx)
    filas[nombre] = dict(recall=r, precision=pr, f1=2 * r * pr / max(1e-9, r + pr),
                         FP=FPx, FN=FNx, auc=roc_auc_score(yv, pmax))
    print(f"  {nombre}")
    print(f"     recall {r:.3f} | precisión {pr:.3f} | F1 {2*r*pr/max(1e-9,r+pr):.3f} | "
          f"AUC {roc_auc_score(yv,pmax):.3f} | FP {FPx} | FN {FNx}")

base = filas["A. relleno de ceros (la actual)"]
mejor = max(filas.items(), key=lambda kv: kv[1]["f1"])
print(f"\n  Mejor F1: {mejor[0]}  ({mejor[1]['f1']:.3f} contra {base['f1']:.3f} de la actual)")
print(f"  Cambio en falsos positivos: {mejor[1]['FP']-base['FP']:+d}")
print("\n  IMPORTANTE: el modelo se ENTRENÓ con fragmentos rellenados, así que esto arregla")
print("  solo la predicción. El arreglo completo es regenerar dataset_fragmentado_500.tsv")
print("  con la opción C y reentrenar: ahí la ganancia debería ser mayor, porque el modelo")
print("  ya no gastaría capacidad en aprender vectores degenerados.")

pd.DataFrame(filas).T.to_csv(RES / f"fragmentacion_id{args.identidad}.csv")
print(f"\nGuardado: {RES / f'fragmentacion_id{args.identidad}.csv'}")
