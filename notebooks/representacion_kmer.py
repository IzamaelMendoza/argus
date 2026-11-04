"""
representacion_kmer.py
----------------------
Prueba si CAMBIAR LA REPRESENTACIÓN de los fragmentos mejora a ARGus (RF, k=6).

Motivo: el análisis de errores mostró que el modelo se apoya en parte en la COMPOSICIÓN
(los falsos positivos se concentran en los genomas de GC extremo; la "rareza de GC" sola
da AUC 0.633). La frecuencia cruda de 6-meros lleva la composición dentro: un fragmento
rico en AT tiene por fuerza muchos 6-meros ricos en AT.

Representaciones comparadas (mismo RF, mismos datos, mismo corte, mismas semillas):
  cruda    frecuencia del 6-mero                                  (lo que se usa hoy)
  oe       observado / esperado por composición de bases          (quita el sesgo de GC)
  oelog    log2((obs + e0) / (esp + e0))                          (igual, pero más estable)

El esperado de un 6-mero es el producto de las frecuencias de sus 6 bases EN ESE FRAGMENTO.
Dividir entre él deja "cuánto aparece de más respecto a lo que le tocaría por azar".

Reglas (iguales que en negativos_dificiles.py):
  * El 1k+1k se excluye del entrenamiento y solo se mide al final.
  * Corte por gen con la misma semilla que argus_k6_id90.
  * Undersampling 1:1 de negativos, solo en el entrenamiento.
  * Varias semillas: una mejora cuenta solo si supera la desviación.

Uso:
    python representacion_kmer.py "/mnt/c/Users/.../Proyecto_Residencias" --identidad 90 --semillas 3
Opciones: --reps cruda,oe,oelog   --arboles 200   --guardar oe
Las representaciones se calculan DE UNA EN UNA para no llenar la RAM.
"""
import argparse, sys, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("raiz", nargs="?", default=".")
ap.add_argument("--identidad", type=int, default=90)
ap.add_argument("--semillas", type=int, default=3)
ap.add_argument("--arboles", type=int, default=200)
ap.add_argument("--reps", default="cruda,oe,oelog")
ap.add_argument("--guardar", default=None, help="representación cuyo modelo (semilla 42) se guarda")
a = ap.parse_args()

K, VENT, SEED = 6, 500, 42
RAIZ = Path(a.raiz).expanduser().resolve()
INTERIM = RAIZ / "data" / "interim"
SALIDA = RAIZ / "results" / f"representacion_id{a.identidad}"
MODELOS = RAIZ / "data" / "models"
SALIDA.mkdir(parents=True, exist_ok=True)

DATASET_FRAG = INTERIM / "dataset_fragmentado_500.tsv"
SET_1K1K = INTERIM / "muestra_genes_labels.tsv"
FASTA_1K1K = INTERIM / "muestra_genes_completos.fasta"
POS_FILTRO = None if a.identidad == 95 else INTERIM / f"positivo_id{a.identidad}_limpio.fasta"
for p in [DATASET_FRAG, SET_1K1K, FASTA_1K1K] + ([POS_FILTRO] if POS_FILTRO else []):
    if not p.exists():
        sys.exit(f"No encuentro: {p}")

# ------------------------------------------------------------------ utilidades
def leer_fasta(path):
    d, cur = {}, None
    for l in open(path):
        l = l.strip()
        if l.startswith(">"):
            cur = l[1:].split()[0]; d[cur] = []
        elif cur:
            d[cur].append(l)
    return {k: "".join(v) for k, v in d.items()}

def fragmentar(seq, ventana=VENT):
    seq = seq.upper(); L = len(seq)
    if L < ventana:
        return [seq + "0" * (ventana - L)]
    n = L // ventana
    fr = [seq[i * ventana:(i + 1) * ventana] for i in range(n)]
    r = L - n * ventana
    if r > 0:
        fr.append(seq[n * ventana:] + "0" * (ventana - r))
    return fr

LUT = np.full(256, 4, dtype=np.int8)
for i, b in enumerate(b"ACGT"):
    LUT[b] = i
POT = (4 ** np.arange(K - 1, -1, -1)).astype(np.int64)
D = 4 ** K

def _conteos(seq):
    """Devuelve (cuentas de k-meros, frecuencias de las 4 bases) usando solo ventanas ACGT."""
    a_ = LUT[np.frombuffer(seq.upper().encode(), dtype=np.uint8)]
    if len(a_) - K + 1 <= 0:
        return None, None
    win = np.lib.stride_tricks.sliding_window_view(a_, K)
    ok = (win < 4).all(axis=1)
    if not ok.any():
        return None, None
    val = win[ok]
    cuentas = np.bincount((val.astype(np.int64) * POT).sum(axis=1), minlength=D).astype(np.float32)
    # composición de bases medida sobre las MISMAS ventanas válidas
    pb = np.bincount(val.ravel(), minlength=4).astype(np.float64)
    return cuentas, pb / pb.sum()

def _esperado(pb):
    """Frecuencia esperada de cada k-mero = producto de las frecuencias de sus bases."""
    e = pb
    for _ in range(K - 1):
        e = np.outer(e, pb).ravel()          # mismo orden de índices que POT (base 4)
    return e

def vector(seq, rep):
    out = np.zeros(D, dtype=np.float32)
    cuentas, pb = _conteos(seq)
    if cuentas is None:
        return out
    n = cuentas.sum()
    obs = cuentas / n
    if rep == "cruda":
        return obs
    esp = _esperado(pb).astype(np.float32)
    if rep == "oe":
        return (obs / (esp + 1e-9)).astype(np.float32)
    if rep == "oelog":
        e0 = np.float32(1.0 / n)             # pseudoconteo = una aparición
        return np.log2((obs + e0) / (esp + e0)).astype(np.float32)
    raise ValueError(rep)

def matriz(seqs, rep):
    X = np.empty((len(seqs), D), dtype=np.float32)
    for i, s in enumerate(seqs):
        X[i] = vector(s, rep)
    return X

# ------------------------------------------------------------------ datos (igual que argus_k6_id90)
t0 = time.time()
frag = pd.read_csv(DATASET_FRAG, sep="\t")
col_id, col_seq, col_lab = "id", "lectura", "etiqueta"
assert {col_id, col_seq, col_lab} <= set(frag.columns), f"columnas: {list(frag.columns)}"
frag[col_lab] = frag[col_lab].astype(int)
frag["gen"] = frag[col_id].str.replace(r"__frag\d+$", "", regex=True)

lab1k = pd.read_csv(SET_1K1K, sep="\t")
col_y1 = next(c for c in lab1k.columns[1:] if pd.to_numeric(lab1k[c], errors="coerce").dropna().isin([0, 1]).all())
ids_1k = set(lab1k.iloc[:, 0].astype(str).str.split().str[0])
genes_frag = set(frag["gen"])
excluir = ids_1k & genes_frag
faltan = ids_1k - excluir
if faltan:
    variantes = {}
    for g in genes_frag:
        variantes.setdefault(g.split("|", 1)[-1], g)
    for f in list(faltan):
        if f in variantes:
            excluir.add(variantes[f]); faltan.discard(f)
assert not faltan, f"No encontré genes del 1k+1k: {list(faltan)[:5]}"
pool = frag[~frag["gen"].isin(excluir)].reset_index(drop=True)

if POS_FILTRO is not None:
    hdrs = [l[1:].strip() for l in open(POS_FILTRO) if l.startswith(">")]
    ids_ok = set(hdrs) | {h.split()[0] for h in hdrs}
    pos_genes = set(pool.loc[pool[col_lab] == 1, "gen"])
    esperadas = [h for h in hdrs if h not in excluir and h.split()[0] not in excluir]
    casan = [h for h in esperadas if h in pos_genes or h.split()[0] in pos_genes]
    assert len(casan) >= 0.9 * len(esperadas), "Los ids del FASTA filtrado no casan con el dataset"
    pool = pool[~((pool[col_lab] == 1) & (~pool["gen"].isin(ids_ok)))].reset_index(drop=True)

y_all = pool[col_lab].values
g_all = pool["gen"].values
seqs = pool[col_seq].astype(str).values
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
tr, te = next(sgkf.split(np.zeros(len(y_all)), y_all, g_all))
assert not set(g_all[tr]) & set(g_all[te]), "FUGA train/test"
print(f"Pool: {len(pool):,} fragmentos | pos={int((y_all==1).sum()):,} neg={int((y_all==0).sum()):,}")
print(f"Train: {len(tr):,} frag ({len(set(g_all[tr])):,} genes) | Test: {len(te):,} frag ({len(set(g_all[te])):,} genes)")

fa = leer_fasta(FASTA_1K1K)
y_1k = dict(zip(lab1k.iloc[:, 0].astype(str).str.split().str[0], lab1k[col_y1].astype(int)))
ids_g = [g for g in fa if g in y_1k]
sq1, gn1 = [], []
for gid in ids_g:
    fr = fragmentar(fa[gid]); sq1 += fr; gn1 += [gid] * len(fr)
gn1 = np.array(gn1)
y1 = np.array([y_1k[g] for g in ids_g])
print(f"1k+1k: {len(ids_g)} genes ({int(y1.sum())} pos) -> {len(sq1):,} fragmentos")

# comprobación rápida de la transformación
_demo = "ACGT" * 125
for r in ("cruda", "oe", "oelog"):
    v = vector(_demo, r)
    print(f"  [{r:5s}] suma={v.sum():.3f} min={v.min():.3f} max={v.max():.3f}")

# ------------------------------------------------------------------ experimento
REPS = [r.strip() for r in a.reps.split(",") if r.strip()]
SEMS = [SEED + i for i in range(a.semillas)]
print(f"\nEntrenando {len(REPS)} representación(es) x {len(SEMS)} semilla(s) = {len(REPS)*len(SEMS)} modelos\n")

filas = []
for rep in REPS:
    t1 = time.time()
    X = matriz(seqs, rep)
    X1k = matriz(sq1, rep)
    print(f"[{rep}] características listas ({time.time()-t1:.0f}s, {X.nbytes/1e9:.2f} GB)")
    for sem in SEMS:
        rng = np.random.default_rng(sem)
        pos = tr[y_all[tr] == 1]; neg = tr[y_all[tr] == 0]
        sel = np.concatenate([pos, rng.choice(neg, size=len(pos), replace=False)])
        rng.shuffle(sel)
        rf = RandomForestClassifier(n_estimators=a.arboles, n_jobs=-1, random_state=sem).fit(X[sel], y_all[sel])

        p_te = rf.predict_proba(X[te])[:, 1]
        gg = pd.DataFrame({"gen": g_all[te], "y": y_all[te], "p": p_te}).groupby("gen").agg(y=("y", "max"), p=("p", "max"))
        p1 = pd.Series(rf.predict_proba(X1k)[:, 1]).groupby(gn1).max().reindex(ids_g).values
        pred = (p1 >= 0.5).astype(int)
        o = dict(rep=rep, semilla=sem,
                 test_frag_auc=roc_auc_score(y_all[te], p_te), test_gen_auc=roc_auc_score(gg.y, gg.p),
                 recall=recall_score(y1, pred), precision=precision_score(y1, pred, zero_division=0),
                 f1=f1_score(y1, pred), auc=roc_auc_score(y1, p1))
        filas.append(o)
        print(f"  semilla {sem} | {rep:6s} | 1k+1k: recall={o['recall']:.3f} prec={o['precision']:.3f} "
              f"F1={o['f1']:.3f} AUC={o['auc']:.3f} | test_frag_auc={o['test_frag_auc']:.3f} ({time.time()-t0:.0f}s)")
        if a.guardar == rep and sem == SEED:
            MODELOS.mkdir(parents=True, exist_ok=True)
            ruta = MODELOS / f"argus_rf_k6_id{a.identidad}_{rep}.joblib"
            joblib.dump({"modelo": rf, "K": K, "ventana": VENT, "relleno": "0", "identidad": a.identidad,
                         "umbral": 0.5, "semilla": sem, "sklearn": sklearn.__version__, "representacion": rep,
                         "AVISO": "la GUI debe usar la MISMA representación para featurizar"}, ruta, compress=3)
            print(f"  modelo guardado: {ruta}")
    del X, X1k
    gc.collect()

res = pd.DataFrame(filas)
res.to_csv(SALIDA / "resultados_por_semilla.csv", index=False)
cols = ["recall", "precision", "f1", "auc", "test_frag_auc", "test_gen_auc"]
mean, sd = res.groupby("rep")[cols].mean().reindex(REPS), res.groupby("rep")[cols].std().reindex(REPS)
mean.join(sd, rsuffix="_sd").round(4).to_csv(SALIDA / "resumen.csv")

print("\n" + "=" * 92)
print("RESUMEN (media ± desviación entre semillas) · 1k+1k por gen, umbral 0.5")
print("=" * 92)
print((mean.round(3).astype(str) + " ±" + sd.round(3).fillna(0).astype(str)).to_string())
if "cruda" in mean.index:
    print("\nDiferencia contra 'cruda' (positivo = mejor):")
    print((mean - mean.loc["cruda"]).drop(index="cruda").round(3).to_string())
    print("\nLo que importa es el AUC: si sube MÁS que la desviación, la nueva representación")
    print("aporta información nueva y no solo mueve el punto de operación.")
print(f"\nArchivos en {SALIDA}")
