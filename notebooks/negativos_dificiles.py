"""
negativos_dificiles.py
----------------------
Prueba si entrenar con "negativos difíciles" mejora a ARGus (RF, k=6).

Idea: hoy el modelo ve negativos elegidos AL AZAR (undersampling 1:1). Aquí se buscan, DENTRO del
entrenamiento y con validación cruzada por gen, los negativos que el modelo confunde con ARG
(los "difíciles") y se prueba entrenar con más de ellos.

Reglas para que la comparación sea honesta:
  * Los negativos difíciles se detectan SOLO con el entrenamiento (predicciones fuera de muestra).
  * El test interno y el 1k+1k NO participan en ninguna decisión; solo se miden al final.
  * Todas las variantes usan el MISMO corte train/test que el notebook argus_k6_id90 (misma semilla).
  * Cada variante se repite con varias semillas: una mejora solo cuenta si supera la desviación (sd).

Variantes:
  base          negativos al azar, 1:1                       (lo que hay hoy; debe reproducir ~.896/.730/.804)
  dificiles_50  1:1, mitad difíciles + mitad al azar
  dificiles_2a1 2:1, tantos difíciles como positivos + tantos al azar

Uso:
    python negativos_dificiles.py "C:\\ruta\\al\\proyecto" --identidad 90
    python negativos_dificiles.py "/mnt/c/Users/.../Proyecto_Residencias" --semillas 1     (prueba rápida)
Opciones: --semillas 3  --arboles 200  --umbral-dificil 0.5  --guardar dificiles_50
No toca el modelo en uso. Con --guardar crea un modelo NUEVO (argus_rf_k6_id90_<variante>.joblib).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, precision_recall_curve

ap = argparse.ArgumentParser()
ap.add_argument("raiz", nargs="?", default=".")
ap.add_argument("--identidad", type=int, default=90)
ap.add_argument("--semillas", type=int, default=3)
ap.add_argument("--arboles", type=int, default=200)
ap.add_argument("--umbral-dificil", type=float, default=0.5, dest="umbral_dificil")
ap.add_argument("--guardar", default=None, help="nombre de la variante cuyo modelo (semilla 42) se guarda")
a = ap.parse_args()

K, VENT, SEED = 6, 500, 42
RAIZ = Path(a.raiz).expanduser().resolve()
INTERIM = RAIZ / "data" / "interim"
SALIDA = RAIZ / "results" / f"negativos_dificiles_id{a.identidad}"
MODELOS = RAIZ / "data" / "models"
SALIDA.mkdir(parents=True, exist_ok=True)

DATASET_FRAG = INTERIM / "dataset_fragmentado_500.tsv"
SET_1K1K = INTERIM / "muestra_genes_labels.tsv"
FASTA_1K1K = INTERIM / "muestra_genes_completos.fasta"
POS_FILTRO = None if a.identidad == 95 else INTERIM / f"positivo_id{a.identidad}_limpio.fasta"
for p in [DATASET_FRAG, SET_1K1K, FASTA_1K1K] + ([POS_FILTRO] if POS_FILTRO else []):
    if not p.exists():
        sys.exit(f"No encuentro: {p}")

# ------------------------------------------------------------------ utilidades (iguales a los notebooks)
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

def vector_kmer(seq):
    a_ = LUT[np.frombuffer(seq.upper().encode(), dtype=np.uint8)]
    out = np.zeros(4 ** K, dtype=np.float32)
    if len(a_) - K + 1 <= 0:
        return out
    win = np.lib.stride_tricks.sliding_window_view(a_, K)
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

# ------------------------------------------------------------------ datos (misma lógica que argus_k6_id90)
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
    quitar = (pool[col_lab] == 1) & (~pool["gen"].isin(ids_ok))
    pool = pool[~quitar].reset_index(drop=True)

y_all = pool[col_lab].values
g_all = pool["gen"].values
seqs = pool[col_seq].astype(str).values
print(f"Pool: {len(pool):,} fragmentos | pos={int((y_all==1).sum()):,} neg={int((y_all==0).sum()):,} "
      f"| memoria de features ≈ {len(pool)*4**K*4/1e9:.2f} GB")

# mismo corte que los notebooks
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
tr, te = next(sgkf.split(np.zeros(len(y_all)), y_all, g_all))
assert not set(g_all[tr]) & set(g_all[te]), "FUGA train/test"
print(f"Train: {len(tr):,} frag ({len(set(g_all[tr])):,} genes) | Test: {len(te):,} frag ({len(set(g_all[te])):,} genes)")

X = matriz(seqs)
print(f"features listas ({time.time()-t0:.0f}s)")

# 1k+1k (genes completos)
fa = leer_fasta(FASTA_1K1K)
y_1k = dict(zip(lab1k.iloc[:, 0].astype(str).str.split().str[0], lab1k[col_y1].astype(int)))
ids_g = [g for g in fa if g in y_1k]
sq1, gn1 = [], []
for gid in ids_g:
    fr = fragmentar(fa[gid]); sq1 += fr; gn1 += [gid] * len(fr)
gn1 = np.array(gn1)
X1k = matriz(sq1)
y1 = np.array([y_1k[g] for g in ids_g])
print(f"1k+1k: {len(ids_g)} genes ({int(y1.sum())} pos) -> {len(sq1):,} fragmentos")

def agrupar_max(genes, p, orden):
    return pd.Series(p).groupby(genes).max().reindex(orden).values

# ------------------------------------------------------------------ paso 1: negativos difíciles (solo con el entrenamiento)
def entrenar(sel, sem):
    rf = RandomForestClassifier(n_estimators=a.arboles, n_jobs=-1, random_state=sem)
    return rf.fit(X[sel], y_all[sel])

print("\n[1/2] Buscando negativos difíciles con validación cruzada dentro del entrenamiento (5 pliegues por gen)...")
p_oof = np.full(len(y_all), np.nan, dtype=np.float32)
skf2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 7)
for k, (ia_, ib_) in enumerate(skf2.split(np.zeros(len(tr)), y_all[tr], g_all[tr]), 1):
    ia, ib = tr[ia_], tr[ib_]
    rng = np.random.default_rng(SEED)
    pos_, neg_ = ia[y_all[ia] == 1], ia[y_all[ia] == 0]
    sel = np.concatenate([pos_, rng.choice(neg_, size=len(pos_), replace=False)])
    m = entrenar(sel, SEED)
    p_oof[ib] = m.predict_proba(X[ib])[:, 1]
    print(f"  pliegue {k}/5 listo ({time.time()-t0:.0f}s)")
auc_oof = roc_auc_score(y_all[tr], p_oof[tr])
neg_tr = tr[y_all[tr] == 0]
duros = neg_tr[p_oof[neg_tr] >= a.umbral_dificil]
faciles = neg_tr[p_oof[neg_tr] < a.umbral_dificil]
print(f"AUC fuera de muestra (fragmentos, entrenamiento): {auc_oof:.3f}")
print(f"Negativos del entrenamiento: {len(neg_tr):,} | difíciles (p >= {a.umbral_dificil}): {len(duros):,} "
      f"({100*len(duros)/len(neg_tr):.1f} %) | genes con algún fragmento difícil: {len(set(g_all[duros])):,}")

dd = pd.DataFrame({"id": pool[col_id].values[duros], "gen": g_all[duros], "p_oof": p_oof[duros]}).sort_values("p_oof", ascending=False)
dd.to_csv(SALIDA / "negativos_dificiles.csv", index=False)
with open(SALIDA / "negativos_dificiles_top100.fasta", "w") as f:       # para BLAST: no toca el 1k+1k
    for _, r in dd.head(100).iterrows():
        s = pool.loc[pool[col_id] == r["id"], col_seq].iloc[0].rstrip("0")
        f.write(f">{r['id']} p_oof={r['p_oof']:.3f}\n")
        for i in range(0, len(s), 70):
            f.write(s[i:i + 70] + "\n")

# ------------------------------------------------------------------ paso 2: variantes
def armar(variante, sem):
    rng = np.random.default_rng(sem)
    pos = tr[y_all[tr] == 1]; neg = tr[y_all[tr] == 0]; n = len(pos)
    if variante == "base":
        sel = np.concatenate([pos, rng.choice(neg, size=n, replace=False)])
    else:
        n_d, n_f = {"dificiles_50": (n // 2, n - n // 2), "dificiles_2a1": (n, n)}[variante]
        sel = np.concatenate([pos,
                              rng.choice(duros, size=min(n_d, len(duros)), replace=False),
                              rng.choice(faciles, size=min(n_f, len(faciles)), replace=False)])
    rng.shuffle(sel)
    return sel

def prec_a_recall(y, p, r0):
    pr, rc, _ = precision_recall_curve(y, p)
    m = rc[:-1] >= r0 - 1e-9
    return float(pr[:-1][m].max()) if m.any() else np.nan

def evaluar(rf, r0):
    o = {}
    p_te = rf.predict_proba(X[te])[:, 1]
    o["test_frag_auc"] = roc_auc_score(y_all[te], p_te)
    gg = pd.DataFrame({"gen": g_all[te], "y": y_all[te], "p": p_te}).groupby("gen").agg(y=("y", "max"), p=("p", "max"))
    o["test_gen_auc"] = roc_auc_score(gg.y, gg.p)
    pr_ = (gg.p >= 0.5).astype(int)
    o["test_gen_recall"] = recall_score(gg.y, pr_); o["test_gen_prec"] = precision_score(gg.y, pr_, zero_division=0)
    p1 = agrupar_max(gn1, rf.predict_proba(X1k)[:, 1], ids_g)
    pred = (p1 >= 0.5).astype(int)
    o["1k_recall"] = recall_score(y1, pred); o["1k_prec"] = precision_score(y1, pred, zero_division=0)
    o["1k_f1"] = f1_score(y1, pred); o["1k_auc"] = roc_auc_score(y1, p1)
    o["1k_prec_mismo_recall"] = prec_a_recall(y1, p1, r0 if r0 is not None else o["1k_recall"])
    return o

VARIANTES = ["base", "dificiles_50", "dificiles_2a1"]
SEMS = [SEED + i for i in range(a.semillas)]
print(f"\n[2/2] Entrenando {len(VARIANTES)} variantes x {len(SEMS)} semilla(s) = {len(VARIANTES)*len(SEMS)} modelos...")
filas = []
for sem in SEMS:
    r0 = None
    for v in VARIANTES:
        sel = armar(v, sem)
        rf = entrenar(sel, sem)
        o = evaluar(rf, r0)
        if v == "base":
            r0 = o["1k_recall"]
            o["1k_prec_mismo_recall"] = o["1k_prec"]
        filas.append(dict(variante=v, semilla=sem, n_train=len(sel), n_pos=int(y_all[sel].sum()),
                          n_neg=int((y_all[sel] == 0).sum()), **o))
        print(f"  semilla {sem} | {v:14s} | train={len(sel):,} | 1k+1k: recall={o['1k_recall']:.3f} prec={o['1k_prec']:.3f} "
              f"F1={o['1k_f1']:.3f} AUC={o['1k_auc']:.3f} ({time.time()-t0:.0f}s)")
        if a.guardar == v and sem == SEED:
            MODELOS.mkdir(parents=True, exist_ok=True)
            ruta = MODELOS / f"argus_rf_k6_id{a.identidad}_{v}.joblib"
            joblib.dump({"modelo": rf, "K": K, "ventana": VENT, "relleno": "0", "identidad": a.identidad,
                         "umbral": 0.5, "semilla": sem, "sklearn": sklearn.__version__, "variante": v,
                         "entrenado_con": f"pool sin 1k+1k; positivos + negativos ({v})"}, ruta, compress=3)
            print(f"  modelo guardado: {ruta}")

res = pd.DataFrame(filas)
res.to_csv(SALIDA / "resultados_por_semilla.csv", index=False)
cols = ["1k_recall", "1k_prec", "1k_f1", "1k_auc", "1k_prec_mismo_recall", "test_frag_auc", "test_gen_auc"]
mean = res.groupby("variante")[cols].mean().reindex(VARIANTES)
sd = res.groupby("variante")[cols].std().reindex(VARIANTES)
resumen = mean.round(3).astype(str) + " ±" + sd.round(3).fillna(0).astype(str)
mean.join(sd, rsuffix="_sd").round(4).to_csv(SALIDA / "resumen.csv")

print("\n" + "=" * 100)
print("RESUMEN (media ± desviación entre semillas)")
print("1k_*  = 1k+1k, por gen, umbral 0.5 | 1k_prec_mismo_recall = precisión si se fuerza el MISMO recall que 'base'")
print("test_* = test interno (proporción natural), AUC (no depende del umbral)")
print("=" * 100)
print(resumen.to_string())
b = mean.loc["base"]
print("\nDiferencia contra 'base' (positivo = mejor que hoy):")
print((mean - b).drop(index="base").round(3).to_string())
print("\nCómo leerlo:")
print("  - La mejora es REAL solo si la diferencia es mayor que la desviación (±) de ambas filas.")
print("  - Compara con 1k_auc y 1k_prec_mismo_recall: cambiar la proporción de negativos mueve el punto de operación,")
print("    pero NO debería subir el AUC si el modelo no aprendió nada nuevo.")
print(f"\nArchivos en {SALIDA}")
