"""
tarea3_sobremuestreo.py -- Tarea 3: sobremuestreo vs class_weight vs ambos,
sobre el modelo de FAMILIAS con DNABERT-6.

LA IDEA
-------
Las 12 familias estan muy desbalanceadas: 291 genes la mas grande y 30 la mas
chica (9.7 a 1). Hay tres formas de atacarlo y la asesora quiere ver las tres:

  1. SOBREMUESTREO   replicar los genes de las familias chicas hasta subirlas
  2. CLASS_WEIGHT    pesar cada familia al inverso de su tamano en la perdida
                     (esto YA esta en familias_dnabert.py, viene de su codigo)
  3. AMBOS

POR QUE ESTE SCRIPT ES BARATO
-----------------------------
Lo lento de DNABERT es calcular los embeddings (~2.3 fragmentos/s en CPU). Pero
el sobremuestreo y el class_weight NO cambian los embeddings: solo cambian como
se entrena la cabeza MLP, que tarda segundos. Asi que se calculan los embeddings
UNA vez y de ahi salen todas las variantes.

Eso deja margen para correr cada variante con varias semillas, que es lo unico
que distingue una mejora real del ruido. Con ~200 genes de test una diferencia
de 0.03 en balanced accuracy puede ser puro azar.

VARIANTES QUE CORRE
-------------------
  nada          ni class_weight ni sobremuestreo   (la referencia de abajo)
  cw            solo class_weight                  (lo que ya teniamos)
  os_<n>        solo sobremuestreo, a nivel n
  cw_os_<n>     ambos
donde n es el objetivo de genes por familia (por defecto 291, 93 y 58).

EL SOBREMUESTREO, EN DETALLE
----------------------------
  - Se hace SOLO sobre el train. El val y el test NUNCA se tocan: si se
    replicaran ahi, el mismo gen estaria varias veces en la evaluacion y el
    numero saldria inflado.
  - El objetivo se cuenta en GENES (291, 93, 58), que es como esta la tabla de
    la asesora, y se traduce a fragmentos con el factor genes_objetivo/genes_que_hay.
  - Solo SUBE. Una familia que ya esta por encima del objetivo se queda igual
    (no hay submuestreo).
  - Replica con reemplazo, que es el RandomOverSampler de siempre.

Uso:
    python scripts/tarea3_sobremuestreo.py --demo      # prueba la tuberia en 1 min
    python scripts/tarea3_sobremuestreo.py --completo  # la corrida real
    python scripts/tarea3_sobremuestreo.py --completo --semillas 10
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--completo", action="store_true", help="corrida real")
ap.add_argument("--demo", action="store_true",
                help="embeddings falsos: prueba toda la tuberia sin DNABERT, en ~1 min")
ap.add_argument("--modelo", default="zhihan1996/DNA_bert_6")
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--ventana", type=int, default=500)
ap.add_argument("--min-pb", type=int, default=150)
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--epocas", type=int, default=80)
ap.add_argument("--paciencia", type=int, default=12)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--semillas", type=int, default=5,
                help="cuantas semillas por variante (para ver si la diferencia es ruido)")
ap.add_argument("--objetivos", default="291,93,58",
                help="niveles de sobremuestreo, en GENES por familia")
args = ap.parse_args()

K, VENTANA, MIN_PB, SEED = args.k, args.ventana, args.min_pb, args.seed
PRUEBA = not (args.completo or args.demo)
OBJETIVOS = [int(x) for x in args.objetivos.split(",") if x.strip()]

# ----------------------------------------------------------------- rutas
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

ETQ = RAIZ / "results" / "familias" / "etiquetas_finales.tsv"
FASTA = RAIZ / "data" / "processed" / "argdb_POSITIVO_final.fasta"
for p, q in [(ETQ, "etiquetas_finales.tsv"), (FASTA, "argdb_POSITIVO_final.fasta")]:
    if not p.exists():
        sys.exit(f"Falta {q}: {p}")

CACHE = RAIZ / "data" / "models" / "dnabert6_emb_familias"
CACHE.mkdir(parents=True, exist_ok=True)
SAL = RAIZ / "results" / "tarea3"
SAL.mkdir(parents=True, exist_ok=True)

print("=" * 96)
print("TAREA 3 -- sobremuestreo vs class_weight vs ambos (familias, DNABERT-6)")
print(f"objetivos de sobremuestreo: {OBJETIVOS} genes por familia")
print(f"semillas por variante: {args.semillas}")
if args.demo:
    print("MODO DEMO: embeddings FALSOS. Solo sirve para ver que la tuberia corre.")
elif PRUEBA:
    print("MODO PRUEBA RAPIDA: muestra reducida. Usa --completo para la real.")
print("=" * 96)

# ----------------------------------------------------------------- dependencias
try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("Falta PyTorch.  pip install torch --index-url https://download.pytorch.org/whl/cpu")
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

torch.manual_seed(SEED)
np.random.seed(SEED)

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
    """Regla de la asesora (5-nov): sin padding, descarta el resto de < min_pb."""
    seq = seq.upper()
    L = len(seq)
    if L < ventana:
        return [seq] if L >= min_pb else []
    n = L // ventana
    fr = [seq[i * ventana:(i + 1) * ventana] for i in range(n)]
    if L - n * ventana >= min_pb:
        fr.append(seq[n * ventana:])
    return fr


def sequence_to_kmers(seq, k=K):
    """La funcion del codigo de la asesora."""
    seq = seq.upper().replace("0", "")
    seq = re.sub(r"[^ACGT]", "N", seq)
    if len(seq) < k:
        return "N" * k
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


seqs, ys, gs, sin_frag = [], [], [], []
for gen, lab in etq.items():
    sq = fa.get(gen)
    if not sq:
        continue
    fr = fragmentar(sq)
    if not fr:
        sin_frag.append((gen, lab, len(sq)))
        continue
    for f in fr:
        seqs.append(f); ys.append(lab); gs.append(gen)
y = np.array(ys)
g = np.array(gs)
clases = sorted(set(y))
y_idx = np.array([clases.index(v) for v in y])

if sin_frag:
    print(f"\n[aviso] {len(sin_frag)} gen(es) de menos de {MIN_PB} pb quedan fuera")

genes_por_fam = Counter({gg: yy for gg, yy in zip(g, y)}.values())
print(f"\n{len(set(g)):,} genes -> {len(seqs):,} fragmentos | {len(clases)} familias")
print(f"\n{'familia':<42} {'genes':>7} {'frags':>8}")
frags_por_fam = Counter(y.tolist())
for fam, n in genes_por_fam.most_common():
    print(f"{str(fam)[:42]:<42} {n:>7,} {frags_por_fam[fam]:>8,}")
MAYOR = genes_por_fam.most_common(1)[0]
print(f"\nla mas grande: {MAYOR[0]} ({MAYOR[1]}) | la mas chica: "
      f"{genes_por_fam.most_common()[-1][0]} ({genes_por_fam.most_common()[-1][1]})")
print(f"desbalance: {MAYOR[1]/genes_por_fam.most_common()[-1][1]:.1f} a 1")

# ----------------------------------------------------------------- cortes
n_splits = max(2, min(5, min(genes_por_fam.values())))
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
tr_full, te = next(sgkf.split(np.zeros(len(y)), y, g))
n_val = max(2, min(5, min(Counter({gg: yy for gg, yy in zip(g[tr_full], y[tr_full])}
                                  .values()).values())))
sgkf2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=SEED + 7)
rel_tr, rel_va = next(sgkf2.split(np.zeros(len(tr_full)), y[tr_full], g[tr_full]))
tr, va = tr_full[rel_tr], tr_full[rel_va]
for a, b, nom in [(tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")]:
    assert not set(g[a]) & set(g[b]), f"FUGA entre {nom}"
print(f"\ntrain {len(tr):,} frag ({len(set(g[tr])):,} genes) | "
      f"val {len(va):,} ({len(set(g[va])):,}) | test {len(te):,} ({len(set(g[te])):,})")

if PRUEBA:
    rng = np.random.default_rng(SEED)
    tr = rng.choice(tr, size=min(300, len(tr)), replace=False)
    va = rng.choice(va, size=min(120, len(va)), replace=False)
    te = rng.choice(te, size=min(160, len(te)), replace=False)
    print(f"  (prueba rapida: train {len(tr)} | val {len(va)} | test {len(te)})")

# ----------------------------------------------------------------- embeddings
if args.demo:
    # embeddings falsos con estructura de clase: cada familia tiene un centro
    # distinto y las chicas tienen mas ruido. Solo para probar la tuberia.
    print("\n[DEMO] generando embeddings falsos (NO son DNABERT)...")
    DIM = 768
    rng = np.random.default_rng(SEED)
    centros = rng.normal(0, 1, (len(clases), DIM)).astype(np.float32)
    E_all = np.empty((len(seqs), DIM), dtype=np.float32)
    for i in range(len(seqs)):
        c = y_idx[i]
        # ruido alto y mayor en las familias chicas, para que el problema no sea
        # trivial y se note si el sobremuestreo ayuda a las minoritarias
        ruido = 7.0 + 9.0 * (1 - genes_por_fam[clases[c]] / MAYOR[1])
        E_all[i] = centros[c] + rng.normal(0, ruido, DIM).astype(np.float32)
    E_tr, E_va, E_te = E_all[tr], E_all[va], E_all[te]
else:
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        sys.exit("Falta transformers.  pip install 'transformers==4.30.2'")
    print(f"\nCargando {args.modelo} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.modelo)
    bert = AutoModel.from_pretrained(args.modelo).eval()   # sin trust_remote_code
    cfg = bert.config
    print(f"  listo en {time.time()-t0:.0f}s | capas {cfg.num_hidden_layers} | "
          f"hidden {cfg.hidden_size} | vocab {cfg.vocab_size}")
    DIM = int(cfg.hidden_size)

    @torch.inference_mode()
    def codificar(lista):
        out = np.empty((len(lista), DIM), dtype=np.float32)
        for i in range(0, len(lista), args.batch):
            lote = [sequence_to_kmers(s) for s in lista[i:i + args.batch]]
            enc = tok(lote, padding=True, truncation=True, max_length=args.max_len,
                      return_tensors="pt")
            hs = bert(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out[i:i + len(lote)] = ((hs * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy()
        return out

    def embeddings(idxs, nombre):
        firma = hashlib.md5(("|".join(seqs[i] for i in idxs)).encode()).hexdigest()[:10]
        d = CACHE / f"{nombre}_{len(idxs)}_{firma}"
        d.mkdir(exist_ok=True)
        partes, t0 = [], time.time()
        for b0 in range(0, len(idxs), args.chunk):
            f = d / f"{b0:07d}.npy"
            if f.exists():
                partes.append(np.load(f)); continue
            e = codificar([seqs[i] for i in idxs[b0:b0 + args.chunk]])
            np.save(f, e); partes.append(e)
            hechos = min(b0 + args.chunk, len(idxs))
            vel = hechos / max(1e-9, time.time() - t0)
            print(f"\r  {nombre}: {hechos}/{len(idxs)}  ({vel:.1f} frag/s, "
                  f"faltan ~{(len(idxs)-hechos)/max(vel,1e-9)/60:.1f} min)   ",
                  end="", flush=True)
        print()
        return np.vstack(partes)

    print("\nCalculando embeddings (la parte lenta; se cachea en disco):")
    E_tr, E_va, E_te = (embeddings(tr, "train"), embeddings(va, "val"),
                        embeddings(te, "test"))

print(f"  train {E_tr.shape} | val {E_va.shape} | test {E_te.shape}")

# normalizacion: mu y sd SOLO del train
mu, sd = E_tr.mean(0, keepdims=True), E_tr.std(0, keepdims=True) + 1e-6
Xtr_t = torch.tensor((E_tr - mu) / sd)
Xva_t = torch.tensor((E_va - mu) / sd)
Xte_t = torch.tensor((E_te - mu) / sd)
ytr_np, yva_np, yte_np = y_idx[tr], y_idx[va], y_idx[te]
g_te = g[te]


# =============================================================================
#  SOBREMUESTREO
# =============================================================================
def indices_sobremuestreados(y_sub, genes_sub, objetivo, semilla):
    """
    Devuelve los indices (dentro del train) despues de replicar las familias
    chicas. El objetivo se cuenta en GENES y se traduce a fragmentos.

    Solo SUBE: una familia que ya tiene mas genes que el objetivo no se toca.
    """
    rng = np.random.default_rng(semilla)
    # OJO: y_sub trae INDICES de clase (0..n-1), no los nombres, asi que este
    # Counter queda con llaves numericas y hay que consultarlo con el indice.
    genes_fam = Counter({gg: yy for gg, yy in zip(genes_sub, y_sub)}.values())
    out = []
    for c in range(len(clases)):
        pos = np.flatnonzero(y_sub == c)
        if len(pos) == 0:
            continue
        n_genes = genes_fam.get(c, 0)
        out.append(pos)
        if n_genes and n_genes < objetivo:
            factor = objetivo / n_genes
            extra = int(round(len(pos) * (factor - 1)))
            if extra > 0:
                out.append(rng.choice(pos, size=extra, replace=True))
    idx = np.concatenate(out)
    rng.shuffle(idx)
    return idx


def pesos_clase(y_sub):
    """class_weight 'balanced', tal como en el codigo de la asesora."""
    cuenta = np.bincount(y_sub, minlength=len(clases)).astype(np.float64)
    w = np.where(cuenta > 0, cuenta.sum() / (len(clases) * np.maximum(cuenta, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def entrenar(usar_cw, objetivo, semilla):
    """Entrena una cabeza y devuelve sus metricas a nivel gen en el test."""
    torch.manual_seed(semilla)
    if objetivo:
        idx = indices_sobremuestreados(ytr_np, g[tr], objetivo, semilla)
    else:
        idx = np.arange(len(ytr_np))
    Xb = Xtr_t[idx]
    yb = torch.tensor(ytr_np[idx], dtype=torch.long)

    perdida = nn.CrossEntropyLoss(weight=pesos_clase(ytr_np[idx]) if usar_cw else None)
    cabeza = nn.Sequential(nn.Linear(DIM, 256), nn.ReLU(), nn.Dropout(0.3),
                           nn.Linear(256, len(clases)))
    opt = torch.optim.Adam(cabeza.parameters(), lr=args.lr, weight_decay=1e-4)

    mejor, sin_mejora, estado = -1.0, 0, None
    for ep in range(args.epocas):
        cabeza.train()
        perm = torch.randperm(len(Xb))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            perdida(cabeza(Xb[b]), yb[b]).backward()
            opt.step()
        cabeza.eval()
        with torch.no_grad():
            pv = cabeza(Xva_t).argmax(1).numpy()
        # checkpoint por BALANCED ACCURACY de val, no por val_loss (decision de la asesora)
        bal = balanced_accuracy_score(yva_np, pv)
        if bal > mejor:
            mejor, sin_mejora = bal, 0
            estado = {k: v.clone() for k, v in cabeza.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                break
    if estado:
        cabeza.load_state_dict(estado)
    cabeza.eval()

    with torch.no_grad():
        prob = torch.softmax(cabeza(Xte_t), dim=1).numpy()
    # nivel gen: promedio de probabilidades (un gen tiene UNA familia)
    por_gen = defaultdict(list)
    for i, gg in enumerate(g_te):
        por_gen[gg].append(i)
    gr, gp = [], []
    for gg, idxs in por_gen.items():
        gr.append(yte_np[idxs[0]])
        gp.append(int(np.argmax(prob[idxs].mean(axis=0))))
    gr, gp = np.array(gr), np.array(gp)
    return dict(bal=balanced_accuracy_score(gr, gp),
                f1=f1_score(gr, gp, average="macro", zero_division=0),
                exac=float((gp == gr).mean()),
                val=mejor, n_train=len(idx), gr=gr, gp=gp)


# =============================================================================
#  CORRER LAS VARIANTES
# =============================================================================
VARIANTES = [("nada", False, None), ("cw", True, None)]
for o in OBJETIVOS:
    VARIANTES.append((f"os_{o}", False, o))
for o in OBJETIVOS:
    VARIANTES.append((f"cw_os_{o}", True, o))

ETIQ = {"nada": "sin nada (referencia)", "cw": "class_weight solo"}
for o in OBJETIVOS:
    ETIQ[f"os_{o}"] = f"sobremuestreo a {o}"
    ETIQ[f"cw_os_{o}"] = f"ambos (cw + sobre. {o})"

print("\n" + "=" * 96)
print(f"ENTRENANDO {len(VARIANTES)} variantes x {args.semillas} semillas "
      f"= {len(VARIANTES)*args.semillas} cabezas")
print("(los embeddings ya estan: esto es lo rapido)")
print("=" * 96)

res = defaultdict(list)
mejor_corrida = {}
t0 = time.time()
for nom, cw, obj in VARIANTES:
    for s in range(args.semillas):
        r = entrenar(cw, obj, SEED + s * 101)
        res[nom].append(r)
        if nom not in mejor_corrida or r["bal"] > mejor_corrida[nom]["bal"]:
            mejor_corrida[nom] = r
    b = np.array([x["bal"] for x in res[nom]])
    print(f"  {ETIQ[nom]:<28} bal.acc {b.mean():.4f} +/- {b.std(ddof=1) if len(b)>1 else 0:.4f}"
          f"   (train: {res[nom][0]['n_train']:,} frag)")
print(f"\n  {time.time()-t0:.0f}s en total")

# =============================================================================
#  RESULTADOS
# =============================================================================
print("\n" + "=" * 96)
print("RESULTADOS (test a nivel gen, promedio de probabilidades entre fragmentos)")
print("=" * 96)
print(f"\n{'variante':<28} {'bal.acc':>9} {'sd':>7} {'F1 macro':>10} {'exactitud':>10} "
      f"{'frags tr':>10}")
tabla = []
for nom, _, _ in VARIANTES:
    b = np.array([x["bal"] for x in res[nom]])
    f = np.array([x["f1"] for x in res[nom]])
    e = np.array([x["exac"] for x in res[nom]])
    sd_b = b.std(ddof=1) if len(b) > 1 else 0.0
    print(f"{ETIQ[nom]:<28} {b.mean():>9.4f} {sd_b:>7.4f} {f.mean():>10.4f} "
          f"{e.mean():>10.4f} {res[nom][0]['n_train']:>10,}")
    tabla.append(dict(variante=nom, etiqueta=ETIQ[nom], bal_acc=b.mean(), sd=sd_b,
                      f1_macro=f.mean(), exactitud=e.mean(),
                      frags_train=res[nom][0]["n_train"], semillas=len(b)))

base = np.array([x["bal"] for x in res["cw"]])
print(f"\n{'-'*96}")
print("CONTRA class_weight solo (que es lo que ya teniamos)")
print(f"{'-'*96}")
print(f"\n{'variante':<28} {'dif':>9} {'t':>8} {'veredicto'}")
for nom, _, _ in VARIANTES:
    if nom == "cw":
        continue
    b = np.array([x["bal"] for x in res[nom]])
    d = b - base                     # comparacion PAREADA: misma semilla
    m = d.mean()
    if len(d) > 1 and d.std(ddof=1) > 0:
        sem = d.std(ddof=1) / np.sqrt(len(d))      # error estandar de la MEDIA
        t = m / sem
        # t critico de dos colas al 5% con len(d)-1 grados de libertad
        tc = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
              7: 2.365, 8: 2.306, 9: 2.262}.get(len(d) - 1, 2.0)
        if abs(t) > tc:
            v = "mejora consistente" if m > 0 else "empeora consistente"
        else:
            v = "dentro del ruido"
    else:
        t, v = float("nan"), "hace falta mas de una semilla"
    print(f"{ETIQ[nom]:<28} {m:>+9.4f} {t:>8.2f}   {v}")

mejor = max(tabla, key=lambda r: r["bal_acc"])
print(f"\n  mejor promedio: {mejor['etiqueta']} ({mejor['bal_acc']:.4f})")
d_mej = np.array([x["bal"] for x in res[mejor["variante"]]]) - base
if mejor["variante"] != "cw" and len(d_mej) > 1:
    sem = d_mej.std(ddof=1) / np.sqrt(len(d_mej))
    if abs(d_mej.mean()) < 2 * sem:
        print(f"  PERO la diferencia contra class_weight ({d_mej.mean():+.4f}) esta")
        print(f"  dentro del ruido: no se puede afirmar que sea mejor.")

import pandas as pd
pd.DataFrame(tabla).to_csv(SAL / "tarea3_comparacion.csv", index=False)
filas = [dict(variante=n, semilla=i, bal_acc=x["bal"], f1_macro=x["f1"],
              exactitud=x["exac"], val_bal_acc=x["val"])
         for n, _, _ in VARIANTES for i, x in enumerate(res[n])]
pd.DataFrame(filas).to_csv(SAL / "tarea3_corridas.csv", index=False)

# --- por familia, para la mejor variante y para cw ---
print(f"\n{'-'*96}")
print(f"ACIERTO POR FAMILIA  (la corrida mediana de cada variante)")
print(f"{'-'*96}")
def acierto_por_clase(r):
    out = {}
    for c in range(len(clases)):
        m = r["gr"] == c
        out[clases[c]] = float((r["gp"][m] == c).mean()) if m.any() else float("nan")
    return out

# el panel por familia compara class_weight contra la mejor variante QUE NO SEA
# class_weight: si la mejor resulta ser cw misma, compararla consigo misma no
# dice nada
rival = max((r for r in tabla if r["variante"] != "cw"), key=lambda r: r["bal_acc"])
a_cw = acierto_por_clase(mejor_corrida["cw"])
a_mj = acierto_por_clase(mejor_corrida[rival["variante"]])
print(f"\n{'familia':<42} {'genes':>6} {'cw':>8} {rival['variante']:>14} {'dif':>8}")
for fam, n in genes_por_fam.most_common():
    c1, c2 = a_cw.get(fam, float("nan")), a_mj.get(fam, float("nan"))
    dif = c2 - c1 if not (np.isnan(c1) or np.isnan(c2)) else float("nan")
    print(f"{str(fam)[:42]:<42} {n:>6} {c1:>8.3f} {c2:>12.3f} {dif:>+8.3f}")

# =============================================================================
#  GRAFICA
# =============================================================================
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))

noms = [n for n, _, _ in VARIANTES]
etqs = [ETIQ[n] for n in noms]
med = [np.mean([x["bal"] for x in res[n]]) for n in noms]
err = [np.std([x["bal"] for x in res[n]], ddof=1) if args.semillas > 1 else 0
       for n in noms]
col = []
for n in noms:
    if n == "nada":
        col.append("#8C8C8C")
    elif n == "cw":
        col.append("#4C72B0")
    elif n.startswith("cw_os"):
        col.append("#55A868")
    else:
        col.append("#DD8452")

b = ax[0].bar(range(len(noms)), med, yerr=err, capsize=4, color=col)
ax[0].axhline(np.mean(base), color="#4C72B0", ls="--", lw=1.3,
              label=f"class_weight solo ({np.mean(base):.3f})")
ax[0].set_xticks(range(len(noms)))
ax[0].set_xticklabels(etqs, rotation=28, ha="right", fontsize=8.5)
ax[0].set_ylabel("balanced accuracy (test, nivel gen)")
ax[0].set_title(f"Familias: sobremuestreo vs class_weight vs ambos\n"
                f"promedio de {args.semillas} semillas, barras = desviacion")
ax[0].legend(fontsize=8)
for r, v in zip(b, med):
    ax[0].text(r.get_x() + r.get_width()/2, v + max(err)*0.25 + 0.004,
               f"{v:.3f}", ha="center", fontsize=8)
lo = min(m - e for m, e in zip(med, err))
hi = max(m + e for m, e in zip(med, err))
ax[0].set_ylim(max(0, lo - 0.05), min(1, hi + 0.06))

fams = [f for f, _ in genes_por_fam.most_common()]
x = np.arange(len(fams)); w = 0.38
ax[1].barh(x - w/2, [a_cw.get(f, 0) for f in fams], w,
           color="#4C72B0", label="class_weight solo")
ax[1].barh(x + w/2, [a_mj.get(f, 0) for f in fams], w,
           color="#55A868", label=ETIQ[rival["variante"]])
ax[1].set_yticks(x)
ax[1].set_yticklabels([f"{str(f)[:30]} ({genes_por_fam[f]})" for f in fams], fontsize=8)
ax[1].invert_yaxis()
ax[1].set_xlabel("acierto en esa familia (recall)")
ax[1].set_title("Por familia\n(entre parentesis, cuantos genes tiene)")
ax[1].legend(fontsize=8, loc="lower right")
ax[1].set_xlim(0, 1)

plt.tight_layout(); plt.savefig(SAL / "tarea3_comparacion.png", dpi=200); plt.close()

print("\n" + "=" * 96)
if args.demo:
    print("ESTO FUE EL DEMO: los numeros NO significan nada (embeddings falsos).")
    print("Sirve para confirmar que la tuberia corre. Ahora: --completo")
elif PRUEBA:
    print("ESTO FUE LA PRUEBA RAPIDA: los numeros NO sirven. Corre con --completo.")
else:
    print("LISTO. En results/tarea3/:")
    print("   tarea3_comparacion.csv   el promedio por variante")
    print("   tarea3_corridas.csv      cada semilla por separado")
    print("   tarea3_comparacion.png   la grafica")
print("=" * 96)
