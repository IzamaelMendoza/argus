"""
familias_desbalance.py -- sobremuestreo vs class_weight vs los dos (Tarea 3, reunion 10-nov).

EL PLANTEAMIENTO
----------------
Las 12 familias estan muy desbalanceadas: 291 genes la mayor, 30 la menor (9.7 a 1).
La asesora pidio probar tres cosas: sobremuestreo, "weightclass" en lugar del submuestreo,
y las dos juntas.

UNA ACLARACION IMPORTANTE ANTES DE EMPEZAR
------------------------------------------
El modelo de familias que ya tienes **nunca uso submuestreo**, y **ya usa class_weight**:

    pesos = cuenta.sum() / (n_clases * cuenta)        # pesos balanceados
    perdida = nn.CrossEntropyLoss(weight=pesos)       # <- en familias_dnabert.py

(El submuestreo es de la ETAPA 1, el modelo binario. Ahi si se submuestrean los negativos.)

Asi que "en vez de submuestreo usar weightclass" ya esta hecho: es la configuracion actual,
la que dio balanced accuracy 0.590. Lo que falta es el sobremuestreo y la combinacion.

Para que la comparacion signifique algo hacen falta las CUATRO celdas, no tres:

    config              sobremuestreo   class_weight   estado
    ----------------------------------------------------------------
    nada (control)          no              no         NUEVO: es el control
    sobremuestreo           si              no         NUEVO
    class_weight            no              si         = el modelo actual (0.590)
    ambos                   si              si         NUEVO

Sin la celda "nada" no se puede saber cuanto aporta cada cosa por separado, porque no hay
contra que comparar. Cuesta lo mismo (la cabeza se entrena en segundos una vez que estan
los embeddings), asi que se corren las cuatro.

CADA CONFIGURACION SE REPITE CON VARIAS SEMILLAS
------------------------------------------------
Leccion de las FASES 8 y 10: con ~200 genes de test, una diferencia de 0.05 esta dentro
del ruido. El script corre cada configuracion con N semillas distintas (lo que cambia es
la inicializacion de la cabeza y el muestreo) y reporta media +- error estandar de la
media, mas una prueba t emparejada contra el control.

EL SOBREMUESTREO
----------------
Se duplican ejemplos de las clases chicas hasta un objetivo. Reglas:

  * SOLO en train. Nunca en validacion ni en test: duplicar ahi inflaria las metricas.
  * Las clases que se tocan son las que elijas con --grupo (por omision, todas menos
    las 2 mas grandes, que fue lo acordado).
  * El objetivo por omision es el tamano de la mayor del grupo, o sea: igualar entre si
    a las 10 chicas sin tocar a las 2 grandes.

Ojo con lo que el sobremuestreo SI y NO puede hacer: duplicar un ejemplo no agrega
informacion nueva. Le dice al modelo "fijate mas en esta clase", que es exactamente lo
mismo que le dice class_weight, solo que por otra via. Por eso es esperable que las dos
den resultados parecidos, y que juntas no sumen. Medirlo es justo el punto.

USO
---
    python familias_desbalance.py                      # prueba rapida
    python familias_desbalance.py --completo           # la corrida real
    python familias_desbalance.py --completo --semillas 5
    python familias_desbalance.py --completo --grupo chicas4    # solo las 4 mas chicas

SALIDAS  (results/familias/)
    desbalance_resultados.csv       una fila por configuracion x semilla
    desbalance_por_clase.csv        F1 por familia y configuracion
    desbalance_resumen.txt
    fig_desbalance.png
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

import numpy as np

# ----------------------------------------------------------------- argumentos
ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--completo", action="store_true")
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
ap.add_argument("--semillas", type=int, default=3,
                help="cuantas repeticiones por configuracion (3 minimo, 5 recomendado)")
ap.add_argument("--grupo", default="sin_las_2_grandes",
                choices=["sin_las_2_grandes", "chicas4", "medianas6", "todas"],
                help="que clases se sobremuestrean")
ap.add_argument("--objetivo", default="max_grupo",
                help="'max_grupo' (por omision), 'max_global', o un numero de fragmentos")
ap.add_argument("--embeddings", default=None,
                help="archivo .npz con embeddings ya calculados (para no recalcular)")
ap.add_argument("--sin-figuras", action="store_true")
args = ap.parse_args()

K, VENTANA, MIN_PB, SEED = args.k, args.ventana, args.min_pb, args.seed
PRUEBA = not args.completo
N_SEM = max(1, args.semillas)

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
CACHE = RAIZ / "data" / "models" / "dnabert6_emb_familias"
CACHE.mkdir(parents=True, exist_ok=True)
SAL = RAIZ / "results" / "familias"
SAL.mkdir(parents=True, exist_ok=True)

for p, q in [(ETQ, "etiquetas_finales.tsv (corre antes modelo_familias.py)"),
             (FASTA, "argdb_POSITIVO_final.fasta")]:
    if not p.exists():
        sys.exit(f"Falta {q}: {p}")

_buf = []


def di(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _buf.append(s)


di("=" * 92)
di("DESBALANCE DE FAMILIAS: sobremuestreo vs class_weight vs ambos")
di("=" * 92)
di(f"K={K} | ventana={VENTANA} | min_pb={MIN_PB} | semillas={N_SEM} | grupo={args.grupo}")
if PRUEBA:
    di("MODO PRUEBA RAPIDA: muestra reducida, los numeros NO sirven. Usa --completo.")

# ----------------------------------------------------------------- dependencias
try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("Falta PyTorch.  pip install torch --index-url https://download.pytorch.org/whl/cpu")
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold

np.random.seed(SEED)

# =============================================================================
#  DATOS
# =============================================================================
etq = {}
with open(ETQ, encoding="utf-8") as fh:
    for f in csv.DictReader(fh, delimiter="\t"):
        if f.get("etiqueta"):
            etq[f["gen"]] = f["etiqueta"]

fa, h, s = {}, None, []
with open(FASTA, encoding="utf-8", errors="replace") as fh:
    for linea in fh:
        linea = linea.rstrip("\n").rstrip("\r")
        if linea.startswith(">"):
            if h:
                fa[h] = "".join(s)
            h, s = linea[1:].split()[0], []
        elif h:
            s.append(linea.strip())
if h:
    fa[h] = "".join(s)


def fragmentar(seq, ventana=VENTANA, min_pb=MIN_PB):
    """Regla de la asesora: sin padding, se descarta el resto si tiene < min_pb."""
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


def sequence_to_kmers(seq, k=K):
    """La funcion del codigo de la asesora."""
    seq = seq.upper().replace("0", "")
    seq = re.sub(r"[^ACGT]", "N", seq)
    if len(seq) < k:
        return "N" * k
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


seqs, ys, gs = [], [], []
descartados = 0
for gen, lab in etq.items():
    sq = fa.get(gen)
    if not sq:
        continue
    fr = fragmentar(sq)
    if not fr:
        descartados += 1
        continue
    for f in fr:
        seqs.append(f)
        ys.append(lab)
        gs.append(gen)
y = np.array(ys)
g = np.array(gs)
clases = sorted(set(y))
y_idx = np.array([clases.index(v) for v in y])
n_cls = len(clases)

genes_por_fam = Counter({gg: yy for gg, yy in zip(g, y)}.values())
di(f"\n{len(set(g)):,} genes -> {len(seqs):,} fragmentos | {n_cls} familias")
if descartados:
    di(f"({descartados} genes mas cortos que {MIN_PB} pb quedaron fuera)")

di(f"\n{'familia':<42} {'genes':>7} {'frags':>8}")
frags_por_fam = Counter(y.tolist())
orden_fam = [f for f, _ in genes_por_fam.most_common()]
for fam in orden_fam:
    di(f"{fam[:42]:<42} {genes_por_fam[fam]:>7} {frags_por_fam[fam]:>8}")

# =============================================================================
#  QUE CLASES SE SOBREMUESTREAN
# =============================================================================
if args.grupo == "sin_las_2_grandes":
    grupo = orden_fam[2:]
elif args.grupo == "chicas4":
    grupo = orden_fam[-4:]
elif args.grupo == "medianas6":
    grupo = orden_fam[2:8]
else:
    grupo = orden_fam[:]
grupo_idx = {clases.index(f) for f in grupo}

di(f"\n{'-'*92}")
di(f"SOBREMUESTREO: se tocan {len(grupo)} de {n_cls} familias ({args.grupo})")
di(f"{'-'*92}")
di("  se sobremuestrean : " + ", ".join(f[:34] for f in grupo[:4]) +
   (f", ... (+{len(grupo)-4})" if len(grupo) > 4 else ""))
intactas = [f for f in orden_fam if f not in set(grupo)]
di("  quedan intactas   : " + (", ".join(f[:34] for f in intactas) if intactas else "ninguna"))

# =============================================================================
#  CORTES (por gen, iguales a familias_dnabert.py)
# =============================================================================
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
di(f"\ntrain {len(tr):,} frag ({len(set(g[tr])):,} genes) | "
   f"val {len(va):,} ({len(set(g[va])):,}) | test {len(te):,} ({len(set(g[te])):,})")

if PRUEBA:
    rng = np.random.default_rng(SEED)
    tr = rng.choice(tr, size=min(300, len(tr)), replace=False)
    va = rng.choice(va, size=min(120, len(va)), replace=False)
    te = rng.choice(te, size=min(160, len(te)), replace=False)
    di(f"  (prueba: train {len(tr)} | val {len(va)} | test {len(te)})")

# =============================================================================
#  EMBEDDINGS
# =============================================================================
if args.embeddings and Path(args.embeddings).exists():
    z = np.load(args.embeddings)
    E_tr, E_va, E_te = z["tr"], z["va"], z["te"]
    DIM = E_tr.shape[1]
    di(f"\nembeddings cargados de {args.embeddings}: {E_tr.shape}")
else:
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        sys.exit("Falta transformers.  pip install 'transformers==4.30.2'")
    di(f"\nCargando {args.modelo} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.modelo)
    bert = AutoModel.from_pretrained(args.modelo).eval()     # sin trust_remote_code
    cfg = bert.config
    DIM = int(cfg.hidden_size)
    di(f"  listo en {time.time()-t0:.0f}s | capas {cfg.num_hidden_layers} | hidden {DIM}")

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
        """Mismo esquema de cache que familias_dnabert.py: si ya estan, no recalcula."""
        firma = hashlib.md5(("|".join(seqs[i] for i in idxs)).encode()).hexdigest()[:10]
        d = CACHE / f"{nombre}_{len(idxs)}_{firma}"
        d.mkdir(exist_ok=True)
        partes, t0 = [], time.time()
        for b0 in range(0, len(idxs), args.chunk):
            f = d / f"{b0:07d}.npy"
            if f.exists():
                partes.append(np.load(f))
                continue
            e = codificar([seqs[i] for i in idxs[b0:b0 + args.chunk]])
            np.save(f, e)
            partes.append(e)
            hechos = min(b0 + args.chunk, len(idxs))
            vel = hechos / max(1e-9, time.time() - t0)
            print(f"\r  {nombre}: {hechos}/{len(idxs)} ({vel:.1f} frag/s)   ", end="", flush=True)
        print()
        return np.vstack(partes)

    di("\nCalculando embeddings (se reusa la cache de familias_dnabert.py si coincide):")
    E_tr, E_va, E_te = embeddings(tr, "train"), embeddings(va, "val"), embeddings(te, "test")
    np.savez_compressed(SAL / "desbalance_embeddings.npz", tr=E_tr, va=E_va, te=E_te)
    di(f"  guardados en {SAL / 'desbalance_embeddings.npz'} (pasalos con --embeddings)")

# normalizacion: media y sd SOLO del train
mu, sd = E_tr.mean(0, keepdims=True), E_tr.std(0, keepdims=True) + 1e-6
Xtr_np = ((E_tr - mu) / sd).astype(np.float32)
Xva = torch.tensor(((E_va - mu) / sd).astype(np.float32))
Xte = torch.tensor(((E_te - mu) / sd).astype(np.float32))
ytr_np = y_idx[tr]
yva_np = y_idx[va]
yte_np = y_idx[te]


# =============================================================================
#  SOBREMUESTREO
# =============================================================================
def indices_sobremuestreados(y_local, rng):
    """
    Devuelve indices (sobre el train) con las clases del grupo repetidas hasta el objetivo.
    SOLO se llama con el train.
    """
    cuenta = Counter(y_local.tolist())
    if args.objetivo == "max_grupo":
        tam = [cuenta[c] for c in grupo_idx if cuenta.get(c)]
        objetivo = max(tam) if tam else 0
    elif args.objetivo == "max_global":
        objetivo = max(cuenta.values())
    else:
        objetivo = int(args.objetivo)

    base = np.arange(len(y_local))
    extra = []
    for c, n in cuenta.items():
        if c not in grupo_idx or n >= objetivo or n == 0:
            continue
        pool = np.flatnonzero(y_local == c)
        extra.append(rng.choice(pool, size=objetivo - n, replace=True))
    return (np.concatenate([base, *extra]) if extra else base), objetivo


def pesos_clase(y_local):
    """class_weight balanceado, igual que en familias_dnabert.py."""
    cuenta = np.bincount(y_local, minlength=n_cls).astype(np.float64)
    return torch.tensor(
        np.where(cuenta > 0, cuenta.sum() / (n_cls * np.maximum(cuenta, 1)), 0.0),
        dtype=torch.float32)


# =============================================================================
#  ENTRENAMIENTO DE UNA CONFIGURACION
# =============================================================================
def entrenar(sobremuestreo, class_weight, semilla):
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)

    if sobremuestreo:
        idx, objetivo = indices_sobremuestreados(ytr_np, rng)
    else:
        idx, objetivo = np.arange(len(ytr_np)), None
    Xt = torch.tensor(Xtr_np[idx])
    yt = torch.tensor(ytr_np[idx], dtype=torch.long)

    pesos = pesos_clase(ytr_np[idx]) if class_weight else None
    cabeza = nn.Sequential(nn.Linear(DIM, 256), nn.ReLU(), nn.Dropout(0.3),
                           nn.Linear(256, n_cls))
    opt = torch.optim.Adam(cabeza.parameters(), lr=args.lr, weight_decay=1e-4)
    perdida = nn.CrossEntropyLoss(weight=pesos)

    mejor, mejor_ep, sin_mejora, estado = -1.0, -1, 0, None
    for ep in range(args.epocas):
        cabeza.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            opt.zero_grad()
            perdida(cabeza(Xt[b]), yt[b]).backward()
            opt.step()
        cabeza.eval()
        with torch.no_grad():
            pv = cabeza(Xva).argmax(1).numpy()
        # checkpoint por balanced accuracy de validacion (decision de la asesora)
        bal = balanced_accuracy_score(yva_np, pv)
        if bal > mejor:
            mejor, mejor_ep, sin_mejora = bal, ep, 0
            estado = {k: v.clone() for k, v in cabeza.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                break
    if estado:
        cabeza.load_state_dict(estado)
    cabeza.eval()

    with torch.no_grad():
        prob = torch.softmax(cabeza(Xte), dim=1).numpy()

    # nivel gen: promedio de probabilidades (la regla validada y la de la GUI)
    por_gen = defaultdict(list)
    for i, gg in enumerate(g[te]):
        por_gen[gg].append(i)
    gr, gp = [], []
    for gg, idxs in por_gen.items():
        gr.append(yte_np[idxs[0]])
        gp.append(int(np.argmax(prob[idxs].mean(axis=0))))
    gr, gp = np.array(gr), np.array(gp)

    return dict(
        bal_gen=balanced_accuracy_score(gr, gp),
        f1_gen=f1_score(gr, gp, average="macro", zero_division=0),
        exac_gen=float((gp == gr).mean()),
        bal_frag=balanced_accuracy_score(yte_np, prob.argmax(1)),
        val_bal=mejor, epoca=mejor_ep + 1, n_train=len(idx), objetivo=objetivo,
        f1_por_clase={clases[c]: f1_score(gr == c, gp == c, zero_division=0)
                      for c in range(n_cls) if (gr == c).any()},
        n_genes_test=len(gr),
    )


# =============================================================================
#  CORRER LAS CUATRO CONFIGURACIONES
# =============================================================================
CONFIGS = [
    ("nada (control)", False, False),
    ("sobremuestreo", True, False),
    ("class_weight", False, True),
    ("ambos", True, True),
]

di(f"\n{'='*92}")
di(f"ENTRENANDO: 4 configuraciones x {N_SEM} semillas = {4*N_SEM} cabezas")
di(f"{'='*92}")

res = defaultdict(list)
por_clase = defaultdict(lambda: defaultdict(list))
t0 = time.time()
for nombre, so, cw in CONFIGS:
    for si in range(N_SEM):
        r = entrenar(so, cw, SEED + si * 101)
        res[nombre].append(r)
        for c, v in r["f1_por_clase"].items():
            por_clase[nombre][c].append(v)
    ult = res[nombre][-1]
    extra = f" | train {ult['n_train']:,} frag" + (
        f" (objetivo {ult['objetivo']}/clase)" if ult["objetivo"] else "")
    di(f"  {nombre:<18} bal.acc {np.mean([x['bal_gen'] for x in res[nombre]]):.4f}{extra}")
di(f"\n  tiempo: {time.time()-t0:.0f}s")


def resumen(lista, campo):
    v = np.array([x[campo] for x in lista])
    sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), sem


# ----------------------------------------------------------------- tabla
di(f"\n{'='*92}")
di(f"RESULTADOS (test por gen, {res['nada (control)'][0]['n_genes_test']} genes, "
   f"media +- error estandar de {N_SEM} semillas)")
di(f"{'='*92}")
di(f"\n  {'configuracion':<18} {'balanced acc':>20} {'F1 macro':>20} {'exactitud':>12}")
for nombre, _, _ in CONFIGS:
    b, bs = resumen(res[nombre], "bal_gen")
    f, fs = resumen(res[nombre], "f1_gen")
    e, _ = resumen(res[nombre], "exac_gen")
    di(f"  {nombre:<18} {b:>12.4f} +-{bs:<6.4f} {f:>12.4f} +-{fs:<6.4f} {e:>12.4f}")

# ----------------------------------------------------------------- comparacion emparejada
di(f"\n{'-'*92}")
di("COMPARACION CONTRA EL CONTROL (diferencias emparejadas por semilla)")
di(f"{'-'*92}")
T_CRIT = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          8: 2.306, 9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093}
ctrl = np.array([x["bal_gen"] for x in res["nada (control)"]])
di(f"\n  {'configuracion':<18} {'dif. media':>12} {'sem':>9} {'t':>8}  veredicto")
for nombre, _, _ in CONFIGS[1:]:
    v = np.array([x["bal_gen"] for x in res[nombre]])
    d = v - ctrl
    m = d.mean()
    if len(d) > 1 and d.std(ddof=1) > 1e-12:
        sem = d.std(ddof=1) / np.sqrt(len(d))
        t = m / sem
        gl = len(d) - 1
        crit = T_CRIT.get(gl, 2.0)
        ver = "consistente" if abs(t) > crit else "dentro del ruido"
        di(f"  {nombre:<18} {m:>+12.4f} {sem:>9.4f} {t:>+8.2f}  {ver} (t critico {crit})")
    else:
        di(f"  {nombre:<18} {m:>+12.4f} {'-':>9} {'-':>8}  hacen falta mas semillas")
if N_SEM < 3:
    di(f"\n  [aviso] con {N_SEM} semilla(s) no se puede juzgar nada. Usa --semillas 5.")

# ----------------------------------------------------------------- por clase
di(f"\n{'-'*92}")
di("F1 POR FAMILIA (media entre semillas) -- aqui es donde se ve si sirvio")
di(f"{'-'*92}")
cab = f"\n  {'familia':<40}"
for nombre, _, _ in CONFIGS:
    cab += f" {nombre[:11]:>12}"
di(cab + f" {'mejor':>14}")
for fam in orden_fam:
    if fam not in por_clase["nada (control)"]:
        continue
    fila = f"  {fam[:40]:<40}"
    vals = {}
    for nombre, _, _ in CONFIGS:
        v = np.mean(por_clase[nombre].get(fam, [0.0]))
        vals[nombre] = v
        fila += f" {v:>12.3f}"
    mejor = max(vals, key=vals.get)
    marca = "*" if fam in set(grupo) else " "
    di(fila + f" {mejor[:13]:>14}{marca}")
di("\n  (* = familia sobremuestreada)")

# ----------------------------------------------------------------- CSV
with open(SAL / "desbalance_resultados.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["configuracion", "semilla", "sobremuestreo", "class_weight",
                "bal_acc_gen", "f1_macro_gen", "exactitud_gen", "bal_acc_frag",
                "val_bal_acc", "mejor_epoca", "n_frag_train"])
    for nombre, so, cw in CONFIGS:
        for i, r in enumerate(res[nombre]):
            w.writerow([nombre, SEED + i * 101, int(so), int(cw),
                        round(r["bal_gen"], 5), round(r["f1_gen"], 5),
                        round(r["exac_gen"], 5), round(r["bal_frag"], 5),
                        round(r["val_bal"], 5), r["epoca"], r["n_train"]])
with open(SAL / "desbalance_por_clase.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["familia", "sobremuestreada"] + [c for c, _, _ in CONFIGS])
    for fam in orden_fam:
        if fam not in por_clase["nada (control)"]:
            continue
        w.writerow([fam, int(fam in set(grupo))] +
                   [round(np.mean(por_clase[n].get(fam, [0.0])), 4) for n, _, _ in CONFIGS])
di(f"\nresultados: {SAL / 'desbalance_resultados.csv'}")
di(f"por clase : {SAL / 'desbalance_por_clase.csv'}")

# ----------------------------------------------------------------- figura
if not args.sin_figuras:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
        noms = [c for c, _, _ in CONFIGS]
        m = [resumen(res[n], "bal_gen")[0] for n in noms]
        e = [resumen(res[n], "bal_gen")[1] for n in noms]
        cols = ["#6C757D", "#4C78A8", "#54A24B", "#C44E52"]
        a1.bar(range(4), m, yerr=e, capsize=6, color=cols)
        a1.axhline(m[0], ls="--", c="gray", lw=1, label="control")
        a1.set_xticks(range(4)); a1.set_xticklabels(noms, rotation=12)
        a1.set_ylabel("balanced accuracy (test por gen)")
        a1.set_title(f"Manejo del desbalance\n(media +- sem de {N_SEM} semillas)")
        a1.legend()
        lo = min(np.array(m) - np.array(e)); hi = max(np.array(m) + np.array(e))
        a1.set_ylim(max(0, lo - 0.05), min(1, hi + 0.05))
        for i, v in enumerate(m):
            a1.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        fams = [f for f in orden_fam if f in por_clase["nada (control)"]][::-1]
        ypos = np.arange(len(fams)); alto = 0.2
        for j, n in enumerate(noms):
            a2.barh(ypos + (j - 1.5) * alto,
                    [np.mean(por_clase[n].get(f, [0.0])) for f in fams],
                    height=alto, label=n, color=cols[j])
        a2.set_yticks(ypos)
        a2.set_yticklabels([("* " if f in set(grupo) else "  ") + f[:30] for f in fams],
                           fontsize=8)
        a2.set_xlabel("F1 de la familia"); a2.legend(fontsize=8)
        a2.set_title("F1 por familia\n(* = sobremuestreada)")
        plt.tight_layout(); plt.savefig(SAL / "fig_desbalance.png", dpi=200); plt.close()
        di(f"figura   : {SAL / 'fig_desbalance.png'}")
    except ImportError:
        di("(matplotlib no instalado: sin figura)")

# ----------------------------------------------------------------- lectura
di(f"\n{'='*92}")
di("COMO LEER ESTO")
di(f"{'='*92}")
b_ctrl = resumen(res["nada (control)"], "bal_gen")[0]
b_so = resumen(res["sobremuestreo"], "bal_gen")[0]
b_cw = resumen(res["class_weight"], "bal_gen")[0]
b_am = resumen(res["ambos"], "bal_gen")[0]
di(f"\n  balanced accuracy, y entre parentesis la diferencia contra el control:")
di(f"    control        {b_ctrl:.4f}")
di(f"    sobremuestreo  {b_so:.4f}  ({b_so-b_ctrl:+.4f})")
di(f"    class_weight   {b_cw:.4f}  ({b_cw-b_ctrl:+.4f})")
di(f"    ambos          {b_am:.4f}  ({b_am-b_ctrl:+.4f})")
di("")
if b_am < max(b_so, b_cw) + 0.002:
    di("  Las dos tecnicas NO se suman. Es lo esperable: duplicar ejemplos de una clase y")
    di("  subirle el peso en la perdida le dicen al modelo exactamente lo mismo por dos")
    di("  vias distintas. Si una ya corrigio el desbalance, la otra no tiene que corregir.")
else:
    di("  Las dos juntas superan a cada una por separado, lo cual no era lo esperable.")
    di("  Vale repetirlo con mas semillas antes de darlo por bueno.")
di("")
di("  Lo que de verdad importa mirar es la tabla POR FAMILIA: el agregado puede moverse")
di("  poco mientras las familias chicas mejoran mucho y las grandes empeoran un poco.")
di("  Ese intercambio es justo el objetivo de tratar el desbalance.")
di("")
di("  Y el limite de fondo: el sobremuestreo no crea informacion. APH tiene 30 genes;")
di("  duplicarlos hasta 93 no le da al modelo ni un gen mas que aprender. Lo que puede")
di("  dar es que el modelo deje de ignorarla. Para mejorar de verdad hacen falta mas")
di("  genes de esas familias, que es lo que menciona el punto 12 de 'lo que falta' en")
di("  la bitacora (ampliar mas alla de CARD).")

if PRUEBA:
    di("\n" + "!" * 92)
    di("ESTO FUE LA PRUEBA RAPIDA: los numeros NO sirven. Corre con --completo.")
    di("!" * 92)

with open(SAL / "desbalance_resumen.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(_buf) + "\n")
print(f"\nResumen en {SAL / 'desbalance_resumen.txt'}")
