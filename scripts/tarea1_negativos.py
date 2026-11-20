"""
tarea1_negativos.py -- Tarea 1, punto 2: meter el SET NEGATIVO COMPLETO al modelo,
ver cuales salen positivos y por que.

Lo que ya sabemos de antes (documento del 29-oct, modelo id90, solo los 1000
negativos del 1k+1k):
  - los negativos vienen de SOLO 6 genomas
  - la tasa de falsos positivos va de 0.134 a 0.567 segun el genoma (4.2x)
  - blastn descarto que los FP se parezcan a algun ARG (0 de 10 candidatos,
    con control positivo 10/10), asi que la causa es el atajo taxonomico
Este script repite la medicion sobre el set negativo ENTERO, que es mucha mas
estadistica, y agrega el analisis de "a que se parecen".

AVISO METODOLOGICO IMPORTANTE
-----------------------------
El set negativo completo incluye genes con los que el modelo ENTRENO. Sobre esos
la tasa de falsos positivos sale artificialmente baja: el modelo ya los vio. Por
eso el script separa en dos grupos y reporta cada uno aparte:

  A) negativos del 1k+1k   -> el modelo NUNCA los vio. Es la cifra honesta.
  B) el resto              -> incluye entrenamiento. Optimista; sirve para ver
                              el reparto por genoma, no para citar una tasa.

Nunca cites la tasa global de los dos juntos.

Uso:
    python scripts/tarea1_negativos.py
    python scripts/tarea1_negativos.py --identidad 95 --version _v2
    python scripts/tarea1_negativos.py --top-vecinos 300
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--identidad", default="95")
ap.add_argument("--version", default="_v2", help="sufijo del modelo ('' = el viejo)")
ap.add_argument("--negativo", default=None, help="FASTA negativo completo")
ap.add_argument("--positivo", default=None, help="FASTA positivo (para el vecino mas cercano)")
ap.add_argument("--umbral", type=float, default=None)
ap.add_argument("--min-pb", type=int, default=None, help="por defecto, el del modelo")
ap.add_argument("--bloque", type=int, default=400, help="genes por bloque (RAM)")
ap.add_argument("--top-vecinos", type=int, default=200,
                help="cuantos FP analizar a fondo (vecino mas cercano)")
args = ap.parse_args()

# ----------------------------------------------------------------- raiz
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

SAL = RAIZ / "results" / "tarea1"
SAL.mkdir(parents=True, exist_ok=True)


def localizar(explicito, patrones, desc):
    if explicito:
        p = Path(explicito).expanduser()
        if not p.is_absolute():
            p = RAIZ / p
        if p.exists():
            return p
        sys.exit(f"No existe el archivo que diste para {desc}: {p}")
    for pat in patrones:
        for p in sorted(RAIZ.glob(pat)):
            if p.is_file():
                return p
    return None


MODELO = RAIZ / "data" / "models" / f"argus_rf_k6_id{args.identidad}{args.version}.joblib"
if not MODELO.exists():
    hay = sorted((RAIZ / "data" / "models").glob("argus_rf_*.joblib"))
    sys.exit(f"No existe {MODELO.name}.\nModelos que si hay:\n" +
             "\n".join(f"   {p.name}" for p in hay))

NEG = localizar(args.negativo,
                ["data/processed/argdb_NEGATIVO_final.fasta",
                 "data/processed/*NEGATIVO*.fasta",
                 "data/processed/*negativ*.fasta",
                 "data/interim/*negativ*.fasta"],
                "el FASTA negativo")
if NEG is None:
    sys.exit("No encontre el FASTA negativo. Pasalo con --negativo\n"
             "FASTAs disponibles:\n" +
             "\n".join(f"   {p.relative_to(RAIZ)}"
                       for p in sorted(RAIZ.glob('data/**/*.fasta'))[:20]))

POS = localizar(args.positivo,
                ["data/processed/argdb_POSITIVO_final.fasta",
                 "data/processed/*POSITIVO*.fasta"],
                "el FASTA positivo")
SET1K = localizar(None, ["data/interim/muestra_genes_labels.tsv"], "el 1k+1k")

# ----------------------------------------------------------------- modelo
paq = joblib.load(MODELO)
rf = paq["modelo"] if isinstance(paq, dict) else paq
UMBRAL = args.umbral if args.umbral is not None else (
    float(paq.get("umbral", 0.5)) if isinstance(paq, dict) else 0.5)
VENTANA = int(paq.get("ventana", 500)) if isinstance(paq, dict) else 500
MIN_PB = args.min_pb if args.min_pb is not None else (
    int(paq.get("min_pb", 150)) if isinstance(paq, dict) else 150)
K = 1
while 4 ** K < rf.n_features_in_:
    K += 1
try:
    rf.set_params(n_jobs=-1)
except Exception:
    pass

print("=" * 94)
print("TAREA 1, PUNTO 2 -- el set negativo completo por el modelo")
print(f"modelo:   {MODELO.name}")
print(f"          K={K} | ventana={VENTANA} | min_pb={MIN_PB} | umbral={UMBRAL}")
print(f"negativo: {NEG.relative_to(RAIZ)}")
if isinstance(paq, dict) and "min_pb" not in paq:
    print("\n  [AVISO] este modelo se entreno con relleno de ceros (no trae 'min_pb').")
    print("  Se le esta aplicando la fragmentacion nueva en prediccion, asi que ve")
    print("  fragmentos distintos a los del entrenamiento. Usa --version _v2.")
print("=" * 94)

# ----------------------------------------------------------------- k-meros
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


def fragmentar(seq, ventana=VENTANA, min_pb=MIN_PB):
    """Regla de la asesora: sin padding, se descarta el resto de < min_pb."""
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


def leer_fasta(path):
    out, h, s = [], None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            linea = linea.rstrip("\n")
            if linea.startswith(">"):
                if h is not None:
                    out.append((h, "".join(s)))
                h, s = linea[1:], []
            elif h is not None:
                s.append(linea.strip())
    if h is not None:
        out.append((h, "".join(s)))
    return out


def genoma_de(encabezado):
    """GCF_000009045.1|NC_000964.3_4136 -> GCF_000009045.1"""
    m = re.search(r"(GC[AF]_\d+\.\d+)", encabezado)
    if m:
        return m.group(1)
    return encabezado.split("|")[0][:28]


def gc(seq):
    s = seq.upper()
    n = len(s) - s.count("N")
    return 100.0 * (s.count("G") + s.count("C")) / n if n else 0.0


# ----------------------------------------------------------------- datos
neg = leer_fasta(NEG)
print(f"\nsecuencias en el negativo: {len(neg):,}")

ids_1k = set()
if SET1K:
    t = pd.read_csv(SET1K, sep="\t")
    ids_1k = set(t[t.iloc[:, 1] == 0].iloc[:, 0].astype(str))
    print(f"negativos del 1k+1k (nunca vistos): {len(ids_1k):,}")
else:
    print("[aviso] no encontre muestra_genes_labels.tsv: no puedo separar los no vistos")

genomas = Counter(genoma_de(h) for h, _ in neg)
print(f"genomas distintos en el negativo: {len(genomas)}")
for g, n in genomas.most_common(12):
    print(f"   {g:<24} {n:>7,} genes")
if len(genomas) > 12:
    print(f"   ... y {len(genomas)-12} mas")

# ----------------------------------------------------------------- predecir
print(f"\nPrediciendo por bloques de {args.bloque} genes...")
filas = []
vec_fp = {}            # id -> vector k-mer promedio (solo para los FP top)
total = len(neg)
for b0 in range(0, total, args.bloque):
    lote = neg[b0:b0 + args.bloque]
    for h, s in lote:
        gid = h.split()[0]
        fr = fragmentar(s)
        if not fr:
            filas.append(dict(gen=gid, genoma=genoma_de(h), largo=len(s), gc=gc(s),
                              n_frag=0, n_pos=0, p_max=np.nan, p_media=np.nan,
                              pred=-1, visto_1k=gid in ids_1k))
            continue
        X = np.array([vector_kmer(f) for f in fr], dtype=np.float32)
        p = rf.predict_proba(X)[:, 1]
        pred = int((p >= UMBRAL).any())
        filas.append(dict(gen=gid, genoma=genoma_de(h), largo=len(s), gc=gc(s),
                          n_frag=len(fr), n_pos=int((p >= UMBRAL).sum()),
                          p_max=float(p.max()), p_media=float(p.mean()),
                          pred=pred, visto_1k=gid in ids_1k))
        if pred:
            vec_fp[gid] = X.mean(axis=0)
    hechos = min(b0 + args.bloque, total)
    print(f"\r  {hechos:,}/{total:,}  ({100*hechos/total:.0f}%)", end="", flush=True)
print()

d = pd.DataFrame(filas)
evaluables = d[d.pred >= 0]
no_eval = d[d.pred < 0]

# =============================================================================
#  RESULTADO
# =============================================================================
print("\n" + "=" * 94)
print("RESULTADO")
print("=" * 94)
print(f"\ngenes negativos            : {len(d):,}")
print(f"  no evaluables (< {MIN_PB} pb): {len(no_eval):,}")
print(f"  evaluados                : {len(evaluables):,}")

fp = evaluables[evaluables.pred == 1]
print(f"\nSALEN POSITIVOS (falsos positivos): {len(fp):,} "
      f"({100*len(fp)/len(evaluables):.1f}% de los evaluados)")

if len(ids_1k):
    a = evaluables[evaluables.visto_1k]
    b = evaluables[~evaluables.visto_1k]
    print(f"\n  SEPARADO, que es como hay que leerlo:")
    if len(a):
        print(f"    A) negativos del 1k+1k (NUNCA vistos): "
              f"{int((a.pred==1).sum()):,} de {len(a):,} = "
              f"{100*(a.pred==1).mean():.1f}%   <-- la cifra honesta")
    print(f"    B) el resto (incluye entrenamiento)  : "
          f"{int((b.pred==1).sum()):,} de {len(b):,} = "
          f"{100*(b.pred==1).mean():.1f}%   <-- optimista, el modelo ya los vio")
    if len(a) and (a.pred == 1).mean() > (b.pred == 1).mean():
        print(f"\n    La diferencia entre A y B mide cuanto del acierto viene de")
        print(f"    memorizar el entrenamiento en lugar de generalizar.")

# ---- por genoma: el hallazgo del atajo taxonomico, ahora con todo el set ----
print("\n" + "-" * 94)
print("TASA DE FALSOS POSITIVOS POR GENOMA")
print("-" * 94)
g = (evaluables.groupby("genoma")
     .agg(genes=("pred", "size"), fp=("pred", "sum"),
          gc_medio=("gc", "mean"), largo_medio=("largo", "mean"))
     .assign(tasa_fp=lambda x: x.fp / x.genes)
     .sort_values("tasa_fp", ascending=False))
print(f"\n{'genoma':<24} {'genes':>8} {'FP':>7} {'tasa':>7} {'%GC':>7} {'largo':>8}")
for gid, r in g.iterrows():
    print(f"{gid:<24} {int(r.genes):>8,} {int(r.fp):>7,} {r.tasa_fp:>7.3f} "
          f"{r.gc_medio:>7.1f} {r.largo_medio:>8.0f}")
if len(g) > 1:
    lo, hi = g.tasa_fp.min(), g.tasa_fp.max()
    veces = (f"{hi/lo:.1f} veces" if lo > 0
             else "el mejor genoma no falla ni una vez")
    print(f"\n  rango: {lo:.3f} a {hi:.3f}  ({veces})")
    if len(g) >= 3:
        c = np.corrcoef(g.gc_medio, g.tasa_fp)[0, 1]
        print(f"  correlacion %GC del genoma con su tasa de FP: {c:+.3f}")
        if c < -0.5:
            print("  (negativa y fuerte: los genomas de GC bajo generan mas FP, que es")
            print("   lo mismo que se vio en el diagnostico del 29-oct)")
g.to_csv(SAL / "negativos_por_genoma.csv")

# ---- por que salen positivos: largo, GC, fragmentos ----
print("\n" + "-" * 94)
print("EN QUE SE DIFERENCIAN LOS QUE FALLAN")
print("-" * 94)
ok = evaluables[evaluables.pred == 0]
print(f"\n{'medida':<28} {'aciertos':>12} {'falsos pos.':>12}")
for col, nom in [("largo", "largo del gen (pb)"), ("gc", "%GC"),
                 ("n_frag", "fragmentos por gen"), ("p_max", "prob. maxima")]:
    print(f"{nom:<28} {ok[col].mean():>12.1f} {fp[col].mean():>12.1f}")

print(f"\ntasa de FP segun cuantos fragmentos tiene el gen:")
print(f"  {'fragmentos':<14} {'genes':>8} {'FP':>8} {'tasa':>8}")
for n, sub in evaluables.groupby(evaluables.n_frag.clip(upper=6)):
    etq = f"{int(n)}" if n < 6 else "6 o mas"
    print(f"  {etq:<14} {len(sub):>8,} {int(sub.pred.sum()):>8,} "
          f"{sub.pred.mean():>8.3f}")
print("  (si sube con el numero de fragmentos es solo la regla 'algun fragmento':")
print("   mas fragmentos = mas oportunidades de pasar el umbral)")

# cuantos son "coherentes" (TODOS sus fragmentos positivos)
coh = fp[(fp.n_frag >= 3) & (fp.n_pos == fp.n_frag)]
print(f"\nfalsos positivos COHERENTES (3+ fragmentos y TODOS positivos): {len(coh):,}")
print("  son los unicos donde cabria esperar parecido real con un ARG;")
print("  el resto son errores difusos, el perfil del atajo taxonomico")

d.to_csv(SAL / "negativos_predicciones.csv", index=False)

# ---- a que se parecen: vecino mas cercano entre los positivos ----
if POS and len(fp):
    print("\n" + "-" * 94)
    print("A QUE SE PARECEN (vecino mas cercano entre los positivos, coseno de k-meros)")
    print("-" * 94)
    top = fp.sort_values("p_max", ascending=False).head(args.top_vecinos)
    print(f"\nanalizando los {len(top):,} falsos positivos de mayor probabilidad...")

    pos = leer_fasta(POS)
    print(f"  comparando contra {len(pos):,} genes positivos")
    P = np.empty((len(pos), 4 ** K), dtype=np.float32)
    for i, (_, s) in enumerate(pos):
        fr = fragmentar(s)
        P[i] = (np.array([vector_kmer(f) for f in fr], dtype=np.float32).mean(axis=0)
                if fr else vector_kmer(s))
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)

    Q = np.array([vec_fp[g] for g in top.gen], dtype=np.float32)
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
    sim = Qn @ Pn.T
    j = sim.argmax(axis=1)
    smax = sim.max(axis=1)

    # control: que tan parecidos son los negativos que SI acierta
    ctrl = ok.sort_values("p_max").head(min(len(top), len(ok)))
    Qc = np.empty((len(ctrl), 4 ** K), dtype=np.float32)
    mapa = {h.split()[0]: s for h, s in neg}
    for i, gid in enumerate(ctrl.gen):
        fr = fragmentar(mapa[gid])
        Qc[i] = (np.array([vector_kmer(f) for f in fr], dtype=np.float32).mean(axis=0)
                 if fr else vector_kmer(mapa[gid]))
    Qcn = Qc / (np.linalg.norm(Qc, axis=1, keepdims=True) + 1e-9)
    sctrl = (Qcn @ Pn.T).max(axis=1)

    print(f"\n  parecido al positivo mas cercano (1.0 = identico en composicion):")
    print(f"    falsos positivos      : media {smax.mean():.4f}  "
          f"mediana {np.median(smax):.4f}  max {smax.max():.4f}")
    print(f"    negativos bien dichos : media {sctrl.mean():.4f}  "
          f"mediana {np.median(sctrl):.4f}  max {sctrl.max():.4f}")
    dif = smax.mean() - sctrl.mean()
    print(f"    diferencia            : {dif:+.4f}")
    if abs(dif) < 0.02:
        print("\n  Los falsos positivos NO se parecen mas a los ARG que los negativos")
        print("  que el modelo acierta. O sea: el error no viene de parecido real.")
        print("  Coincide con lo que ya dijo el BLAST el 29-oct (0 de 10 candidatos).")
    else:
        print(f"\n  Hay una diferencia de {dif:+.4f} en composicion de k-meros.")
        print("  Vale la pena confirmarla con blastn antes de concluir nada.")

    top = top.copy()
    top["positivo_mas_cercano"] = [pos[k][0][:70] for k in j]
    top["similitud"] = smax.round(4)
    top.to_csv(SAL / "negativos_falsos_positivos_top.csv", index=False)

    print(f"\n  Los 10 falsos positivos mas seguros y a que se parecen:")
    for _, r in top.head(10).iterrows():
        print(f"    p={r.p_max:.3f} {r.n_pos}/{r.n_frag} frag | {r.gen[:44]}")
        print(f"       se parece a ({r.similitud:.3f}): {r.positivo_mas_cercano[:66]}")

    # FASTA de los FP para BLAST
    f_out = SAL / "negativos_falsos_positivos.fasta"
    with open(f_out, "w", encoding="utf-8") as fh:
        for _, r in top.iterrows():
            fh.write(f">{r.gen} p={r.p_max:.3f} frags={r.n_pos}/{r.n_frag}\n"
                     f"{mapa[r.gen]}\n")
    print(f"\n  FASTA para BLAST: {f_out.relative_to(RAIZ)} ({len(top)} secuencias)")

# ----------------------------------------------------------------- graficas
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))

gg = g.sort_values("tasa_fp")
c = ["#C44E52" if t > gg.tasa_fp.median() else "#4C72B0" for t in gg.tasa_fp]
ax[0].barh(range(len(gg)), gg.tasa_fp, color=c)
ax[0].set_yticks(range(len(gg)))
ax[0].set_yticklabels([str(i)[:22] for i in gg.index], fontsize=8)
ax[0].set_xlabel("tasa de falsos positivos")
ax[0].set_title("Falsos positivos por genoma")
for i, (t, n) in enumerate(zip(gg.tasa_fp, gg.genes)):
    ax[0].text(t + .01, i, f"{t:.3f}  (n={int(n):,})", va="center", fontsize=7.5)
ax[0].set_xlim(0, max(gg.tasa_fp.max() * 1.35, 0.1))

ax[1].hist(ok.p_max.dropna(), bins=50, alpha=.75, color="#4C72B0", label="acierta (no ARG)")
ax[1].hist(fp.p_max.dropna(), bins=50, alpha=.75, color="#C44E52", label="falso positivo")
ax[1].axvline(UMBRAL, color="k", ls="--", lw=1.3, label=f"umbral {UMBRAL}")
ax[1].set_xlabel("probabilidad maxima del gen"); ax[1].set_ylabel("genes")
ax[1].set_title("Distribucion de probabilidades"); ax[1].legend(fontsize=8)

ax[2].scatter(ok.gc, ok.p_max, s=3, alpha=.18, color="#4C72B0", label="acierta")
ax[2].scatter(fp.gc, fp.p_max, s=3, alpha=.30, color="#C44E52", label="falso positivo")
ax[2].axhline(UMBRAL, color="k", ls="--", lw=1.2)
ax[2].set_xlabel("%GC del gen"); ax[2].set_ylabel("probabilidad maxima")
ax[2].set_title("%GC contra probabilidad"); ax[2].legend(fontsize=8, markerscale=3)

plt.tight_layout(); plt.savefig(SAL / "negativos_analisis.png", dpi=200); plt.close()

print("\n" + "=" * 94)
print("LISTO. Archivos nuevos en results/tarea1/:")
for n in ("negativos_predicciones.csv", "negativos_por_genoma.csv",
          "negativos_falsos_positivos_top.csv", "negativos_falsos_positivos.fasta",
          "negativos_analisis.png"):
    p = SAL / n
    if p.exists():
        print(f"   {n}")
print("=" * 94)
