"""
coherencia_familias.py -- comprueba si los genes de cada familia se parecen entre
si, que es la hipotesis que planteo la asesora al ver el reporte:

   "tal vez estos ejemplos aunque pertenecen a la misma familia no se parezcan
    mucho, pero se tiene que comprobar"

COMO SE COMPRUEBA
-----------------
Para cada familia se mide:

  A) De cuantas familias CURADAS DE CARD esta hecha. Nuestras 12 familias son
     agrupaciones de las ~368 familias del aro_index. Si una de las nuestras
     junta 40 familias distintas de CARD, es un cajon por construccion.

  B) Cuanto se parecen sus genes ENTRE SI (similitud intra) contra cuanto se
     parecen a los genes de las OTRAS familias (similitud inter). La medida es
     el coseno de los vectores de k-meros del gen completo.

       razon = intra / inter

     Si la razon es ~1, los genes de la familia no se parecen entre si mas que a
     cualquier otro gen: la familia no tiene coherencia interna.

  C) Si eso explica el desempeno: se correlaciona la coherencia de cada familia
     con su recall (sacado de matriz_confusion.csv, que produce
     reporte_familias.py).

     Si la correlacion es fuerte y positiva, queda comprobado que lo que limita
     al modelo es la coherencia de las clases y no cuantos ejemplos tienen.
     Eso encaja con lo que ya se vio en la Tarea 3: la correlacion entre numero
     de genes y acierto era -0.05, o sea ninguna.

Uso:
    python scripts/coherencia_familias.py
    python scripts/coherencia_familias.py --blast     # agrega identidad real por blastn
"""
import argparse
import csv
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--blast", action="store_true",
                help="ademas, identidad real con blastn todos-contra-todos por familia")
ap.add_argument("--max-blast", type=int, default=60,
                help="maximo de genes por familia al blastear (el tiempo crece al cuadrado)")
args = ap.parse_args()
K = args.k

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
CM = RAIZ / "results" / "familias" / "matriz_confusion.csv"
SAL = RAIZ / "results" / "familias"
SAL.mkdir(parents=True, exist_ok=True)

ARO = None
for pat in ("data/raw/1_CARD/aro_index.tsv", "data/raw/**/aro_index.tsv"):
    hits = sorted(RAIZ.glob(pat))
    if hits:
        ARO = hits[0]
        break

for p, q in [(ETQ, "etiquetas_finales.tsv"), (FASTA, "argdb_POSITIVO_final.fasta")]:
    if not p.exists():
        sys.exit(f"Falta {q}: {p}")

print("=" * 96)
print("COHERENCIA DE LAS FAMILIAS")
print("  Hipotesis a comprobar: los genes de una misma familia podrian no")
print("  parecerse entre si. Si es asi, el modelo no tiene nada que aprender.")
print("=" * 96)

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

genes = [gg for gg in etq if gg in fa]
fam = np.array([etq[gg] for gg in genes])
clases = sorted(set(fam))
print(f"\n{len(genes):,} genes con secuencia y familia | {len(clases)} familias")

# =============================================================================
#  A. DE QUE FAMILIAS DE CARD ESTA HECHA CADA UNA DE LAS NUESTRAS
# =============================================================================
print(f"\n{'='*96}")
print("A. COMPOSICION: de cuantas familias curadas de CARD esta hecha cada una")
print(f"{'='*96}")

card_de = {}
if ARO and ARO.exists():
    aro_fam = {}
    with open(ARO, encoding="utf-8", errors="replace") as fh:
        r = csv.reader(fh, delimiter="\t")
        cab = next(r)
        # columna 1 = ARO Accession, columna 9 = AMR Gene Family (1-indexado)
        i_aro = 0
        i_fam = 8 if len(cab) > 8 else len(cab) - 1
        for fila in r:
            if len(fila) > i_fam:
                aro_fam[fila[i_aro].strip()] = fila[i_fam].strip()
    for gg in genes:
        m = re.search(r"(ARO:\d+)", gg)
        if m and m.group(1) in aro_fam:
            card_de[gg] = aro_fam[m.group(1)]
    print(f"\n  aro_index: {ARO.relative_to(RAIZ)}  ({len(aro_fam):,} ARO)")
    print(f"  genes con familia CARD resuelta: {len(card_de):,} de {len(genes):,}")
else:
    print("\n  [aviso] no encontre aro_index.tsv: se salta esta parte")

composicion = {}
if card_de:
    print(f"\n{'nuestra familia':<40}{'genes':>7}{'fams CARD':>11}{'la mayor':>10}  ejemplos")
    for c in clases:
        gs = [gg for gg in genes if etq[gg] == c and gg in card_de]
        if not gs:
            continue
        cc = Counter(card_de[gg] for gg in gs)
        mayor = cc.most_common(1)[0]
        composicion[c] = cc
        ej = ", ".join(f"{k[:26]}({v})" for k, v in cc.most_common(3))
        print(f"{c[:40]:<40}{len(gs):>7}{len(cc):>11}{100*mayor[1]/len(gs):>9.0f}%  {ej}")
    print("\n  'la mayor' = que porcentaje de la familia ocupa su familia CARD mas comun.")
    print("  Un valor bajo con muchas familias CARD = agrupacion heterogenea.")

# =============================================================================
#  B. SIMILITUD DENTRO vs FUERA DE LA FAMILIA
# =============================================================================
print(f"\n{'='*96}")
print("B. SIMILITUD: se parecen entre si los genes de una misma familia?")
print(f"{'='*96}")

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


print(f"\n  calculando vectores de {4**K} k-meros para {len(genes):,} genes...")
X = np.empty((len(genes), 4 ** K), dtype=np.float32)
for i, gg in enumerate(genes):
    X[i] = vector_kmer(fa[gg])
Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
S = Xn @ Xn.T                                   # matriz de similitud coseno
np.fill_diagonal(S, np.nan)                     # un gen consigo mismo no cuenta

filas = []
for c in clases:
    m = fam == c
    intra = np.nanmean(S[np.ix_(m, m)])
    inter = np.nanmean(S[np.ix_(m, ~m)])
    filas.append(dict(familia=c, genes=int(m.sum()),
                      intra=float(intra), inter=float(inter),
                      razon=float(intra / inter) if inter else np.nan,
                      fams_card=len(composicion.get(c, {})) or None))
t = pd.DataFrame(filas).sort_values("razon", ascending=False)

print(f"\n{'familia':<40}{'genes':>7}{'intra':>8}{'inter':>8}{'razon':>8}{'CARD':>7}")
for _, r in t.iterrows():
    nc = f"{int(r.fams_card)}" if r.fams_card else "-"
    print(f"{r.familia[:40]:<40}{r.genes:>7}{r.intra:>8.3f}{r.inter:>8.3f}"
          f"{r.razon:>8.3f}{nc:>7}")
print("\n  intra = cuanto se parecen entre si los genes DE la familia")
print("  inter = cuanto se parecen a los genes de las OTRAS familias")
print("  razon = intra/inter. Cerca de 1 -> la familia no es mas homogenea que el resto.")

# =============================================================================
#  C. EXPLICA ESTO EL DESEMPENO DEL MODELO?
# =============================================================================
print(f"\n{'='*96}")
print("C. LA COHERENCIA EXPLICA EL ACIERTO DEL MODELO?")
print(f"{'='*96}")

if CM.exists():
    cm = pd.read_csv(CM, index_col=0)
    rec = {}
    for c in cm.index:
        tot = cm.loc[c].sum()
        if tot:
            rec[c] = cm.loc[c, c] / tot
    t["recall"] = t.familia.map(rec)
    sub = t.dropna(subset=["recall"])
    if len(sub) >= 4:
        n = len(sub)
        # t critico de dos colas al 5%, segun los grados de libertad (n-2)
        TC = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
              9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145}

        def corr(x, y, nom):
            r = float(np.corrcoef(x, y)[0, 1])
            t_ = r * np.sqrt(n - 2) / np.sqrt(max(1 - r ** 2, 1e-12))
            tc = TC.get(n - 2, 2.0)
            sig = "significativa" if abs(t_) > tc else "NO significativa"
            print(f"  {nom:<36} r={r:+.3f}  t={t_:+.2f}  {sig}")
            return r, abs(t_) > tc

        print(f"\n  Correlaciones con el recall (n={n} familias, t critico 5% = "
              f"{TC.get(n-2, 2.0)}):")
        cr, sig_r = corr(sub.razon, sub.recall, "coherencia por k-meros")
        cg, _ = corr(sub.genes, sub.recall, "numero de genes")
        if sub.fams_card.notna().sum() >= 4:
            sc = sub.dropna(subset=["fams_card"])
            rc = float(np.corrcoef(sc.fams_card, sc.recall)[0, 1])
            print(f"  {'numero de familias CARD':<36} r={rc:+.3f}")

        # cuantas familias rompen la relacion: coherencia baja pero recall alto,
        # o al reves. Una correlacion puede ser significativa y aun asi tener
        # contraejemplos que la hacen poco util para decidir.
        med_r, med_c = sub.recall.median(), sub.razon.median()
        rotas = sub[((sub.razon < med_c) & (sub.recall > med_r)) |
                    ((sub.razon > med_c) & (sub.recall < med_r))]
        print(f"\n  {'-'*86}")
        if sig_r and abs(cr) > abs(cg) + 0.2:
            print("  La coherencia por k-meros predice el recall mejor que el tamano,")
            print("  y la correlacion es significativa.")
        elif abs(cr) > 0.3:
            print("  Hay relacion entre coherencia y recall, pero no es fuerte.")
        else:
            print("  La coherencia por k-meros NO predice el recall.")
        if len(rotas):
            print(f"\n  PERO {len(rotas)} de {n} familias rompen la relacion:")
            for _, r in rotas.iterrows():
                print(f"      {str(r.familia)[:38]:<38} coherencia {r.razon:.3f}  "
                      f"recall {r.recall:.3f}")
        print("\n  CUIDADO AL INTERPRETAR: el coseno de k-meros mide COMPOSICION global,")
        print("  que refleja sobre todo el organismo de origen, no la funcion ni la")
        print("  homologia. Dos genes sin ningun parentesco pueden salir 'parecidos' si")
        print("  vienen de bacterias con el mismo uso de codones. Para responder de")
        print("  verdad si los genes de una familia se parecen, la medida correcta es")
        print("  la identidad de secuencia: vuelve a correr con --blast.")
        if composicion:
            sc = t.dropna(subset=["fams_card", "recall"])
            if len(sc) >= 4:
                cf = np.corrcoef(sc.fams_card, sc.recall)[0, 1]
                print(f"\n  correlacion NUM. DE FAMILIAS CARD con el recall: {cf:+.3f}")
                print("  (negativa = mientras mas familias de CARD junta, peor acierta)")
    print(f"\n{'familia':<40}{'razon':>8}{'recall':>9}{'genes':>7}{'CARD':>7}")
    for _, r in t.sort_values("recall").iterrows():
        rc = f"{r.recall:.3f}" if pd.notna(r.get("recall")) else "-"
        nc = f"{int(r.fams_card)}" if r.fams_card else "-"
        print(f"{r.familia[:40]:<40}{r.razon:>8.3f}{rc:>9}{r.genes:>7}{nc:>7}")
else:
    print(f"\n  [aviso] no encontre {CM.name}: corre antes reporte_familias.py")

t.to_csv(SAL / "coherencia_familias.csv", index=False)

# =============================================================================
#  D. LAS DOS FAMILIAS PROBLEMATICAS, EN DETALLE
# =============================================================================
peores = t.nsmallest(2, "razon").familia.tolist()
if "recall" in t.columns and t.recall.notna().any():
    peores = list(dict.fromkeys(peores + t.nsmallest(2, "recall").familia.tolist()))[:3]

print(f"\n{'='*96}")
print("D. LAS FAMILIAS MENOS COHERENTES, EN DETALLE")
print(f"{'='*96}")
for c in peores:
    m = fam == c
    print(f"\n--- {c}  ({int(m.sum())} genes) ---")
    sim = S[np.ix_(m, m)]
    v = sim[~np.isnan(sim)]
    print(f"  similitud entre sus genes: media {v.mean():.3f} | "
          f"mediana {np.median(v):.3f} | min {v.min():.3f} | max {v.max():.3f}")
    if c in composicion:
        cc = composicion[c]
        print(f"  esta hecha de {len(cc)} familias curadas de CARD:")
        for k, n in cc.most_common(12):
            print(f"      {n:>4}  {k}")
        if len(cc) > 12:
            print(f"      ... y {len(cc)-12} familias mas")

# =============================================================================
#  E. BLAST (opcional): identidad real de secuencia
# =============================================================================
if args.blast:
    print(f"\n{'='*96}")
    print("E. IDENTIDAD REAL POR BLASTN (todos contra todos, dentro de cada familia)")
    print(f"{'='*96}")
    if not subprocess.run(["which", "blastn"], capture_output=True).returncode == 0:
        print("\n  [aviso] no encontre blastn en el PATH; se salta")
    else:
        res = []
        for c in clases:
            gs = [gg for gg in genes if etq[gg] == c][:args.max_blast]
            if len(gs) < 2:
                continue
            with tempfile.TemporaryDirectory() as td:
                f = Path(td) / "fam.fasta"
                with open(f, "w") as fh:
                    for gg in gs:
                        fh.write(f">{gg}\n{fa[gg]}\n")
                subprocess.run(["makeblastdb", "-in", str(f), "-dbtype", "nucl",
                                "-out", str(Path(td) / "db")],
                               capture_output=True)
                out = subprocess.run(
                    ["blastn", "-query", str(f), "-db", str(Path(td) / "db"),
                     "-outfmt", "6 qseqid sseqid pident", "-evalue", "1e-5",
                     "-max_target_seqs", str(len(gs))],
                    capture_output=True, text=True).stdout
            pares, ids = set(), []
            for ln in out.splitlines():
                p = ln.split("\t")
                if len(p) >= 3 and p[0] != p[1]:
                    pares.add(frozenset((p[0], p[1])))
                    ids.append(float(p[2]))
            posibles = len(gs) * (len(gs) - 1) / 2
            res.append(dict(familia=c, genes=len(gs),
                            pares_con_hit=len(pares),
                            pct_pares=100 * len(pares) / posibles if posibles else 0,
                            identidad_media=np.mean(ids) if ids else np.nan))
            print(f"  {c[:40]:<40} {len(pares):>5} de {int(posibles):>5} pares "
                  f"({100*len(pares)/posibles if posibles else 0:>5.1f}%)  "
                  f"identidad media {np.mean(ids) if ids else float('nan'):>5.1f}%")
        if res:
            pd.DataFrame(res).to_csv(SAL / "coherencia_blast.csv", index=False)
            print("\n  'pares con hit' = cuantas parejas de genes de la familia se")
            print("  alinean entre si. Un porcentaje bajo = la familia no comparte")
            print("  homologia, solo una etiqueta.")

# =============================================================================
#  FIGURA
# =============================================================================
if "recall" in t.columns and t.recall.notna().sum() >= 4:
    sub = t.dropna(subset=["recall"])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    sc = ax[0].scatter(sub.razon, sub.recall, s=sub.genes * 1.6,
                       c=sub.genes, cmap="viridis", alpha=.8, edgecolor="k", linewidth=.6)
    for _, r in sub.iterrows():
        ax[0].annotate(str(r.familia)[:22], (r.razon, r.recall), fontsize=7.5,
                       xytext=(5, 4), textcoords="offset points")
    ax[0].set_xlabel("coherencia  (similitud dentro / fuera de la familia)")
    ax[0].set_ylabel("recall del modelo")
    ax[0].set_title(f"Coherencia contra acierto\ncorrelacion "
                    f"{np.corrcoef(sub.razon, sub.recall)[0,1]:+.3f}  "
                    f"(el tamano del punto = genes)")
    ax[0].grid(alpha=.25)
    fig.colorbar(sc, ax=ax[0], label="genes en la familia")

    o = t.sort_values("razon")
    col = ["#C44E52" if r < o.razon.median() else "#4C72B0" for r in o.razon]
    ax[1].barh(range(len(o)), o.razon, color=col)
    ax[1].axvline(1.0, color="k", ls="--", lw=1.2)
    ax[1].text(1.02, len(o) - 1.4, "1.0 = igual de parecidos\ndentro que fuera",
               fontsize=8, va="top")
    ax[1].set_yticks(range(len(o)))
    ax[1].set_yticklabels([f"{str(f)[:30]} ({g})" for f, g in zip(o.familia, o.genes)],
                          fontsize=8)
    ax[1].set_xlabel("coherencia (intra / inter)")
    ax[1].set_title("Que tan homogenea es cada familia")
    plt.tight_layout()
    plt.savefig(SAL / "coherencia_familias.png", dpi=200)
    plt.close()

print(f"\n{'='*96}")
print("LISTO. En results/familias/:")
print("   coherencia_familias.csv")
if (SAL / "coherencia_familias.png").exists():
    print("   coherencia_familias.png")
if args.blast and (SAL / "coherencia_blast.csv").exists():
    print("   coherencia_blast.csv")
print("=" * 96)