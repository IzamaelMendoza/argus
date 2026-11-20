"""
tarea1_sets.py -- Tarea 1 de la reunion del 12-nov, la parte que solo necesita
los FASTAs (corre en segundos).

Responde:
  1. Hay genes unicos en el set positivo? (duplicados exactos, ids repetidos,
     secuencias que aparecen en mas de una base de datos)
  3. Cuantos ejemplos hay por familia (las 12) -> tabla + grafica
  4. Grafico de antes de cd-hit, ya normalizado -> antes vs despues por base de datos
  5. Cual es el gen mas largo y el mas corto del positivo -> + distribucion

El punto 2 (meter el set negativo al modelo) va en tarea1_negativos.py, que es
el lento porque tiene que cargar el modelo y predecir.

Uso:
    python scripts/tarea1_sets.py
    python scripts/tarea1_sets.py --raiz "ruta/al/Proyecto_Residencias"

Todo lo que produce queda en results/tarea1/.
"""
import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--positivo", default=None, help="FASTA positivo final (despues de cd-hit)")
ap.add_argument("--antes", default=None, help="FASTA positivo ANTES de cd-hit, ya normalizado")
ap.add_argument("--familias", default=None, help="TSV gen -> familia")
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

print("=" * 94)
print("TAREA 1 -- analisis de los sets (parte FASTA)")
print(f"proyecto: {RAIZ}")
print(f"salida:   {SAL}")
print("=" * 94)


# ----------------------------------------------------------------- utilidades
def leer_fasta(path):
    """Devuelve lista de (encabezado_completo, id, secuencia)."""
    out, h, s = [], None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            linea = linea.rstrip("\n")
            if linea.startswith(">"):
                if h is not None:
                    out.append((h, h.split()[0], "".join(s)))
                h, s = linea[1:], []
            elif h is not None:
                s.append(linea.strip())
    if h is not None:
        out.append((h, h.split()[0], "".join(s)))
    return out


def fuente_de(encabezado):
    """
    De que base de datos viene la secuencia, por el formato del encabezado.
    Los seis formatos que se mezclaron al armar el set:

      CARD          CARD|gb|AY769962.1|+|2434-5611|ARO:3000781|adeJ [organismo]
      MEGARes       ...|MEG_1234|Drugs|betalactams|ClassA|TEM
      ARGANNOT      argannot~~~(Bla)TEM-1~~~1-861~~~
      AMRFinderPlus ...|prot|nucl|n|n|SIMBOLO|simbolo|descripcion   (8 campos)
      ResFinderFG   ...|mecanismo|accesion|fuente|antibiotico       (5 campos)
      ResFinder     nombre_variante_accesion                        (2 campos)
    """
    h = encabezado
    hl = h.lower()
    if hl.startswith("argannot") or "~~~" in h:
        return "ARGANNOT"
    if h.startswith("CARD|") or "ARO:" in h:
        return "CARD"
    if re.search(r"\bMEG_\d+", h):
        return "MEGARes"
    n = h.count("|") + 1
    if n >= 8:
        return "AMRFinderPlus"
    if n == 5:
        return "ResFinderFG"
    if n == 2:
        return "ResFinder"
    # ResFinder tambien aparece sin pipes: nombre_variante_accesion
    if n == 1 and re.match(r"^[A-Za-z][\w()'-]*_\d+_[A-Z]{1,3}\d{4,}", h):
        return "ResFinder"
    return f"sin identificar ({n} campos)"


def localizar(explicito, patrones, descripcion):
    """Busca un archivo: primero el que se dio a mano, si no, por patrones."""
    if explicito:
        p = Path(explicito).expanduser()
        if not p.is_absolute():
            p = RAIZ / p
        if p.exists():
            return p
        sys.exit(f"No existe el archivo que diste para {descripcion}: {p}")
    vistos = []
    for pat in patrones:
        for p in sorted(RAIZ.glob(pat)):
            if p.is_file() and p not in vistos:
                vistos.append(p)
    return vistos[0] if vistos else None


# =============================================================================
#  INVENTARIO
# =============================================================================
print("\n### ARCHIVOS ###")

POS = localizar(args.positivo,
                ["data/processed/argdb_POSITIVO_final.fasta",
                 "data/processed/*POSITIVO*final*.fasta",
                 "data/processed/*positiv*.fasta",
                 "data/interim/*positiv*.fasta"],
                "el FASTA positivo")
if POS is None:
    sys.exit("No encontre el FASTA positivo. Pasalo con --positivo")
print(f"positivo (final):  {POS.relative_to(RAIZ)}")

ANTES = localizar(args.antes,
                  ["data/interim/*normaliz*.fasta",
                   "data/interim/*antes*cdhit*.fasta",
                   "data/interim/*pre*cdhit*.fasta",
                   "data/interim/*combinad*.fasta",
                   "data/interim/*unificad*.fasta",
                   "data/interim/*todos*.fasta",
                   "data/processed/*normaliz*.fasta"],
                  "el FASTA antes de cd-hit")
if ANTES:
    print(f"antes de cd-hit:   {ANTES.relative_to(RAIZ)}")
else:
    print("antes de cd-hit:   NO ENCONTRADO")

FAM = localizar(args.familias,
                ["results/familias/etiquetas_finales.tsv",
                 "results/familias/*etiqueta*.tsv"],
                "el TSV de familias")
if FAM:
    print(f"familias:          {FAM.relative_to(RAIZ)}")
else:
    print("familias:          NO ENCONTRADO (corre antes modelo_familias.py)")

if ANTES is None:
    print("\n  Para el punto 4 hace falta el FASTA de antes de cd-hit. Si existe con")
    print("  otro nombre, pasalo:  --antes data/interim/<nombre>.fasta")
    print("  Estos son los .fasta que hay en el proyecto:")
    for p in sorted(RAIZ.glob("data/**/*.fasta"))[:25]:
        print(f"      {p.relative_to(RAIZ)}  ({p.stat().st_size/1e6:.1f} MB)")

# =============================================================================
#  PUNTO 1: genes unicos en el set positivo
# =============================================================================
print("\n" + "=" * 94)
print("PUNTO 1 -- genes unicos en el set positivo")
print("=" * 94)

pos = leer_fasta(POS)
print(f"\nsecuencias en el archivo: {len(pos):,}")

# 1a. ids repetidos
ids = [i for _, i, _ in pos]
ids_rep = {k: v for k, v in Counter(ids).items() if v > 1}
print(f"ids distintos: {len(set(ids)):,}")
if ids_rep:
    print(f"  [!] {len(ids_rep)} id(s) aparecen mas de una vez "
          f"({sum(ids_rep.values())-len(ids_rep)} secuencias de mas)")
    for k, v in list(Counter(ids_rep).most_common(5)):
        print(f"      {v}x  {k[:70]}")
else:
    print("  todos los ids son unicos")

# 1b. secuencias identicas (hash exacto, ignorando mayusculas)
por_seq = defaultdict(list)
for h, i, s in pos:
    por_seq[hashlib.md5(s.upper().encode()).hexdigest()].append((h, i, s))
dups = {k: v for k, v in por_seq.items() if len(v) > 1}
n_seq_unicas = len(por_seq)
print(f"\nsecuencias DISTINTAS: {n_seq_unicas:,} de {len(pos):,} "
      f"({100*n_seq_unicas/len(pos):.1f}%)")
if dups:
    extra = sum(len(v) for v in dups.values()) - len(dups)
    print(f"  [!] {len(dups):,} secuencia(s) aparecen repetidas "
          f"-> {extra:,} entradas de mas")
    print("\n  Los 5 grupos de duplicados mas grandes:")
    for k, v in sorted(dups.items(), key=lambda kv: -len(kv[1]))[:5]:
        fuentes = Counter(fuente_de(h) for h, _, _ in v)
        print(f"      {len(v)}x  {len(v[0][2])} pb  | fuentes: {dict(fuentes)}")
        for h, _, _ in v[:3]:
            print(f"            {h[:88]}")
    # cuantos duplicados cruzan bases de datos
    cruzan = sum(1 for v in dups.values()
                 if len({fuente_de(h) for h, _, _ in v}) > 1)
    print(f"\n  de esos grupos, {cruzan:,} tienen la MISMA secuencia en bases de")
    print("  datos distintas (es la redundancia que cd-hit deberia haber quitado)")
else:
    print("  no hay secuencias repetidas: todos los genes son unicos")

# 1c. reparto por base de datos
fuentes = Counter(fuente_de(h) for h, _, _ in pos)
print(f"\nreparto por base de datos de origen:")
for f, n in fuentes.most_common():
    print(f"  {f:<22} {n:>7,}  {100*n/len(pos):>5.1f}%")

# si algun formato no se reconocio, mostrar ejemplos para poder afinar fuente_de()
desconocidas = [f for f in fuentes if f.startswith("sin identificar")]
if desconocidas:
    print("\n  [!] hay encabezados que no reconoci. Ejemplos de cada formato,")
    print("      para ajustar fuente_de() en este script si hace falta:")
    for f in desconocidas:
        ej = [h for h, _, _ in pos if fuente_de(h) == f][:3]
        print(f"      --- {f} ({fuentes[f]:,} secuencias) ---")
        for e in ej:
            print(f"          {e[:86]}")

pd.DataFrame([{"fuente": f, "secuencias": n, "pct": round(100*n/len(pos), 2)}
              for f, n in fuentes.most_common()]).to_csv(
    SAL / "positivo_por_fuente.csv", index=False)

# =============================================================================
#  PUNTO 5: el mas largo y el mas corto
# =============================================================================
print("\n" + "=" * 94)
print("PUNTO 5 -- largo de los genes del set positivo")
print("=" * 94)

largos = np.array([len(s) for _, _, s in pos])
orden = np.argsort(largos)
print(f"\n  minimo : {largos.min():>8,} pb")
print(f"  maximo : {largos.max():>8,} pb")
print(f"  mediana: {int(np.median(largos)):>8,} pb")
print(f"  media  : {largos.mean():>8,.0f} pb")

print("\n  Los 5 MAS CORTOS:")
for k in orden[:5]:
    h, i, s = pos[k]
    print(f"      {len(s):>6,} pb  [{fuente_de(h)}]  {h[:70]}")
print("\n  Los 5 MAS LARGOS:")
for k in orden[-5:][::-1]:
    h, i, s = pos[k]
    print(f"      {len(s):>6,} pb  [{fuente_de(h)}]  {h[:70]}")

# cuantos se pierden con la regla de los 150 pb
for umbral in (150, 500):
    n = int((largos < umbral).sum())
    print(f"\n  genes de menos de {umbral} pb: {n:,} ({100*n/len(largos):.2f}%)")
    if umbral == 150 and n:
        print("    (estos son los 'no evaluables' con la regla de la asesora)")

pd.DataFrame({"id": [i for _, i, _ in pos],
              "fuente": [fuente_de(h) for h, _, _ in pos],
              "largo_pb": largos}).to_csv(SAL / "positivo_largos.csv", index=False)

# --- grafica de la distribucion de largos ---
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].hist(largos, bins=60, color="#4C72B0", edgecolor="white", linewidth=.4)
ax[0].axvline(150, color="#C44E52", ls="--", lw=1.4, label="150 pb (minimo)")
ax[0].axvline(500, color="#55A868", ls="--", lw=1.4, label="500 pb (ventana)")
ax[0].set_xlabel("largo del gen (pb)"); ax[0].set_ylabel("numero de genes")
ax[0].set_title("Distribucion de largos -- set positivo"); ax[0].legend(fontsize=8)

ax[1].hist(largos, bins=np.logspace(np.log10(max(largos.min(), 1)),
                                    np.log10(largos.max()), 60),
           color="#4C72B0", edgecolor="white", linewidth=.4)
ax[1].set_xscale("log")
ax[1].axvline(150, color="#C44E52", ls="--", lw=1.4)
ax[1].axvline(500, color="#55A868", ls="--", lw=1.4)
ax[1].set_xlabel("largo del gen (pb, escala log)")
ax[1].set_title(f"min {largos.min():,} | mediana {int(np.median(largos)):,} | "
                f"max {largos.max():,} pb")
plt.tight_layout(); plt.savefig(SAL / "positivo_largos.png", dpi=200); plt.close()
print(f"\n  grafica: {(SAL / 'positivo_largos.png').relative_to(RAIZ)}")

# =============================================================================
#  PUNTO 3: ejemplos por familia
# =============================================================================
print("\n" + "=" * 94)
print("PUNTO 3 -- ejemplos por familia en el set positivo")
print("=" * 94)

if FAM is None:
    print("\n  [saltado] no encontre el TSV de familias.")
else:
    t = pd.read_csv(FAM, sep="\t")
    col_gen = t.columns[0]
    col_fam = "etiqueta" if "etiqueta" in t.columns else t.columns[1]
    t = t[t[col_fam].notna() & (t[col_fam].astype(str).str.strip() != "")]
    cuenta = t[col_fam].value_counts()
    total = int(cuenta.sum())

    print(f"\n{len(cuenta)} familias | {total:,} genes etiquetados "
          f"(de {len(pos):,} positivos)")
    print(f"\n{'familia':<42} {'genes':>7} {'%':>7}")
    for fam, n in cuenta.items():
        print(f"{str(fam)[:42]:<42} {n:>7,} {100*n/total:>6.1f}%")

    print(f"\n  la mas grande: {cuenta.index[0]} ({cuenta.iloc[0]:,})")
    print(f"  la mas chica : {cuenta.index[-1]} ({cuenta.iloc[-1]:,})")
    print(f"  desbalance   : {cuenta.iloc[0]/cuenta.iloc[-1]:.1f} a 1")

    # cuantos genes positivos NO tienen familia
    sin_fam = len(pos) - total
    print(f"\n  genes positivos SIN familia: {sin_fam:,} "
          f"({100*sin_fam/len(pos):.1f}%)")
    print("  (son los que no vienen de CARD: sin nombre de gen no hay como")
    print("   sacarles la familia de la ontologia)")

    cuenta.rename("genes").to_frame().assign(
        pct=lambda d: (100*d.genes/total).round(2)).to_csv(SAL / "familias_conteo.csv")

    # --- grafica ---
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    col = ["#C44E52" if n < 60 else "#4C72B0" for n in cuenta.values]
    b = ax.barh(range(len(cuenta)), cuenta.values, color=col)
    ax.set_yticks(range(len(cuenta)))
    ax.set_yticklabels([str(f)[:38] for f in cuenta.index], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("numero de genes")
    ax.set_title(f"Ejemplos por familia -- set positivo ({len(cuenta)} familias, "
                 f"{total:,} genes)")
    for r, n in zip(b, cuenta.values):
        ax.text(n + total*0.004, r.get_y() + r.get_height()/2,
                f"{n:,}  ({100*n/total:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(0, cuenta.max()*1.22)
    ax.text(0.98, 0.04, "rojo = menos de 60 genes", transform=ax.transAxes,
            ha="right", fontsize=8, color="#C44E52")
    plt.tight_layout(); plt.savefig(SAL / "familias_conteo.png", dpi=200); plt.close()
    print(f"\n  grafica: {(SAL / 'familias_conteo.png').relative_to(RAIZ)}")

# =============================================================================
#  PUNTO 4: antes vs despues de cd-hit
# =============================================================================
print("\n" + "=" * 94)
print("PUNTO 4 -- antes de cd-hit (ya normalizado) vs despues")
print("=" * 94)

if ANTES is None:
    print("\n  [saltado] no encontre el FASTA de antes de cd-hit.")
    print("  Corre otra vez con:  --antes data/interim/<el archivo>.fasta")
else:
    antes = leer_fasta(ANTES)
    f_antes = Counter(fuente_de(h) for h, _, _ in antes)
    f_desp = fuentes
    l_antes = np.array([len(s) for _, _, s in antes])

    print(f"\n  antes de cd-hit : {len(antes):,} secuencias")
    print(f"  despues         : {len(pos):,} secuencias")
    red = 100*(1 - len(pos)/len(antes))
    print(f"  cd-hit quito    : {len(antes)-len(pos):,} ({red:.1f}%)")

    todas = sorted(set(f_antes) | set(f_desp),
                   key=lambda f: -f_antes.get(f, 0))
    print(f"\n{'base de datos':<18} {'antes':>9} {'despues':>9} {'quitado':>9} {'%':>7}")
    filas = []
    for f in todas:
        a, d = f_antes.get(f, 0), f_desp.get(f, 0)
        pc = 100*(1-d/a) if a else 0.0
        print(f"{f:<18} {a:>9,} {d:>9,} {a-d:>9,} {pc:>6.1f}%")
        filas.append({"fuente": f, "antes": a, "despues": d,
                      "quitado": a-d, "pct_quitado": round(pc, 2)})
    print(f"{'TOTAL':<18} {len(antes):>9,} {len(pos):>9,} "
          f"{len(antes)-len(pos):>9,} {red:>6.1f}%")
    filas.append({"fuente": "TOTAL", "antes": len(antes), "despues": len(pos),
                  "quitado": len(antes)-len(pos), "pct_quitado": round(red, 2)})
    pd.DataFrame(filas).to_csv(SAL / "cdhit_antes_despues.csv", index=False)

    print(f"\n  largo antes : min {l_antes.min():,} | mediana "
          f"{int(np.median(l_antes)):,} | max {l_antes.max():,} pb")
    print(f"  largo despues: min {largos.min():,} | mediana "
          f"{int(np.median(largos)):,} | max {largos.max():,} pb")

    # --- grafica: barras por base de datos + distribucion de largos ---
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))
    etq = [f for f in todas]
    x = np.arange(len(etq)); w = 0.38
    ax[0].bar(x - w/2, [f_antes.get(f, 0) for f in etq], w,
              label=f"antes de cd-hit ({len(antes):,})", color="#8C8C8C")
    ax[0].bar(x + w/2, [f_desp.get(f, 0) for f in etq], w,
              label=f"despues ({len(pos):,})", color="#4C72B0")
    ax[0].set_xticks(x); ax[0].set_xticklabels(etq, rotation=20, ha="right", fontsize=8)
    ax[0].set_ylabel("secuencias")
    ax[0].set_title("Set positivo por base de datos\n(normalizado, antes y despues de cd-hit)")
    ax[0].legend(fontsize=8)
    for xi, f in zip(x, etq):
        a, d = f_antes.get(f, 0), f_desp.get(f, 0)
        if a:
            ax[0].text(xi, max(a, d)*1.02, f"-{100*(1-d/a):.0f}%",
                       ha="center", fontsize=7.5, color="#C44E52")

    bins = np.logspace(np.log10(max(min(l_antes.min(), largos.min()), 1)),
                       np.log10(max(l_antes.max(), largos.max())), 60)
    ax[1].hist(l_antes, bins=bins, color="#8C8C8C", alpha=.75, label="antes")
    ax[1].hist(largos, bins=bins, color="#4C72B0", alpha=.75, label="despues")
    ax[1].set_xscale("log"); ax[1].set_xlabel("largo (pb, escala log)")
    ax[1].set_ylabel("secuencias"); ax[1].legend(fontsize=8)
    ax[1].set_title("Distribucion de largos")
    plt.tight_layout(); plt.savefig(SAL / "cdhit_antes_despues.png", dpi=200); plt.close()
    print(f"\n  grafica: {(SAL / 'cdhit_antes_despues.png').relative_to(RAIZ)}")

# =============================================================================
print("\n" + "=" * 94)
print("LISTO. Archivos en results/tarea1/:")
for p in sorted(SAL.iterdir()):
    print(f"   {p.name}")
print("\nFalta el punto 2 (set negativo por el modelo): corre tarea1_negativos.py")
print("=" * 94)
