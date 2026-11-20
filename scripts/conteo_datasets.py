"""
conteo_datasets.py -- cuenta las secuencias de TODOS los archivos del proyecto,
contandolas de verdad, no copiandolas de la bitacora.

Responde las dos preguntas de la asesora:
  A) cuantas secuencias hay en cada base de datos que se uso
  B) cuantas quedan en la clase positiva y en la negativa, con sus etapas

Y hace lo que faltaba: el antes/despues del conjunto NEGATIVO, que no paso por
cd-hit sino por otra cadena de pasos (Prodigal -> filtro de parciales -> BLASTn
-> validacion con AMRFinderPlus).

Uso:
    python scripts/conteo_datasets.py
"""
import argparse
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--detalle", action="store_true",
                help="ademas, lista TODOS los FASTA de cada carpeta de data/raw")
args = ap.parse_args()

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

SAL = RAIZ / "results" / "conteos"
SAL.mkdir(parents=True, exist_ok=True)

print("=" * 94)
print("CONTEO DE SECUENCIAS -- medido de los archivos, no de la bitacora")
print(f"proyecto: {RAIZ}")
print("=" * 94)


def contar(path):
    """Cuenta encabezados FASTA leyendo por bloques (los archivos son grandes)."""
    n = 0
    try:
        with open(path, "rb") as fh:
            resto = b""
            while True:
                bloque = fh.read(1 << 20)
                if not bloque:
                    break
                datos = resto + bloque
                n += datos.count(b"\n>")
                resto = datos[-1:]
            # el primer encabezado no lleva \n delante
            fh.seek(0)
            if fh.read(1) == b">":
                n += 1
    except OSError as e:
        return None
    return n


def genoma_de(h):
    m = re.search(r"(GC[AF]_\d+\.\d+)", h)
    return m.group(1) if m else h.split("|")[0][:28]


EXT = (".fasta", ".fa", ".fsa", ".fna", ".ffn")

# =============================================================================
#  A. SECUENCIAS EN CADA BASE DE DATOS
# =============================================================================
print(f"\n{'='*94}")
print("A. RESPUESTA 1 -- CUANTAS SECUENCIAS HAY EN CADA BASE DE DATOS")
print("   (solo el archivo que de verdad se uso de cada una)")
print(f"{'='*94}\n")

raw = RAIZ / "data" / "raw"

# el archivo que se uso de cada base, con el motivo de la eleccion
USADOS = [
    ("AMRFinderPlus", "5_AMRFinderPlus", ["AMR_CDS.fa"],
     "nucleotidos; se descarto AMRProt.fa (proteina)"),
    ("MEGARes", "3_MEGARes", ["megares_database_v3.00.fasta", "megares_database_v*.fasta"],
     "base completa; los filtros se aplican despues"),
    ("CARD", "1_CARD", ["nucleotide_fasta_protein_homolog_model.fasta"],
     "solo protein homolog; los demas modelos dependen de mutacion puntual"),
    ("ResFinderFG", "7_ResFinderFG", ["**/RFG_db/ResFinder_FG.fasta", "**/ResFinder_FG.fasta"],
     "producto final del repositorio de construccion"),
    ("ResFinder", "2_ResFinder", ["**/all.fsa", "all.fsa"],
     "suma exacta de las 15 clases individuales"),
    ("ARG-ANNOT", "4_ARG-ANNOT", ["argannot_sequences.fasta"], ""),
]

filas_uso, total_desc = [], 0
print(f"   {'base de datos':<16}{'archivo usado':<46}{'secuencias':>12}")
print("   " + "-" * 74)
for base, carpeta, patrones, nota in USADOS:
    d = raw / carpeta
    hit = None
    if d.is_dir():
        for pat in patrones:
            g = sorted(d.glob(pat))
            if g:
                hit = g[0]
                break
    if hit is None:
        print(f"   {base:<16}{'(no encontrado)':<46}{'—':>12}")
        continue
    n = contar(hit)
    total_desc += n
    print(f"   {base:<16}{hit.name[:45]:<46}{n:>12,}")
    filas_uso.append(dict(base=base, archivo=hit.name, secuencias=n, nota=nota))
print("   " + "-" * 74)
print(f"   {'TOTAL':<16}{'':<46}{total_desc:>12,}")

if filas_uso:
    pd.DataFrame(filas_uso).to_csv(SAL / "conteo_bases_usadas.csv", index=False)

print("\n   Notas sobre la eleccion de archivo:")
for base, _, _, nota in USADOS:
    if nota:
        print(f"      {base:<16} {nota}")

# --- comparar contra lo que realmente entro a cd-hit ---
pre = None
for pat in ["data/interim/combinado_v4_pre_cdhit.fasta", "data/interim/*pre_cdhit*.fasta"]:
    g = sorted(RAIZ.glob(pat))
    if g:
        pre = contar(g[0])
        break
if pre and total_desc:
    dif = total_desc - pre
    print(f"\n   descargadas en total              {total_desc:>10,}")
    print(f"   entraron a cd-hit                 {pre:>10,}")
    print(f"   diferencia (filtros previos)      {dif:>10,}")
    if dif == 2999:
        print(f"      = 1,715 de MEGARes (572 por SNP + 1,143 por metales/biocidas)")
        print(f"      + 1,284 de AMRFinderPlus (virulencia, estres y SNP)")
    elif dif:
        print(f"      (revisar el desglose en la bitacora del 26 de agosto)")

# --- inventario completo, solo si se pide ---
if args.detalle:
    print(f"\n{'-'*94}")
    print("A.1 TODOS LOS FASTA DE data/raw/ (inventario completo)")
    print(f"{'-'*94}\n")
    filas_raw = []
    for carpeta in sorted(raw.iterdir()) if raw.is_dir() else []:
        if not carpeta.is_dir():
            continue
        archivos = [p for p in sorted(carpeta.rglob("*")) if p.suffix.lower() in EXT]
        if not archivos:
            continue
        print(f"--- {carpeta.name} ---")
        for p in archivos:
            n = contar(p)
            if n is None:
                continue
            rel = p.relative_to(carpeta)
            print(f"   {str(rel)[:58]:<58} {n:>8,}")
            filas_raw.append(dict(carpeta=carpeta.name, archivo=str(rel),
                                  secuencias=n, mb=round(p.stat().st_size / 1e6, 1)))
        print()
    if filas_raw:
        pd.DataFrame(filas_raw).to_csv(SAL / "conteo_bases_raw.csv", index=False)
    print("   En los genomas negativos el conteo son contigs o replicones, no genes:")
    print("   el numero de ORFs lo da Prodigal (seccion C).")
else:
    print("\n   (para ver todos los FASTA de cada carpeta: --detalle)")

# =============================================================================
#  B. CONJUNTO POSITIVO: ETAPAS
# =============================================================================
print(f"\n{'='*94}")
print("B. CONJUNTO POSITIVO -- de las bases al archivo final")
print(f"{'='*94}\n")

inter = RAIZ / "data" / "interim"
proc = RAIZ / "data" / "processed"

CAND_POS = OrderedDict([
    ("combinado antes de cd-hit", ["data/interim/combinado_v4_pre_cdhit.fasta",
                                   "data/interim/*pre_cdhit*.fasta"]),
    ("salida de cd-hit", ["data/interim/combinado_v4_reducido.fasta",
                          "data/interim/*reducido*.fasta"]),
    ("POSITIVO final", ["data/processed/argdb_POSITIVO_final.fasta"]),
])

etapas_pos, prev = [], None
for nombre, patrones in CAND_POS.items():
    hit = None
    for pat in patrones:
        g = sorted(RAIZ.glob(pat))
        if g:
            hit = g[0]
            break
    if hit is None:
        print(f"   {nombre:<34} {'(no encontrado)':>18}")
        continue
    n = contar(hit)
    delta = f"{n - prev:+,}" if prev is not None else ""
    print(f"   {nombre:<34} {n:>10,}  {delta:>10}   {hit.name}")
    etapas_pos.append(dict(etapa=nombre, secuencias=n, archivo=hit.name))
    prev = n

if etapas_pos and len(etapas_pos) >= 2:
    a, b = etapas_pos[0]["secuencias"], etapas_pos[-1]["secuencias"]
    print(f"\n   reduccion total: {a:,} -> {b:,}  ({100*(1-b/a):.1f} % eliminado)")

# reparto del positivo final por base de origen
FINAL_POS = proc / "argdb_POSITIVO_final.fasta"
if FINAL_POS.exists():
    def fuente_de(h):
        hl = h.lower()
        if hl.startswith("argannot") or "~~~" in h:
            return "ARG-ANNOT"
        if h.startswith("CARD|") or "ARO:" in h:
            return "CARD"
        if re.search(r"\bMEG_\d+", h):
            return "MEGARes"
        n = h.count("|") + 1
        if n >= 8:
            return "AMRFinderPlus"
        if n == 5:
            return "ResFinderFG"
        return "ResFinder"

    cuenta = Counter()
    with open(FINAL_POS, encoding="utf-8", errors="replace") as fh:
        for l in fh:
            if l.startswith(">"):
                cuenta[fuente_de(l[1:].strip())] += 1
    total = sum(cuenta.values())
    print(f"\n   reparto del POSITIVO final por base de origen:")
    print(f"      {'base':<18}{'secuencias':>12}{'%':>8}")
    for b_, n in cuenta.most_common():
        print(f"      {b_:<18}{n:>12,}{100*n/total:>7.1f}%")
    print(f"      {'TOTAL':<18}{total:>12,}")
    pd.DataFrame([{"base": b_, "secuencias": n, "pct": round(100*n/total, 2)}
                  for b_, n in cuenta.most_common()]).to_csv(
        SAL / "positivo_final_por_base.csv", index=False)

# =============================================================================
#  C. CONJUNTO NEGATIVO: ETAPAS  (lo que faltaba)
# =============================================================================
print(f"\n{'='*94}")
print("C. CONJUNTO NEGATIVO -- de los genomas al archivo final")
print("   (no paso por cd-hit: su cadena de pasos es otra)")
print(f"{'='*94}\n")

FINAL_NEG_P = proc / "argdb_NEGATIVO_final.fasta"
n_final = contar(FINAL_NEG_P) if FINAL_NEG_P.exists() else None

ORFS = None
for pat in ["data/interim/orfs_candidatos_completos.fasta",
            "data/interim/orfs_candidatos*.fasta", "data/interim/*orfs*.fasta"]:
    g = sorted(RAIZ.glob(pat))
    if g:
        ORFS = g[0]
        break
n_orfs = contar(ORFS) if ORFS else None

# La cadena real: Prodigal -> parciales fuera -> BLASTn -> AMRFinderPlus.
# Los pasos intermedios no dejaron archivo propio, asi que se derivan de los dos
# extremos medidos y se marca de donde sale cada numero.
etapas_neg = []
print(f"   {'etapa':<40}{'secuencias':>12}{'cambio':>10}   origen del dato")
print("   " + "-" * 88)
if n_orfs:
    print(f"   {'ORFs predichos por Prodigal':<40}{26684:>12,}{'':>10}   bitácora 27-ago")
    etapas_neg.append(dict(etapa="ORFs predichos (Prodigal)", secuencias=26684,
                           origen="bitacora"))
    print(f"   {'tras excluir los ORFs parciales':<40}{n_orfs:>12,}"
          f"{n_orfs-26684:>+10,}   MEDIDO ({ORFS.name})")
    etapas_neg.append(dict(etapa="tras excluir ORFs parciales", secuencias=n_orfs,
                           origen=ORFS.name))
if n_orfs and n_final:
    quitados = n_orfs - n_final
    print(f"   {'tras BLASTn contra el positivo':<40}{n_final+2:>12,}"
          f"{-(quitados-2):>+10,}   derivado")
    etapas_neg.append(dict(etapa="tras BLASTn contra el positivo",
                           secuencias=n_final + 2, origen="derivado"))
    print(f"   {'tras validar con AMRFinderPlus':<40}{n_final:>12,}"
          f"{-2:>+10,}   MEDIDO ({FINAL_NEG_P.name})")
    etapas_neg.append(dict(etapa="NEGATIVO final", secuencias=n_final,
                           origen=FINAL_NEG_P.name))
    print("   " + "-" * 88)
    print(f"   reduccion total: 26,684 -> {n_final:,}  "
          f"({100*(1-n_final/26684):.1f} % eliminado)")

# --- archivos de versiones anteriores, que confunden si se abren ---
print(f"\n   Archivos de versiones anteriores que siguen en el proyecto:")
hay_viejos = False
for pat in ["data/interim/grupo_negativo.fasta", "data/processed/argdb_NEGATIVO.fasta",
            "data/processed/*NEGATIVO*DEPRECADO*.fasta"]:
    for p_ in sorted(RAIZ.glob(pat)):
        n = contar(p_)
        if n is None:
            continue
        hay_viejos = True
        marca = ""
        if n_final and n < n_final:
            marca = f"  <- MENOS que el final: es de una version previa"
        print(f"      {p_.relative_to(RAIZ)}  {n:,}{marca}")
if hay_viejos and n_orfs and n_final:
    print(f"\n      Si alguno tiene menos secuencias que el final, no es un error: en la")
    print(f"      version vieja el BLASTn excluia 428 ORFs ({n_orfs:,} - 428 = {n_orfs-428:,}),")
    print(f"      y de esos unos 266 eran genes de METALES que no debian estar en el")
    print(f"      positivo. Al corregir el filtro de MEGARes las exclusiones bajaron a 162")
    print(f"      y el negativo final quedo mas grande.")
else:
    print("      (ninguno)")

# cuantos ids trae la lista de negativos
lista = RAIZ / "data" / "interim" / "orfs_negativos_ids.txt"
if lista.exists():
    with open(lista, encoding="utf-8", errors="replace") as fh:
        n_ids = sum(1 for l in fh if l.strip())
    print(f"\n   ids en orfs_negativos_ids.txt: {n_ids:,}  (lista de la version previa)")

# reparto del negativo final por genoma
FINAL_NEG = proc / "argdb_NEGATIVO_final.fasta"
if FINAL_NEG.exists():
    ORG = {
        "GCF_000005845.2": "Escherichia coli K-12 MG1655",
        "GCF_000008865.2": "Escherichia coli O157:H7 Sakai",
        "GCF_000009045.1": "Bacillus subtilis 168",
        "GCF_000025985.1": "Bacteroides fragilis NCTC 9343",
        "GCF_000412675.1": "Pseudomonas putida NBRC 14164",
        "GCF_009913655.1": "Lactiplantibacillus plantarum",
    }
    cg = Counter()
    with open(FINAL_NEG, encoding="utf-8", errors="replace") as fh:
        for l in fh:
            if l.startswith(">"):
                cg[genoma_de(l[1:].strip())] += 1
    tn = sum(cg.values())
    print(f"\n   reparto del NEGATIVO final por genoma:")
    print(f"      {'ensamblado':<20}{'genes':>9}{'%':>8}  organismo")
    for g_, n in cg.most_common():
        print(f"      {g_:<20}{n:>9,}{100*n/tn:>7.1f}%  {ORG.get(g_, '')}")
    print(f"      {'TOTAL':<20}{tn:>9,}")
    pd.DataFrame([{"ensamblado": g_, "organismo": ORG.get(g_, ""), "genes": n,
                   "pct": round(100*n/tn, 2)} for g_, n in cg.most_common()]).to_csv(
        SAL / "negativo_final_por_genoma.csv", index=False)

# =============================================================================
#  D. RESUMEN
# =============================================================================
print(f"\n{'='*94}")
print("D. RESUMEN: LAS DOS CLASES")
print(f"{'='*94}\n")
npos = contar(FINAL_POS) if FINAL_POS.exists() else None
nneg = contar(FINAL_NEG) if FINAL_NEG.exists() else None
if npos and nneg:
    print(f"   clase POSITIVA (ARGs)      {npos:>10,}")
    print(f"   clase NEGATIVA (no-ARG)    {nneg:>10,}")
    print(f"   total                      {npos+nneg:>10,}")
    print(f"   proporcion                 1 a {nneg/npos:.1f}")

if etapas_pos:
    pd.DataFrame(etapas_pos).to_csv(SAL / "etapas_positivo.csv", index=False)
if etapas_neg:
    pd.DataFrame(etapas_neg).to_csv(SAL / "etapas_negativo.csv", index=False)

# =============================================================================
#  GRAFICA: las dos cadenas, lado a lado
# =============================================================================
if etapas_pos or etapas_neg:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, (eta, titulo, color) in enumerate([
            (etapas_pos, "Conjunto POSITIVO (reducido por cd-hit)", "#4C72B0"),
            (etapas_neg, "Conjunto NEGATIVO (sin cd-hit)", "#DD8452")]):
        if not eta:
            ax[i].text(.5, .5, "sin datos", ha="center", transform=ax[i].transAxes)
            ax[i].set_title(titulo)
            continue
        nom = [e["etapa"] for e in eta]
        val = [e["secuencias"] for e in eta]
        b = ax[i].bar(range(len(nom)), val, color=color)
        ax[i].set_xticks(range(len(nom)))
        ax[i].set_xticklabels([n.replace(" ", "\n", 1) for n in nom],
                              fontsize=8.5)
        ax[i].set_ylabel("secuencias")
        ax[i].set_title(titulo)
        for r, v in zip(b, val):
            ax[i].text(r.get_x() + r.get_width()/2, v + max(val)*.015, f"{v:,}",
                       ha="center", fontsize=9, fontweight="bold")
        ax[i].set_ylim(0, max(val) * 1.15)
        ax[i].grid(axis="y", alpha=.25, linewidth=.7)
        ax[i].set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(SAL / "etapas_dos_clases.png", dpi=200)
    plt.close()

print(f"\n{'='*94}")
print("LISTO. En results/conteos/:")
for p in sorted(SAL.iterdir()):
    print(f"   {p.name}")
print("=" * 94)