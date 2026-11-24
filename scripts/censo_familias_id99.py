# -*- coding: utf-8 -*-
"""
censo_familias_id99.py  (v2)

Cuenta familias y ejemplos por familia en los conjuntos reducidos al 95 % y al
99 % de identidad, y los compara.

Reproduce EXACTAMENTE el etiquetado de scripts/modelo_familias.py:
  - solo genes de CARD (unica base con ontologia de familias)
  - si un gen tiene varias familias separadas por ';', se toma LA PRIMERA
  - la tabla GRUPOS se aplica EN ORDEN, ignorando mayusculas (re.I)

La v1 de este script difería del modelo real en dos familias por no hacer esas
dos cosas. Si el bloque del 95 % no reproduce 12 clases y 1,011 genes, algo
sigue sin cuadrar y hay que revisarlo antes de usar los numeros.

Uso, desde la raiz del proyecto:
    python scripts/censo_familias_id99.py --min-clase 30
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--id95", default="data/interim/combinado_v4_reducido.fasta")
ap.add_argument("--id99", default="data/interim/combinado_v4_reducido_id99.fasta")
ap.add_argument("--aro",  default="data/raw/1_CARD/aro_index.tsv")
ap.add_argument("--min-clase", type=int, default=30)
args = ap.parse_args()

# -------------------------------------------------- tabla de scripts/modelo_familias.py
GRUPOS = [
    (r"^OXA",                                        "OXA beta-lactamasa"),
    (r"^(IMP|VIM|NDM|KPC|GES|SPM|SIM|GIM)\b",        "carbapenemasa (IMP/VIM/NDM/KPC)"),
    (r"^(CTX-M|TEM|SHV|PER|VEB)\b",                  "beta-lactamasa de espectro extendido"),
    (r"^(ADC|ACT|CMY|FOX|MIR|DHA|MOX)\b",            "cefalosporinasa AmpC (ADC/ACT/CMY)"),
    (r"beta-lactamase",                              "otras beta-lactamasas"),
    (r"^AAC\(",                                      "AAC acetiltransferasa"),
    (r"^APH\(",                                      "APH fosfotransferasa"),
    (r"^ANT\(",                                      "ANT nucleotidiltransferasa"),
    (r"16S rRNA methyltransferase",                  "metiltransferasa 16S rRNA"),
    (r"aminoglycoside",                              "otros aminoglucósidos"),
    (r"^(van[A-Z]|Van ligase)",                      "van / glucopéptidos"),
    (r"glycopeptide resistance gene cluster",        "van / glucopéptidos"),
    (r"resistance-nodulation-cell division \(RND\)", "eflujo RND"),
    (r"major facilitator superfamily \(MFS\)",       "eflujo MFS"),
    (r"ATP-binding cassette \(ABC\)",                "eflujo ABC"),
    (r"small multidrug resistance \(SMR\)",          "eflujo SMR"),
    (r"multidrug and toxic compound extrusion",      "eflujo MATE"),
    (r"efflux regulatory protein",                   "regulador de eflujo"),
    (r"efflux pump",                                 "otras bombas de eflujo"),
    (r"^Erm|23S rRNA methyltransferase",             "Erm metiltransferasa 23S"),
    (r"macrolide phosphotransferase",                "MPH fosfotransferasa de macrólidos"),
    (r"lincosamide nucleotidyltransferase",          "LNU nucleotidiltransferasa"),
    (r"ABC-F",                                       "ABC-F protección ribosomal"),
    (r"ribosomal protection protein",                "protección ribosomal (tet)"),
    (r"tetracycline inactivation",                   "inactivación de tetraciclina"),
    (r"dihydrofolate reductase",                     "dfr dihidrofolato reductasa"),
    (r"chloramphenicol acetyltransferase",           "CAT cloranfenicol acetiltransferasa"),
    (r"fosfomycin thiol transferase",                "fosfomicina tiol transferasa"),
    (r"quinolone resistance protein",                "qnr quinolonas"),
    (r"phosphoethanolamine transferase",             "MCR fosfoetanolamina transferasa"),
    (r"rifampin|ADP-ribosyltransferase",             "rifampicina (arr)"),
    (r"sulfonamide",                                 "sulfonamidas (sul)"),
]


def agrupar(nombre):
    """Igual que en modelo_familias.py: primera regla que casa, ignorando mayusculas."""
    for patron, grupo in GRUPOS:
        if re.search(patron, nombre, flags=re.I):
            return grupo
    return None


def exigir(rel):
    p = RAIZ / rel
    if not p.exists():
        sys.exit(f"[FALTA] no encuentro {rel}")
    return p


# -------------------------------------------------- indice ARO -> familia
import csv
idx = {}
with open(exigir(args.aro), encoding="utf-8", errors="replace") as fh:
    lector = csv.DictReader(fh, delimiter="\t")
    cols = lector.fieldnames or []
    c_aro = next((c for c in cols if c.strip().lower() == "aro accession"), None)
    c_fam = next((c for c in cols if c.strip().lower() == "amr gene family"), None)
    if not c_aro or not c_fam:
        sys.exit(f"[ERROR] columnas no encontradas en aro_index.tsv: {cols}")
    for f in lector:
        idx[(f[c_aro] or "").strip()] = (f[c_fam] or "").strip()
print(f"aro_index.tsv: {len(idx):,} entradas")


def censo(ruta_rel, etiqueta):
    ruta = exigir(ruta_rel)
    regs = []
    with open(ruta, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            if not linea.startswith(">CARD|"):
                continue
            m = re.search(r"(ARO:\d+)", linea)
            if not m:
                continue
            fam = idx.get(m.group(1), "")
            if not fam:
                continue
            # IGUAL QUE EL SCRIPT REAL: varias familias con ';' -> se toma la primera
            crudas = [x.strip() for x in fam.split(";") if x.strip()]
            if not crudas:
                continue
            regs.append(crudas[0])

    grupos = [agrupar(c) for c in regs]
    cnt = Counter(g for g in grupos if g)
    sin_regla = sum(1 for g in grupos if not g)

    print(f"\n{'='*78}\n{etiqueta}\n{'='*78}")
    print(f"  genes de CARD con familia : {len(regs):,}")
    print(f"  sin regla de agrupacion   : {sin_regla:,}")
    print(f"  agrupados                 : {sum(cnt.values()):,}")
    print(f"  grupos distintos          : {len(cnt)}")

    usables = [g for g, n in cnt.items() if n >= args.min_clase]
    total_us = sum(cnt[g] for g in usables)
    print(f"\n  {'familia':<44} {'ejemplos':>9}   70/15/15")
    for g, n in cnt.most_common():
        tr, va = round(.70 * n), round(.15 * n)
        te = n - tr - va
        marca = "" if n >= args.min_clase else "   <- bajo el minimo"
        print(f"  {g[:44]:<44} {n:>9,}   {tr}/{va}/{te}{marca}")
    print(f"\n  FAMILIAS CON >= {args.min_clase} EJEMPLOS : {len(usables)}")
    print(f"  TOTAL DE GENES EN ESAS FAMILIAS: {total_us:,}")
    return cnt, set(usables)


c95, u95 = censo(args.id95, f"IDENTIDAD 95 %  (el conjunto actual)")
c99, u99 = censo(args.id99, f"IDENTIDAD 99 %  (el nuevo)")

# -------------------------------------------------- comprobacion
print(f"\n{'='*78}\nCOMPROBACION\n{'='*78}")
if len(u95) == 12 and sum(c95[g] for g in u95) == 1011:
    print("  [OK] el bloque del 95 % reproduce las 12 clases y los 1,011 genes")
    print("       del modelo actual, asi que el etiquetado es el correcto.")
else:
    print(f"  [AVISO] al 95 % salen {len(u95)} clases y "
          f"{sum(c95[g] for g in u95):,} genes.")
    print("          El modelo actual tiene 12 clases y 1,011 genes.")
    print("          Revisa antes de usar estos numeros en un documento.")

# -------------------------------------------------- comparacion
print(f"\n{'='*78}\nCOMPARACION\n{'='*78}")
todas = sorted(set(c95) | set(c99), key=lambda g: -c99.get(g, 0))
print(f"  {'familia':<44} {'95%':>6} {'99%':>6} {'x':>6}  entra")
for g in todas:
    a, b = c95.get(g, 0), c99.get(g, 0)
    x = f"{b/a:.2f}" if a else "—"
    e = ("95 y 99" if g in u95 and g in u99 else
         "solo 99" if g in u99 else
         "solo 95" if g in u95 else "ninguno")
    print(f"  {g[:44]:<44} {a:>6,} {b:>6,} {x:>6}  {e}")

print(f"\n{'='*78}\nRESUMEN\n{'='*78}")
print(f"  {'':<28}{'95 %':>12}{'99 %':>12}")
print(f"  {'familias':<28}{len(u95):>12}{len(u99):>12}")
print(f"  {'genes etiquetados':<28}{sum(c95[g] for g in u95):>12,}"
      f"{sum(c99[g] for g in u99):>12,}")

nuevas = sorted(u99 - u95, key=lambda g: -c99[g])
if nuevas:
    print(f"\n  Familias nuevas al 99 % ({len(nuevas)}):")
    for g in nuevas:
        print(f"    {g:<44} {c95.get(g,0):>4} -> {c99[g]:>4}")
perdidas = sorted(u95 - u99)
if perdidas:
    print(f"\n  Familias que se pierden ({len(perdidas)}):")
    for g in perdidas:
        print(f"    {g:<44} {c95[g]:>4} -> {c99.get(g,0):>4}")

# composicion beta-lactamica, que es lo que mas crece
BL = [g for g in u99 if re.search(r"lactam|OXA|AmpC|carbapenem|espectro", g, re.I)]
b95 = sum(c95.get(g, 0) for g in BL if g in u95)
b99 = sum(c99.get(g, 0) for g in BL)
t95, t99 = sum(c95[g] for g in u95), sum(c99[g] for g in u99)
print(f"\n  Genes de beta-lactamasas sobre el total:")
print(f"    95 % : {b95:,} de {t95:,}  ({100*b95/max(t95,1):.1f} %)")
print(f"    99 % : {b99:,} de {t99:,}  ({100*b99/max(t99,1):.1f} %)")
