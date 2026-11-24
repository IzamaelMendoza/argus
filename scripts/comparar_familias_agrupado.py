# -*- coding: utf-8 -*-
"""
comparar_familias_agrupado.py

Segunda parte de la comparacion de taxonomias.

El primer analisis dio pureza media 0.288 y parecia indicar que ARGus y
DeepARG clasifican cosas distintas. Mirando el cruce se ve que no es asi:
cada familia de ARGus se reparte entre varias etiquetas de DeepARG, pero
todas son variantes del mismo gen (AAC(3)-I, AAC(3)-IIIA, AAC(6')-I ... son
todas AAC).

La diferencia es de NIVEL DE AGREGACION, no de taxonomia: DeepARG etiqueta
subfamilias y ARGus agrupa.

Este script agrupa las etiquetas de DeepARG por su termino principal y vuelve
a medir la correspondencia. Si sube mucho, queda demostrado que las dos
herramientas responden la misma pregunta a distinta resolucion.

Uso, desde la raiz del proyecto:
    python scripts/comparar_familias_agrupado.py
"""
from pathlib import Path
import re
import sys

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "results"
SALIDA = RES / "comparacion_taxonomias"
SALIDA.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Terminos principales. Se aplica el mas largo que coincida al inicio de
# la etiqueta, para que "AAC(6')-I" caiga en AAC y no se parta de otro modo.
# La tabla es explicita a proposito: es juicio biologico, no debe quedar
# escondido en una expresion regular.
# ---------------------------------------------------------------------
TERMINOS = [
    "AAC", "APH", "ANT", "AAD",              # modificacion de aminoglucosidos
    "DFR",                                    # dihidrofolato reductasa
    "VAN",                                    # glucopeptidos
    "ERM", "MSR", "VGA", "LSA", "CLBA",      # MLS / proteccion ribosomal
    "OXA", "ADC", "ACT", "AMPC", "CMY",      # beta-lactamasas con nombre propio
    "CLASS_A", "CLASS_B", "CLASS_C",         # beta-lactamasas por clase
    "NDM", "VIM", "IMI", "IND", "TLA",       # otras beta-lactamasas
    "MEX", "ACR", "ADE", "AMR", "EMR",       # bombas RND y MFS
    "TET", "CMX", "TAP", "QAC", "MDT", "MDS",
]
TERMINOS = sorted(set(TERMINOS), key=len, reverse=True)


def termino_principal(etiqueta: str) -> str:
    """Reduce una etiqueta de DeepARG a su termino principal."""
    e = str(etiqueta).upper().strip()
    for t in TERMINOS:
        if e.startswith(t):
            return t
    # sin termino conocido: se deja la etiqueta tal cual, pero sin el sufijo
    # numerico, para no inflar artificialmente el numero de grupos
    return re.sub(r"[-_(].*$", "", e)


def titulo(t):
    print(f"\n{t}\n" + "=" * len(t))


# ===================================================== cargar
ruta_arg = RES / "familias" / "etiquetas_finales.tsv"
ruta_dp = RES / "deeparg_genes_id30.mapping.ARG"
for r in (ruta_arg, ruta_dp):
    if not r.exists():
        sys.exit(f"[FALTA] {r.relative_to(RAIZ)}")

arg = pd.read_csv(ruta_arg, sep="\t")[["gen", "etiqueta"]]
arg = arg.rename(columns={"etiqueta": "familia_argus"}).drop_duplicates("gen")

dp = pd.read_csv(ruta_dp, sep="\t")
dp = dp.rename(columns={"#ARG": "familia_deeparg", "read_id": "gen"})
if "probability" in dp.columns:
    dp = dp.sort_values("probability", ascending=False)
dp = dp.drop_duplicates("gen")[["gen", "familia_deeparg"]]
dp["grupo_deeparg"] = dp["familia_deeparg"].map(termino_principal)

m = arg.merge(dp, on="gen", how="inner")

titulo("Efecto de agrupar las etiquetas de DeepARG")
print(f"  genes comparables                      {len(m):>5}")
print(f"  etiquetas distintas de DeepARG         {m['familia_deeparg'].nunique():>5}")
print(f"  grupos tras reducir al termino principal {m['grupo_deeparg'].nunique():>3}")


def pureza(col):
    filas = []
    for fam, sub in m.groupby("familia_argus"):
        c = sub[col].value_counts()
        filas.append({"familia_argus": fam, "genes": len(sub),
                      "distintas": len(c), "mas_frecuente": c.index[0],
                      "pureza": round(c.iloc[0] / len(sub), 3)})
    return pd.DataFrame(filas).sort_values("pureza", ascending=False)


antes = pureza("familia_deeparg")
despues = pureza("grupo_deeparg")

comp = antes[["familia_argus", "genes", "pureza"]].rename(
    columns={"pureza": "pureza_sin_agrupar"}).merge(
    despues[["familia_argus", "pureza", "distintas", "mas_frecuente"]].rename(
        columns={"pureza": "pureza_agrupada",
                 "distintas": "grupos",
                 "mas_frecuente": "grupo_principal"}),
    on="familia_argus")
comp["mejora"] = (comp["pureza_agrupada"] - comp["pureza_sin_agrupar"]).round(3)
comp = comp.sort_values("pureza_agrupada", ascending=False)

titulo("Correspondencia por familia")
print(comp.to_string(index=False))

comp.to_csv(SALIDA / "correspondencia_agrupada.csv", index=False)
pd.crosstab(m["familia_argus"], m["grupo_deeparg"]).to_csv(
    SALIDA / "cruce_argus_deeparg_agrupado.csv")

titulo("Resumen")
print(f"  pureza media sin agrupar   {antes['pureza'].mean():.3f}")
print(f"  pureza media agrupada      {despues['pureza'].mean():.3f}")
alto = (comp["pureza_agrupada"] >= 0.90).sum()
print(f"  familias con pureza >= 0.90  {alto} de {len(comp)}")
print(f"  familias que caen en un solo grupo  "
      f"{(comp['grupos'] == 1).sum()} de {len(comp)}")

print("\n  Si la pureza agrupada es alta, las dos herramientas responden la")
print("  misma pregunta a distinta resolucion: DeepARG etiqueta subfamilias y")
print("  ARGus agrupa. Si sigue baja en alguna familia, ahi si hay desacuerdo")
print("  real sobre a que grupo pertenece el gen.")

titulo("Archivos generados")
for f in sorted(SALIDA.iterdir()):
    print(f"  {f.relative_to(RAIZ)}")
