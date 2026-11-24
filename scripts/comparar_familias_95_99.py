#!/usr/bin/env python3
"""
comparar_familias_95_99.py -- cuantas familias y cuantos ejemplos tenemos ahora,
contra lo que teniamos con el conjunto al 95 %.

Lee las dos tablas de etiquetas finales (la nueva en results/familias/ y la vieja
en respaldo_id95/) y las pone lado a lado.

Tambien compara el min_clase de cada corrida. Eso importa: el numero de familias
no depende solo de cuantos genes hay, sino del minimo de ejemplos que le exigimos
a una familia para entrar. Si las dos corridas usaron minimos distintos, parte de
la diferencia viene de ahi y no de los datos, y conviene saberlo antes de
reportarla.

Uso:
    python scripts/comparar_familias_95_99.py
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--nueva", default="results/familias/etiquetas_finales.tsv")
ap.add_argument("--vieja", default="respaldo_id95/etiquetas_finales.tsv")
args = ap.parse_args()

if args.raiz:
    RAIZ = Path(args.raiz).expanduser()
else:
    RAIZ = None
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "results").is_dir():
            RAIZ = cand
            break
    if RAIZ is None:
        sys.exit("No encontre el proyecto. Usa --raiz")


def columna_etiqueta(d):
    """La columna con el nombre de la familia; si no la reconozco, la ultima."""
    for c in d.columns:
        if c.lower() in ("grupo", "familia", "etiqueta", "clase", "label"):
            return c
    return d.columns[-1]


def censo(ruta):
    if not ruta.exists():
        return None
    d = pd.read_csv(ruta, sep="\t")
    col = columna_etiqueta(d)
    return d[col].value_counts()


def meta(ruta):
    if not ruta.exists():
        return {}
    try:
        return json.load(open(ruta, encoding="utf-8"))
    except Exception:
        return {}


nueva = censo(RAIZ / args.nueva)
vieja = censo(RAIZ / args.vieja)

if nueva is None:
    sys.exit(f"No encontre {args.nueva}")

m_nueva = meta(RAIZ / "results/familias/modelo_familias_meta.json")
m_vieja = meta(RAIZ / "respaldo_id95/modelo_familias_meta.json")

print("=" * 84)
print("FAMILIAS Y EJEMPLOS -- CONJUNTO AL 99 % CONTRA EL DEL 95 %")
print("=" * 84)

print("\n  Como se corrio cada una:")
for nombre, m, c in (("95 %", m_vieja, vieja), ("99 %", m_nueva, nueva)):
    if c is None:
        print(f"    {nombre:<6} (no encontre su tabla de etiquetas)")
        continue
    mc = m.get("min_clase", "?")
    print(f"    {nombre:<6} min_clase {str(mc):<4} | {len(c):>2} familias | "
          f"{int(c.sum()):>5,} genes etiquetados")

mc_v, mc_n = m_vieja.get("min_clase"), m_nueva.get("min_clase")
if mc_v is not None and mc_n is not None and mc_v != mc_n:
    print(f"\n  [OJO] Los minimos no son iguales: {mc_v} contra {mc_n}.")
    print("  Una familia entra al modelo si tiene al menos ese numero de ejemplos,")
    print("  asi que parte del cambio en el numero de familias viene del minimo y no")
    print("  de los datos. Para separar las dos cosas, corre el censo del conjunto")
    print("  nuevo con el minimo viejo y compara contra ese:")
    print(f"      python scripts/modelo_familias.py --solo-censo --sin-otras --min-clase {mc_v}")
elif mc_v is None or mc_n is None:
    print("\n  [OJO] A una de las dos corridas no le pude leer el min_clase.")
    print("  Revisa que el numero de familias de las dos se haya decidido con el")
    print("  mismo minimo antes de reportar la diferencia.")

if vieja is None:
    print(f"\n  No encontre {args.vieja}, asi que solo muestro el censo nuevo.\n")
    print(f"  {'familia':<42} {'n':>6}")
    print("  " + "-" * 50)
    for fam, n in nueva.items():
        print(f"  {fam:<42} {n:>6,}")
    print(f"\n  {len(nueva)} familias | {int(nueva.sum()):,} genes")
    sys.exit(0)

# ------------------------------------------------------------------ lado a lado
todas = sorted(set(nueva.index) | set(vieja.index),
               key=lambda f: -int(nueva.get(f, 0)))

print("\n" + "=" * 84)
print(f"  {'familia':<42} {'95 %':>7} {'99 %':>7} {'cambio':>9}   ")
print("-" * 84)
for fam in todas:
    v, n = int(vieja.get(fam, 0)), int(nueva.get(fam, 0))
    if v == 0:
        nota = "  <- nueva"
        cambio = f"+{n}"
    elif n == 0:
        nota = "  <- ya no entra"
        cambio = f"-{v}"
    else:
        cambio = f"{n - v:+d}"
        nota = "  <- se multiplico" if v and n >= 2 * v else ""
    print(f"  {fam:<42} {v:>7,} {n:>7,} {cambio:>9}{nota}")

print("-" * 84)
print(f"  {'TOTAL':<42} {int(vieja.sum()):>7,} {int(nueva.sum()):>7,} "
      f"{int(nueva.sum()) - int(vieja.sum()):>+9,}")
print(f"  {'familias':<42} {len(vieja):>7} {len(nueva):>7} "
      f"{len(nueva) - len(vieja):>+9}")
print("=" * 84)

nuevas = [f for f in nueva.index if f not in vieja.index]
perdidas = [f for f in vieja.index if f not in nueva.index]
if nuevas:
    print("\n  Familias que antes no se podian modelar y ahora si:")
    for f in sorted(nuevas, key=lambda x: -int(nueva[x])):
        print(f"    {int(nueva[f]):>5,}  {f}")
if perdidas:
    print("\n  Familias que estaban y ya no:")
    for f in sorted(perdidas, key=lambda x: -int(vieja[x])):
        print(f"    {int(vieja[f]):>5,}  {f}")

salida = RAIZ / "results" / "familias" / "comparacion_familias_95_99.csv"
pd.DataFrame({"familia": todas,
              "n_id95": [int(vieja.get(f, 0)) for f in todas],
              "n_id99": [int(nueva.get(f, 0)) for f in todas]}).to_csv(salida, index=False)
print(f"\nGuardado: {salida}")
