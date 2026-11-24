#!/usr/bin/env python3
"""
comparar_1k1k.py -- compara las corridas del modelo binario sobre el conjunto
fijo de evaluacion 1k+1k.

Las metricas se recalculan aqui desde los CSV de predicciones, no se leen de los
metadatos de cada modelo. Eso importa porque las versiones viejas del script no
guardaban las metricas del 1k+1k, y porque asi las cuatro corridas se miden con
exactamente la misma regla y sobre exactamente los mismos genes.

Uso:
    python scripts/comparar_1k1k.py
    python scripts/comparar_1k1k.py --umbral 0.5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--umbral", type=float, default=0.5,
                help="umbral sobre p para llamar ARG (el script usa 0.5)")
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

RES = RAIZ / "results"

CORRIDAS = [
    ("90 %",                    "r1k_id90_v2.csv"),
    ("95 %",                    "r1k_id95_v2.csv"),
    ("99 % completo",           "r1k_id99_completo.csv"),
    ("99 % sin parientes",      "r1k_id99_sinparientes.csv"),
]

print("=" * 92)
print("MODELO BINARIO (ETAPA 1) SOBRE EL CONJUNTO FIJO 1k+1k")
print(f"regla: gen marcado como ARG si p >= {args.umbral}")
print("=" * 92)

datos, conjuntos = {}, {}
for etiqueta, archivo in CORRIDAS:
    ruta = RES / archivo
    if not ruta.exists():
        print(f"  [falta] {archivo}  -> se omite {etiqueta}")
        continue
    d = pd.read_csv(ruta)
    faltan = {"gen", "y", "pred", "p"} - set(d.columns)
    if faltan:
        print(f"  [raro] {archivo} no trae {faltan} -> se omite")
        continue
    datos[etiqueta] = d
    conjuntos[etiqueta] = set(d["gen"].astype(str))

if not datos:
    sys.exit("No encontre ningun CSV de predicciones.")

# -------------------------------------------------- los genes deben ser los mismos
print("\nGenes evaluados por corrida:")
for etiqueta, d in datos.items():
    print(f"  {etiqueta:22} {len(d):,} genes  "
          f"(pos {int((d['y'] == 1).sum()):,} | neg {int((d['y'] == 0).sum()):,})")

base = next(iter(conjuntos.values()))
if all(c == base for c in conjuntos.values()):
    comun = base
    print("\n  Las corridas evaluan exactamente los mismos genes: la comparacion es directa.")
else:
    comun = set.intersection(*conjuntos.values())
    print(f"\n  AVISO: los conjuntos no son identicos. Se usan los {len(comun):,} genes")
    print("  comunes a todas las corridas, para no comparar sobre poblaciones distintas.")

# -------------------------------------------------- metricas
def metricas(d, umbral):
    d = d[d["gen"].astype(str).isin(comun)]
    y = d["y"].to_numpy(dtype=int)
    p = d["p"].to_numpy(dtype=float)
    yhat = (p >= umbral).astype(int)

    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())

    rec = tp / (tp + fn) if tp + fn else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else float("nan")
    esp = tn / (tn + fp) if tn + fp else float("nan")
    try:
        auc = roc_auc_score(y, p)
    except ValueError:
        auc = float("nan")
    # discrepancia entre la columna pred guardada y recalcular con el umbral
    dif = int((d["pred"].to_numpy(dtype=int) != yhat).sum())
    return dict(rec=rec, pre=pre, f1=f1, esp=esp, auc=auc,
                tp=tp, fp=fp, fn=fn, tn=tn, dif=dif)

filas = {e: metricas(d, args.umbral) for e, d in datos.items()}

print("\n" + "=" * 92)
print(f"  {'corrida':<22} {'recall':>8} {'precision':>10} {'F1':>8} "
      f"{'especif.':>9} {'AUC':>8} {'FP':>6} {'FN':>6}")
print("-" * 92)
for e, m in filas.items():
    print(f"  {e:<22} {m['rec']:>8.3f} {m['pre']:>10.3f} {m['f1']:>8.3f} "
          f"{m['esp']:>9.3f} {m['auc']:>8.3f} {m['fp']:>6d} {m['fn']:>6d}")
print("=" * 92)

raras = {e: m["dif"] for e, m in filas.items() if m["dif"]}
if raras:
    print("\n  AVISO: en estas corridas la columna 'pred' guardada no coincide con")
    print(f"  recalcular p >= {args.umbral}. Puede que se guardaran con otro umbral:")
    for e, n in raras.items():
        print(f"    {e}: {n:,} genes de diferencia")

# -------------------------------------------------- lecturas
print("\n" + "=" * 92)
print("LECTURA")
print("=" * 92)

if "99 % completo" in filas and "99 % sin parientes" in filas:
    a, b = filas["99 % completo"], filas["99 % sin parientes"]
    df1 = b["f1"] - a["f1"]
    print(f"\n  Efecto de la fuga por variantes aleicas (completo -> sin parientes):")
    print(f"    F1 {a['f1']:.3f} -> {b['f1']:.3f}   ({df1:+.3f})")
    print(f"    AUC {a['auc']:.3f} -> {b['auc']:.3f}  ({b['auc']-a['auc']:+.3f})")
    if abs(df1) < 0.02:
        print("    El cambio es menor que 0.02: quitar los positivos con parientes")
        print("    cercanos en el set de evaluacion no mueve el resultado, asi que la")
        print("    mejora del 99 % no se explica por contaminacion.")
    else:
        print("    El cambio es apreciable: parte del resultado del conjunto completo")
        print("    venia de entrenar con casi-copias de lo que luego se evalua.")

if "95 %" in filas and "99 % completo" in filas:
    a, b = filas["95 %"], filas["99 % completo"]
    print(f"\n  Efecto de subir la identidad de CD-HIT (95 % -> 99 %):")
    print(f"    recall    {a['rec']:.3f} -> {b['rec']:.3f}  ({b['rec']-a['rec']:+.3f})")
    print(f"    precision {a['pre']:.3f} -> {b['pre']:.3f}  ({b['pre']-a['pre']:+.3f})")
    print(f"    F1        {a['f1']:.3f} -> {b['f1']:.3f}  ({b['f1']-a['f1']:+.3f})")
    print(f"    AUC       {a['auc']:.3f} -> {b['auc']:.3f}  ({b['auc']-a['auc']:+.3f})")
    print(f"    FP        {a['fp']:,} -> {b['fp']:,}")
    print(f"    FN        {a['fn']:,} -> {b['fn']:,}")

salida = RES / "comparacion_1k1k_identidades.csv"
pd.DataFrame(filas).T.to_csv(salida)
print(f"\nGuardado: {salida}")
print("=" * 92)
