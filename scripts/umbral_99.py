#!/usr/bin/env python3
"""
umbral_99.py -- a que umbral hay que poner el modelo del 99 % para que se
comporte como el del 95 %, y si ahi le gana.

El problema que resuelve: comparar dos modelos a umbral fijo de 0.5 no dice cual
es mejor cuando sus distribuciones de probabilidad estan corridas. Uno puede
parecer peor solo porque el 0.5 le cae en otro punto de su curva. La comparacion
justa es a igual recall: pones los dos a cazar la misma proporcion de genes
reales y ves cual ensucia menos.

Lee los CSV de predicciones que deja reentrenar_fragmentacion.py (gen,y,pred,p)
y responde tres cosas:

  1. Con que umbral el modelo del 99 % alcanza el recall del 95 %, y con cuanta
     precision se queda ahi.
  2. Que umbral le da el mejor F1.
  3. Cuantos genes marcaria la GUI con cada umbral, que es el numero que se ve
     en pantalla.

Uso:
    python scripts/umbral_99.py
    python scripts/umbral_99.py --sin-figura
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--referencia", default="95 %",
                help="corrida cuyo recall hay que igualar")
ap.add_argument("--sin-figura", action="store_true")
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
    ("90 %",               "r1k_id90_v2.csv"),
    ("95 %",               "r1k_id95_v2.csv"),
    ("99 % completo",      "r1k_id99_completo.csv"),
    ("99 % sin parientes", "r1k_id99_sinparientes.csv"),
]

datos = {}
for etiqueta, archivo in CORRIDAS:
    ruta = RES / archivo
    if ruta.exists():
        d = pd.read_csv(ruta)
        if {"gen", "y", "p"} <= set(d.columns):
            datos[etiqueta] = d
if not datos:
    sys.exit("No encontre los CSV de predicciones en results/.")

# los mismos genes en todas, si no la comparacion no vale
comun = set.intersection(*(set(d["gen"].astype(str)) for d in datos.values()))
datos = {e: d[d["gen"].astype(str).isin(comun)].reset_index(drop=True)
         for e, d in datos.items()}

print("=" * 88)
print("CALIBRACION DEL UMBRAL -- MODELO BINARIO SOBRE EL 1k+1k")
print(f"{len(comun):,} genes comunes a todas las corridas")
print("=" * 88)


def en_umbral(d, t):
    y = d["y"].to_numpy(dtype=int)
    yhat = (d["p"].to_numpy(dtype=float) >= t).astype(int)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else float("nan")
    return dict(t=t, marcados=tp + fp, tp=tp, fp=fp, fn=fn, rec=rec, pre=pre, f1=f1)


# ------------------------------------------------------------------ referencia
if args.referencia not in datos:
    sys.exit(f"No tengo la corrida de referencia '{args.referencia}'.")

ref = en_umbral(datos[args.referencia], 0.5)
print(f"\nReferencia: {args.referencia} a umbral 0.50")
print(f"  recall {ref['rec']:.3f} | precision {ref['pre']:.3f} | F1 {ref['f1']:.3f} "
      f"| marcados {ref['marcados']:,}")

OBJETIVO = ref["rec"]

# ------------------------------------------------------------------ por corrida
print("\n" + "=" * 88)
print(f"  {'corrida':<22} {'umbral':>7} {'recall':>8} {'precision':>10} "
      f"{'F1':>7} {'marcados':>9}   que es")
print("-" * 88)

resumen = {}
for etiqueta, d in datos.items():
    y = d["y"].to_numpy(dtype=int)
    p = d["p"].to_numpy(dtype=float)
    pre_c, rec_c, umb_c = precision_recall_curve(y, p)
    # precision_recall_curve devuelve un punto mas que umbrales
    pre_c, rec_c = pre_c[:-1], rec_c[:-1]

    fila = []
    fila.append((en_umbral(d, 0.5), "umbral de fabrica"))

    # mejor F1
    f1_c = np.where(pre_c + rec_c > 0, 2 * pre_c * rec_c / (pre_c + rec_c), 0.0)
    t_f1 = float(umb_c[int(np.argmax(f1_c))])
    fila.append((en_umbral(d, t_f1), "mejor F1"))

    # igualar el recall de la referencia
    viable = rec_c >= OBJETIVO
    if viable.any():
        i = int(np.argmax(np.where(viable, pre_c, -1)))
        t_rec = float(umb_c[i])
        fila.append((en_umbral(d, t_rec), f"recall >= {OBJETIVO:.3f}"))
    else:
        fila.append((None, f"no alcanza recall {OBJETIVO:.3f} con ningun umbral"))

    resumen[etiqueta] = dict(ap=average_precision_score(y, p), filas=fila,
                             pre=pre_c, rec=rec_c)

    for m, que in fila:
        if m is None:
            print(f"  {etiqueta:<22} {'--':>7} {'--':>8} {'--':>10} {'--':>7} "
                  f"{'--':>9}   {que}")
        else:
            print(f"  {etiqueta:<22} {m['t']:>7.3f} {m['rec']:>8.3f} {m['pre']:>10.3f} "
                  f"{m['f1']:>7.3f} {m['marcados']:>9,}   {que}")
    print()
print("=" * 88)

# ------------------------------------------------------------------ veredicto
print("\nCOMPARACION A IGUAL RECALL")
print("-" * 88)
cand = [e for e in datos if e.startswith("99") and e in resumen]
for e in cand:
    m_ref = ref
    m_99 = resumen[e]["filas"][2][0]
    if m_99 is None:
        print(f"  {e}: no alcanza el recall de la referencia a ningun umbral.")
        continue
    dp = m_99["pre"] - m_ref["pre"]
    print(f"\n  {args.referencia} vs {e}, los dos a recall ~{OBJETIVO:.3f}:")
    print(f"    precision {m_ref['pre']:.3f} -> {m_99['pre']:.3f}  ({dp:+.3f})")
    print(f"    falsas alarmas {m_ref['fp']:,} -> {m_99['fp']:,}")
    if dp > 0.02:
        print("    -> A igual recall el del 99 % ensucia menos. La ganancia existia,")
        print("       estaba escondida por el umbral: hay que recalibrarlo, no cambiar")
        print("       de modelo.")
    elif dp < -0.02:
        print("    -> A igual recall el del 99 % ensucia mas. Subir la identidad no")
        print("       ayudo a la deteccion binaria.")
    else:
        print("    -> A igual recall los dos son equivalentes. Subir la identidad no")
        print("       cambio la capacidad de distinguir, solo movio el punto de")
        print("       operacion. La eleccion entre ellos es de uso, no de metrica.")

print("\n  Area bajo la curva precision-recall (no depende del umbral):")
for e in datos:
    print(f"    {e:<22} {resumen[e]['ap']:.3f}")

# ------------------------------------------------------------------ barrido
objetivo = next((e for e in datos if e.startswith("99 % completo")), None)
if objetivo:
    print("\n" + "=" * 88)
    print(f"BARRIDO DEL UMBRAL -- {objetivo}")
    print("  (la columna 'marcados' es el numero que vas a ver en la GUI)")
    print("-" * 88)
    print(f"  {'umbral':>7} {'marcados':>9} {'aciertos':>9} {'falsas al.':>11} "
          f"{'se escapan':>11} {'recall':>8} {'precision':>10} {'F1':>7}")
    for t in np.arange(0.20, 0.71, 0.05):
        m = en_umbral(datos[objetivo], float(t))
        print(f"  {m['t']:>7.2f} {m['marcados']:>9,} {m['tp']:>9,} {m['fp']:>11,} "
              f"{m['fn']:>11,} {m['rec']:>8.3f} {m['pre']:>10.3f} {m['f1']:>7.3f}")
    print("=" * 88)
    print("\n  Para mover la GUI a uno de estos puntos, cambia en app/argus_gui.py:")
    print("      UMBRAL_DEFECTO = <el umbral que elijas>")

# ------------------------------------------------------------------ figura
if not args.sin_figura:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        estilos = ["-", "--", "-.", ":"]
        for (e, r), est in zip(resumen.items(), estilos):
            ax.plot(r["rec"], r["pre"], est, linewidth=1.4,
                    label=f"{e}  (AP {r['ap']:.3f})")
        ax.set_xlabel("Recall (proporción de genes de resistencia detectados)")
        ax.set_ylabel("Precisión")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, loc="lower left", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        salida = RES / "curva_pr_identidades.png"
        fig.savefig(salida, dpi=200)
        print(f"\nFigura: {salida}")
    except Exception as e:
        print(f"\n(no se pudo generar la figura: {e})")

salida_csv = RES / "umbral_99_barrido.csv"
if objetivo:
    pd.DataFrame([en_umbral(datos[objetivo], float(t))
                  for t in np.arange(0.05, 0.96, 0.01)]).to_csv(salida_csv, index=False)
    print(f"Barrido completo: {salida_csv}")
