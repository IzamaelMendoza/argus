# -*- coding: utf-8 -*-
"""
diagnostico_argnet.py

Comprueba si el acierto de 0.071 de ARGNet es real o un artefacto del cruce.

Con 14 clases, 0.071 es exactamente 1/14, es decir, el azar. Eso puede
significar tres cosas muy distintas:

  a) ARGNet realmente fallo (poco probable en un modelo publicado)
  b) las predicciones quedaron desalineadas de los genes (zip sobre dos
     listas que no estaban en el mismo orden)
  c) se comparo contra la etiqueta equivocada

Las tres dejan huellas distintas y este script las separa.

Uso, desde la raiz del proyecto:
    python scripts/diagnostico_argnet.py
"""
from pathlib import Path
import sys

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA = RAIZ / "results" / "concordancia_categoria_argnet.csv"
if not RUTA.exists():
    sys.exit(f"[FALTA] {RUTA}")

df = pd.read_csv(RUTA)
ac = df["acierto"]
if ac.dtype == object:
    ac = ac.astype(str).str.lower() == "true"


def titulo(t):
    print(f"\n{t}\n" + "=" * len(t))


titulo("1. Lo basico")
print(f"  filas                 {len(df)}")
print(f"  genes distintos       {df['gid'].nunique()}")
print(f"  acierto               {ac.mean():.3f}")
print(f"  clases reales         {df['clase_real'].nunique()}")
print(f"  clases predichas      {df['clase_pred'].nunique()}")
az = 1 / max(df['clase_real'].nunique(), 1)
print(f"  azar (1/n_clases)     {az:.3f}")

titulo("2. ¿Predice siempre lo mismo?")
vc = df["clase_pred"].value_counts()
print(vc.head(10).to_string())
top = vc.iloc[0] / len(df)
print(f"\n  la clase mas predicha cubre el {100*top:.1f} % de los casos")
if top > 0.5:
    print("  -> SOSPECHOSO: el modelo esta colapsando a una sola clase.")

titulo("3. ¿Coinciden los vocabularios?")
r, p = set(df["clase_real"]), set(df["clase_pred"])
print(f"  solo en clase_real : {sorted(r - p)}")
print(f"  solo en clase_pred : {sorted(p - r)}")
print(f"  en ambas           : {len(r & p)}")
if not (r & p):
    print("  -> El problema es de vocabulario: no comparten ni una etiqueta.")

titulo("4. Matriz de confusion (real en filas)")
cz = pd.crosstab(df["clase_real"], df["clase_pred"])
print(cz.to_string())

titulo("5. La prueba del desplazamiento")
print("  Si las predicciones quedaron corridas respecto de los genes, al")
print("  desplazarlas una o varias posiciones el acierto deberia SUBIR mucho.")
base = ac.mean()
mejor, k_mejor = base, 0
for k in list(range(1, 6)) + [-1, -2, -3]:
    corr = (df["clase_real"] == df["clase_pred"].shift(k)).mean()
    marca = "  <-- mucho mayor" if corr > base * 3 and corr > 0.3 else ""
    print(f"    desplazamiento {k:>3}: acierto {corr:.3f}{marca}")
    if corr > mejor:
        mejor, k_mejor = corr, k

titulo("Veredicto")
if mejor > base * 3 and mejor > 0.3:
    print(f"  Las predicciones estan DESALINEADAS: con un desplazamiento de")
    print(f"  {k_mejor} el acierto pasa de {base:.3f} a {mejor:.3f}.")
    print("  Hay que rehacer el cruce emparejando por identificador, no por orden.")
elif not (r & p):
    print("  Los dos vocabularios de clases no se tocan: se esta comparando")
    print("  contra una taxonomia distinta. El 0.071 no mide a ARGNet.")
elif top > 0.5:
    print("  ARGNet colapsa a una sola clase. Suele pasar al usar el modelo de")
    print("  lecturas cortas (argnet-s) sobre genes completos, o al reves.")
    print("  Hay que revisar con que modelo se corrio.")
else:
    print(f"  No aparece desalineacion ni problema de vocabulario. El {base:.3f}")
    print("  podria ser real, pero conviene revisar con que modelo de ARGNet se")
    print("  corrio y si la entrada era del tipo que ese modelo espera.")
