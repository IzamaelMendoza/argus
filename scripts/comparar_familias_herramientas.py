# -*- coding: utf-8 -*-
"""
comparar_familias_herramientas.py

Responde la tarea: ¿los modelos que ya existen clasifican las MISMAS familias
que ARGus?

Compara tres taxonomias sobre los mismos genes:

  ARGus    12 familias de gen, de la ontologia de CARD
           ("OXA beta-lactamasa", "AAC acetiltransferasa", ...)
  DeepARG  familia de gen (columna #ARG: "OXA", "AAC(6')-I") + clase de farmaco
  ARGNet   solo clase de farmaco ("beta-lactam", "MLS", ...)

La pregunta "cual acierta mas" solo tiene sentido donde las taxonomias se
corresponden. Por eso el script mide primero la correspondencia y solo
despues el desempeno.

Uso, desde la raiz del proyecto:
    python scripts/comparar_familias_herramientas.py
"""
from pathlib import Path
import sys

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "results"
SALIDA = RES / "comparacion_taxonomias"
SALIDA.mkdir(parents=True, exist_ok=True)


def titulo(t):
    print(f"\n{t}\n" + "=" * len(t))


def exigir(ruta):
    if not ruta.exists():
        sys.exit(f"[FALTA] no encuentro {ruta.relative_to(RAIZ)}")
    return ruta


# ===================================================== 1. cargar
titulo("1. Datos de partida")

arg = pd.read_csv(exigir(RES / "familias" / "etiquetas_finales.tsv"), sep="\t")
arg = arg[["gen", "etiqueta"]].rename(columns={"etiqueta": "familia_argus"})
arg = arg.drop_duplicates("gen")
print(f"  ARGus     {len(arg):>5} genes etiquetados, "
      f"{arg['familia_argus'].nunique()} familias")

dp = pd.read_csv(exigir(RES / "deeparg_genes_id30.mapping.ARG"), sep="\t")
dp = dp.rename(columns={"#ARG": "familia_deeparg",
                        "read_id": "gen",
                        "predicted_ARG-class": "clase_deeparg"})
# un gen puede traer varias filas; se conserva la de mayor probabilidad
if "probability" in dp.columns:
    dp = dp.sort_values("probability", ascending=False)
dp = dp.drop_duplicates("gen")[["gen", "familia_deeparg", "clase_deeparg"]]
print(f"  DeepARG   {len(dp):>5} genes predichos, "
      f"{dp['familia_deeparg'].nunique()} familias distintas")

ruta_an = RES / "concordancia_categoria_argnet.csv"
an = None
if ruta_an.exists():
    an = pd.read_csv(ruta_an).rename(columns={"gid": "gen"})
    print(f"  ARGNet    {len(an):>5} genes, "
          f"{an['clase_pred'].nunique()} clases de farmaco distintas")
else:
    print("  ARGNet    (no encontrado, se omite)")

# ===================================================== 2. cobertura
titulo("2. Cobertura: sobre cuantos genes se puede comparar")

m = arg.merge(dp, on="gen", how="left")
con_dp = m["familia_deeparg"].notna().sum()
print(f"  genes etiquetados por ARGus            {len(arg):>5}")
print(f"  de esos, DeepARG tambien predijo algo  {con_dp:>5}  "
      f"({100*con_dp/len(arg):.1f} %)")
print(f"  DeepARG no los detecto                 {len(arg)-con_dp:>5}")
print("\n  Nota: los no detectados no son errores de familia, son genes que")
print("  DeepARG no reconocio como ARG. No entran en la comparacion.")

m = m.dropna(subset=["familia_deeparg"])
if m.empty:
    sys.exit("\nNo hay genes en comun: no se puede comparar.")

# ===================================================== 3. correspondencia
titulo("3. ¿Son la misma taxonomia?")

cruz = pd.crosstab(m["familia_argus"], m["familia_deeparg"])
cruz.to_csv(SALIDA / "cruce_argus_deeparg.csv")

filas = []
for fam, sub in m.groupby("familia_argus"):
    conteo = sub["familia_deeparg"].value_counts()
    filas.append({
        "familia_argus": fam,
        "genes": len(sub),
        "familias_deeparg_distintas": len(conteo),
        "deeparg_mas_frecuente": conteo.index[0],
        "pureza": round(conteo.iloc[0] / len(sub), 3),
    })
corr = pd.DataFrame(filas).sort_values("pureza", ascending=False)
corr.to_csv(SALIDA / "correspondencia_por_familia.csv", index=False)

print(corr.to_string(index=False))

pureza_media = corr["pureza"].mean()
una_sola = (corr["familias_deeparg_distintas"] == 1).sum()
print(f"\n  pureza media                      {pureza_media:.3f}")
print(f"  familias que caen en una sola     {una_sola} de {len(corr)}")
print("\n  Lectura: pureza 1.0 significa que todos los genes de esa familia de")
print("  ARGus reciben la misma etiqueta en DeepARG, es decir, que las dos")
print("  taxonomias se corresponden en ese grupo. Valores bajos significan que")
print("  la familia de ARGus se reparte entre varias de DeepARG.")

# ===================================================== 4. desempeno
titulo("4. Desempeno, solo donde la correspondencia es alta")

comparables = corr[corr["pureza"] >= 0.90]["familia_argus"].tolist()
print(f"  familias con pureza >= 0.90: {len(comparables)} de {len(corr)}")
for f in comparables:
    print(f"    - {f}")

if not comparables:
    print("\n  Ninguna familia alcanza correspondencia suficiente.")
    print("  Comparar el acierto de las dos herramientas no tendria sentido:")
    print("  estarian respondiendo preguntas distintas.")
else:
    sub = m[m["familia_argus"].isin(comparables)]
    print(f"\n  genes en esas familias: {len(sub)}")
    print("  Sobre este subconjunto si se pueden comparar los dos modelos.")
    print("  El acierto de ARGus por familia esta en results/familias/ ")
    print("  (reporte_familias.py); el de DeepARG se mide contra la misma")
    print("  etiqueta de referencia usando el cruce de arriba.")

# ===================================================== 5. ARGNet
if an is not None:
    titulo("5. ARGNet: otra taxonomia")
    acc = an["acierto"].mean() if an["acierto"].dtype != object else \
          (an["acierto"].astype(str).str.lower() == "true").mean()
    print(f"  ARGNet predice CLASE DE FARMACO, no familia de gen.")
    print(f"  clases distintas que usa: {an['clase_pred'].nunique()}")
    print(f"  acierto sobre su propia taxonomia: {acc:.3f}")
    print("\n  No es comparable con las 12 familias de ARGus: una beta-lactamasa")
    print("  resiste penicilinas, cefalosporinas y carbapenemicos a la vez, asi")
    print("  que la clase de farmaco es un problema multi-etiqueta y la familia")
    print("  de gen no lo es.")

# ===================================================== 6. archivos
titulo("Archivos generados")
for f in sorted(SALIDA.iterdir()):
    print(f"  {f.relative_to(RAIZ)}")
