"""
filtrar_prioritarios.py -- separa los falsos positivos "coherentes" en dos grupos:
los que son artefacto del relleno de ceros y los que son candidatos reales para BLAST.

Por qué: un gen de 504 pb se parte en un fragmento de 500 pb reales y otro de 4 pb
reales + 496 ceros. Con K = 6 ese segundo fragmento no tiene NINGUNA ventana de 6
letras ACGT, su vector de 4096 frecuencias queda en cero y el bosque le da una
probabilidad fija de ~0.714 > umbral. O sea: positivo automático por un error de
código, no por parecerse a un ARG. Mandarlos a BLAST es perder el tiempo.

El criterio es el RESTO de la división entre 500:
    resto 1-5 pb     -> vector degenerado, probabilidad fija  (DESCARTAR)
    resto 6-99 pb    -> vector distorsionado, frecuencias ~10x infladas (SOSPECHOSO)
    resto >= 100 pb  -> fragmentación limpia  (CANDIDATO)

Uso:
    python filtrar_prioritarios.py
    python filtrar_prioritarios.py --min-frags 3
"""
import argparse
import csv
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--identidad", default="90")
ap.add_argument("--ventana", type=int, default=500)
ap.add_argument("--min-frags", type=int, default=3,
                help="mínimo de fragmentos para considerarlo evidencia fuerte")
args = ap.parse_args()

if args.raiz:
    RAIZ = Path(args.raiz).expanduser()
else:
    RAIZ = None
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "data" / "interim").is_dir():
            RAIZ = cand
            break
    if RAIZ is None:
        sys.exit('No encontré el proyecto. Usa --raiz "ruta\\al\\Proyecto_Residencias"')

DIR = RAIZ / "results" / f"falsos_positivos_id{args.identidad}"
CSV_PRED = DIR / "predicciones_1k1k.csv"
FASTA_FUENTE = RAIZ / "data" / "interim" / "muestra_genes_completos.fasta"
for p, q in [(CSV_PRED, "predicciones_1k1k.csv"), (FASTA_FUENTE, "muestra_genes_completos.fasta")]:
    if not p.exists():
        sys.exit(f"No encontré {q}:\n  {p}")


def leer_fasta(path):
    d, cur = {}, None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            linea = linea.strip()
            if linea.startswith(">"):
                cur = linea[1:].split()[0]
                d[cur] = []
            elif cur:
                d[cur].append(linea)
    return {k: "".join(v) for k, v in d.items()}


fa = leer_fasta(FASTA_FUENTE)

filas = []
with open(CSV_PRED, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        largo = int(r["largo_pb"])
        resto = largo % args.ventana if largo >= args.ventana else largo
        filas.append({
            "gen": r["gen"], "y": int(r["y_real"]), "pred": int(r["pred"]),
            "p": float(r["p_max"]), "fpos": int(r["n_frags_pos"]),
            "nfrag": int(r["n_frags"]), "largo": largo,
            "resto": resto,
            "clase": ("degenerado" if 0 < resto < 6 else
                      "distorsionado" if resto < 100 else "limpio"),
        })

FP = [f for f in filas if f["y"] == 0 and f["pred"] == 1]
coherentes = [f for f in FP if f["nfrag"] >= 2 and f["fpos"] == f["nfrag"]]

descartar = [f for f in coherentes if f["clase"] == "degenerado"]
sospechosos = [f for f in coherentes if f["clase"] == "distorsionado"]
limpios = [f for f in coherentes if f["clase"] == "limpio"]
candidatos = sorted([f for f in limpios if f["nfrag"] >= args.min_frags], key=lambda x: -x["p"])
debiles = sorted([f for f in limpios if f["nfrag"] < args.min_frags], key=lambda x: -x["p"])

SALIDA = DIR / "blast"
SALIDA.mkdir(parents=True, exist_ok=True)


def escribir(grupo, nombre, etiqueta):
    ruta = SALIDA / nombre
    with open(ruta, "w", encoding="utf-8") as fh:
        for f in grupo:
            seq = fa.get(f["gen"])
            if not seq:
                continue
            fh.write(f">{f['gen']} {etiqueta} p={f['p']:.3f} frags={f['fpos']}/{f['nfrag']} "
                     f"largo={f['largo']}pb resto={f['resto']}pb\n")
            for i in range(0, len(seq), 70):
                fh.write(seq[i:i + 70] + "\n")
    return ruta


def tabla(grupo, titulo):
    print(f"\n{titulo}  ({len(grupo)})")
    if not grupo:
        print("   (ninguno)")
        return
    print(f"   {'p_max':>6}  {'frags':>6}  {'largo':>7}  {'resto':>6}  gen")
    for f in grupo:
        print(f"   {f['p']:>6.3f}  {f['fpos']:>2}/{f['nfrag']:<3}  {f['largo']:>5}pb  "
              f"{f['resto']:>4}pb  {f['gen']}")


print("=" * 78)
print(f"Falsos positivos: {len(FP)}  |  coherentes (todos sus fragmentos pasaron): {len(coherentes)}")

tabla(descartar, "ARTEFACTO -- resto de 1-5 pb, probabilidad fija: NO mandar a BLAST")
tabla(sospechosos, "SOSPECHOSOS -- resto de 6-99 pb, vector distorsionado: baja prioridad")
tabla(candidatos, f"CANDIDATOS REALES -- fragmentación limpia y {args.min_frags}+ fragmentos")
tabla(debiles, "EVIDENCIA DÉBIL -- limpios pero solo 2 fragmentos")

print("\n" + "=" * 78)
print("Archivos escritos:")
for g, n, e in [(candidatos, "3_candidatos_blast.fasta", "CANDIDATO"),
                (debiles, "4_debiles.fasta", "DEBIL"),
                (sospechosos, "5_sospechosos_relleno.fasta", "SOSPECHOSO"),
                (descartar, "6_artefacto_relleno.fasta", "ARTEFACTO")]:
    r = escribir(g, n, e)
    print(f"   {r}  ({len(g)})")

if candidatos:
    pb = sum(f["largo"] for f in candidatos)
    print(f"\nManda 3_candidatos_blast.fasta al BLAST web: {len(candidatos)} secuencias, "
          f"{pb:,} pb. Es un solo lote, sin problema de límites.")
