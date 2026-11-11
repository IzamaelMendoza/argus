"""
censo_familias.py -- PASO 0 del modelo de familias.

Lo que pidió la asesora: "Checar cuantas familias tenemos (cuantos ejemplos hay de cada
familia, cuales son los porcentajes, el porcentaje estipulado 70/15/15). Esto checarlo
ANTES de empezar a entrenar."

Este script no entrena nada. Hace tres cosas:

 A. DESCUBRE la estructura real de los encabezados del FASTA de positivos, agrupados por
    base de datos de origen, y muestra ejemplos. Hace falta porque cada base guarda la
    información de familia en un lugar distinto del encabezado.
 B. EXTRAE etiquetas candidatas con varios esquemas (clase de fármaco, familia de gen,
    base de origen) y hace el censo de cada uno: cuántas clases, cuántos ejemplos por
    clase, porcentajes.
 C. EVALÚA si un corte 70/15/15 estratificado es viable: avisa de las clases con tan pocos
    ejemplos que no alcanzan para repartirse en tres conjuntos.

Uso:
    python censo_familias.py
    python censo_familias.py --raiz "C:\\Users\\r4gbi\\Desktop\\Proyecto_Residencias"
    python censo_familias.py --min-clase 50       # umbral para considerar una clase usable
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--fasta", default=None, help="ruta explícita al FASTA de positivos")
ap.add_argument("--min-clase", type=int, default=50,
                help="mínimo de genes para que una clase sea usable (def. 50)")
ap.add_argument("--ejemplos", type=int, default=4, help="encabezados de muestra por base")
args = ap.parse_args()

# ---------------------------------------------------------------- rutas
if args.raiz:
    RAIZ = Path(args.raiz).expanduser()
else:
    RAIZ = None
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "data").is_dir():
            RAIZ = cand
            break
    if RAIZ is None:
        sys.exit('No encontré el proyecto. Usa --raiz "ruta\\al\\Proyecto_Residencias"')

if args.fasta:
    FASTA = Path(args.fasta).expanduser()
else:
    posibles = [RAIZ / "data" / "processed" / "argdb_POSITIVO_final.fasta",
                RAIZ / "data" / "interim" / "positivo_id95_limpio.fasta",
                RAIZ / "data" / "interim" / "positivo_id90_limpio.fasta"]
    FASTA = next((p for p in posibles if p.exists()), None)
    if FASTA is None:
        sys.exit("No encontré el FASTA de positivos. Pásalo con --fasta\nProbé:\n  "
                 + "\n  ".join(str(p) for p in posibles))

print("=" * 90)
print("CENSO DE FAMILIAS -- paso previo al entrenamiento")
print("Proyecto:", RAIZ)
print("FASTA   :", FASTA.name)
print("=" * 90)

# ---------------------------------------------------------------- leer encabezados
encabezados, largos = [], []
cur = 0
with open(FASTA, encoding="utf-8", errors="replace") as fh:
    for linea in fh:
        linea = linea.rstrip("\n")
        if linea.startswith(">"):
            if cur:
                largos.append(cur)
            cur = 0
            encabezados.append(linea[1:].strip())
        else:
            cur += len(linea.strip())
if cur:
    largos.append(cur)

print(f"\nSecuencias: {len(encabezados):,}")
if largos:
    largos_o = sorted(largos)
    print(f"Largo (pb): min {largos_o[0]} | p25 {largos_o[len(largos_o)//4]} | "
          f"mediana {largos_o[len(largos_o)//2]} | p75 {largos_o[3*len(largos_o)//4]} | "
          f"max {largos_o[-1]}")
    cab512 = sum(1 for L in largos if L <= 500)
    print(f"Genes que caben enteros en 500 pb: {cab512:,} ({100*cab512/len(largos):.1f} %)")
    print("  -> los demás hay que fragmentar: DNABERT-6 admite 512 tokens y un fragmento")
    print("     de 500 pb da 495 6-meros + [CLS] + [SEP] = 497. Un gen de 2000 pb daría 1995.")

# ---------------------------------------------------------------- A. estructura
def base_de(h):
    t = h.split()[0]
    return t.split("|", 1)[0] if "|" in t else (t.split("~", 1)[0] if "~" in t else t)

por_base = defaultdict(list)
for h in encabezados:
    por_base[base_de(h)].append(h)

print("\n" + "=" * 90)
print("A. ESTRUCTURA DE LOS ENCABEZADOS, POR BASE DE ORIGEN")
print("=" * 90)
for base, hs in sorted(por_base.items(), key=lambda x: -len(x[1])):
    campos = Counter(h.split()[0].count("|") + 1 for h in hs)
    print(f"\n{base}  ({len(hs):,} secuencias, {100*len(hs)/len(encabezados):.1f} %)")
    print(f"  campos separados por '|': {dict(campos)}")
    for h in hs[:args.ejemplos]:
        print(f"    {h[:110]}")

# ---------------------------------------------------------------- B. esquemas de etiqueta
# Cada base guarda la información en un lugar distinto. Estos extractores cubren los
# formatos conocidos; los que no casen quedan como "SIN_ETIQUETA" y se reportan.

RE_ARGANNOT = re.compile(r"\(([A-Za-z]+)\)\s*([A-Za-z0-9_\-'()]+)")

def etiqueta_clase_farmaco(h):
    """Clase de fármaco / de resistencia (betalactams, aminoglycosides, ...)."""
    t = h.split()[0]
    p = t.split("|")
    b = base_de(h).upper()
    if b.startswith("MEGARES") and len(p) >= 4:
        return p[3]                                   # MEGARes|MEG_x|Drugs|<clase>|...
    if b.startswith("ARGANNOT"):
        m = RE_ARGANNOT.search(t)
        if m:
            return m.group(1)                         # (Bla), (AGly), (Tet) ...
    if b.startswith("RESFINDERFG") and len(p) >= 2:
        return p[1].split("-")[0]                     # tet-protection -> tet
    return None

def etiqueta_familia_gen(h):
    """Familia del gen a partir de su nombre (OXA-837 -> OXA, AAC(6')-Ih -> AAC(6'))."""
    t = h.split()[0]
    p = t.split("|")
    b = base_de(h).upper()
    nombre = None
    if b.startswith("ARGANNOT"):
        m = RE_ARGANNOT.search(t)
        if m:
            nombre = m.group(2)
    elif b.startswith("MEGARES") and len(p) >= 5:
        nombre = p[4]
    elif b.startswith("RESFINDERFG") and len(p) >= 2:
        nombre = p[1]
    elif len(p) >= 2:
        nombre = p[1]
    if not nombre:
        return None
    nombre = nombre.strip()
    # quitar el sufijo de variante: OXA-837 -> OXA ; blaSFH-1 -> blaSFH ; tet(A) -> tet(A)
    fam = re.sub(r"[-_]\d+[A-Za-z]*$", "", nombre)
    fam = re.sub(r"^(bla|aac|aad|ant|aph)(?=[A-Z(])", lambda m: m.group(1).lower(), fam)
    return fam or None

ESQUEMAS = {
    "clase_farmaco": etiqueta_clase_farmaco,
    "familia_gen":   etiqueta_familia_gen,
    "base_origen":   lambda h: base_de(h),
}

print("\n" + "=" * 90)
print("B. ESQUEMAS DE ETIQUETA CANDIDATOS")
print("=" * 90)

resultados = {}
for nombre_esq, fn in ESQUEMAS.items():
    etiquetas = [fn(h) for h in encabezados]
    sin = sum(1 for e in etiquetas if not e)
    cnt = Counter(e for e in etiquetas if e)
    resultados[nombre_esq] = (cnt, sin, etiquetas)
    cobertura = 100 * (len(encabezados) - sin) / max(1, len(encabezados))
    usables = {k: v for k, v in cnt.items() if v >= args.min_clase}
    print(f"\n--- {nombre_esq} ---")
    print(f"  cobertura: {cobertura:.1f} %  ({sin:,} sin etiqueta)")
    print(f"  clases distintas: {len(cnt)}  |  con >= {args.min_clase} ejemplos: {len(usables)}")
    if usables:
        n_us = sum(usables.values())
        print(f"  secuencias en clases usables: {n_us:,} "
              f"({100*n_us/len(encabezados):.1f} % del total)")
    print(f"  {'clase':<26} {'n':>7} {'%':>7}   70/15/15")
    for k, v in cnt.most_common(25):
        tr, va, te = round(.70*v), round(.15*v), v - round(.70*v) - round(.15*v)
        aviso = ""
        if v < 7:
            aviso = "  <- IMPOSIBLE repartir"
        elif v < args.min_clase:
            aviso = "  <- muy pocos"
        print(f"  {str(k)[:26]:<26} {v:>7,} {100*v/len(encabezados):>6.2f}%   "
              f"{tr}/{va}/{te}{aviso}")
    if len(cnt) > 25:
        resto = sum(v for _, v in cnt.most_common()[25:])
        print(f"  ... y {len(cnt)-25} clases más, {resto:,} secuencias en total")

# ---------------------------------------------------------------- C. viabilidad del corte
print("\n" + "=" * 90)
print("C. VIABILIDAD DEL CORTE 70 / 15 / 15")
print("=" * 90)
for nombre_esq, (cnt, sin, _) in resultados.items():
    if not cnt:
        continue
    imposibles = [k for k, v in cnt.items() if v < 7]
    pocas = [k for k, v in cnt.items() if 7 <= v < args.min_clase]
    ok = [k for k, v in cnt.items() if v >= args.min_clase]
    n_ok = sum(cnt[k] for k in ok)
    print(f"\n{nombre_esq}:")
    print(f"  clases con >= {args.min_clase} ejemplos : {len(ok):>4}  ({n_ok:,} secuencias)")
    print(f"  clases con 7-{args.min_clase-1} ejemplos     : {len(pocas):>4}  "
          f"({sum(cnt[k] for k in pocas):,} secuencias)")
    print(f"  clases con < 7 ejemplos      : {len(imposibles):>4}  "
          f"({sum(cnt[k] for k in imposibles):,} secuencias)  <- no se pueden estratificar")
    if ok:
        mayor = max(cnt[k] for k in ok); menor = min(cnt[k] for k in ok)
        print(f"  desbalance entre clases usables: {mayor:,} vs {menor:,} "
              f"(razón {mayor/max(1,menor):.1f}x)")
        if mayor / max(1, menor) > 10:
            print("    -> desbalance fuerte: hacen falta class_weights Y sampler,")
            print("       como en el código de la asesora (balance_strategy='both'),")
            print("       y el checkpoint debe ir por balanced_accuracy, no por val_loss.")

# ---------------------------------------------------------------- guardar
SALIDA = RAIZ / "results" / "familias"
SALIDA.mkdir(parents=True, exist_ok=True)
import csv
ruta = SALIDA / "censo_familias.csv"
with open(ruta, "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["esquema", "clase", "n", "pct"])
    for nombre_esq, (cnt, _, _) in resultados.items():
        for k, v in cnt.most_common():
            w.writerow([nombre_esq, k, v, round(100*v/len(encabezados), 3)])
print(f"\nGuardado: {ruta}")

ruta2 = SALIDA / "etiquetas_por_gen.tsv"
with open(ruta2, "w", encoding="utf-8") as fh:
    fh.write("gen\tbase_origen\tclase_farmaco\tfamilia_gen\tencabezado\n")
    for h in encabezados:
        gid = h.split()[0]
        fh.write(f"{gid}\t{base_de(h)}\t{etiqueta_clase_farmaco(h) or ''}\t"
                 f"{etiqueta_familia_gen(h) or ''}\t{h}\n")
print(f"Guardado: {ruta2}  ({len(encabezados):,} filas)")

print("\n" + "=" * 90)
print("Pásame esta salida y con los números reales decidimos qué va a ser 'familia'")
print("y qué clases entran al modelo.")
