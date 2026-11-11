"""
familias_card.py -- PASO 1 (corregido): familia del gen tomada de la ONTOLOGÍA CURADA
de CARD, en vez de adivinarla de los nombres con expresiones regulares.

Por qué este cambio: el intento anterior deducía la familia del nombre del gen
(dfrF -> dfr) y producía 1,096 clases, 1,055 de ellas con menos de 7 ejemplos.
Parte era un bug mío y parte es que un nombre de gen no es una familia.

CARD ya trae la familia curada a mano. El encabezado de cada secuencia incluye el
ARO, y `aro_index.tsv` lo cruza con la columna "AMR Gene Family":

  CARD|gb|AF028812.1|+|392-887|ARO:3002867|dfrF [Enterococcus faecalis]
                                    ^^^^^^^^^ llave

  aro_index.tsv (12 columnas):
     1 ARO Accession          <- la llave
     9 AMR Gene Family        <- la etiqueta que queremos
    10 Drug Class
    11 Resistance Mechanism

El script reporta las tres granularidades (familia, clase de fármaco y mecanismo)
porque salen gratis del mismo cruce, y así la asesora decide con números.

Uso:
    python familias_card.py
    python familias_card.py --min-clase 30
    python familias_card.py --incluir-otras      # suma MEGARes y las demás bases
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--fasta", default=None)
ap.add_argument("--aro-index", default=None)
ap.add_argument("--min-clase", type=int, default=50)
ap.add_argument("--incluir-otras", action="store_true",
                help="añade MEGARes/ResFinder/etc. (vocabularios distintos: ver aviso)")
ap.add_argument("--top", type=int, default=30)
args = ap.parse_args()

# ------------------------------------------------------------------ rutas
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

FASTA = Path(args.fasta).expanduser() if args.fasta else next(
    (p for p in [RAIZ / "data" / "processed" / "argdb_POSITIVO_final.fasta"] if p.exists()), None)
if FASTA is None:
    sys.exit("No encontré el FASTA de positivos. Pásalo con --fasta")

ARO_IDX = Path(args.aro_index).expanduser() if args.aro_index else next(
    (p for p in RAIZ.rglob("aro_index.tsv")), None)
if ARO_IDX is None:
    sys.exit("No encontré aro_index.tsv. Pásalo con --aro-index data/raw/1_CARD/aro_index.tsv")

print("=" * 94)
print("FAMILIA DEL GEN desde la ontología curada de CARD")
print("FASTA    :", FASTA.name)
print("aro_index:", ARO_IDX)
print("=" * 94)

# ------------------------------------------------------------------ cargar aro_index
aro = {}
with open(ARO_IDX, encoding="utf-8", errors="replace") as fh:
    r = csv.DictReader(fh, delimiter="\t")
    faltan = [c for c in ("ARO Accession", "AMR Gene Family") if c not in (r.fieldnames or [])]
    if faltan:
        sys.exit(f"aro_index.tsv no tiene las columnas esperadas: faltan {faltan}\n"
                 f"columnas vistas: {r.fieldnames}")
    for fila in r:
        k = (fila.get("ARO Accession") or "").strip()
        if k:
            aro[k] = {
                "familia":   (fila.get("AMR Gene Family") or "").strip(),
                "farmaco":   (fila.get("Drug Class") or "").strip(),
                "mecanismo": (fila.get("Resistance Mechanism") or "").strip(),
                "nombre":    (fila.get("ARO Name") or "").strip(),
            }
print(f"\naro_index: {len(aro):,} entradas cargadas")

# ------------------------------------------------------------------ leer el FASTA
regs = []
h, s = None, []
with open(FASTA, encoding="utf-8", errors="replace") as fh:
    for linea in fh:
        linea = linea.rstrip("\n")
        if linea.startswith(">"):
            if h is not None:
                regs.append((h, "".join(s)))
            h, s = linea[1:].strip(), []
        else:
            s.append(linea.strip())
if h is not None:
    regs.append((h, "".join(s)))
print(f"FASTA    : {len(regs):,} secuencias")


def base_de(x):
    t = x.split()[0]
    return t.split("|", 1)[0] if "|" in t else t


RE_ARO = re.compile(r"(ARO:\d+)")

# ------------------------------------------------------------------ cruce
filas, sin_aro, aro_no_encontrado = [], [], []
for enc, seq in regs:
    b = base_de(enc)
    m = RE_ARO.search(enc)
    fam = farmaco = mec = ""
    aro_id = m.group(1) if m else ""
    if b.upper() == "CARD":
        if not aro_id:
            sin_aro.append(enc)
        elif aro_id not in aro:
            aro_no_encontrado.append(aro_id)
        else:
            d = aro[aro_id]
            fam, farmaco, mec = d["familia"], d["farmaco"], d["mecanismo"]
    elif args.incluir_otras:
        p = enc.split("|")
        if b.upper() == "MEGARES" and len(p) >= 5:
            fam, farmaco = p[4].strip(), p[3].strip()
        elif b.upper() == "RESFINDER" and len(p) >= 2:
            fam = re.sub(r"_\d+_[A-Za-z]{1,3}[\d_.]+.*$", "", p[1]).strip()
        elif b.upper() == "ARGANNOT":
            mm = re.search(r"\(([A-Za-z]+)\)\s*([A-Za-z0-9_\-'()]+)", enc)
            if mm:
                fam, farmaco = mm.group(2), mm.group(1)
        elif b.upper() == "AMRFINDERPLUS" and len(p) >= 8:
            fam = p[7].split()[0].strip()

    filas.append(dict(gen=enc.split()[0], base=b, aro=aro_id, familia=fam,
                      farmaco=farmaco, mecanismo=mec, largo=len(seq), encabezado=enc))

n_card = sum(1 for f in filas if f["base"].upper() == "CARD")
n_card_ok = sum(1 for f in filas if f["base"].upper() == "CARD" and f["familia"])
print(f"\nCruce con CARD: {n_card_ok:,} de {n_card:,} secuencias de CARD obtuvieron familia "
      f"({100*n_card_ok/max(1,n_card):.1f} %)")
if sin_aro:
    print(f"  {len(sin_aro)} encabezados de CARD sin ARO en el texto")
if aro_no_encontrado:
    c = Counter(aro_no_encontrado)
    print(f"  {len(aro_no_encontrado)} con un ARO que no está en aro_index "
          f"(ej.: {', '.join(list(c)[:5])})")
    print("    -> suele ser desfase de versión entre el FASTA y los metadatos de CARD")


# ------------------------------------------------------------------ reportes
def reporte(nombre, campo, universo):
    """Un valor puede traer varias etiquetas separadas por ';' (p. ej. Drug Class)."""
    sub = [f for f in universo if f[campo]]
    multi = sum(1 for f in sub if ";" in f[campo])
    cnt = Counter()
    for f in sub:
        for parte in f[campo].split(";"):
            parte = parte.strip()
            if parte:
                cnt[parte] += 1
    total = len(universo)
    us = {k: v for k, v in cnt.items() if v >= args.min_clase}
    print(f"\n{'='*94}\n{nombre}\n{'='*94}")
    print(f"  con etiqueta: {len(sub):,} de {total:,} ({100*len(sub)/max(1,total):.1f} %)"
          + (f"  |  {multi:,} con varias etiquetas (';')" if multi else ""))
    print(f"  clases distintas: {len(cnt)}  |  con >= {args.min_clase}: {len(us)}  "
          f"|  con < 7: {sum(1 for v in cnt.values() if v < 7)}")
    if us:
        print(f"  secuencias en clases usables: {sum(us.values()):,}")
        mx, mn = max(us.values()), min(us.values())
        print(f"  desbalance entre usables: {mx:,} vs {mn:,} (razón {mx/max(1,mn):.1f}x)")
    print(f"\n  {'clase':<48} {'n':>6} {'%':>7}   70/15/15")
    for k, v in cnt.most_common(args.top):
        tr, va = round(.70*v), round(.15*v)
        te = v - tr - va
        av = "  <- IMPOSIBLE" if v < 7 else ("  <- pocos" if v < args.min_clase else "")
        print(f"  {k[:48]:<48} {v:>6,} {100*v/max(1,total):>6.2f}%   {tr}/{va}/{te}{av}")
    if len(cnt) > args.top:
        print(f"  ... y {len(cnt)-args.top} clases más, "
              f"{sum(v for _, v in cnt.most_common()[args.top:]):,} secuencias")
    return cnt


universo = ([f for f in filas if f["base"].upper() == "CARD"]
            if not args.incluir_otras else filas)
etq = "CARD solamente" if not args.incluir_otras else "todas las bases"
print(f"\n### Universo del reporte: {etq} ({len(universo):,} secuencias) ###")

cnt_fam = reporte("AMR GENE FAMILY (familia del gen, curada por CARD)", "familia", universo)
cnt_far = reporte("DRUG CLASS (clase de fármaco)", "farmaco", universo)
cnt_mec = reporte("RESISTANCE MECHANISM (mecanismo)", "mecanismo", universo)

if args.incluir_otras:
    print(f"\n{'='*94}\nAVISO SOBRE --incluir-otras\n{'='*94}")
    print("  Cada base nombra las familias con su propio vocabulario: CARD dice")
    print("  'Erm-like 23S rRNA methyltransferase' y MEGARes dice otra cosa para lo mismo.")
    print("  Mezclarlas sin una tabla de equivalencias crea clases duplicadas y, peor,")
    print("  deja que el modelo aprenda a reconocer la BASE DE DATOS en vez de la familia.")
    print("  Para el primer modelo conviene CARD solo: un vocabulario, curado, sin mezclar.")

# ------------------------------------------------------------------ confundido
print(f"\n{'='*94}\nCHEQUEO DE CONFUNDIDO: familia x base de origen\n{'='*94}")
cruz = defaultdict(Counter)
for f in universo:
    if f["familia"]:
        cruz[f["familia"].split(";")[0].strip()][f["base"]] += 1
usables = [k for k, v in cnt_fam.items() if v >= args.min_clase]
if len(set(f["base"] for f in universo)) == 1:
    print("  Universo de una sola base: no aplica (no hay confundido posible por origen).")
elif usables:
    riesgo = []
    print(f"  {'familia':<44} {'n':>6} {'bases':>6}  {'dominante':<14} {'%':>7}")
    for k in sorted(usables, key=lambda x: -cnt_fam[x])[:args.top]:
        c = cruz.get(k, Counter())
        if not c:
            continue
        dom, nd = c.most_common(1)[0]
        pct = 100 * nd / sum(c.values())
        if pct >= 90:
            riesgo.append(k)
        print(f"  {k[:44]:<44} {cnt_fam[k]:>6,} {len(c):>6}  {dom:<14} {pct:>6.1f}%"
              + ("  <- MONOPOLIZADA" if pct >= 90 else ""))
    if riesgo:
        print(f"\n  AVISO: {len(riesgo)} familia(s) vienen >=90 % de una sola base.")

# ------------------------------------------------------------------ guardar
SAL = RAIZ / "results" / "familias"
SAL.mkdir(parents=True, exist_ok=True)

with open(SAL / "familias_card.tsv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, delimiter="\t",
                       fieldnames=["gen", "base", "aro", "familia", "farmaco",
                                   "mecanismo", "largo", "encabezado"])
    w.writeheader()
    w.writerows(filas)

with open(SAL / "revision_familias_card.txt", "w", encoding="utf-8") as fh:
    fh.write("FAMILIAS CURADAS DE CARD, con los genes que caen en cada una.\n")
    fh.write("Son las etiquetas oficiales de CARD, no reglas nuestras.\n\n")
    for k, v in cnt_fam.most_common():
        miembros = [f for f in universo if k in [x.strip() for x in f["familia"].split(";")]]
        fh.write(f"{'='*74}\n{k}   n={v}\n")
        nombres = Counter(aro.get(f["aro"], {}).get("nombre", f["gen"]) for f in miembros)
        for n, c in nombres.most_common(30):
            fh.write(f"    {c:>4}x  {n}\n")
        if len(nombres) > 30:
            fh.write(f"    ... y {len(nombres)-30} nombres más\n")

print(f"\nGuardado: {SAL/'familias_card.tsv'}")
print(f"Guardado: {SAL/'revision_familias_card.txt'}")
print("\n" + "=" * 94)
print("Lo que decide el diseño: cuántas AMR Gene Family llegan al mínimo y cuántas")
print("secuencias cubren. Si son 6-12 clases con 50+ cada una, el modelo es viable.")
