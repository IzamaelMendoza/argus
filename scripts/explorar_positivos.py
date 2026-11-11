"""
explorar_positivos.py -- radiografia del conjunto POSITIVO (Tarea 1 de la reunion del 10-nov).

Responde cuatro de las cinco preguntas de la asesora:

  1. Hay genes unicos en el set positivo?  -> duplicados exactos, duplicados por
     reverso-complemento, ids repetidos, y cuantas secuencias quedan realmente unicas.
  2. Cuantos ejemplos hay por familia?     -> conteo por familia desde etiquetas_finales.tsv
  3. Cual es el mas largo y el mas corto?  -> extremos, cuartiles y distribucion
  4. Grafico antes de cd-hit               -> composicion por base de datos ANTES y DESPUES
                                              del clustering, con el porcentaje de reduccion

(La quinta, meter el negativo al modelo, va en negativos_al_modelo.py: es otro tipo
de trabajo y tarda mucho mas.)

USO
---
    python explorar_positivos.py
    python explorar_positivos.py --pre-cdhit data/interim/argdb_combinado.fasta

Si no se le dice cual es el archivo de antes de cd-hit, lo busca entre los candidatos
habituales y lista lo que encuentre para que elijas.

SALIDAS
-------
    results/exploracion/positivos_resumen.txt     todo lo que imprime, para pegar en el documento
    results/exploracion/positivos_por_base.csv    tabla antes/despues por base de datos
    results/exploracion/positivos_largos.csv      largo de cada gen, con su base y familia
    results/exploracion/duplicados.csv            los grupos de secuencias repetidas
    results/exploracion/fig_antes_despues_cdhit.png
    results/exploracion/fig_largos.png
    results/exploracion/fig_familias.png
"""
import argparse
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ----------------------------------------------------------------- argumentos
ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--positivo", default=None,
                help="FASTA positivo final (por omision data/processed/argdb_POSITIVO_final.fasta)")
ap.add_argument("--pre-cdhit", default=None,
                help="FASTA combinado ANTES de cd-hit (las ~30,014 secuencias)")
ap.add_argument("--etiquetas", default=None,
                help="etiquetas_finales.tsv (gen -> familia)")
ap.add_argument("--sin-figuras", action="store_true")
args = ap.parse_args()

# ----------------------------------------------------------------- raiz
if args.raiz:
    RAIZ = Path(args.raiz).expanduser()
else:
    RAIZ = None
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "data").is_dir():
            RAIZ = cand
            break
    if RAIZ is None:
        sys.exit('No encontre el proyecto. Usa --raiz "ruta/al/Proyecto_Residencias"')

POS = Path(args.positivo) if args.positivo else RAIZ / "data" / "processed" / "argdb_POSITIVO_final.fasta"
ETQ = Path(args.etiquetas) if args.etiquetas else RAIZ / "results" / "familias" / "etiquetas_finales.tsv"
SAL = RAIZ / "results" / "exploracion"
SAL.mkdir(parents=True, exist_ok=True)

if not POS.exists():
    sys.exit(f"No existe el FASTA positivo: {POS}")

# ----------------------------------------------------------------- salida doble
_buf = []


def di(*a):
    """imprime en pantalla y guarda para el .txt"""
    s = " ".join(str(x) for x in a)
    print(s)
    _buf.append(s)


# ----------------------------------------------------------------- FASTA
def leer_fasta(path):
    """Devuelve [(encabezado_completo, secuencia), ...] en el orden del archivo."""
    out, h, s = [], None, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for linea in fh:
            linea = linea.rstrip("\n").rstrip("\r")
            if linea.startswith(">"):
                if h is not None:
                    out.append((h, "".join(s)))
                h, s = linea[1:], []
            elif h is not None:
                s.append(linea.strip())
    if h is not None:
        out.append((h, "".join(s)))
    return out


# ----------------------------------------------------------------- base de datos de origen
def base_de(enc):
    """
    De que base de datos viene esta secuencia, segun el formato de su encabezado.

    Los seis formatos (documentados en la bitacora, FASE 1):
      CARD           gb|accesion|hebra|coords|ARO:n|NOMBRE [organismo]
      ARG-ANNOT      argannot~~~(CLASE)NOMBRE~~~coords~~~
      MEGARes        MEG_n|tipo|clase|mecanismo|GRUPO
      AMRFinderPlus  prot|nucl|n|n|SIMBOLO|simbolo|descripcion   (8 campos)
      ResFinderFG    nombre|mecanismo|accesion|fuente|antibiotico (5 campos)
      ResFinder      nombre_variante_accesion                    (2 campos)

    El orden de las reglas importa: se prueban de la mas especifica a la mas general.
    """
    t = enc.strip()
    low = t.lower()

    # marcadores inequivocos primero
    if "aro:" in low:
        return "CARD"
    if "argannot" in low or "~~~" in t:
        return "ARG-ANNOT"
    if t.startswith("MEG_") or low.startswith("meg_"):
        return "MEGARes"

    campos = t.split("|")
    n = len(campos)
    if n >= 8:
        return "AMRFinderPlus"
    if n == 5:
        return "ResFinderFG"
    if n in (2, 3):
        return "ResFinder"
    return "desconocido"


# --- prueba de la funcion, con un ejemplo real de cada base ---
_CASOS = [
    ("gb|AY769962.1|+|2434-5611|ARO:3000781|adeJ [Acinetobacter baumannii]", "CARD"),
    ("argannot~~~(Bla)TEM-1~~~1-861~~~", "ARG-ANNOT"),
    ("MEG_1|Drugs|beta-lactams|Class_A_betalactamases|TEM", "MEGARes"),
    ("prot|NG_047831.1|1|1|blaTEM-1|blaTEM|class A beta-lactamase|extra", "AMRFinderPlus"),
    ("ERR1600439_k99_12345|beta-lactamase|ERR1600439|gut|amoxicillin", "ResFinderFG"),
    ("blaTEM-1_1_AF309824", "desconocido"),
    ("tet(A)_1|AJ517790", "ResFinder"),
]
_fallos = [(e, esp, base_de(e)) for e, esp in _CASOS if base_de(e) != esp]
if _fallos:
    print("[aviso] la deteccion de base de datos falla en estos casos de prueba:")
    for e, esp, got in _fallos:
        print(f"   esperaba {esp:<15} obtuvo {got:<15}  {e[:60]}")
    print("   -> revisa base_de(); los conteos por base pueden salir mal\n")


# ----------------------------------------------------------------- utilidades
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s):
    return s.translate(_COMP)[::-1]


def gc(s):
    s = s.upper()
    n = sum(s.count(b) for b in "ACGT")
    return 100.0 * (s.count("G") + s.count("C")) / n if n else 0.0


def pct(x, total):
    return f"{100.0*x/total:.1f}%" if total else "-"


# =============================================================================
#  CARGA
# =============================================================================
di("=" * 86)
di("EXPLORACION DEL CONJUNTO POSITIVO")
di("=" * 86)
di(f"archivo: {POS}")

pos = leer_fasta(POS)
di(f"secuencias: {len(pos):,}")
if not pos:
    sys.exit("El FASTA esta vacio.")

ids = [h.split()[0] for h, _ in pos]
seqs = [s.upper() for _, s in pos]
largos = [len(s) for s in seqs]

# =============================================================================
#  1. GENES UNICOS
# =============================================================================
di("\n" + "=" * 86)
di("1. GENES UNICOS EN EL SET POSITIVO")
di("=" * 86)

# --- ids repetidos ---
rep_id = {k: v for k, v in Counter(ids).items() if v > 1}
di(f"\nidentificadores repetidos: {len(rep_id)}")
if rep_id:
    di("  (dos secuencias con el mismo id rompen el agrupamiento por gen)")
    for k, v in list(rep_id.items())[:10]:
        di(f"    {v} veces  {k[:70]}")

# --- secuencias identicas ---
por_hash = defaultdict(list)
for i, s in enumerate(seqs):
    por_hash[hashlib.md5(s.encode()).hexdigest()].append(i)
grupos_dup = [v for v in por_hash.values() if len(v) > 1]
n_en_dup = sum(len(v) for v in grupos_dup)
di(f"\nsecuencias EXACTAMENTE identicas: {n_en_dup:,} secuencias en {len(grupos_dup):,} grupos")
di(f"  -> sobran {n_en_dup - len(grupos_dup):,} copias")

# --- identicas por reverso-complemento ---
#     cd-hit-est compara ambas hebras, asi que esto deberia ser 0; si no lo es,
#     es que algo entro despues del clustering
canon = defaultdict(list)
for i, s in enumerate(seqs):
    rc = revcomp(s)
    canon[hashlib.md5(min(s, rc).encode()).hexdigest()].append(i)
grupos_rc = [v for v in canon.values() if len(v) > 1]
extra_rc = len(grupos_rc) - len(grupos_dup)
di(f"secuencias identicas contando el reverso-complemento: {len(grupos_rc):,} grupos "
   f"({extra_rc:+,} respecto a solo hebra directa)")

unicas = len(por_hash)
di(f"\nSECUENCIAS UNICAS: {unicas:,} de {len(pos):,}  ({pct(unicas, len(pos))})")
if unicas == len(pos):
    di("  -> no hay ni una secuencia repetida. Es lo esperado despues de cd-hit al 95%:")
    di("     si dos fueran identicas, cd-hit las habria colapsado en una.")
else:
    di("  -> OJO: hay repetidas DESPUES de cd-hit. Vale revisar si algo se concateno")
    di("     al FASTA final despues de clusterizar.")

if grupos_dup:
    with open(SAL / "duplicados.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["grupo", "n_copias", "largo", "base", "id"])
        for gi, idxs in enumerate(grupos_dup, 1):
            for i in idxs:
                w.writerow([gi, len(idxs), largos[i], base_de(pos[i][0]), ids[i]])
    di(f"  detalle en {SAL / 'duplicados.csv'}")

# --- cuantos vienen de cada base, y si una base aporta mas duplicados ---
bases = [base_de(h) for h, _ in pos]
di(f"\nPor base de datos (en el positivo final):")
cb = Counter(bases)
for b, n in cb.most_common():
    di(f"  {n:>6}  {pct(n, len(pos)):>6}  {b}")
if cb.get("desconocido"):
    di(f"\n  [aviso] {cb['desconocido']} encabezados sin clasificar. Ejemplos:")
    for h, _ in pos:
        if base_de(h) == "desconocido":
            di(f"     {h[:78]}")
            if sum(1 for x in pos[:pos.index((h, _))] if base_de(x[0]) == "desconocido") >= 4:
                break

# =============================================================================
#  3. MAS LARGO Y MAS CORTO
# =============================================================================
di("\n" + "=" * 86)
di("3. LARGO DE LOS GENES POSITIVOS")
di("=" * 86)

orden = sorted(range(len(seqs)), key=lambda i: largos[i])
q = lambda p: largos[orden[min(len(orden) - 1, int(p * len(orden)))]]

di(f"\n  minimo   {largos[orden[0]]:>8,} pb")
di(f"  Q1       {q(0.25):>8,} pb")
di(f"  mediana  {q(0.50):>8,} pb")
di(f"  Q3       {q(0.75):>8,} pb")
di(f"  maximo   {largos[orden[-1]]:>8,} pb")
di(f"  promedio {sum(largos)/len(largos):>8,.0f} pb")

di("\nEL MAS CORTO:")
for i in orden[:5]:
    di(f"  {largos[i]:>7,} pb  [{base_de(pos[i][0])}]  {ids[i][:60]}")
di("\nEL MAS LARGO:")
for i in orden[-5:][::-1]:
    di(f"  {largos[i]:>7,} pb  [{base_de(pos[i][0])}]  {ids[i][:60]}")

# --- lo que importa para la fragmentacion ---
MIN_PB, VENT = 150, 500
cortos = sum(1 for L in largos if L < MIN_PB)
un_frag = sum(1 for L in largos if MIN_PB <= L < VENT)
di(f"\nRelevante para la fragmentacion (ventana {VENT}, minimo {MIN_PB}):")
di(f"  mas cortos que {MIN_PB} pb -> NO producen ningun fragmento: {cortos:,} ({pct(cortos, len(pos))})")
di(f"  entre {MIN_PB} y {VENT} pb -> un solo fragmento parcial:    {un_frag:,} ({pct(un_frag, len(pos))})")
di(f"  de {VENT} pb o mas         -> al menos un fragmento entero:  "
   f"{len(pos)-cortos-un_frag:,} ({pct(len(pos)-cortos-un_frag, len(pos))})")

# --- largo por base de datos: delata si una base aporta secuencias atipicas ---
di("\nLargo por base de datos:")
di(f"  {'base':<16} {'n':>6} {'min':>8} {'mediana':>9} {'max':>9}")
por_base = defaultdict(list)
for b, L in zip(bases, largos):
    por_base[b].append(L)
for b, Ls in sorted(por_base.items(), key=lambda kv: -len(kv[1])):
    Ls = sorted(Ls)
    di(f"  {b:<16} {len(Ls):>6} {Ls[0]:>8,} {Ls[len(Ls)//2]:>9,} {Ls[-1]:>9,}")

# =============================================================================
#  2. EJEMPLOS POR FAMILIA
# =============================================================================
di("\n" + "=" * 86)
di("2. EJEMPLOS POR FAMILIA")
di("=" * 86)

familia_de = {}
if ETQ.exists():
    with open(ETQ, encoding="utf-8") as fh:
        for f in csv.DictReader(fh, delimiter="\t"):
            if f.get("etiqueta"):
                familia_de[f["gen"]] = f["etiqueta"]
    cf = Counter(familia_de.values())
    tot = sum(cf.values())
    di(f"\n{len(cf)} familias | {tot:,} genes etiquetados "
       f"(de {len(pos):,} positivos: el resto no tiene familia en CARD)")
    di(f"\n  {'n':>6} {'%':>7}  {'acumulado':>10}  familia")
    acum = 0
    for fam, n in cf.most_common():
        acum += n
        di(f"  {n:>6} {100*n/tot:>6.1f}%  {100*acum/tot:>9.1f}%  {fam}")
    di(f"\n  razon mayor/menor: {max(cf.values())/min(cf.values()):.1f} a 1")
    di(f"  la familia mas chica tiene {min(cf.values())} genes")
else:
    di(f"\n  No encontre {ETQ}")
    di("  Corre antes: python scripts/modelo_familias.py --solo-censo")
    cf = Counter()

# =============================================================================
#  4. ANTES Y DESPUES DE CD-HIT
# =============================================================================
di("\n" + "=" * 86)
di("4. COMPOSICION ANTES Y DESPUES DE CD-HIT")
di("=" * 86)

pre_path = Path(args.pre_cdhit) if args.pre_cdhit else None
if pre_path is None:
    # buscar el combinado entre los nombres mas probables
    cands = []
    for d in (RAIZ / "data" / "interim", RAIZ / "data" / "processed", RAIZ / "data" / "raw"):
        if d.is_dir():
            for p in d.glob("*.fasta"):
                nl = p.name.lower()
                if any(t in nl for t in ("combinad", "merge", "todas", "all", "pre_cdhit",
                                         "pre-cdhit", "sin_cdhit", "crudo", "unido")):
                    cands.append(p)
    if len(cands) == 1:
        pre_path = cands[0]
        di(f"\nusando el combinado que encontre: {pre_path.name}")
    elif cands:
        di("\nHay varios candidatos para el archivo de ANTES de cd-hit. Elige uno con --pre-cdhit:")
        for p in cands:
            di(f"   {p}")
    else:
        di("\nNo encontre el FASTA combinado de antes de cd-hit.")
        di("Segun la bitacora deben ser ~30,014 secuencias (la entrada de cd-hit-est).")
        di("Pasalo con:  --pre-cdhit ruta/al/archivo.fasta")
        di("Para localizarlo:")
        di(f"   grep -c '>' {RAIZ}/data/interim/*.fasta {RAIZ}/data/processed/*.fasta")

pre_conteo = None
if pre_path and Path(pre_path).exists():
    pre = leer_fasta(pre_path)
    pre_conteo = Counter(base_de(h) for h, _ in pre)
    di(f"\nantes de cd-hit : {len(pre):,} secuencias  ({pre_path})")
    di(f"despues         : {len(pos):,} secuencias  ({POS.name})")
    di(f"reduccion       : {len(pre)-len(pos):,} secuencias "
       f"({100*(len(pre)-len(pos))/len(pre):.1f}% colapsadas por redundancia)")

    di(f"\n  {'base':<16} {'antes':>8} {'despues':>8} {'quedan':>8}  {'% del final':>11}")
    todas = sorted(set(pre_conteo) | set(cb), key=lambda b: -pre_conteo.get(b, 0))
    for b in todas:
        a, d_ = pre_conteo.get(b, 0), cb.get(b, 0)
        tasa = f"{100*d_/a:.1f}%" if a else "-"
        di(f"  {b:<16} {a:>8,} {d_:>8,} {tasa:>8}  {pct(d_, len(pos)):>11}")

    with open(SAL / "positivos_por_base.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["base", "antes_cdhit", "despues_cdhit", "tasa_supervivencia", "pct_del_final"])
        for b in todas:
            a, d_ = pre_conteo.get(b, 0), cb.get(b, 0)
            w.writerow([b, a, d_, round(100*d_/a, 2) if a else "", round(100*d_/len(pos), 2)])
    di(f"\n  tabla en {SAL / 'positivos_por_base.csv'}")
    di("\n  La 'tasa de supervivencia' dice que tan redundante era cada base: una tasa")
    di("  baja significa que esa base traia muchas variantes casi identicas entre si.")

# ----------------------------------------------------------------- CSV de largos
with open(SAL / "positivos_largos.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "base", "familia", "largo_pb", "gc_pct", "n_fragmentos_150_500"])
    for i, (h, s) in enumerate(pos):
        L = largos[i]
        nf = 0 if L < MIN_PB else (1 if L < VENT else L // VENT + (1 if L % VENT >= MIN_PB else 0))
        w.writerow([ids[i], bases[i], familia_de.get(ids[i], ""), L, round(gc(s), 2), nf])
di(f"\nlargo de cada gen en {SAL / 'positivos_largos.csv'}")

# =============================================================================
#  FIGURAS
# =============================================================================
if not args.sin_figuras:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- antes vs despues de cd-hit ---
        if pre_conteo:
            todas = sorted(set(pre_conteo) | set(cb), key=lambda b: -pre_conteo.get(b, 0))
            x = range(len(todas))
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
            a1.bar([i - 0.2 for i in x], [pre_conteo.get(b, 0) for b in todas],
                   width=0.4, label=f"antes de cd-hit (n={sum(pre_conteo.values()):,})")
            a1.bar([i + 0.2 for i in x], [cb.get(b, 0) for b in todas],
                   width=0.4, label=f"despues (n={len(pos):,})")
            a1.set_xticks(list(x)); a1.set_xticklabels(todas, rotation=20, ha="right")
            a1.set_ylabel("secuencias"); a1.legend(loc="upper right")
            a1.set_title("Composicion por base de datos")
            a1.set_ylim(0, max(pre_conteo.values()) * 1.25)   # aire para las etiquetas
            for i, b in enumerate(todas):
                a1.text(i - 0.2, pre_conteo.get(b, 0), f"{pre_conteo.get(b,0):,}",
                        ha="center", va="bottom", fontsize=7, rotation=90)
                a1.text(i + 0.2, cb.get(b, 0), f"{cb.get(b,0):,}",
                        ha="center", va="bottom", fontsize=7, rotation=90)

            tasas = [100 * cb.get(b, 0) / pre_conteo[b] for b in todas if pre_conteo.get(b)]
            nn = [b for b in todas if pre_conteo.get(b)]
            a2.bar(range(len(nn)), tasas, color="#6C757D")
            a2.axhline(100 * len(pos) / sum(pre_conteo.values()), ls="--", c="crimson",
                       label=f"global {100*len(pos)/sum(pre_conteo.values()):.1f}%")
            a2.set_xticks(range(len(nn))); a2.set_xticklabels(nn, rotation=20, ha="right")
            a2.set_ylabel("% que sobrevive a cd-hit"); a2.legend()
            a2.set_title("Redundancia interna de cada base\n(barra baja = muchas variantes casi iguales)")
            for i, t in enumerate(tasas):
                a2.text(i, t, f"{t:.0f}%", ha="center", va="bottom", fontsize=8)
            plt.tight_layout()
            plt.savefig(SAL / "fig_antes_despues_cdhit.png", dpi=200)
            plt.close()
            di(f"figura: {SAL / 'fig_antes_despues_cdhit.png'}")

        # --- distribucion de largos ---
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
        a1.hist(largos, bins=60, color="#4C78A8")
        a1.axvline(MIN_PB, c="crimson", ls="--", label=f"{MIN_PB} pb (minimo)")
        a1.axvline(VENT, c="darkorange", ls="--", label=f"{VENT} pb (ventana)")
        a1.set_xlabel("largo del gen (pb)"); a1.set_ylabel("genes"); a1.legend()
        a1.set_title("Distribucion de largos")
        rec = [L for L in largos if L <= 6000]
        a2.hist(rec, bins=60, color="#4C78A8")
        a2.axvline(MIN_PB, c="crimson", ls="--"); a2.axvline(VENT, c="darkorange", ls="--")
        a2.set_xlabel("largo del gen (pb), recortado a 6000"); a2.set_ylabel("genes")
        a2.set_title(f"Mismo histograma sin la cola\n({len(largos)-len(rec)} genes de mas de 6000 pb fuera)")
        plt.tight_layout(); plt.savefig(SAL / "fig_largos.png", dpi=200); plt.close()
        di(f"figura: {SAL / 'fig_largos.png'}")

        # --- familias ---
        if cf:
            fams = [f for f, _ in cf.most_common()][::-1]
            vals = [cf[f] for f in fams]
            fig, ax = plt.subplots(figsize=(9, 5))
            cols = ["#C44E52" if v < 50 else "#4C78A8" for v in vals]
            ax.barh(range(len(fams)), vals, color=cols)
            ax.set_yticks(range(len(fams)))
            ax.set_yticklabels([f[:38] for f in fams], fontsize=8)
            ax.set_xlabel("genes")
            ax.set_title(f"Ejemplos por familia ({len(cf)} familias, {sum(cf.values()):,} genes)\n"
                         "en rojo las de menos de 50 genes")
            for i, v in enumerate(vals):
                ax.text(v, i, f" {v}", va="center", fontsize=8)
            plt.tight_layout(); plt.savefig(SAL / "fig_familias.png", dpi=200); plt.close()
            di(f"figura: {SAL / 'fig_familias.png'}")
    except ImportError:
        di("\n(matplotlib no esta instalado: no se generaron figuras)")

# ----------------------------------------------------------------- guardar texto
with open(SAL / "positivos_resumen.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(_buf) + "\n")
print(f"\nResumen completo guardado en {SAL / 'positivos_resumen.txt'}")
