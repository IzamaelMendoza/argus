"""
negativos_al_modelo.py -- pasa el conjunto NEGATIVO COMPLETO por el modelo y caracteriza
los que salen positivos (Tarea 1, punto 2 de la reunion del 10-nov).

La pregunta de la asesora: "meter el set negativo completo al modelo y los que salgan
positivos ver por que salen, que similitud tienen y por que".

OJO con la interpretacion: una parte grande de estos negativos SI se uso para entrenar,
asi que la tasa de falsos positivos global que salga aqui NO es una estimacion honesta
del error. El script separa los dos grupos y lo reporta por separado:

  * negativos que el modelo VIO al entrenar   -> memorizados, su tasa es optimista
  * negativos que NO vio (test + 1k+1k)       -> esta es la tasa que vale

Lo que si es valido con todo el conjunto es la pregunta de la asesora: QUE caracteriza
a los que fallan. Para eso el script mide, para cada falso positivo:

  - genoma de origen (de los seis de Prodigal)
  - %GC del gen y del genoma
  - largo y numero de fragmentos
  - probabilidad maxima y cuantos de sus fragmentos salen positivos
  - distancia composicional al centroide de los positivos (ver abajo)

La "similitud" se mide de dos formas distintas, porque son dos preguntas distintas:

  1. COMPOSICIONAL: que tan cerca esta su vector de 6-meros del centro de masa de los
     positivos, comparado con el de los negativos. Es lo que el modelo literalmente ve.
  2. HOMOLOGA: eso lo responde BLAST, no este script. En la FASE 7 ya se hizo y dio
     0 de 10. Aqui se escribe un FASTA listo para blastn por si se quiere repetir con
     el conjunto completo.

USO
---
    python negativos_al_modelo.py                    # prueba con 2000 negativos
    python negativos_al_modelo.py --completo         # los 26,513
    python negativos_al_modelo.py --completo --modelo data/models/argus_rf_k6_id95_v2.joblib

SALIDAS  (en results/negativos/)
    negativos_predicciones.csv    un renglon por gen, con todo lo medido
    negativos_falsos_positivos.fasta
    negativos_resumen.txt
    fig_negativos.png
"""
import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--modelo", default=None,
                help="por omision data/models/argus_rf_k6_id95_v2.joblib")
ap.add_argument("--negativo", default=None,
                help="por omision data/processed/argdb_NEGATIVO_final.fasta")
ap.add_argument("--positivo", default=None)
ap.add_argument("--completo", action="store_true", help="sin esto, solo 2000 negativos de prueba")
ap.add_argument("--umbral", type=float, default=None, help="por omision, el del .joblib")
ap.add_argument("--lote", type=int, default=4000, help="fragmentos por lote (cuida la RAM)")
ap.add_argument("--sin-figuras", action="store_true")
args = ap.parse_args()

# ----------------------------------------------------------------- raiz y archivos
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

MODELO = Path(args.modelo) if args.modelo else RAIZ / "data" / "models" / "argus_rf_k6_id95_v2.joblib"
NEG = Path(args.negativo) if args.negativo else RAIZ / "data" / "processed" / "argdb_NEGATIVO_final.fasta"
POS = Path(args.positivo) if args.positivo else RAIZ / "data" / "processed" / "argdb_POSITIVO_final.fasta"
SET_1K1K = RAIZ / "data" / "interim" / "muestra_genes_labels.tsv"
SAL = RAIZ / "results" / "negativos"
SAL.mkdir(parents=True, exist_ok=True)

for p, q in [(MODELO, "el modelo .joblib"), (NEG, "el FASTA negativo")]:
    if not p.exists():
        sys.exit(f"Falta {q}: {p}")

_buf = []


def di(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _buf.append(s)


# ----------------------------------------------------------------- modelo
import joblib

paq = joblib.load(MODELO)
rf = paq["modelo"] if isinstance(paq, dict) else paq
meta = paq if isinstance(paq, dict) else {}
UMBRAL = args.umbral if args.umbral is not None else float(meta.get("umbral", 0.5))
VENTANA = int(meta.get("ventana", 500))
MIN_PB = int(meta.get("min_pb", 150))
K = 1
while 4 ** K < rf.n_features_in_:
    K += 1
if 4 ** K != rf.n_features_in_:
    sys.exit(f"El modelo espera {rf.n_features_in_} columnas; no corresponde a 4**K.")
try:
    rf.set_params(n_jobs=-1)
except Exception:
    pass

di("=" * 88)
di("EL CONJUNTO NEGATIVO COMPLETO CONTRA EL MODELO")
di("=" * 88)
di(f"modelo : {MODELO.name}")
di(f"         K={K} | ventana={VENTANA} | min_pb={MIN_PB} | umbral={UMBRAL}")
if "min_pb" not in meta:
    di("         [aviso] el modelo no registra min_pb: es de antes de la regla de la asesora.")
    di(f"         Se usa {MIN_PB} de todos modos, pero el modelo entreno con relleno.")

# ----------------------------------------------------------------- k-meros
LUT = np.full(256, 4, dtype=np.int8)
for _i, _b in enumerate(b"ACGT"):
    LUT[_b] = _i
POT = (4 ** np.arange(K - 1, -1, -1)).astype(np.int64)


def vector_kmer(seq):
    a = LUT[np.frombuffer(seq.upper().encode("ascii", "replace"), dtype=np.uint8)]
    out = np.zeros(4 ** K, dtype=np.float32)
    if len(a) - K + 1 <= 0:
        return out
    win = np.lib.stride_tricks.sliding_window_view(a, K)
    ok = (win < 4).all(axis=1)
    if not ok.any():
        return out
    idx = (win[ok].astype(np.int64) * POT).sum(axis=1)
    out += np.bincount(idx, minlength=4 ** K).astype(np.float32)
    return out / out.sum()


def fragmentar(seq, ventana=VENTANA, min_pb=MIN_PB):
    """Regla de la asesora: bloques de `ventana`; el resto se queda si tiene >= min_pb."""
    seq = seq.upper()
    L = len(seq)
    if L < ventana:
        return [seq] if L >= min_pb else []
    n = L // ventana
    fr = [seq[i * ventana:(i + 1) * ventana] for i in range(n)]
    r = L - n * ventana
    if r >= min_pb:
        fr.append(seq[n * ventana:])
    return fr


def leer_fasta(path):
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


def gc(s):
    s = s.upper()
    n = sum(s.count(b) for b in "ACGT")
    return 100.0 * (s.count("G") + s.count("C")) / n if n else 0.0


def genoma_de(enc):
    """Los negativos vienen de Prodigal sobre seis genomas: el prefijo es el ensamble."""
    t = enc.split()[0]
    return t.split("|")[0] if "|" in t else t.rsplit("_", 1)[0]


# ----------------------------------------------------------------- datos
neg = leer_fasta(NEG)
di(f"\nnegativos en el archivo: {len(neg):,}")
if not args.completo:
    rng = np.random.default_rng(42)
    sel = sorted(rng.choice(len(neg), size=min(2000, len(neg)), replace=False).tolist())
    neg = [neg[i] for i in sel]
    di(f"MODO PRUEBA: solo {len(neg):,} de muestra. Usa --completo para el conjunto entero.")

ids_1k = set()
if SET_1K1K.exists():
    with open(SET_1K1K, encoding="utf-8") as fh:
        next(fh, None)
        for linea in fh:
            p = linea.split("\t")
            if p:
                ids_1k.add(p[0].strip())
    di(f"genes del 1k+1k conocidos: {len(ids_1k):,} (se marcan aparte: el modelo no los vio)")
else:
    di(f"[aviso] no encontre {SET_1K1K.name}: no se puede separar 'visto' de 'no visto'.")

# ----------------------------------------------------------------- predecir por lotes
di(f"\nPrediciendo (lotes de {args.lote:,} fragmentos)...")
t0 = time.time()
filas = []
buf_vec, buf_ref = [], []       # acumulador de fragmentos entre lotes
acum = defaultdict(list)        # gen -> [probabilidades de sus fragmentos]
sin_frag = []


def vaciar():
    if not buf_vec:
        return
    X = np.asarray(buf_vec, dtype=np.float32)
    p = rf.predict_proba(X)[:, 1]
    for ref, pi in zip(buf_ref, p):
        acum[ref].append(float(pi))
    buf_vec.clear()
    buf_ref.clear()


for i, (h, s) in enumerate(neg):
    gid = h.split()[0]
    fr = fragmentar(s)
    if not fr:
        sin_frag.append((gid, len(s)))
        continue
    for f in fr:
        buf_vec.append(vector_kmer(f))
        buf_ref.append(gid)
    if len(buf_vec) >= args.lote:
        vaciar()
    if (i + 1) % 2000 == 0:
        vel = (i + 1) / max(1e-9, time.time() - t0)
        di(f"  {i+1:,}/{len(neg):,}  ({vel:.0f} genes/s, "
           f"faltan ~{(len(neg)-i-1)/max(vel,1e-9)/60:.1f} min)")
vaciar()
di(f"  listo en {time.time()-t0:.0f}s")

# ----------------------------------------------------------------- armar la tabla
di(f"\ngenes sin ningun fragmento (< {MIN_PB} pb, no evaluables): {len(sin_frag):,}")

for h, s in neg:
    gid = h.split()[0]
    ps = acum.get(gid)
    if not ps:
        continue
    ps = np.asarray(ps)
    filas.append(dict(
        gen=gid, encabezado=h, genoma=genoma_de(h), largo=len(s), gc=gc(s),
        n_frag=len(ps), n_frag_pos=int((ps >= UMBRAL).sum()),
        p_max=float(ps.max()), p_media=float(ps.mean()),
        pred=int((ps >= UMBRAL).any()),
        visto_en_train=int(gid not in ids_1k) if ids_1k else -1,
    ))

n = len(filas)
fp = [f for f in filas if f["pred"] == 1]
di(f"\n{'='*88}")
di("RESULTADO GLOBAL")
di(f"{'='*88}")
di(f"\ngenes evaluados    : {n:,}")
di(f"salen POSITIVOS    : {len(fp):,}  ({100*len(fp)/n:.1f}%)")
di(f"salen negativos    : {n-len(fp):,}  ({100*(n-len(fp))/n:.1f}%)")

# ----------------------------------------------------------------- visto vs no visto
if ids_1k:
    vistos = [f for f in filas if f["visto_en_train"] == 1]
    nuevos = [f for f in filas if f["visto_en_train"] == 0]
    if nuevos:
        tv = 100 * sum(f["pred"] for f in vistos) / max(1, len(vistos))
        tn = 100 * sum(f["pred"] for f in nuevos) / max(1, len(nuevos))
        di(f"\n{'-'*88}")
        di("SEPARANDO LO QUE EL MODELO YA HABIA VISTO")
        di(f"{'-'*88}")
        di(f"\n  negativos del pool (entrenamiento + test interno) : {len(vistos):>7,}   "
           f"tasa de FP {tv:>5.1f}%")
        di(f"  negativos del 1k+1k (apartados desde el inicio)   : {len(nuevos):>7,}   "
           f"tasa de FP {tn:>5.1f}%")
        di(f"  diferencia: {tn-tv:+.1f} puntos porcentuales")
        di("\n  La cifra que vale es la SEGUNDA: esos 1,000 genes se apartaron antes de")
        di("  entrenar y el modelo no los vio nunca.")
        di("  La primera mezcla dos cosas y hay que leerla con cuidado: la mayoria de")
        di("  esos genes se usaron para entrenar (el bosque ya los conoce), pero un ~10%")
        di("  cayo en el test interno y tampoco se vio. No se pueden separar sin el")
        di("  corte exacto del cuaderno, asi que esta tasa no estima nada por si sola.")
        if tv > tn:
            di(f"\n  Aqui salio MAS ALTA en el pool ({tv:.1f}% contra {tn:.1f}%), que es lo")
            di("  contrario de lo que haria la memorizacion. Vale revisar si el 1k+1k")
            di("  tiene una composicion distinta del resto (por ejemplo, si quedo cargado")
            di("  a un genoma con GC alto, que es el que menos falsos positivos da).")

# ----------------------------------------------------------------- por genoma
di(f"\n{'-'*88}")
di("POR GENOMA DE ORIGEN")
di(f"{'-'*88}")
por_g = defaultdict(lambda: {"n": 0, "fp": 0, "gc": []})
for f in filas:
    d = por_g[f["genoma"]]
    d["n"] += 1
    d["fp"] += f["pred"]
    d["gc"].append(f["gc"])
di(f"\n  {'genoma':<28} {'genes':>7} {'FP':>7} {'tasa':>7} {'%GC medio':>10}")
orden_g = sorted(por_g.items(), key=lambda kv: -kv[1]["fp"] / max(1, kv[1]["n"]))
for g, d in orden_g:
    di(f"  {g[:28]:<28} {d['n']:>7,} {d['fp']:>7,} "
       f"{100*d['fp']/d['n']:>6.1f}% {np.mean(d['gc']):>9.1f}%")

if len(por_g) > 2:
    gcs = np.array([np.mean(d["gc"]) for _, d in por_g.items()])
    tas = np.array([d["fp"] / max(1, d["n"]) for _, d in por_g.items()])
    if gcs.std() > 1e-9 and tas.std() > 1e-9:
        r = float(np.corrcoef(gcs, tas)[0, 1])
        di(f"\n  correlacion %GC del genoma vs tasa de FP: {r:+.3f}")
        di("  (en la FASE 7 dio -0.883 sobre los cinco genomas de GC < 60: los organismos")
        di("   de GC bajo, los mas parecidos a ResFinderFG, son los que mas FP generan)")

# ----------------------------------------------------------------- similitud composicional
di(f"\n{'-'*88}")
di("SIMILITUD COMPOSICIONAL: QUE TAN 'POSITIVO' SE VE CADA UNO")
di(f"{'-'*88}")
if POS.exists():
    pos = leer_fasta(POS)
    rng = np.random.default_rng(7)
    muestra_pos = [pos[i] for i in rng.choice(len(pos), size=min(1500, len(pos)), replace=False)]
    muestra_neg = [neg[i] for i in rng.choice(len(neg), size=min(1500, len(neg)), replace=False)]

    def centroide(lista):
        V = []
        for _, s in lista:
            fr = fragmentar(s)
            if fr:
                V.append(vector_kmer(fr[0]))
        return np.mean(V, axis=0) if V else None

    c_pos, c_neg = centroide(muestra_pos), centroide(muestra_neg)
    if c_pos is not None and c_neg is not None:
        def coseno(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            return float(a @ b / (na * nb)) if na and nb else 0.0

        di(f"\n  centroides calculados con {len(muestra_pos)} positivos y "
           f"{len(muestra_neg)} negativos (primer fragmento de cada uno)")
        di(f"  coseno entre los dos centroides: {coseno(c_pos, c_neg):.4f}")
        di("  (cerca de 1 = los dos grupos se ven casi iguales en composicion de 6-meros)")

        # para cada gen: a cual centroide se parece mas
        sub = fp[:] if len(fp) <= 4000 else fp[:4000]
        ctrl = [f for f in filas if f["pred"] == 0][:len(sub)]
        seq_de = {h.split()[0]: s for h, s in neg}

        def margen(lote):
            out = []
            for f in lote:
                fr = fragmentar(seq_de.get(f["gen"], ""))
                if not fr:
                    continue
                v = vector_kmer(fr[0])
                out.append(coseno(v, c_pos) - coseno(v, c_neg))
            return np.asarray(out)

        m_fp, m_tn = margen(sub), margen(ctrl)
        if len(m_fp) and len(m_tn):
            di(f"\n  margen = coseno(gen, centroide positivo) - coseno(gen, centroide negativo)")
            di(f"  un margen > 0 significa que el gen se parece mas a los positivos\n")
            di(f"  {'grupo':<34} {'n':>6} {'margen medio':>14} {'% con margen>0':>16}")
            di(f"  {'negativos que salen POSITIVOS':<34} {len(m_fp):>6} "
               f"{m_fp.mean():>+14.5f} {100*(m_fp>0).mean():>15.1f}%")
            di(f"  {'negativos que salen negativos':<34} {len(m_tn):>6} "
               f"{m_tn.mean():>+14.5f} {100*(m_tn>0).mean():>15.1f}%")
            dif = m_fp.mean() - m_tn.mean()
            di(f"\n  diferencia: {dif:+.5f}")
            if abs(dif) < 1e-4:
                di("  -> practicamente la misma composicion. Los falsos positivos NO se")
                di("     distinguen por parecerse mas a los ARG en frecuencias de 6-meros.")
            else:
                di("  -> los que fallan si estan composicionalmente corridos hacia los positivos.")
else:
    di(f"\n  (no encontre {POS.name}: no se pudo medir la similitud composicional)")

# ----------------------------------------------------------------- perfil de los FP
if fp:
    di(f"\n{'-'*88}")
    di(f"PERFIL DE LOS {len(fp):,} FALSOS POSITIVOS")
    di(f"{'-'*88}")
    altos = [f for f in fp if f["p_max"] >= 0.9]
    medios = [f for f in fp if 0.7 <= f["p_max"] < 0.9]
    bajos = [f for f in fp if f["p_max"] < 0.7]
    di(f"\n  por confianza:  p>=0.90 {len(altos):>6,}   0.70-0.90 {len(medios):>6,}   "
       f"<0.70 {len(bajos):>6,}")
    di("  (los de p cercana al umbral se corrigen subiendolo; los de p>=0.9 son")
    di("   errores de fondo del modelo y son los que vale la pena mirar)")

    todos_frag = [f for f in fp if f["n_frag_pos"] == f["n_frag"] and f["n_frag"] >= 3]
    di(f"\n  con TODOS sus fragmentos positivos y 3 o mas fragmentos: {len(todos_frag):,}")
    di("  (estos son los candidatos limpios: el error no viene de un solo fragmento raro)")

    di(f"\n  los 15 peores (mayor probabilidad):")
    di(f"    {'p_max':>7} {'frags':>8} {'largo':>8} {'%GC':>6}  gen")
    for f in sorted(fp, key=lambda x: -x["p_max"])[:15]:
        di(f"    {f['p_max']:>7.4f} {f['n_frag_pos']:>3}/{f['n_frag']:<4} "
           f"{f['largo']:>8,} {f['gc']:>5.1f}  {f['gen'][:46]}")

    with open(SAL / "negativos_falsos_positivos.fasta", "w", encoding="utf-8") as fh:
        seq_de = {h.split()[0]: (h, s) for h, s in neg}
        for f in sorted(fp, key=lambda x: -x["p_max"]):
            h, s = seq_de[f["gen"]]
            fh.write(f">{h} p_max={f['p_max']:.4f} frags={f['n_frag_pos']}/{f['n_frag']}\n{s}\n")
    di(f"\n  FASTA para BLAST: {SAL / 'negativos_falsos_positivos.fasta'}")

# ----------------------------------------------------------------- CSV
campos = ["gen", "genoma", "largo", "gc", "n_frag", "n_frag_pos", "p_max", "p_media",
          "pred", "visto_en_train", "encabezado"]
with open(SAL / "negativos_predicciones.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
    w.writeheader()
    for f in sorted(filas, key=lambda x: -x["p_max"]):
        w.writerow(f)
di(f"\ntabla completa: {SAL / 'negativos_predicciones.csv'}")

# ----------------------------------------------------------------- figura
if not args.sin_figuras and filas:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 2, figsize=(13, 9))
        P = np.array([f["p_max"] for f in filas])
        ax[0, 0].hist(P, bins=60, color="#4C78A8")
        ax[0, 0].axvline(UMBRAL, c="crimson", ls="--", label=f"umbral {UMBRAL}")
        ax[0, 0].set_xlabel("probabilidad maxima del gen"); ax[0, 0].set_ylabel("genes")
        ax[0, 0].set_title("Distribucion de probabilidad\n(todo esto deberia estar a la izquierda del umbral)")
        ax[0, 0].legend()

        gg = [g for g, _ in orden_g]
        tt = [100 * por_g[g]["fp"] / por_g[g]["n"] for g in gg]
        ax[0, 1].barh(range(len(gg)), tt, color="#C44E52")
        ax[0, 1].set_yticks(range(len(gg)))
        ax[0, 1].set_yticklabels([g[:24] for g in gg], fontsize=8)
        ax[0, 1].set_xlabel("% de falsos positivos")
        ax[0, 1].set_title("Tasa de falsos positivos por genoma")
        for i, t in enumerate(tt):
            ax[0, 1].text(t, i, f" {t:.1f}%", va="center", fontsize=8)

        ax[1, 0].scatter([f["gc"] for f in filas], P, s=3, alpha=0.25, c="#4C78A8")
        ax[1, 0].axhline(UMBRAL, c="crimson", ls="--")
        ax[1, 0].set_xlabel("%GC del gen"); ax[1, 0].set_ylabel("probabilidad maxima")
        ax[1, 0].set_title("Probabilidad contra %GC\n(en la FASE 7 la correlacion fue -0.024: no es un detector de GC)")

        L = np.array([f["largo"] for f in filas])
        ax[1, 1].scatter(L, P, s=3, alpha=0.25, c="#4C78A8")
        ax[1, 1].axhline(UMBRAL, c="crimson", ls="--")
        ax[1, 1].set_xscale("log")
        ax[1, 1].set_xlabel("largo del gen (pb, escala log)")
        ax[1, 1].set_ylabel("probabilidad maxima")
        ax[1, 1].set_title("Probabilidad contra largo\n(mas largo = mas fragmentos = mas oportunidades con la regla OR)")

        plt.tight_layout(); plt.savefig(SAL / "fig_negativos.png", dpi=200); plt.close()
        di(f"figura: {SAL / 'fig_negativos.png'}")
    except ImportError:
        di("(matplotlib no instalado: sin figuras)")

if not args.completo:
    di("\n" + "!" * 88)
    di("ESTO FUE LA PRUEBA con una muestra. Corre con --completo para el conjunto entero.")
    di("!" * 88)

with open(SAL / "negativos_resumen.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(_buf) + "\n")
print(f"\nResumen en {SAL / 'negativos_resumen.txt'}")
