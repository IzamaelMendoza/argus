"""
modelo_familias.py -- ETAPA 2 de ARGus: dado un gen ya clasificado como ARG, decir
de qué FAMILIA es.

Produce exactamente lo que la GUI ya espera en `predecir_mag()`:

    ruta_familias = os.path.join(base_dir, "modelo_familias.joblib")
    modelo_etapa2 = joblib.load(ruta_familias)
    familia_pred = str(modelo_etapa2.predict(X_frags[i_max:i_max+1])[0])

O sea: un estimador SUELTO (no un dict), que recibe los mismos vectores de k-meros
k=6 que ya calcula la etapa 1, y devuelve el nombre de la familia como texto.
Por eso usa RandomForestClassifier y no XGBoost: sklearn acepta etiquetas de texto
y `.predict()` las devuelve tal cual, mientras XGBoost exige enteros.
=> No hay que cambiar NI UNA LÍNEA de la GUI.

Etiquetas: AMR Gene Family curada de CARD, agrupando las subfamilias que CARD separa
(OXA sin subfamilia + OXA-213-like + OXA-134-like -> una sola familia OXA). La tabla
de agrupación está abajo, explícita y editable, para que la asesora la revise.

Uso:
    python modelo_familias.py
    python modelo_familias.py --min-clase 30
    python modelo_familias.py --sin-otras        # excluye en vez de agrupar las raras
    python modelo_familias.py --solo-censo       # mide y no entrena
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--min-clase", type=int, default=50)
ap.add_argument("--sin-otras", action="store_true",
                help="excluye las familias raras en vez de juntarlas en 'Otras familias'")
ap.add_argument("--solo-censo", action="store_true", help="mide el agrupamiento y no entrena")
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--ventana", type=int, default=500)
ap.add_argument("--arboles", type=int, default=400)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

K, VENTANA, SEED = args.k, args.ventana, args.seed

# ===========================================================================
# TABLA DE AGRUPACIÓN  --  REVISAR CON LA ASESORA
# Se aplica EN ORDEN: gana la primera regla que casa. Por eso las reglas
# específicas (OXA) van antes del comodín general (beta-lactamase).
# El patrón es una expresión regular contra el nombre de familia de CARD.
# ===========================================================================
GRUPOS = [
    # --- betalactamasas: OXA primero, luego el resto ---
    (r"^OXA",                                        "OXA beta-lactamasa"),
    (r"^(IMP|VIM|NDM|KPC|GES|SPM|SIM|GIM)\b",        "carbapenemasa (IMP/VIM/NDM/KPC)"),
    (r"^(CTX-M|TEM|SHV|PER|VEB)\b",                  "beta-lactamasa de espectro extendido"),
    (r"^(ADC|ACT|CMY|FOX|MIR|DHA|MOX)\b",            "cefalosporinasa AmpC (ADC/ACT/CMY)"),
    (r"beta-lactamase",                              "otras beta-lactamasas"),

    # --- aminoglucósidos, por enzima ---
    (r"^AAC\(",                                      "AAC acetiltransferasa"),
    (r"^APH\(",                                      "APH fosfotransferasa"),
    (r"^ANT\(",                                       "ANT nucleotidiltransferasa"),
    (r"16S rRNA methyltransferase",                  "metiltransferasa 16S rRNA"),
    (r"aminoglycoside",                              "otros aminoglucósidos"),

    # --- glucopéptidos (van) ---
    (r"^(van[A-Z]|Van ligase)",                      "van / glucopéptidos"),
    (r"glycopeptide resistance gene cluster",        "van / glucopéptidos"),

    # --- bombas de eflujo, cada superfamilia aparte ---
    (r"resistance-nodulation-cell division \(RND\)", "eflujo RND"),
    (r"major facilitator superfamily \(MFS\)",       "eflujo MFS"),
    (r"ATP-binding cassette \(ABC\)",                "eflujo ABC"),
    (r"small multidrug resistance \(SMR\)",          "eflujo SMR"),
    (r"multidrug and toxic compound extrusion",      "eflujo MATE"),
    (r"efflux regulatory protein",                   "regulador de eflujo"),
    (r"efflux pump",                                 "otras bombas de eflujo"),

    # --- MLS ---
    (r"^Erm|23S rRNA methyltransferase",             "Erm metiltransferasa 23S"),
    (r"macrolide phosphotransferase",                "MPH fosfotransferasa de macrólidos"),
    (r"lincosamide nucleotidyltransferase",          "LNU nucleotidiltransferasa"),

    # --- tetraciclinas ---
    (r"ABC-F",                                       "ABC-F protección ribosomal"),
    (r"ribosomal protection protein",                "protección ribosomal (tet)"),
    (r"tetracycline inactivation",                   "inactivación de tetraciclina"),

    # --- otros con nombre propio ---
    (r"dihydrofolate reductase",                     "dfr dihidrofolato reductasa"),
    (r"chloramphenicol acetyltransferase",           "CAT cloranfenicol acetiltransferasa"),
    (r"fosfomycin thiol transferase",                "fosfomicina tiol transferasa"),
    (r"quinolone resistance protein",                "qnr quinolonas"),
    (r"phosphoethanolamine transferase",             "MCR fosfoetanolamina transferasa"),
    (r"rifampin|ADP-ribosyltransferase",             "rifampicina (arr)"),
    (r"sulfonamide",                                 "sulfonamidas (sul)"),
]


def agrupar(nombre):
    for patron, grupo in GRUPOS:
        if re.search(patron, nombre, flags=re.I):
            return grupo
    return None


# --------------------------------------------------------------------- rutas
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

TSV = RAIZ / "results" / "familias" / "familias_card.tsv"
if not TSV.exists():
    sys.exit(f"Falta {TSV}\nCorre primero: python scripts/familias_card.py")
FASTA = RAIZ / "data" / "processed" / "argdb_POSITIVO_final.fasta"
if not FASTA.exists():
    sys.exit(f"Falta {FASTA}")

print("=" * 92)
print("ETAPA 2 -- MODELO DE FAMILIAS")
print("=" * 92)

# --------------------------------------------------------------------- etiquetas
regs = []
with open(TSV, encoding="utf-8") as fh:
    for f in csv.DictReader(fh, delimiter="\t"):
        if f["base"].upper() != "CARD" or not f["familia"]:
            continue
        crudas = [x.strip() for x in f["familia"].split(";") if x.strip()]
        regs.append(dict(gen=f["gen"], cruda=crudas[0], n_crudas=len(crudas)))

multi = sum(1 for r in regs if r["n_crudas"] > 1)
print(f"\nGenes de CARD con familia: {len(regs):,}")
print(f"  {multi} tienen varias familias (';'): se toma la primera")

cnt_cruda = Counter(r["cruda"] for r in regs)
for r in regs:
    r["grupo"] = agrupar(r["cruda"])

sin_regla = Counter(r["cruda"] for r in regs if not r["grupo"])
cnt_grupo = Counter(r["grupo"] for r in regs if r["grupo"])

print(f"\nAgrupación: {len(cnt_cruda)} familias de CARD -> {len(cnt_grupo)} grupos")
print(f"  genes agrupados: {sum(cnt_grupo.values()):,}")
print(f"  sin regla: {sum(sin_regla.values()):,} genes en {len(sin_regla)} familias de CARD")

print(f"\n{'='*92}\nCENSO DESPUÉS DE AGRUPAR\n{'='*92}")
print(f"  {'grupo':<42} {'n':>6} {'%':>7}   70/15/15")
usables = []
for g, n in cnt_grupo.most_common():
    tr, va = round(.70 * n), round(.15 * n)
    te = n - tr - va
    ok = n >= args.min_clase
    if ok:
        usables.append(g)
    print(f"  {g[:42]:<42} {n:>6,} {100*n/len(regs):>6.2f}%   {tr}/{va}/{te}"
          + ("" if ok else "  <- bajo el mínimo"))

print(f"\n  grupos con >= {args.min_clase}: {len(usables)}  "
      f"({sum(cnt_grupo[g] for g in usables):,} genes)")
if sin_regla:
    print(f"\n  Familias de CARD sin regla de agrupación (las 12 mayores):")
    for nombre, n in sin_regla.most_common(12):
        print(f"    {n:>4}  {nombre[:76]}")
    print("    -> si alguna merece su propio grupo, agrégala a la tabla GRUPOS.")

# --------------------------------------------------------------------- etiqueta final
OTRAS = "Otras familias"
for r in regs:
    g = r["grupo"]
    r["etiqueta"] = g if (g in usables) else (None if args.sin_otras else OTRAS)

usados = [r for r in regs if r["etiqueta"]]
clases = Counter(r["etiqueta"] for r in usados)
print(f"\n{'='*92}\nETIQUETAS FINALES\n{'='*92}")
for c, n in clases.most_common():
    print(f"  {c[:52]:<52} {n:>6,}")
print(f"\n  {len(clases)} clases | {len(usados):,} genes "
      f"| descartados: {len(regs)-len(usados):,}")
if len(clases) < 2:
    sys.exit("\nCon menos de 2 clases no hay nada que entrenar. Baja --min-clase.")
if not args.sin_otras and clases.get(OTRAS, 0) > 0.5 * len(usados):
    print(f"\n  AVISO: '{OTRAS}' es más de la mitad de los datos. Es una clase")
    print("  heterogénea y va a atraer predicciones. Considera --min-clase más bajo.")

SAL = RAIZ / "results" / "familias"
with open(SAL / "etiquetas_finales.tsv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, delimiter="\t",
                       fieldnames=["gen", "cruda", "grupo", "etiqueta", "n_crudas"])
    w.writeheader()
    w.writerows(regs)
print(f"\nGuardado: {SAL/'etiquetas_finales.tsv'}")

if args.solo_censo:
    print("\n--solo-censo: no se entrena. Revisa el censo y vuelve a correr sin la bandera.")
    sys.exit(0)

# --------------------------------------------------------------------- k-meros
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


def fragmentar(seq, ventana=VENTANA):
    # misma fragmentación CORREGIDA de la etapa 1: el último bloque se corre
    # hacia atrás en vez de rellenarse con ceros.
    seq = seq.upper()
    L = len(seq)
    if L < ventana:
        return [seq + "0" * (ventana - L)]
    n = L // ventana
    fr = [seq[i * ventana:(i + 1) * ventana] for i in range(n)]
    if L % ventana:
        fr.append(seq[-ventana:])
    return fr


# --------------------------------------------------------------------- dataset
fa, h, s = {}, None, []
with open(FASTA, encoding="utf-8", errors="replace") as fh:
    for linea in fh:
        linea = linea.rstrip("\n")
        if linea.startswith(">"):
            if h:
                fa[h] = "".join(s)
            h, s = linea[1:].split()[0], []
        else:
            s.append(linea.strip())
if h:
    fa[h] = "".join(s)

print(f"\n{'='*92}\nFRAGMENTANDO Y CALCULANDO K-MEROS (K={K}, ventana={VENTANA})\n{'='*92}")
t0 = time.time()
Xs, ys, gs = [], [], []
sin_sec = 0
for r in usados:
    seq = fa.get(r["gen"])
    if not seq:
        sin_sec += 1
        continue
    for fr in fragmentar(seq):
        Xs.append(vector_kmer(fr))
        ys.append(r["etiqueta"])
        gs.append(r["gen"])
X = np.array(Xs, dtype=np.float32)
y = np.array(ys)
g = np.array(gs)
print(f"  {len(usados)-sin_sec:,} genes -> {len(X):,} fragmentos | "
      f"X {X.shape} {X.nbytes/1e6:.0f} MB | {time.time()-t0:.0f}s")
if sin_sec:
    print(f"  {sin_sec} genes sin secuencia en el FASTA (omitidos)")

# --------------------------------------------------------------------- partición POR GEN
# Un gen y todos sus fragmentos van al mismo lado. Igual que en la etapa 1.
# n_splits no puede pasar del número de GENES de la clase más pequeña,
# porque StratifiedGroupKFold necesita al menos un gen de cada clase por pliegue.
genes_por_clase = Counter({gg: yy for gg, yy in zip(g, y)}.values())
n_splits = max(2, min(5, min(genes_por_clase.values())))
print(f"\nGenes por clase (mínimo {min(genes_por_clase.values())}): "
      f"{dict(genes_por_clase.most_common())}")
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
tr, te = next(sgkf.split(X, y, g))
assert not set(g[tr]) & set(g[te]), "FUGA: un gen quedó en train y test"
print(f"\nPartición por gen ({n_splits} pliegues, se usa 1 como test):")
print(f"  train {len(tr):,} fragmentos ({len(set(g[tr])):,} genes)")
print(f"  test  {len(te):,} fragmentos ({len(set(g[te])):,} genes)")

# --------------------------------------------------------------------- entrenar
print(f"\nEntrenando RandomForest multiclase ({args.arboles} árboles, "
      f"class_weight='balanced')...")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=args.arboles, class_weight="balanced",
                            n_jobs=-1, random_state=SEED).fit(X[tr], y[tr])
print(f"  listo en {time.time()-t0:.0f}s")

# --------------------------------------------------------------------- evaluar
def evaluar(nombre, Xe, ye, ge=None):
    pred = rf.predict(Xe)
    bal = balanced_accuracy_score(ye, pred)
    f1m = f1_score(ye, pred, average="macro", zero_division=0)
    print(f"\n--- {nombre} ---")
    print(f"  balanced accuracy: {bal:.3f}   F1 macro: {f1m:.3f}   "
          f"exactitud: {(pred==ye).mean():.3f}")
    print(classification_report(ye, pred, zero_division=0, digits=3))
    return bal, f1m


print(f"\n{'='*92}\nRESULTADOS\n{'='*92}")
evaluar("TEST, nivel fragmento", X[te], y[te])

# nivel gen: voto mayoritario entre los fragmentos (NO 'algún fragmento':
# un gen tiene UNA familia, así que la regla de la etapa 1 no aplica aquí)
proba = rf.predict_proba(X[te])
clases_rf = rf.classes_
por_gen = defaultdict(list)
for i, gg in enumerate(g[te]):
    por_gen[gg].append(i)
gy, gp = [], []
for gg, idxs in por_gen.items():
    gy.append(y[te][idxs[0]])
    gp.append(clases_rf[int(np.argmax(proba[idxs].mean(axis=0)))])
gy, gp = np.array(gy), np.array(gp)
print("\n--- TEST, nivel gen (promedio de probabilidades entre fragmentos) ---")
print(f"  balanced accuracy: {balanced_accuracy_score(gy, gp):.3f}   "
      f"F1 macro: {f1_score(gy, gp, average='macro', zero_division=0):.3f}   "
      f"exactitud: {(gp==gy).mean():.3f}")
print(classification_report(gy, gp, zero_division=0, digits=3))

print("Matriz de confusión a nivel gen (filas = real, columnas = predicho):")
etiquetas = sorted(set(gy) | set(gp))
cm = confusion_matrix(gy, gp, labels=etiquetas)
anchos = max(len(e) for e in etiquetas)
print("   " + " ".join(f"{i:>4}" for i in range(len(etiquetas))))
for i, (e, fila) in enumerate(zip(etiquetas, cm)):
    print(f"  {i:>2} " + " ".join(f"{v:>4}" for v in fila) + f"   {e[:anchos]}")

# --------------------------------------------------------------------- guardar
APP = RAIZ / "app"
APP.mkdir(exist_ok=True)
destino = APP / "modelo_familias.joblib"

# IMPORTANTE: estimador SUELTO, no dict. Es lo que espera la GUI:
#   modelo_etapa2.predict(X_frags[i:i+1])[0]
rf.set_params(n_jobs=1)          # la GUI corre en un hilo; evita sobre-suscripción
joblib.dump(rf, destino, compress=3)
print(f"\n{'='*92}")
print(f"Guardado para la GUI: {destino}  ({destino.stat().st_size/1e6:.0f} MB)")

# copia con metadatos, para el expediente (la GUI NO lee esta)
meta = {
    "algoritmo": "RandomForestClassifier", "n_estimators": args.arboles,
    "K": K, "ventana": VENTANA, "class_weight": "balanced", "semilla": SEED,
    "regla_gen": "promedio de probabilidades entre fragmentos",
    "etiqueta": "AMR Gene Family de CARD, agrupada (ver tabla GRUPOS)",
    "universo": "solo CARD (único con ontología curada)",
    "min_clase": args.min_clase, "clases": list(rf.classes_),
    "n_genes": len(set(g)), "n_fragmentos": int(len(X)),
    "test_gen_balanced_acc": float(balanced_accuracy_score(gy, gp)),
    "test_gen_f1_macro": float(f1_score(gy, gp, average="macro", zero_division=0)),
}
joblib.dump({"modelo": rf, **meta}, RAIZ / "data" / "models" /
            f"argus_familias_k{K}_card.joblib", compress=3)
with open(SAL / "modelo_familias_meta.json", "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2, ensure_ascii=False)

# verificación del contrato con la GUI
chk = joblib.load(destino)
prueba = chk.predict(X[te][:3])
assert all(isinstance(p, str) for p in prueba), "la GUI espera etiquetas de texto"
assert chk.n_features_in_ == 4 ** K, "la GUI deduce K de n_features_in_"
print(f"Contrato con la GUI verificado: .predict() devuelve texto "
      f"({prueba[0][:30]}...) y n_features_in_ = {chk.n_features_in_}")
print("\nNo hace falta cambiar nada en la GUI: ya busca app/modelo_familias.joblib.")
print("=" * 92)
