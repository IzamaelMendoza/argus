"""
===============================================================================
 familias_dnabert.py -- ETAPA 2 de ARGus: la FAMILIA del gen, con DNABERT-6
===============================================================================

QUE HACE ESTE SCRIPT, EN UNA FRASE
----------------------------------
Toma los genes que YA se sabe que son resistentes (el set positivo) y entrena un
clasificador que dice a que familia de resistencia pertenece cada uno. Es la
segunda etapa: la primera (el Random Forest de k-meros) solo dice si/no.

COMO LO HACE
------------
DNABERT-6 convierte cada fragmento de ADN en un vector de 768 numeros. Ese
cuerpo NO se entrena: se usa congelado. Lo unico que se entrena es una cabeza
MLP pequena (768 -> 256 -> n_familias, unos 200 mil parametros) encima de esos
vectores.

  * Por que congelado: el fine-tuning completo son ~90 millones de parametros
    sobre 1,011 genes. En CPU tardaria horas y se sobreajustaria. Con GPU en
    Colab si conviene probarlo (es donde corrio el codigo de la asesora).

DNABERT-6 es `zhihan1996/DNA_bert_6`. Confirmado de su config: 12 capas
transformer, 12 cabezas de atencion, hidden 768, max_position_embeddings 512,
vocabulario 4101 (los 4096 6-meros + 5 tokens especiales).

  Un fragmento de 500 pb -> 495 6-meros + [CLS] + [SEP] = 497 tokens. Entra en
  los 512. Un gen completo de 2000 pb daria 1995 y NO entra: de ahi que haya
  que fragmentar.
-------------------------------------------------------------------------------
Son cuatro decisiones. Estan anotadas tal como se leyeron de su codigo:

  1. `sequence_to_kmers`: la secuencia se convierte en k-meros separados por
     espacio antes de tokenizar. Es su funcion.

  2. El checkpoint va por BALANCED ACCURACY de validacion, NO por val_loss.
     Ella lo cambio a proposito y dejo el comentario: con clases desbalanceadas
     el val_loss premia a un modelo que solo acierta las clases mayoritarias.

  3. NO se usa `trust_remote_code`. DNA_bert_6 es un BERT estandar y las
     versiones actuales de transformers lo cargan nativo; con esa bandera
     chocan las clases BertConfig y truena con "config_class not consistent".

  4. Pesos de clase balanceados en la funcion de perdida.

Y de los pasos que apunto en la reunion:
  - k = 6                 -> argumento --k, default 6
  - 512 tokens            -> argumento --max-len, default 512
  - 70 / 15 / 15          -> seccion 4 (cortes), y el corte es POR GEN
  - "checar cuantas familias hay antes de entrenar"  -> seccion 3
  - "checar el numero de transformer"                -> seccion 5, se imprime


-------------------------------------------------------------------------------
 LO QUE CAMBIO EL 6-NOV                  -> marcado abajo con  # [CAMBIO 6-nov]
-------------------------------------------------------------------------------
La fragmentacion. Antes este script usaba la "opcion C": cuando el gen no era
multiplo de 500, el ultimo bloque se corria hacia atras (`seq[-500:]`). Eso es
justo lo que la asesora rechazo el 5-nov, y con razon: en un gen de 650 pb los
fragmentos serian [0:500] y [150:650], o sea 350 bases se cuentan DOS VECES y
esa region pesa doble.

Ahora usa su regla:
  * bloques de 500 pb;
  * el resto final se conserva con su largo real, SIN rellenar, si tiene >= 150 pb;
  * si tiene menos de 150 pb se descarta;
  * nunca se usa padding -> los fragmentos miden entre 150 y 500 pb.

Por que importaba tambien aqui, y no solo en la etapa 1: la GUI ya fragmenta con
esta regla, asi que le estaba pasando a DNABERT fragmentos de largo variable
mientras la cabeza se habia entrenado con fragmentos todos de 500 pb y con
solapamiento. Es la misma desalineacion entrenamiento/inferencia que se arreglo
en el modelo binario.

OJO AL CORRER: la cache de embeddings se identifica por un md5 de las
secuencias, asi que al cambiar la fragmentacion la cache vieja queda invalida y
los embeddings se recalculan desde cero (es la parte lenta, ~2.3 fragmentos/s en
CPU). La carpeta vieja se puede borrar a mano:
    rm -rf data/models/dnabert6_emb_familias

COMPARABILIDAD: usa las MISMAS etiquetas, la MISMA fragmentacion, el MISMO corte
por gen y la MISMA semilla que `modelo_familias.py` (el RandomForest). Para que
la comparacion siga siendo valida, hay que correr los dos con la regla nueva.

USO
---
    python familias_dnabert.py                  # prueba rapida primero
    python familias_dnabert.py --completo       # la corrida real
===============================================================================
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# =============================================================================
#  SECCION 1: argumentos
# =============================================================================
# Todo lo configurable esta aqui, nada esta escrito a mano mas abajo. Los
ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None,
                help="raiz del proyecto; si no se da, se detecta subiendo carpetas")
ap.add_argument("--prueba", action="store_true",
                help="prueba rapida con muestra reducida (sin esto, corrida completa)")
ap.add_argument("--modelo", default="zhihan1996/DNA_bert_6",
                help="el DNABERT original con tokenizacion de 6-meros")
ap.add_argument("--k", type=int, default=6)              # k = 6
ap.add_argument("--ventana", type=int, default=500)      # tamano del bloque
ap.add_argument("--min-pb", type=int, default=150)       # [CAMBIO 6-nov] minimo de bases reales
ap.add_argument("--max-len", type=int, default=512)      # 512 tokens
ap.add_argument("--batch", type=int, default=16, help="lote para codificar")
ap.add_argument("--chunk", type=int, default=256, help="bloque que se guarda en disco")
ap.add_argument("--epocas", type=int, default=80)
ap.add_argument("--paciencia", type=int, default=12, help="epocas sin mejorar antes de parar")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

K, VENTANA, MIN_PB, SEED = args.k, args.ventana, args.min_pb, args.seed
PRUEBA = args.prueba

# =============================================================================
#  SECCION 2: rutas
# =============================================================================
# La raiz se detecta subiendo carpetas hasta encontrar data/, asi el script corre
# igual desde scripts/ que desde la raiz.
if args.raiz:
    RAIZ = Path(args.raiz).expanduser()
else:
    RAIZ = None
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "data").is_dir():
            RAIZ = cand
            break
    if RAIZ is None:
        sys.exit('No encontre el proyecto. Usa --raiz "ruta\\al\\Proyecto_Residencias"')

# etiquetas_finales.tsv lo produce modelo_familias.py: es el gen -> familia que
# salio de la ontologia curada de CARD (columna "AMR Gene Family" del aro_index)
ETQ = RAIZ / "results" / "familias" / "etiquetas_finales.tsv"
FASTA = RAIZ / "data" / "processed" / "argdb_POSITIVO_final.fasta"
for p, q in [(ETQ, "etiquetas_finales.tsv (corre antes modelo_familias.py)"),
             (FASTA, "argdb_POSITIVO_final.fasta")]:
    if not p.exists():
        sys.exit(f"Falta {q}: {p}")

# los embeddings se cachean en disco porque calcularlos es lo que tarda
CACHE = RAIZ / "data" / "models" / "dnabert6_emb_familias"
CACHE.mkdir(parents=True, exist_ok=True)
SAL = RAIZ / "results" / "familias"

print("=" * 92)
print("ETAPA 2 con DNABERT-6  (familia del gen)")
print(f"modelo: {args.modelo} | K={K} | ventana={VENTANA} | min_pb={MIN_PB} | "
      f"max_len={args.max_len}")
if PRUEBA:
    print("MODO PRUEBA RAPIDA: muestra reducida. Usa --completo para la corrida real.")
print("=" * 92)

# ----------------------------------------------------------------- dependencias
try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("Falta PyTorch.  pip install torch --index-url https://download.pytorch.org/whl/cpu")
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    sys.exit("Falta transformers.  pip install 'transformers==4.30.2'")
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedGroupKFold

# semillas fijas: sin esto dos corridas dan numeros distintos y no se puede
# comparar nada
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
print(f"torch {torch.__version__} | hilos {torch.get_num_threads()} | CPU")

# =============================================================================
#  SECCION 3: datos  ("recibe mis archivos / leer las lecturas" de su lista)
# =============================================================================
# 3a. las etiquetas: gen -> familia
etq = {}
with open(ETQ, encoding="utf-8") as fh:
    for f in csv.DictReader(fh, delimiter="\t"):
        if f.get("etiqueta"):
            etq[f["gen"]] = f["etiqueta"]

# 3b. las secuencias: se lee el FASTA a mano (sin biopython) para no depender de
#     nada mas. El id del gen es la primera palabra del encabezado.
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


# 3c. fragmentacion
def fragmentar(seq, ventana=VENTANA, min_pb=MIN_PB):
    """
    [CAMBIO 6-nov] Regla de la reuncion (reunion del 5-nov).

    Bloques de `ventana` pb. El resto final se conserva con su largo REAL, sin
    rellenar, si alcanza `min_pb` bases; si no, se descarta. Nunca mete ceros,
    nunca solapa. Los fragmentos quedan entre min_pb y ventana pb.

    Un gen mas corto que min_pb devuelve [] -> ese gen no se puede usar.

    ANTES (opcion C, la que ella rechazo):
        if L % ventana: fr.append(seq[-ventana:])
    Eso hacia que en un gen de 650 pb los fragmentos fueran [0:500] y [150:650]:
    350 bases contadas dos veces.
    """
    seq = seq.upper()
    L = len(seq)
    if L < ventana:
        return [seq] if L >= min_pb else []
    n = L // ventana
    fr = [seq[i * ventana:(i + 1) * ventana] for i in range(n)]
    r = L - n * ventana
    if r >= min_pb:
        fr.append(seq[n * ventana:])          # tal cual, sin rellenar
    return fr


def sequence_to_kmers(seq, k=K):
    """
    [Original] Su funcion. La secuencia se parte en k-meros separados por espacio,
    que es lo que el tokenizador de DNABERT espera: cada 6-mero es un token del
    vocabulario de 4096.

    El `.replace("0", "")` queda por compatibilidad con fragmentos viejos que
    traian relleno; con la regla nueva ya no llega ninguno con ceros.
    Lo que no sea ACGT se vuelve N (el tokenizador lo manda a [UNK]).
    """
    seq = seq.upper().replace("0", "")
    seq = re.sub(r"[^ACGT]", "N", seq)
    if len(seq) < k:
        return "N" * k
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


# 3d. armar la lista de fragmentos, cada uno con su familia y su gen de origen.
#     El gen se guarda (en `gs`) porque los cortes de la seccion 4 se hacen POR
#     GEN: si dos fragmentos del mismo gen caen en train y test, hay fuga.
seqs, ys, gs = [], [], []
sin_frag = []                                    # [CAMBIO 6-nov]
for gen, lab in etq.items():
    sq = fa.get(gen)
    if not sq:
        continue
    fr = fragmentar(sq)
    if not fr:                                   # [CAMBIO 6-nov] gen < MIN_PB pb
        sin_frag.append((gen, lab, len(sq)))
        continue
    for f in fr:
        seqs.append(f)
        ys.append(lab)
        gs.append(gen)
y = np.array(ys)
g = np.array(gs)
clases = sorted(set(y))
y_idx = np.array([clases.index(v) for v in y])

# [CAMBIO 6-nov] reportar los genes que se quedaron fuera, y de que familias.
# Si una familia pierde demasiados genes se nota aqui, no a la mitad del
# entrenamiento.
if sin_frag:
    print(f"\n[aviso] {len(sin_frag)} gen(es) mas cortos que {MIN_PB} pb: no producen "
          f"ningun fragmento y quedan fuera")
    for fam, n in Counter(f for _, f, _ in sin_frag).most_common():
        print(f"           {n:>3}  {fam}")

print(f"\n{len(set(g)):,} genes -> {len(seqs):,} fragmentos | {len(clases)} clases")

# 3e. [Original] "checar cuantas familias tenemos, cuantos ejemplos hay de cada
#     una, cuales son los porcentajes" -- esto es ese paso, hecho ANTES de
#     entrenar como ella pidio. Se cuenta por GEN, no por fragmento: un gen
#     largo da mas fragmentos y inflaria su familia.
genes_por_fam = Counter({gg: yy for gg, yy in zip(g, y)}.values())
total_genes = sum(genes_por_fam.values())
print(f"\n{'familia':<42} {'genes':>7} {'%':>7} {'frags':>8}")
frags_por_fam = Counter(y.tolist())
for fam, n in genes_por_fam.most_common():
    print(f"{fam[:42]:<42} {n:>7} {100*n/total_genes:>6.1f}% {frags_por_fam[fam]:>8}")
flaca = min(genes_por_fam.values())
print(f"\nla familia mas chica tiene {flaca} genes "
      f"(eso limita en cuantos pliegues se puede cortar)")
if flaca < 3:
    print("  [aviso] con menos de 3 genes la familia no puede quedar en train, val y test")

# =============================================================================
#  SECCION 4: cortes train / val / test   ([ORIGINAL] "70 entrenar, 15 validar, 15 probar")
# =============================================================================
# El corte es POR GEN con StratifiedGroupKFold: ningun gen aparece en dos lados.
# Es lo mismo que se hizo en la etapa 1, y por el mismo motivo: dos fragmentos
# del mismo gen son casi la misma secuencia, asi que si uno esta en train y otro
# en test el modelo no esta siendo evaluado, se esta autoevaluando.
#
# Nota sobre los porcentajes: no son exactamente 70/15/15. El numero de pliegues
# lo limita la familia mas chica (no se puede cortar en 5 si una familia tiene 3
# genes), asi que se usa el maximo posible hasta 5 -> test ~20% con 5 pliegues.
# Con la lista de familias de arriba se ve cual es la que manda.
#
# Se usa el mismo corte y la misma semilla que modelo_familias.py para que la
# comparacion contra el RandomForest sea directa.
n_splits = max(2, min(5, min(genes_por_fam.values())))
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
tr_full, te = next(sgkf.split(np.zeros(len(y)), y, g))

# val se saca del train, tambien por gen
n_val = max(2, min(5, min(Counter({gg: yy for gg, yy in zip(g[tr_full], y[tr_full])}
                                  .values()).values())))
sgkf2 = StratifiedGroupKFold(n_splits=n_val, shuffle=True, random_state=SEED + 7)
rel_tr, rel_va = next(sgkf2.split(np.zeros(len(tr_full)), y[tr_full], g[tr_full]))
tr, va = tr_full[rel_tr], tr_full[rel_va]

# los assert son la red de seguridad: si alguna vez se rompe el agrupamiento,
# el script se detiene en lugar de dar un numero bonito y falso
for a, b, nom in [(tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")]:
    assert not set(g[a]) & set(g[b]), f"FUGA entre {nom}"
print(f"\ntrain {len(tr):,} frag ({len(set(g[tr])):,} genes) | "
      f"val {len(va):,} ({len(set(g[va])):,}) | test {len(te):,} ({len(set(g[te])):,})")
print(f"  pliegues: {n_splits} para test, {n_val} para val "
      f"(los limita la familia mas chica)")

if PRUEBA:
    # la prueba rapida solo sirve para ver que el script corre de punta a punta
    rng = np.random.default_rng(SEED)
    tr = rng.choice(tr, size=min(200, len(tr)), replace=False)
    va = rng.choice(va, size=min(80, len(va)), replace=False)
    te = rng.choice(te, size=min(120, len(te)), replace=False)
    print(f"  (prueba rapida: train {len(tr)} | val {len(va)} | test {len(te)})")

# =============================================================================
#  SECCION 5: cargar DNABERT-6   ([ORIGINAL] "cargar el modelo, checar el numero
#                                  de transformer, verificar que k=6")
# =============================================================================
print(f"\nCargando {args.modelo} ...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(args.modelo)
# [ORIGINAL] sin trust_remote_code: con esa bandera chocan las clases BertConfig
modelo = AutoModel.from_pretrained(args.modelo).eval()
cfg = modelo.config
# esta linea es la que responde "checar el numero de transformer": imprime 12
print(f"  listo en {time.time()-t0:.0f}s | capas {cfg.num_hidden_layers} | "
      f"cabezas {cfg.num_attention_heads} | hidden {cfg.hidden_size} | "
      f"vocab {cfg.vocab_size} | max_pos {cfg.max_position_embeddings}")
DIM = int(cfg.hidden_size)
# el vocab de 4101 = 4096 6-meros + 5 especiales es la confirmacion de que el
# modelo es de k=6 y no de k=3 o k=4
assert cfg.vocab_size >= 4 ** K, (
    f"el modelo tiene vocab {cfg.vocab_size}, muy chico para k={K}: "
    f"revisa que sea DNA_bert_{K}")

# .eval() y inference_mode: el cuerpo esta CONGELADO, no se le calculan
# gradientes ni se actualiza nada. Es la decision central del diseno.


@torch.inference_mode()
def codificar(lista):
    """
    Convierte fragmentos de ADN en vectores de 768 numeros.

    Mean pooling enmascarado: se promedian los estados de la ultima capa sobre
    los tokens REALES del fragmento. La mascara importa porque el tokenizador
    rellena el lote hasta el fragmento mas largo; sin enmascarar, el relleno
    arrastraria el promedio hacia abajo y un fragmento de 150 pb saldria
    distorsionado respecto a uno de 500.
    """
    out = np.empty((len(lista), DIM), dtype=np.float32)
    for i in range(0, len(lista), args.batch):
        lote = [sequence_to_kmers(s) for s in lista[i:i + args.batch]]
        enc = tok(lote, padding=True, truncation=True, max_length=args.max_len,
                  return_tensors="pt")
        hs = modelo(**enc).last_hidden_state                 # (B, T, 768)
        m = enc["attention_mask"].unsqueeze(-1).float()      # 1 = token real
        out[i:i + len(lote)] = ((hs * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy()
    return out


def embeddings(idxs, nombre):
    """
    Codifica por bloques y los guarda en disco. Si la corrida se corta, al
    volver a correr reanuda donde iba en lugar de empezar de cero.

    La firma md5 es de las secuencias: si cambia la fragmentacion, cambia la
    firma y los embeddings se recalculan solos. Por eso el cambio del 6-nov
    obliga a recalcular todo (la carpeta vieja se puede borrar a mano).
    """
    firma = hashlib.md5(("|".join(seqs[i] for i in idxs)).encode()).hexdigest()[:10]
    d = CACHE / f"{nombre}_{len(idxs)}_{firma}"
    d.mkdir(exist_ok=True)
    partes, t0 = [], time.time()
    for b0 in range(0, len(idxs), args.chunk):
        f = d / f"{b0:07d}.npy"
        if f.exists():
            partes.append(np.load(f))
            continue
        e = codificar([seqs[i] for i in idxs[b0:b0 + args.chunk]])
        np.save(f, e)
        partes.append(e)
        hechos = min(b0 + args.chunk, len(idxs))
        vel = hechos / max(1e-9, time.time() - t0)
        print(f"\r  {nombre}: {hechos}/{len(idxs)}  ({vel:.1f} frag/s, "
              f"faltan ~{(len(idxs)-hechos)/max(vel,1e-9)/60:.1f} min)   ", end="", flush=True)
    print()
    return np.vstack(partes)


print("\nCalculando embeddings (es la parte lenta; se cachea en disco):")
E_tr, E_va, E_te = (embeddings(tr, "train"), embeddings(va, "val"), embeddings(te, "test"))
print(f"  train {E_tr.shape} | val {E_va.shape} | test {E_te.shape}")

# =============================================================================
#  SECCION 6: la cabeza MLP -- lo UNICO que se entrena
# =============================================================================
# Normalizacion: media y desviacion se calculan SOLO del train y se aplican a
# val y test. Calcularlas con todo seria fuga (el test influiria en como se
# escala el train). Estos dos vectores se guardan al final porque la GUI los
# necesita para normalizar igual.
mu, sd = E_tr.mean(0, keepdims=True), E_tr.std(0, keepdims=True) + 1e-6
Xtr = torch.tensor((E_tr - mu) / sd)
Xva = torch.tensor((E_va - mu) / sd)
Xte = torch.tensor((E_te - mu) / sd)
ytr = torch.tensor(y_idx[tr], dtype=torch.long)
yva = torch.tensor(y_idx[va], dtype=torch.long)

# [ORIGINAL] Pesos de clase balanceados: cada familia pesa en la perdida al
# inverso de cuantos ejemplos tiene. Sin esto el modelo aprende a contestar
# siempre la familia mas comun y acierta "bien" en exactitud pero no sirve.
cuenta = np.bincount(y_idx[tr], minlength=len(clases)).astype(np.float64)
pesos = torch.tensor(np.where(cuenta > 0, cuenta.sum() / (len(clases) * np.maximum(cuenta, 1)), 0.0),
                     dtype=torch.float32)
print(f"\nPesos de clase (balanced, como en el codigo de original): "
      f"min {pesos[pesos>0].min():.2f} max {pesos.max():.2f}")

# La cabeza: 768 -> 256 -> ReLU -> Dropout(0.3) -> n_clases.
# Son ~200 mil parametros contra los ~90 millones congelados de DNABERT.
# El Dropout es lo que evita que se memorice el train, que con 1,011 genes es
# un riesgo real.
cabeza = nn.Sequential(nn.Linear(DIM, 256), nn.ReLU(), nn.Dropout(0.3),
                       nn.Linear(256, len(clases)))
opt = torch.optim.Adam(cabeza.parameters(), lr=args.lr, weight_decay=1e-4)
perdida = nn.CrossEntropyLoss(weight=pesos)

# En modo PRUEBA el checkpoint va a un archivo aparte: si no, una corrida rapida
# de 200 fragmentos pisa el modelo bueno que usa la GUI.
CKPT = RAIZ / "data" / "models" / (
    "dnabert6_familias_cabeza_PRUEBA.pt" if PRUEBA else "dnabert6_familias_cabeza.pt")
if PRUEBA:
    print(f"\n[prueba] el checkpoint va a {CKPT.name}, NO se toca el modelo de la GUI")
mejor, mejor_ep, sin_mejora, estado = -1.0, -1, 0, None
print(f"\nEntrenando la cabeza (checkpoint por BALANCED ACCURACY de val, "
      f"no por val_loss):")
for ep in range(args.epocas):
    cabeza.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(perm), 128):
        b = perm[i:i + 128]
        opt.zero_grad()
        l = perdida(cabeza(Xtr[b]), ytr[b])
        l.backward()
        opt.step()

    # --- validacion al final de cada epoca ---
    cabeza.eval()
    with torch.no_grad():
        pv = cabeza(Xva).argmax(1).numpy()

    # [ORIGINAL] ESTA es la decision que se cambio. El checkpoint se
    # guarda cuando mejora la BALANCED ACCURACY de validacion, no cuando baja el
    # val_loss. Con clases desbalanceadas el val_loss premia a un modelo que solo
    # acierta las familias mayoritarias: bajaria "bien" mientras el modelo se
    # vuelve inutil para las familias chicas, que son justo las que interesan.
    bal = balanced_accuracy_score(y_idx[va], pv)
    mejoro = bal > mejor
    if mejoro:
        mejor, mejor_ep, sin_mejora = bal, ep, 0
        estado = {k: v.clone() for k, v in cabeza.state_dict().items()}
        torch.save(estado, CKPT)        # se guarda el MEJOR, no el ultimo
    else:
        sin_mejora += 1
    if ep % 5 == 0 or mejoro:
        print(f"  epoca {ep+1:>3}  val balanced acc {bal:.4f}" + ("  *" if mejoro else ""))
    if sin_mejora >= args.paciencia:
        print(f"  early stopping en la epoca {ep+1} ({args.paciencia} sin mejorar)")
        break

# se recarga el mejor estado: si las ultimas epocas empeoraron, se descartan
if estado:
    cabeza.load_state_dict(estado)
cabeza.eval()
print(f"  mejor: epoca {mejor_ep+1}, val balanced accuracy {mejor:.4f}")

# =============================================================================
#  SECCION 7: evaluacion en el TEST
# =============================================================================
with torch.no_grad():
    prob_te = torch.softmax(cabeza(Xte), dim=1).numpy()
pred_te = prob_te.argmax(1)
real_te = y_idx[te]
nombres = [clases[i] for i in range(len(clases))]

print(f"\n{'='*92}\nRESULTADOS -- DNABERT\n{'='*92}")

# 7a. nivel fragmento. Util para diagnostico, pero NO es la metrica que importa:
#     lo que se entrega es la familia del GEN.
print("\n--- TEST, nivel fragmento ---")
print(f"  balanced accuracy: {balanced_accuracy_score(real_te, pred_te):.3f}   "
      f"F1 macro: {f1_score(real_te, pred_te, average='macro', zero_division=0):.3f}   "
      f"exactitud: {(pred_te==real_te).mean():.3f}")

# 7b. nivel gen: se PROMEDIAN las probabilidades de los fragmentos del gen.
#
#     Por que promedio y no "algun fragmento" como en la etapa 1: un gen tiene
#     UNA sola familia, asi que la pregunta es "en conjunto, a que se parece
#     mas", no "hay algun pedazo que parezca X". En la etapa 1 si era "alguno",
#     porque ahi basta un pedazo resistente para que el gen lo sea.
#
#     Esta es la misma regla que usa la GUI, y por eso debe ser la misma con la
#     que se mide aqui.
por_gen = defaultdict(list)
for i, gg in enumerate(g[te]):
    por_gen[gg].append(i)
gr, gp, gmax = [], [], []
for gg, idxs in por_gen.items():
    pr = prob_te[idxs].mean(axis=0)
    gr.append(real_te[idxs[0]])
    gp.append(int(np.argmax(pr)))
    gmax.append(float(pr.max()))
gr, gp = np.array(gr), np.array(gp)
bal_g = balanced_accuracy_score(gr, gp)
f1_g = f1_score(gr, gp, average="macro", zero_division=0)

# balanced accuracy y no exactitud: con familias desbalanceadas la exactitud se
# puede ver alta contestando siempre la mas comun. La balanced accuracy es el
# promedio del acierto POR FAMILIA, asi que una familia chica pesa igual que una
# grande.
print("\n--- TEST, nivel gen (promedio de probabilidades) ---")
print(f"  balanced accuracy: {bal_g:.3f}   F1 macro: {f1_g:.3f}   "
      f"exactitud: {(gp==gr).mean():.3f}")
print(f"  genes evaluados: {len(gr)}")
pres = sorted(set(gr.tolist()) | set(gp.tolist()))
print(classification_report(gr, gp, labels=pres,
                            target_names=[nombres[i] for i in pres],
                            zero_division=0, digits=3))

# la matriz de confusion es donde se ve si alguna familia esta actuando de "iman"
# (se predice para todo). Fila = familia real, columna = familia predicha.
print("Matriz de confusion a nivel gen:")
cm = confusion_matrix(gr, gp, labels=pres)
print("   " + " ".join(f"{i:>4}" for i in range(len(pres))))
for i, (ci, fila) in enumerate(zip(pres, cm)):
    print(f"  {i:>2} " + " ".join(f"{v:>4}" for v in fila) + f"   {nombres[ci][:40]}")

# =============================================================================
#  SECCION 8: comparacion contra el RandomForest
# =============================================================================
# Sirve para justificar por que la GUI usa DNABERT y no el RF. Para que la
# comparacion valga, los dos deben haberse corrido con la MISMA fragmentacion.
print(f"\n{'='*92}\nCOMPARACION CONTRA EL RANDOM FOREST\n{'='*92}")
ref = SAL / "modelo_familias_meta.json"
if ref.exists():
    with open(ref, encoding="utf-8") as fh:
        m = json.load(fh)
    print(f"  {'modelo':<34} {'balanced acc':>13} {'F1 macro':>10}")
    print(f"  {'RandomForest k-meros k=6':<34} "
          f"{m.get('test_gen_balanced_acc', float('nan')):>13.3f} "
          f"{m.get('test_gen_f1_macro', float('nan')):>10.3f}")
    print(f"  {'DNABERT-6 congelado + MLP':<34} {bal_g:>13.3f} {f1_g:>10.3f}")
    d = bal_g - m.get("test_gen_balanced_acc", 0)
    gana = "DNABERT-6" if d > 0 else "RandomForest"
    print(f"\n  diferencia: {d:+.3f}  ->  gana {gana}")
    # [CAMBIO 6-nov] advertencia de comparabilidad
    if int(m.get("min_pb", -1)) != MIN_PB:
        print(f"\n  [AVISO] el RandomForest de referencia se entreno con "
              f"min_pb={m.get('min_pb', 'sin registrar')} y este con {MIN_PB}.")
        print("  La comparacion NO es directa. Corre modelo_familias.py con la")
        print("  fragmentacion nueva antes de reportar esta diferencia.")
    else:
        print("  Mismas etiquetas, misma fragmentacion, mismo corte por gen y misma")
        print("  semilla, asi que la comparacion es directa.")
    # con ~200 genes de test, una diferencia chica entra en el ruido
    if abs(d) < 0.05:
        print(f"\n  OJO: con {len(gr)} genes de test una diferencia de {abs(d):.3f}")
        print("  esta dentro del ruido. Para afirmar que uno gana habria que")
        print("  repetir con varias semillas.")
else:
    print(f"  No encontre {ref.name}: corre antes modelo_familias.py para comparar.")

if PRUEBA:
    print("\n" + "!" * 92)
    print("ESTO FUE LA PRUEBA RAPIDA: los numeros NO sirven. Corre con --completo.")
    print("!" * 92)
    sys.exit(0)

# =============================================================================
#  SECCION 9: guardar
# =============================================================================
# Tres archivos, y los tres los necesita app/familias_dnabert_gui.py:
#   dnabert6_familias_cabeza.pt     los pesos (ya se guardo en el bucle)
#   familias_dnabert_meta.json      las clases EN ORDEN + la configuracion
#   dnabert6_norm.npy               mu y sd del train
#
# El orden de `clases` es critico: la cabeza devuelve un indice, y ese indice se
# traduce a nombre con esta lista. Si se reordena, la GUI reporta familias
# equivocadas sin dar ningun error.
meta = {
    "modelo_base": args.modelo, "modo": "embeddings congelados + cabeza MLP",
    "capas_transformer": int(cfg.num_hidden_layers), "hidden": DIM,
    "K": K, "ventana": VENTANA, "max_len": args.max_len,
    # [CAMBIO 6-nov] queda registrada la regla de fragmentacion, para que la GUI
    # y el script del RandomForest puedan comprobar que coinciden
    "min_pb": MIN_PB,
    "relleno": "ninguno",
    "fragmentacion": (f"bloques de {VENTANA} pb; el resto final se conserva con su largo "
                      f"real sin rellenar si tiene >= {MIN_PB} pb, y se descarta si no"),
    "genes_descartados_cortos": len(sin_frag),
    "clases": clases, "semilla": SEED,
    "checkpoint_por": "balanced accuracy de validacion",
    "mejor_epoca": int(mejor_ep + 1), "val_balanced_acc": float(mejor),
    "test_gen_balanced_acc": float(bal_g), "test_gen_f1_macro": float(f1_g),
    "test_n_genes": int(len(gr)),
    "regla_gen": "promedio de probabilidades entre fragmentos",
    "n_genes": int(len(set(g))), "n_fragmentos": int(len(seqs)),
}
with open(SAL / "familias_dnabert_meta.json", "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2, ensure_ascii=False)
np.save(SAL / "dnabert6_norm.npy", np.vstack([mu, sd]))
print(f"\nGuardado:")
print(f"  {CKPT}")
print(f"  {SAL / 'familias_dnabert_meta.json'}  ({len(clases)} familias)")
print(f"  {SAL / 'dnabert6_norm.npy'}")
print("\nLa GUI (app/familias_dnabert_gui.py) lee estos tres y ya los usa:")
print("MODELO_FAMILIA = 'dnabert' en argus_gui.py. No hay que cambiar nada mas.")
