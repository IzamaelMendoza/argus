"""
familias_dnabert_gui.py -- predictor de FAMILIA con DNABERT-6 para la etapa 2 de ARGus.

Va en la misma carpeta que la GUI (app/) y se importa desde ahí. Encapsula todo lo
que DNABERT necesita para que `predecir_mag` quede con tres líneas de cambio.

Cómo funciona:
  - Carga perezosa: DNABERT se carga la PRIMERA vez que hace falta, no al abrir la
    GUI. Si el MAG no trae ningún gen resistente, nunca se carga.
  - Una sola carga por sesión: queda en memoria a nivel de módulo, así que varios
    análisis seguidos no vuelven a pagar el costo.
  - Varias versiones del modelo: el cuerpo de DNABERT es el MISMO para todas (es
    el preentrenado de Hugging Face), así que al cambiar de versión solo se
    recarga la cabeza, que pesa menos de 1 MB. Cambiar de 99 % a 99.5 % cuesta
    milisegundos, no los 400 MB de la primera carga.
  - Si algo falla (sin internet la primera vez, falta un archivo, falta torch),
    `cargar()` devuelve False y la GUI cae al RandomForest sin tronar.

Archivos que necesita por versión (los genera scripts/familias_dnabert.py):
  data/models/versiones/id<V>/dnabert_cabeza.pt     pesos de la cabeza MLP
  data/models/versiones/id<V>/dnabert_meta.json     lista de clases y configuración
  data/models/versiones/id<V>/dnabert_norm.npy      media y desviación del train

Si esa carpeta no existe, cae a las rutas de siempre, que es donde el script de
entrenamiento deja los archivos recién salidos:
  data/models/dnabert6_familias_cabeza.pt
  results/familias/familias_dnabert_meta.json
  results/familias/dnabert6_norm.npy

La primera vez descarga el modelo de Hugging Face (359 MB) y lo deja en la caché
de huggingface_hub; después abre en ~1 segundo.
"""
import json
import re
from pathlib import Path

import numpy as np

MODELO_HF = "zhihan1996/DNA_bert_6"
K = 6
MAX_LEN = 512
BATCH = 16

# El cuerpo de DNABERT es compartido entre versiones; la cabeza no.
_E = {"estado": "sin_cargar", "error": None, "tok": None, "bert": None,
      "cabeza": None, "clases": None, "mu": None, "sd": None, "torch": None,
      "version": None}


def _rutas(base_dir, version=None):
    """Rutas de los tres archivos del modelo entrenado.

    base_dir es la carpeta de la GUI (app/); el proyecto es su carpeta madre.
    Con `version`, busca primero en la carpeta de esa versión y solo cae a las
    rutas planas si esa carpeta no está completa. Así conviven las dos cosas:
    el archivo recién entrenado, que siempre aterriza en las rutas planas, y las
    versiones ya archivadas.
    """
    raiz = Path(base_dir).resolve().parent
    planas = {
        "cabeza": raiz / "data" / "models" / "dnabert6_familias_cabeza.pt",
        "meta":   raiz / "results" / "familias" / "familias_dnabert_meta.json",
        "norm":   raiz / "results" / "familias" / "dnabert6_norm.npy",
    }
    if version is None:
        return planas

    carpeta = raiz / "data" / "models" / "versiones" / f"id{version}"
    porversion = {
        "cabeza": carpeta / "dnabert_cabeza.pt",
        "meta":   carpeta / "dnabert_meta.json",
        "norm":   carpeta / "dnabert_norm.npy",
    }
    return porversion if all(p.exists() for p in porversion.values()) else planas


def archivos_listos(base_dir, version=None):
    """True si están los tres archivos del modelo entrenado. No carga nada."""
    return all(p.exists() for p in _rutas(base_dir, version).values())


def motivo_no_disponible():
    return _E["error"]


def version_cargada():
    return _E["version"]


def _cargar_cuerpo():
    """Tokenizador y BERT preentrenado. Son los mismos para todas las versiones."""
    if _E["tok"] is not None and _E["bert"] is not None:
        return True
    from transformers import AutoTokenizer, AutoModel
    # sin trust_remote_code: DNA_bert_6 es un BERT estándar y con esa bandera
    # chocan las clases BertConfig (lo documentó la asesora en su código)
    _E["tok"] = AutoTokenizer.from_pretrained(MODELO_HF)
    _E["bert"] = AutoModel.from_pretrained(MODELO_HF).eval()
    return True


def cargar(base_dir, version=None):
    """Carga DNABERT y la cabeza de `version`. True si quedó listo. Idempotente.

    Si ya está cargada otra versión, cambia solo la cabeza y deja el cuerpo en
    memoria.
    """
    if _E["estado"] == "listo" and _E["version"] == version:
        return True

    r = _rutas(base_dir, version)
    faltan = [k for k, p in r.items() if not p.exists()]
    if faltan:
        _E["error"] = ("Faltan archivos del modelo DNABERT: "
                       + ", ".join(str(r[k].name) for k in faltan)
                       + ". Corre scripts/familias_dnabert.py")
        _E["estado"] = "fallido"
        _E["version"] = version
        return False

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        _E["error"] = ("Falta PyTorch en este ambiente. Abre la GUI desde el ambiente "
                       "'dnabert', o instala:  pip install torch "
                       "--index-url https://download.pytorch.org/whl/cpu")
        _E["estado"] = "fallido"
        return False
    try:
        import transformers  # noqa: F401
    except ImportError:
        _E["error"] = "Falta transformers. Instala: pip install 'transformers==4.30.2'"
        _E["estado"] = "fallido"
        return False

    try:
        with open(r["meta"], encoding="utf-8") as fh:
            meta = json.load(fh)
        clases = list(meta["clases"])
        dim = int(meta.get("hidden", 768))

        norm = np.load(r["norm"])
        mu, sd = norm[0:1].astype(np.float32), norm[1:2].astype(np.float32)

        _cargar_cuerpo()

        # misma arquitectura que en el entrenamiento
        cabeza = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3),
                               nn.Linear(256, len(clases)))
        cabeza.load_state_dict(torch.load(r["cabeza"], map_location="cpu"))
        cabeza.eval()

        _E.update(estado="listo", error=None, cabeza=cabeza, clases=clases,
                  mu=mu, sd=sd, torch=torch, version=version)
        return True
    except Exception as e:
        _E["error"] = f"No se pudo cargar DNABERT: {type(e).__name__}: {e}"
        _E["estado"] = "fallido"
        _E["version"] = version
        return False


def sequence_to_kmers(seq, k=K):
    """Igual que en el entrenamiento: k-meros separados por espacio, sin el relleno."""
    seq = seq.upper().replace("0", "")
    seq = re.sub(r"[^ACGT]", "N", seq)
    if len(seq) < k:
        return "N" * k
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


def _embeddings(frags):
    torch = _E["torch"]
    tok, bert = _E["tok"], _E["bert"]
    salida = np.empty((len(frags), _E["mu"].shape[1]), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, len(frags), BATCH):
            lote = [sequence_to_kmers(s) for s in frags[i:i + BATCH]]
            enc = tok(lote, padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt")
            hs = bert(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            salida[i:i + len(lote)] = (
                (hs * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy()
    return salida


def predecir(frags, umbral=0.0):
    """
    frags: lista de fragmentos crudos de UN gen (cadenas de 500 pb).
    Devuelve (familia, probabilidad). Si la clase ganadora no llega al umbral,
    la familia es "Sin determinar".

    Promedia las probabilidades entre los fragmentos del gen, que es la regla con
    la que se validó el modelo (NO "algún fragmento": un gen tiene UNA familia).
    """
    if _E["estado"] != "listo":
        return "Sin clasificar", 0.0
    torch = _E["torch"]
    E = (_embeddings(frags) - _E["mu"]) / _E["sd"]
    with torch.inference_mode():
        pr = torch.softmax(_E["cabeza"](torch.tensor(E)), dim=1).numpy().mean(axis=0)
    j = int(np.argmax(pr))
    p = float(pr[j])
    if p < umbral:
        return "Sin determinar", p
    return _E["clases"][j], p


def info():
    """Para mostrar en la GUI qué modelo de familias está activo."""
    if _E["estado"] != "listo":
        return f"DNABERT no cargado ({_E['estado']})"
    v = f" · conjunto {_E['version']}" if _E["version"] else ""
    return f"DNABERT-6 · {len(_E['clases'])} familias{v}"
