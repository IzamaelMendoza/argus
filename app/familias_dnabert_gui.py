"""
familias_dnabert_gui.py -- predictor de FAMILIA con DNABERT-6 para la etapa 2 de ARGus.

Va en la misma carpeta que la GUI (app/) y se importa desde ahí. Encapsula todo lo
que DNABERT necesita para que `predecir_mag` quede con tres líneas de cambio.

Cómo funciona:
  - Carga perezosa: DNABERT se carga la PRIMERA vez que hace falta, no al abrir la
    GUI. Si el MAG no trae ningún gen resistente, nunca se carga.
  - Una sola carga por sesión: queda en memoria a nivel de módulo, así que varios
    análisis seguidos no vuelven a pagar el costo.
  - Si algo falla (sin internet la primera vez, falta un archivo, falta torch),
    `disponible()` devuelve False y la GUI cae al RandomForest sin tronar.

Archivos que necesita (los genera scripts/familias_dnabert.py --completo):
  data/models/dnabert6_familias_cabeza.pt        pesos de la cabeza MLP
  results/familias/familias_dnabert_meta.json    lista de clases y configuración
  results/familias/dnabert6_norm.npy             media y desviación del train

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

_E = {"estado": "sin_cargar", "error": None, "tok": None, "bert": None,
      "cabeza": None, "clases": None, "mu": None, "sd": None, "torch": None}


def _rutas(base_dir):
    """base_dir es la carpeta de la GUI (app/); el proyecto es su carpeta madre."""
    raiz = Path(base_dir).resolve().parent
    return {
        "cabeza": raiz / "data" / "models" / "dnabert6_familias_cabeza.pt",
        "meta":   raiz / "results" / "familias" / "familias_dnabert_meta.json",
        "norm":   raiz / "results" / "familias" / "dnabert6_norm.npy",
    }


def archivos_listos(base_dir):
    """True si están los tres archivos del modelo entrenado. No carga nada."""
    return all(p.exists() for p in _rutas(base_dir).values())


def motivo_no_disponible():
    return _E["error"]


def cargar(base_dir):
    """Carga DNABERT y la cabeza. Devuelve True si quedó listo. Idempotente."""
    if _E["estado"] == "listo":
        return True
    if _E["estado"] == "fallido":
        return False

    r = _rutas(base_dir)
    faltan = [k for k, p in r.items() if not p.exists()]
    if faltan:
        _E["error"] = ("Faltan archivos del modelo DNABERT: "
                       + ", ".join(str(r[k].name) for k in faltan)
                       + ". Corre scripts/familias_dnabert.py --completo")
        _E["estado"] = "fallido"
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
        from transformers import AutoTokenizer, AutoModel
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

        # sin trust_remote_code: DNA_bert_6 es un BERT estándar y con esa bandera
        # chocan las clases BertConfig (lo documentó la asesora en su código)
        tok = AutoTokenizer.from_pretrained(MODELO_HF)
        bert = AutoModel.from_pretrained(MODELO_HF).eval()

        # misma arquitectura que en el entrenamiento
        cabeza = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3),
                               nn.Linear(256, len(clases)))
        cabeza.load_state_dict(torch.load(r["cabeza"], map_location="cpu"))
        cabeza.eval()

        _E.update(estado="listo", error=None, tok=tok, bert=bert, cabeza=cabeza,
                  clases=clases, mu=mu, sd=sd, torch=torch)
        return True
    except Exception as e:
        _E["error"] = f"No se pudo cargar DNABERT: {type(e).__name__}: {e}"
        _E["estado"] = "fallido"
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
    return f"DNABERT-6 · {len(_E['clases'])} familias"
