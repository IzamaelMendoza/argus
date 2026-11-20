"""
verificar_instalacion.py -- comprueba que ARGus pueda arrancar.

Revisa cuatro cosas y dice exactamente que hacer si algo falta:

  1. la version de Python
  2. los paquetes necesarios
  3. los archivos del modelo (y si quedaron como punteros de Git LFS)
  4. la cache de DNABERT

Se escribio sin dependencias externas a proposito: usa solo la biblioteca
estandar, para que funcione aunque el entorno este a medio instalar.

Uso:
    python app/verificar_instalacion.py
"""
import importlib
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent

OK, AVISO, FALLO = "  [ok]   ", "  [aviso]", "  [FALTA]"
problemas, avisos = [], []


def titulo(t):
    print(f"\n{t}\n" + "-" * len(t))


print("=" * 72)
print("ARGus -- verificacion de la instalacion")
print(f"proyecto: {RAIZ}")
print("=" * 72)

# =====================================================================
titulo("1. Python")
v = sys.version_info
print(f"{OK if v >= (3, 10) else FALLO} Python {v.major}.{v.minor}.{v.micro}")
if v < (3, 10):
    problemas.append("Python 3.10 o superior. Crea el entorno con: conda create -n argus python=3.11")
elif v >= (3, 14):
    avisos.append(f"Python {v.major}.{v.minor} es mas nuevo de lo probado (3.10-3.13); "
                  "si algun paquete falla al instalar, usa 3.11")

# =====================================================================
titulo("2. Paquetes")
PAQUETES = [
    ("PySide6", "la interfaz grafica", True),
    ("matplotlib", "las graficas", True),
    ("numpy", "calculo numerico", True),
    ("pandas", "tablas y exportacion a CSV", True),
    ("joblib", "cargar el modelo binario", True),
    ("sklearn", "el Random Forest de la etapa 1", True),
    ("torch", "DNABERT (clasificacion por familia)", False),
    ("transformers", "DNABERT (clasificacion por familia)", False),
]
for mod, para, obligatorio in PAQUETES:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"{OK} {mod:<14} {ver:<12} {para}")
        if mod == "transformers" and ver[0:1].isdigit() and int(ver.split(".")[0]) >= 5:
            avisos.append(f"transformers {ver}: la rama 5.x cambia la carga de modelos BERT "
                          "antiguos y DNA_bert_6 puede fallar. Si pasa: "
                          "pip install 'transformers>=4.30,<5'")
    except ImportError:
        marca = FALLO if obligatorio else AVISO
        print(f"{marca} {mod:<14} {'':<12} {para}")
        if obligatorio:
            problemas.append(f"falta {mod}. Instala todo con: pip install -r requirements.txt")
        else:
            avisos.append(f"falta {mod}: ARGus funcionara, pero sin clasificacion por familia "
                          "(todas las resistencias saldran como 'Sin clasificar')")

# =====================================================================
titulo("3. Archivos del modelo")

PUNTERO = b"version https://git-lfs"

ARCHIVOS = [
    (RAIZ / "data" / "models" / "argus_rf_k6_id95_v2.joblib",
     "modelo binario (etapa 1)", 1_000_000, True),
    (RAIZ / "data" / "models" / "dnabert6_familias_cabeza.pt",
     "cabeza de familias (etapa 2)", 100_000, False),
    (RAIZ / "results" / "familias" / "familias_dnabert_meta.json",
     "lista de las 12 familias", 100, False),
    (RAIZ / "results" / "familias" / "dnabert6_norm.npy",
     "normalizacion de embeddings", 100, False),
]

hay_punteros = False
for ruta, desc, minimo, obligatorio in ARCHIVOS:
    if not ruta.exists():
        marca = FALLO if obligatorio else AVISO
        print(f"{marca} {desc}")
        print(f"           no existe: {ruta.relative_to(RAIZ)}")
        (problemas if obligatorio else avisos).append(f"falta {ruta.name}")
        continue

    tam = ruta.stat().st_size
    try:
        with open(ruta, "rb") as fh:
            cabeza = fh.read(64)
    except OSError:
        cabeza = b""

    if cabeza.startswith(PUNTERO):
        hay_punteros = True
        print(f"{FALLO} {desc}")
        print(f"           {ruta.relative_to(RAIZ)} es un PUNTERO de Git LFS ({tam} bytes),")
        print(f"           no el archivo real")
        problemas.append(f"{ruta.name} no se descargo de verdad")
    elif tam < minimo:
        print(f"{AVISO} {desc}")
        print(f"           {ruta.relative_to(RAIZ)} pesa solo {tam:,} bytes, parece incompleto")
        avisos.append(f"{ruta.name} parece incompleto")
    else:
        mb = tam / 1e6
        print(f"{OK} {desc:<34} {mb:>8.1f} MB")

if hay_punteros:
    problemas.append("Git LFS no descargo los modelos. Ejecuta:  git lfs install && git lfs pull")

# el meta.json ademas debe traer las clases
meta = RAIZ / "results" / "familias" / "familias_dnabert_meta.json"
if meta.exists():
    try:
        import json
        with open(meta, encoding="utf-8") as fh:
            m = json.load(fh)
        n = len(m.get("clases", []))
        bal = m.get("test_gen_balanced_acc")
        extra = f" | balanced accuracy {bal:.3f}" if isinstance(bal, (int, float)) else ""
        print(f"{OK} el modelo de familias reconoce {n} familias{extra}")
        if m.get("min_pb") is None:
            avisos.append("el modelo de familias no registra 'min_pb': se entreno con la "
                          "fragmentacion antigua. Conviene reentrenarlo")
    except Exception as e:
        avisos.append(f"no se pudo leer familias_dnabert_meta.json: {e}")

# =====================================================================
titulo("4. Cache de DNABERT")
posibles = [
    Path.home() / ".cache" / "huggingface" / "hub" / "models--zhihan1996--DNA_bert_6",
    Path(os.environ.get("HF_HOME", "")) / "hub" / "models--zhihan1996--DNA_bert_6",
]
cache = next((p for p in posibles if p.exists()), None)
if cache:
    mb = sum(f.stat().st_size for f in cache.rglob("*") if f.is_file()) / 1e6
    print(f"{OK} descargado ({mb:.0f} MB) en {cache}")
else:
    print(f"{AVISO} DNABERT todavia no esta descargado")
    print( "           Se bajara solo (359 MB) la primera vez que analices un archivo.")
    print( "           Hace falta conexion a internet en ese primer analisis.")

# =====================================================================
print("\n" + "=" * 72)
if problemas:
    print("NO SE PUEDE ARRANCAR TODAVIA")
    print("=" * 72)
    for i, p_ in enumerate(dict.fromkeys(problemas), 1):
        print(f"  {i}. {p_}")
    if avisos:
        print("\n  Ademas:")
        for a in dict.fromkeys(avisos):
            print(f"     - {a}")
    sys.exit(1)

print("TODO LISTO" + (" (con avisos)" if avisos else ""))
print("=" * 72)
for a in dict.fromkeys(avisos):
    print(f"  - {a}")
print("\n  Para abrir ARGus:   python app/argus_gui.py")
sys.exit(0)
