# -*- coding: utf-8 -*-
"""
parche_fragmentacion.py

Pone en scripts/modelo_familias.py la MISMA fragmentacion que usan el modelo
binario y DNABERT: bloques de 500 pb, el ultimo se conserva con su largo real
si tiene >= min_pb, se descarta si no, y NUNCA se rellena con ceros.

Cambia cuatro cosas:
  1. agrega el argumento --min-pb (por defecto 150)
  2. define MIN_PB junto a K, VENTANA y SEED
  3. reemplaza la funcion fragmentar()
  4. registra min_pb y relleno en el meta.json, como hace el modelo binario

Guarda una copia en scripts/modelo_familias.py.antes_del_parche

Uso, desde la raiz del proyecto:
    python parche_fragmentacion.py
"""
import re
import shutil
import sys
from pathlib import Path

RUTA = Path("scripts/modelo_familias.py")
if not RUTA.exists():
    sys.exit("[ERROR] corre esto desde la raiz del proyecto "
             "(no encuentro scripts/modelo_familias.py)")

src = RUTA.read_text(encoding="utf-8")
original = src
cambios = []

# ---------------------------------------------------------------- 1. argumento
if "--min-pb" in src:
    cambios.append("1. --min-pb ya existia, no se toca")
else:
    m = re.search(r'(ap\.add_argument\("--ventana"[^\n]*\n)', src)
    if not m:
        sys.exit("[ERROR] no encuentro el argumento --ventana")
    src = src[:m.end()] + \
        'ap.add_argument("--min-pb", type=int, default=150,\n' \
        '                help="minimo de bases reales del bloque final; '\
        'por debajo se descarta")\n' + src[m.end():]
    cambios.append("1. agregado el argumento --min-pb")

# ---------------------------------------------------------------- 2. constante
if re.search(r"^MIN_PB\s*=", src, re.M):
    cambios.append("2. MIN_PB ya estaba definido")
else:
    m = re.search(r"^(K, VENTANA, SEED = args\.k, args\.ventana, args\.seed)$",
                  src, re.M)
    if not m:
        sys.exit("[ERROR] no encuentro la linea que define K, VENTANA y SEED")
    src = src[:m.end()] + "\nMIN_PB = args.min_pb" + src[m.end():]
    cambios.append("2. definido MIN_PB")

# ---------------------------------------------------------------- 3. funcion
NUEVA = '''def fragmentar(seq, ventana=VENTANA, min_pb=MIN_PB):
    """Parte la secuencia en bloques de `ventana` pb.

    El bloque final se conserva CON SU LARGO REAL si tiene al menos `min_pb`
    bases, y se descarta si no llega. Nunca se rellena con ceros y los bloques
    nunca se solapan: es la misma regla del modelo binario y de DNABERT.

    Un gen de menos de `min_pb` no produce ningun fragmento (lista vacia).
    """
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
'''
patron = re.compile(r"def fragmentar\(.*?\n    return fr\n", re.S)
m = patron.search(src)
if not m:
    sys.exit("[ERROR] no encuentro la funcion fragmentar() completa.\n"
             "        Revisa el archivo a mano; no se cambio nada.")
viejo = m.group(0)
if 'return [seq]' in viejo and '"0"' not in viejo:
    cambios.append("3. fragmentar() ya estaba sin relleno, no se toca")
else:
    src = src[:m.start()] + NUEVA + src[m.end():]
    cambios.append("3. reemplazada fragmentar() (sin relleno, sin solapamiento)")

# ---------------------------------------------------------------- 4. meta
if '"min_pb"' in src:
    cambios.append("4. el meta ya registraba min_pb")
else:
    m = re.search(r'("K": K, "ventana": VENTANA, )', src)
    if not m:
        cambios.append("4. [AVISO] no encontre el dict del meta; "
                       "agrega min_pb a mano si lo quieres registrado")
    else:
        src = src[:m.end()] + '"min_pb": MIN_PB, "relleno": "ninguno", ' + src[m.end():]
        cambios.append("4. el meta ahora registra min_pb y relleno")

# ---------------------------------------------------------------- guardar
if src == original:
    print("No habia nada que cambiar.")
    sys.exit(0)

copia = RUTA.with_suffix(".py.antes_del_parche")
shutil.copy2(RUTA, copia)
RUTA.write_text(src, encoding="utf-8")

print("=" * 70)
for c in cambios:
    print("  " + c)
print("=" * 70)
print(f"  respaldo: {copia}")

# comprobacion de sintaxis
import ast
try:
    ast.parse(src)
    print("  el archivo sigue siendo Python valido")
except SyntaxError as e:
    shutil.copy2(copia, RUTA)
    sys.exit(f"  [ERROR] quedo mal: {e}\n  se restauro el original")

print("\n  La nueva fragmentacion, sobre tres casos:")
exec(compile(NUEVA.replace("ventana=VENTANA", "ventana=500")
                  .replace("min_pb=MIN_PB", "min_pb=150"), "<test>", "exec"))
for L in (5297, 831, 620, 162, 100):
    fr = fragmentar("A" * L)
    print(f"    gen de {L:>5} pb -> {len(fr)} fragmentos "
          f"{[len(x) for x in fr]}")
