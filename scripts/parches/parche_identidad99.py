#!/usr/bin/env python3
"""
parche_identidad99.py -- habilita el nivel de identidad 99 % en
scripts/reentrenar_fragmentacion.py

Tres cambios:

  1. --identidad acepta "99" (antes solo 95, 90, 85).

  2. Se agrega --filtro, un FASTA que restringe el pool de positivos y que
     sustituye al que se deduce de --identidad. Sirve para la corrida "limpia",
     donde queremos excluir las secuencias con parientes cercanos en el
     conjunto de evaluacion.

  3. Se arregla la deduccion de FILTRO. Antes era:

         FILTRO = None if args.identidad == "95" else INTER / "positivo_idNN_limpio.fasta"

     El supuesto era que 90 y 85 son SUBCONJUNTOS del 95 %, asi que un filtro
     basta para bajar de nivel. Pero el 99 % tiene MAS secuencias (6,257 contra
     4,228), y un filtro solo puede quitar, nunca agregar. Como POS ya apunta al
     FASTA del 99 %, para ese nivel la respuesta correcta es no filtrar nada.

Corre esto desde la raiz del proyecto:

    python parche_identidad99.py

Deja respaldo en scripts/reentrenar_fragmentacion.py.antes_del_parche99 y
restaura solo si el resultado no es Python valido.
"""
import ast
import shutil
import sys
from pathlib import Path

DESTINO = Path("scripts/reentrenar_fragmentacion.py")
RESPALDO = Path("scripts/reentrenar_fragmentacion.py.antes_del_parche99")

if not DESTINO.exists():
    sys.exit(f"No encontre {DESTINO}. Corre esto desde la raiz del proyecto.")

original = DESTINO.read_text(encoding="utf-8")
texto = original
hechos = []

# ---------------------------------------------------------------- 1 y 2
VIEJO_ARG = 'ap.add_argument("--identidad", default="95", choices=["95", "90", "85"])'
NUEVO_ARG = (
    'ap.add_argument("--identidad", default="95", choices=["95", "90", "85", "99"])\n'
    'ap.add_argument("--filtro", default=None,\n'
    '                help="FASTA que restringe el pool de positivos; '
    'sustituye al que se deduce de --identidad")'
)

if VIEJO_ARG in texto:
    texto = texto.replace(VIEJO_ARG, NUEVO_ARG, 1)
    hechos.append("1. --identidad acepta 99")
    hechos.append("2. agregado el argumento --filtro")
elif '"99"' in texto and "--filtro" in texto:
    hechos.append("1 y 2. ya estaban aplicados")
else:
    sys.exit("No reconoci la linea de --identidad. No toco el archivo.")

# ---------------------------------------------------------------- 3
VIEJO_FILTRO = (
    'FILTRO = None if args.identidad == "95" '
    'else INTER / f"positivo_id{args.identidad}_limpio.fasta"'
)
NUEVO_FILTRO = '''# El filtro solo puede QUITAR positivos del pool, nunca agregar. 90 y 85 son
# subconjuntos del 95 %, asi que ahi un filtro alcanza. Para 95 y 99 el FASTA de
# POS ya contiene exactamente el conjunto de ese nivel, asi que no se filtra.
if args.filtro:
    FILTRO = Path(args.filtro).expanduser()
elif args.identidad in ("95", "99"):
    FILTRO = None
else:
    FILTRO = INTER / f"positivo_id{args.identidad}_limpio.fasta"'''

if VIEJO_FILTRO in texto:
    texto = texto.replace(VIEJO_FILTRO, NUEVO_FILTRO, 1)
    hechos.append("3. arreglada la deduccion de FILTRO")
elif "if args.filtro:" in texto:
    hechos.append("3. ya estaba aplicado")
else:
    sys.exit("No reconoci la linea de FILTRO. No toco el archivo.")

# ---------------------------------------------------------------- escribir
if texto == original:
    print("=" * 70)
    print("  El archivo ya estaba parchado. No cambie nada.")
    print("=" * 70)
    sys.exit(0)

shutil.copy2(DESTINO, RESPALDO)
DESTINO.write_text(texto, encoding="utf-8")

try:
    ast.parse(texto)
except SyntaxError as e:
    shutil.copy2(RESPALDO, DESTINO)
    sys.exit(f"Quedo mal (linea {e.lineno}: {e.msg}). Restaure el original.")

print("=" * 70)
for h in hechos:
    print(f"  {h}")
print("=" * 70)
print(f"  respaldo: {RESPALDO}")
print("  el archivo sigue siendo Python valido")
print()
print("  Como queda la logica del filtro:")
print("    --identidad 99                 -> sin filtro, usa POS completo (6,257)")
print("    --identidad 99 --filtro X.fasta -> restringe los positivos a X.fasta")
print("    --identidad 90                 -> busca positivo_id90_limpio.fasta")
print("=" * 70)
