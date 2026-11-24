#!/usr/bin/env python3
"""
parche_gui_min150.py -- pone la GUI al dia con los modelos del 99 %.

Seis cambios en app/argus_gui.py:

  1. Define MIN_PB = 150 y NO_EVALUABLE.

  2. Reemplaza fragmentar() por la regla de la asesora (opcion D): bloques de
     500 pb, el resto final se conserva SIN rellenar si llega a 150 pb reales y
     se descarta si no. Sin ceros y sin solapamiento. La version que estaba
     rellenaba los genes cortos y corria el ultimo bloque hacia atras, asi que
     le daba a los modelos fragmentos que nunca vieron en entrenamiento.

  3. Maneja el caso de un gen que no deja ningun fragmento. Antes la GUI se
     caia ahi: np.array([]) sale con forma (0,) y predict_proba truena. Ahora
     el gen aparece en la tabla como "No evaluable" con su razon, que es lo
     correcto: un gen que no se pudo medir no es un gen limpio.

  4. En la tabla, "No evaluable" se dibuja en gris con su propio icono y la
     columna de probabilidad muestra un guion en lugar de 0.00, para que no se
     confunda con un gen que el modelo evaluo y descarto con confianza.

  5. IDENTIDAD_MODELO pasa de 95 a 99.

  6. MODELO_FAMILIA pasa de "dnabert" a "rf", y se corrigen los numeros de los
     comentarios, que seguian siendo los de 12 clases.

El punto 6 se puede revertir a mano: la cabeza de DNABERT existe y es la nueva
de 17 clases, pero esta en data/models/ y no en app/. Con 0.736 contra 0.722,
la diferencia no justifica la dependencia de GPU.

Corre esto desde la raiz del proyecto:

    python parche_gui_min150.py

Deja respaldo en app/argus_gui.py.antes_del_parche_min150 y restaura solo si el
resultado no compila.
"""
import ast
import re
import shutil
import sys
from pathlib import Path

DESTINO = Path("app/argus_gui.py")
RESPALDO = Path("app/argus_gui.py.antes_del_parche_min150")

if not DESTINO.exists():
    sys.exit(f"No encontre {DESTINO}. Corre esto desde la raiz del proyecto.")

original = DESTINO.read_text(encoding="utf-8")
texto = original
hechos, avisos = [], []


def unico(aguja, donde):
    """Exige que el ancla aparezca exactamente una vez."""
    n = donde.count(aguja)
    if n != 1:
        sys.exit(f"El ancla aparece {n} veces y esperaba 1:\n  {aguja.strip()[:80]}")
    return True


# ------------------------------------------------------------------ 1. constantes
ANCLA_VENTANA = "VENTANA = 500\n"
NUEVAS_CONST = '''VENTANA = 500
MIN_PB = 150            # bases reales minimas de un fragmento (regla de la asesora)
NO_EVALUABLE = "No evaluable"   # gen demasiado corto: no se pudo medir
'''

if "MIN_PB" in texto and "NO_EVALUABLE" in texto:
    hechos.append("1. las constantes ya estaban")
else:
    unico(ANCLA_VENTANA, texto)
    texto = texto.replace(ANCLA_VENTANA, NUEVAS_CONST, 1)
    hechos.append("1. definidas MIN_PB = 150 y NO_EVALUABLE")

# ------------------------------------------------------------------ 2. fragmentar
NUEVA_FRAG = '''def fragmentar(seq, ventana=VENTANA, min_pb=MIN_PB):
    """Parte la secuencia en bloques de `ventana` pb.

    El resto final se conserva con su largo REAL si alcanza `min_pb` bases, y
    se descarta si no llega. Nunca rellena con ceros y nunca solapa bloques,
    asi que cada fragmento es un trozo real y distinto de la secuencia.

    Un gen mas corto que `min_pb` devuelve lista vacia: no es evaluable. Quien
    llame a esta funcion tiene que tratar ese caso aparte, no como negativo.

    Es la misma regla con la que se entrenaron los dos modelos (ver
    reentrenar_fragmentacion.py y modelo_familias.py). Si se cambia aqui, hay
    que reentrenar.
    """
    seq = seq.upper()
    L = len(seq)
    if L < ventana:
        return [seq] if L >= min_pb else []
    n = L // ventana
    frags = [seq[i * ventana:(i + 1) * ventana] for i in range(n)]
    r = L - n * ventana
    if r >= min_pb:
        frags.append(seq[n * ventana:])
    return frags
'''

# Hay dos "def fragmentar" en el archivo, pero la segunda esta dentro de un
# string (el bloque """ //ANTERIOR ... """), asi que es comentario. count=1
# reemplaza la primera, que es la que de verdad corre.
patron_frag = re.compile(r"def fragmentar\(seq, ventana=VENTANA\):.*?\n    return frags\n", re.S)

if "min_pb=MIN_PB" in texto:
    hechos.append("2. fragmentar() ya estaba actualizada")
elif patron_frag.search(texto):
    texto = patron_frag.sub(lambda m: NUEVA_FRAG, texto, count=1)
    hechos.append("2. reemplazada fragmentar() (sin relleno, sin solapamiento)")
else:
    sys.exit("No reconoci fragmentar(). No toco el archivo.")

# ------------------------------------------------------------------ 3. gen sin fragmentos
ANCLA_FRAGS = "        frags = fragmentar(seq, ventana=VENTANA)\n"
GUARDIA = '''        frags = fragmentar(seq, ventana=VENTANA)

        # Con la regla de MIN_PB un gen demasiado corto no deja ningun
        # fragmento. No se puede evaluar, y reportarlo como "No" seria afirmar
        # que esta limpio cuando en realidad nunca se midio: un error que el
        # usuario no tendria forma de notar. Se marca aparte.
        if not frags:
            if "|" in sid:
                _mag = sid.split("|")[0]
                _gen = sid.split("|")[-1][:25]
            else:
                _mag = nombre_defecto if nombre_defecto else sid[:15]
                _gen = sid[:25]
            datos.append({
                "mag": _mag,
                "gen": _gen,
                "presencia": NO_EVALUABLE,
                "familia": f"gen < {MIN_PB} pb",
                "probabilidad": 0.0,
            })
            if callback_progreso:
                callback_progreso(idx, total)
            continue

'''

if "if not frags:" in texto:
    hechos.append("3. el caso sin fragmentos ya estaba cubierto")
else:
    unico(ANCLA_FRAGS, texto)
    texto = texto.replace(ANCLA_FRAGS, GUARDIA, 1)
    hechos.append("3. agregado el caso del gen sin fragmentos -> No evaluable")

# ------------------------------------------------------------------ 4. tabla
# Se insertan lineas en lugar de reemplazar las que existen, para no tener que
# calzar los emoji y los acentos del archivo.
pasos_tabla = [
    (
        "            item_presencia = QTableWidgetItem(texto_presencia)\n",
        '            if d["presencia"] == NO_EVALUABLE:\n'
        '                texto_presencia = f"⚪ {d[\'presencia\']}"\n'
        "            item_presencia = QTableWidgetItem(texto_presencia)\n",
        "icono propio para No evaluable",
    ),
    (
        "            self.tabla.setItem(fila, 2, item_presencia)\n",
        '            if d["presencia"] == NO_EVALUABLE:\n'
        '                item_presencia.setForeground(QBrush(QColor("#ADB5BD")))\n'
        "            self.tabla.setItem(fila, 2, item_presencia)\n",
        "No evaluable en gris",
    ),
    (
        "            item_prob.setTextAlignment(Qt.AlignCenter)\n",
        "            item_prob.setTextAlignment(Qt.AlignCenter)\n"
        '            if d["presencia"] == NO_EVALUABLE:\n'
        '                item_prob.setText("—")\n',
        "guion en la columna de probabilidad",
    ),
]

if 'item_prob.setText("—")' in texto:
    hechos.append("4. la tabla ya estaba actualizada")
else:
    detalles = []
    for ancla, nuevo, que in pasos_tabla:
        unico(ancla, texto)
        texto = texto.replace(ancla, nuevo, 1)
        detalles.append(que)
    hechos.append("4. tabla: " + "; ".join(detalles))

# ------------------------------------------------------------------ 5. identidad
if "IDENTIDAD_MODELO = 95" in texto:
    unico("IDENTIDAD_MODELO = 95", texto)
    texto = texto.replace("IDENTIDAD_MODELO = 95", "IDENTIDAD_MODELO = 99", 1)
    hechos.append("5. IDENTIDAD_MODELO: 95 -> 99")
elif "IDENTIDAD_MODELO = 99" in texto:
    hechos.append("5. IDENTIDAD_MODELO ya era 99")
else:
    avisos.append("no encontre IDENTIDAD_MODELO; revisa a mano a que modelo apunta")

# ------------------------------------------------------------------ 6. etapa 2
patron_coment = re.compile(
    r'# "dnabert" -> DNABERT-6 \(balanced accuracy [\d.]+ en el test por gen\)\n'
    r'# "rf"\s+-> RandomForest sobre k-meros \([\d.]+, pero instant.neo\)\n'
)
NUEVO_COMENT = (
    '# Conjunto al 99 % de identidad, 17 familias, misma particion por gen:\n'
    '#   "dnabert" -> DNABERT-6, balanced accuracy 0.736 (necesita el .pt y GPU)\n'
    '#   "rf"      -> RandomForest sobre k-meros, 0.722 (instantaneo, sin GPU)\n'
    '# La diferencia es de 0.014, asi que la dependencia de DNABERT no se paga.\n'
)

if patron_coment.search(texto):
    texto = patron_coment.sub(NUEVO_COMENT, texto, count=1)
    hechos.append("6a. corregidos los numeros de los comentarios de etapa 2")
else:
    avisos.append('no reconoci los comentarios de MODELO_FAMILIA; sus numeros '
                  'pueden seguir siendo los de 12 clases')

if 'MODELO_FAMILIA = "dnabert"' in texto:
    unico('MODELO_FAMILIA = "dnabert"', texto)
    texto = texto.replace('MODELO_FAMILIA = "dnabert"', 'MODELO_FAMILIA = "rf"', 1)
    hechos.append('6b. MODELO_FAMILIA: "dnabert" -> "rf"')
elif 'MODELO_FAMILIA = "rf"' in texto:
    hechos.append('6b. MODELO_FAMILIA ya era "rf"')
else:
    avisos.append("no encontre MODELO_FAMILIA; revisa a mano que modelo de familias usa")

# ------------------------------------------------------------------ escribir
if texto == original:
    print("=" * 74)
    print("  El archivo ya estaba parchado. No cambie nada.")
    print("=" * 74)
    sys.exit(0)

shutil.copy2(DESTINO, RESPALDO)
DESTINO.write_text(texto, encoding="utf-8")

try:
    ast.parse(texto)
except SyntaxError as e:
    shutil.copy2(RESPALDO, DESTINO)
    sys.exit(f"Quedo mal (linea {e.lineno}: {e.msg}). Restaure el original.")

print("=" * 74)
for h in hechos:
    print(f"  {h}")
for a in avisos:
    print(f"  [AVISO] {a}")
print("=" * 74)
print(f"  respaldo: {RESPALDO}")
print("  el archivo sigue siendo Python valido")
print()

# ------------------------------------------------------------------ autoprueba
ns = {}
exec(compile(NUEVA_FRAG, "<fragmentar>", "exec"),
     {"VENTANA": 500, "MIN_PB": 150}, ns)
frag = ns["fragmentar"]

print("  Lo que hace la fragmentacion nueva:")
for L in (5297, 831, 620, 162, 100):
    fr = frag(("ACGT" * (L // 4 + 1))[:L])
    largos = [len(x) for x in fr]
    nota = ""
    if L == 620:
        nota = "  <- descarta 120 pb"
    elif L == 162:
        nota = "  <- se queda con su largo real"
    elif L == 100:
        nota = "  <- NO EVALUABLE"
    print(f"    gen de {L:>5} pb -> {len(fr):>2} fragmento(s) {largos}{nota}")
print("=" * 74)
