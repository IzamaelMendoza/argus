#!/usr/bin/env python3
"""
parche_gui_progreso.py -- barra de carga con tiempo transcurrido y estimacion
de lo que falta, en app/argus_gui.py

Lo que agrega:

  1. QProgressBar debajo de la barra de botones, oculta mientras no haya un
     analisis corriendo.
  2. _formato_tiempo(): segundos -> "1 h 04 min" / "3 min 20 s" / "45 s".
  3. Marca de tiempo al iniciar y reinicio de la barra.
  4. actualizar_progreso() rehecho: mueve la barra y escribe
     "Analizando 340 de 2,000 (17 %) - transcurrido 2 min 10 s - falta ~10 min".
  5. Al terminar, la barra se esconde y el mensaje final dice cuanto tardo.
  6. Estilo de la barra en las dos paletas, clara y oscura.

Dos decisiones que vale la pena conocer:

  * La estimacion usa el promedio de TODO lo corrido, no el de los ultimos
    segundos. Aqui el costo por secuencia es muy desigual: solo las que dan
    positivo en la etapa 1 pasan por DNABERT, y esas tardan ~1.3 s mientras que
    una negativa tarda milisegundos. Un promedio corto brincaria segun la racha
    de positivos que toque. El promedio global se corrige solo y converge.

  * Por lo mismo no se muestra la estimacion hasta llevar 20 secuencias: antes
    de eso el numero seria una mentira con aspecto de dato.

  * El texto se redibuja 5 veces por segundo como maximo, aunque la senal llegue
    una vez por secuencia. La barra si se mueve en cada una.

Corre esto desde la raiz del proyecto:

    python parche_gui_progreso.py
"""
import ast
import re
import shutil
import sys
from pathlib import Path

DESTINO = Path("app/argus_gui.py")
RESPALDO = Path("app/argus_gui.py.antes_del_progreso")

if not DESTINO.exists():
    sys.exit(f"No encontre {DESTINO}. Corre esto desde la raiz del proyecto.")

original = DESTINO.read_text(encoding="utf-8")
texto = original
hechos = []


def unico(aguja, donde):
    n = donde.count(aguja)
    if n != 1:
        sys.exit(f"El ancla aparece {n} veces y esperaba 1:\n  {aguja.strip()[:90]}")


# ------------------------------------------------------------------ 1. import
A_IMPORT = "    QLineEdit\n)"
if "QProgressBar" in texto:
    hechos.append("1. QProgressBar ya estaba importado")
else:
    unico(A_IMPORT, texto)
    texto = texto.replace(A_IMPORT, "    QLineEdit, QProgressBar\n)", 1)
    hechos.append("1. importado QProgressBar")

# ------------------------------------------------------------------ 2. helper
A_HILO = "class HiloPrediccion(QThread):"
AYUDA = '''def _formato_tiempo(segundos):
    """Segundos -> '1 h 04 min', '3 min 20 s' o '45 s'."""
    segundos = max(0, int(round(segundos)))
    h, resto = divmod(segundos, 3600)
    m, s = divmod(resto, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


class HiloPrediccion(QThread):'''

if "_formato_tiempo" in texto:
    hechos.append("2. _formato_tiempo ya estaba")
else:
    unico(A_HILO, texto)
    texto = texto.replace(A_HILO, AYUDA, 1)
    hechos.append("2. agregado _formato_tiempo()")

# ------------------------------------------------------------------ 3. widget
A_LAYOUT = "        layout.addLayout(barra)\n"
WIDGET = '''        layout.addLayout(barra)

        # Barra de avance: oculta mientras no haya un analisis corriendo, para
        # no dejar un cero permanente en pantalla cuando no hay nada que medir.
        self.barra_progreso = QProgressBar()
        self.barra_progreso.setObjectName("barraProgreso")
        self.barra_progreso.setMinimum(0)
        self.barra_progreso.setMaximum(100)
        self.barra_progreso.setValue(0)
        self.barra_progreso.setTextVisible(True)
        self.barra_progreso.setFormat("%p %")
        self.barra_progreso.hide()
        layout.addWidget(self.barra_progreso)

        self._t_inicio = None          # cuando arranco el analisis en curso
        self._t_ultimo_texto = 0.0     # para no redibujar el texto en cada secuencia
'''

if "self.barra_progreso" in texto:
    hechos.append("3. la barra ya estaba en la interfaz")
else:
    unico(A_LAYOUT, texto)
    texto = texto.replace(A_LAYOUT, WIDGET, 1)
    hechos.append("3. agregada la barra de avance a la interfaz")

# ------------------------------------------------------------------ 4. iniciar
A_INICIAR = "        nombre_mag = os.path.splitext(os.path.basename(self.ruta_actual))[0]\n"
INICIAR = '''        self._t_inicio = time.time()
        self._t_ultimo_texto = 0.0
        self.barra_progreso.setMaximum(max(1, len(self.secuencias_cargadas)))
        self.barra_progreso.setValue(0)
        self.barra_progreso.show()

        nombre_mag = os.path.splitext(os.path.basename(self.ruta_actual))[0]
'''

if "self._t_inicio = time.time()" in texto:
    hechos.append("4. el inicio del cronometro ya estaba")
else:
    unico(A_INICIAR, texto)
    texto = texto.replace(A_INICIAR, INICIAR, 1)
    hechos.append("4. el cronometro arranca con el analisis")

# ------------------------------------------------------------------ 5. progreso
NUEVO_PROGRESO = '''    def actualizar_progreso(self, actual, total):
        """Mueve la barra y escribe avance, tiempo corrido y lo que falta.

        La estimacion usa el promedio de TODO lo corrido y no el de los ultimos
        segundos. Aqui el costo por secuencia es muy desigual: solo las que dan
        positivo en la etapa 1 pasan por la etapa 2, y esas tardan ordenes de
        magnitud mas. Un promedio corto brincaria segun la racha de positivos
        que toque; el global se corrige solo conforme avanza.

        Por eso mismo no se muestra hasta llevar 20 secuencias: antes de eso
        seria una adivinanza con cara de dato.
        """
        if total <= 0:
            return

        self.barra_progreso.setMaximum(total)
        self.barra_progreso.setValue(actual)

        ahora = time.time()
        if actual < total and (ahora - self._t_ultimo_texto) < 0.2:
            return          # 5 refrescos por segundo bastan para el texto
        self._t_ultimo_texto = ahora

        transcurrido = (ahora - self._t_inicio) if self._t_inicio else 0.0
        porcentaje = int(actual / total * 100)
        partes = [f"Analizando {actual:,} de {total:,} ({porcentaje} %)",
                  f"transcurrido {_formato_tiempo(transcurrido)}"]
        if actual >= 20 and transcurrido > 0:
            falta = transcurrido / actual * (total - actual)
            partes.append(f"falta ~{_formato_tiempo(falta)}")
        self.label_estado.setText("   ·   ".join(partes))
        self.label_estado.setStyleSheet("color: #FD7E14; font-weight: bold;")

'''

patron_prog = re.compile(
    r"    def actualizar_progreso\(self, actual, total\):.*?\n(?=    def )", re.S)

if "self._t_ultimo_texto = ahora" in texto:
    hechos.append("5. actualizar_progreso ya estaba rehecha")
elif patron_prog.search(texto):
    texto = patron_prog.sub(lambda m: NUEVO_PROGRESO, texto, count=1)
    hechos.append("5. rehecha actualizar_progreso (barra + tiempos)")
else:
    sys.exit("No reconoci actualizar_progreso(). No toco el archivo.")

# ------------------------------------------------------------------ 6. terminar
A_TERMINADO = "    def analisis_terminado(self, resultados, fue_cancelado):\n"
TERMINADO = '''    def analisis_terminado(self, resultados, fue_cancelado):
        self.barra_progreso.setValue(self.barra_progreso.maximum())
        self.barra_progreso.hide()
'''

A_ERROR = "    def analisis_error(self, mensaje_error):\n"
CIERRE = '''        if self._t_inicio is not None:
            self.label_estado.setText(
                self.label_estado.text()
                + f"\\nTiempo total: {_formato_tiempo(time.time() - self._t_inicio)}")
            self._t_inicio = None

    def analisis_error(self, mensaje_error):
        self.barra_progreso.hide()
        self._t_inicio = None
'''

if "Tiempo total:" in texto:
    hechos.append("6. el cierre del cronometro ya estaba")
else:
    unico(A_TERMINADO, texto)
    texto = texto.replace(A_TERMINADO, TERMINADO, 1)
    unico(A_ERROR, texto)
    texto = texto.replace(A_ERROR, CIERRE, 1)
    hechos.append("6. al terminar: barra oculta y tiempo total en el mensaje")

# ------------------------------------------------------------------ 7. estilo
A_ESTILO = ('QToolButton#botonCerrarVentana:hover '
            '{ background-color: #E81123; color: white; }\n"""')
ESTILO = ('QToolButton#botonCerrarVentana:hover '
          '{ background-color: #E81123; color: white; }\n'
          'QProgressBar#barraProgreso { background-color: $tab; border: 1px solid $borde; '
          'border-radius: 7px; min-height: 16px; max-height: 16px; text-align: center; '
          'color: $texto_ui; font-size: 8pt; font-weight: bold; }\n'
          'QProgressBar#barraProgreso::chunk { background-color: $acento; '
          'border-radius: 6px; margin: 1px; }\n"""')

if "barraProgreso" in texto and "QProgressBar#barraProgreso" in texto:
    hechos.append("7. el estilo de la barra ya estaba")
else:
    unico(A_ESTILO, texto)
    texto = texto.replace(A_ESTILO, ESTILO, 1)
    hechos.append("7. estilo de la barra en las paletas clara y oscura")

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
print("=" * 74)
print(f"  respaldo: {RESPALDO}")
print("  el archivo sigue siendo Python valido")
print()
print("  Como se vera la linea de estado:")
print("    Analizando 340 de 2,000 (17 %)   ·   transcurrido 2 min 10 s   ·   falta ~10 min")
print("=" * 74)
