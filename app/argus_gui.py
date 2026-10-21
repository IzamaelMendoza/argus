import sys
import os
import time
import json
import random
import joblib
from datetime import datetime
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QTabWidget, QLabel,
    QHeaderView, QComboBox, QFileDialog, QMessageBox, QToolButton, QDialog
)
from PySide6.QtGui import QFont, QColor, QBrush, QPixmap, QIcon
from PySide6.QtCore import Qt, QThread, Signal, QSettings, QEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from itertools import product
from collections import Counter

# Barra de título propia (oscura). Pon False para usar la barra nativa del sistema.
BARRA_TITULO_PROPIA = True
MARGEN_VENTANA = 6          # grosor (px) de la zona para redimensionar la ventana

# ============================================================
#  1. FEATURIZACIÓN Y FRAGMENTACIÓN (Idéntico al Notebook)
# ============================================================
VENTANA = 500
K = 6   # se ajusta solo al cargar el modelo (ver cargar_modelo)
KMERS = ["".join(p) for p in product("ACGT", repeat=K)]
KMER_IDX = {km: i for i, km in enumerate(KMERS)}

def fragmentar(seq, ventana=VENTANA):
    """Divide la secuencia en bloques de 500 pb rellenando con '0'."""
    seq = seq.upper()
    L = len(seq)
    frags = []
    if L < ventana:
        frags.append(seq + "0" * (ventana - L))
    else:
        n = L // ventana
        for i in range(n):
            frags.append(seq[i*ventana:(i+1)*ventana])
        resto = L - n * ventana
        if resto > 0:
            frags.append(seq[n*ventana:] + "0" * (ventana - resto))
    return frags

def vector_kmer(seq):
    """Convierte un fragmento de 500 pb en un vector de 4**K frecuencias."""
    seq = seq.upper()
    counts = Counter(seq[i:i+K] for i in range(len(seq) - K + 1))
    # Suma SOLO k-mers válidos (ACGT), ignorando el relleno de '0'
    total = sum(c for km, c in counts.items() if km in KMER_IDX)
    vec = [0.0] * len(KMERS)
    if total > 0:
        for km, c in counts.items():
            if km in KMER_IDX:
                vec[KMER_IDX[km]] = c / total
    return vec

# ============================================================
#  2. LECTOR DE FASTA (Valida que sea FASTA, no FASTQ)
# ============================================================
def leer_fasta(ruta):
    with open(ruta, encoding="utf-8", errors="ignore") as f:
        primera = ""
        for linea in f:
            if linea.strip():
                primera = linea.strip()
                break
    if primera.startswith("@"):
        raise ValueError("El archivo parece FASTQ (empieza con '@'). "
                         "El sistema solo acepta FASTA (encabezados con '>').")
    if not primera.startswith(">"):
        raise ValueError("No parece un archivo FASTA válido (no empieza con '>').")

    seqs = []
    header, chunks = None, []
    with open(ruta, encoding="utf-8", errors="ignore") as f:
        for linea in f:
            linea = linea.rstrip("\n")
            if linea.startswith(">"):
                if header is not None:
                    seqs.append((header, "".join(chunks)))
                header = linea[1:].split()[0]
                chunks = []
            else:
                chunks.append(linea.strip())
        if header is not None:
            seqs.append((header, "".join(chunks)))
    return seqs

# ============================================================
#  3. CONEXIÓN REAL CON EL MODELO ENTRENADO (.joblib)
# ============================================================
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(RAIZ, "data", "models", "argus_rf_k6_id95.joblib")
UMBRAL_DEFECTO = 0.5

def _fijar_k(k):
    """Reconstruye la tabla de k-mers si el modelo usa otra K."""
    global K, KMERS, KMER_IDX
    K = k
    KMERS = ["".join(p) for p in product("ACGT", repeat=K)]
    KMER_IDX = {km: i for i, km in enumerate(KMERS)}

def cargar_modelo(ruta=RUTA_MODELO):
    """Carga el .joblib (dict de ARGus v2 o RF suelto) y ajusta K automáticamente."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo en:\n{ruta}\n\n"
            "Ejecuta 'git lfs pull' o revisa la ruta."
        )
    obj = joblib.load(ruta)
    if isinstance(obj, dict):
        rf = obj["modelo"]
        umbral = float(obj.get("umbral", UMBRAL_DEFECTO))
    else:
        rf, umbral = obj, UMBRAL_DEFECTO
    k = 1
    while 4 ** k < rf.n_features_in_:
        k += 1
    if 4 ** k != rf.n_features_in_:
        raise ValueError(f"El modelo espera {rf.n_features_in_} columnas; no corresponde a 4**K.")
    _fijar_k(k)
    try:
        rf.set_params(n_jobs=1)
    except Exception:
        pass
    return rf, umbral

def predecir_mag(secuencias, callback_progreso=None, esta_detenido=None, nombre_defecto=None):
    """
    secuencias: lista de (id, secuencia) leídas del MAG.
    Regla de gen: ARG si algún fragmento tiene probabilidad >= umbral.
    nombre_defecto: nombre del MAG cuando el encabezado no trae 'MAG|...'
                    (normalmente el nombre del archivo, para que el mapa de calor
                    tenga UNA columna por MAG y no una por secuencia).
    """
    modelo_etapa1, umbral = cargar_modelo()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_familias = os.path.join(base_dir, "modelo_familias.joblib")
    modelo_etapa2 = joblib.load(ruta_familias) if os.path.exists(ruta_familias) else None

    datos = []
    total = len(secuencias)

    for idx, (sid, seq) in enumerate(secuencias, start=1):
        if esta_detenido and esta_detenido():
            break

        frags = fragmentar(seq, ventana=VENTANA)
        X_frags = np.array([vector_kmer(f) for f in frags], dtype=np.float32)
        probs_frags = modelo_etapa1.predict_proba(X_frags)[:, 1]

        prob_maxima = float(np.max(probs_frags))
        es_resistente = prob_maxima >= umbral

        if "|" in sid:
            partes = sid.split("|")
            nombre_mag = partes[0]
            nombre_gen = partes[-1][:25]
        else:
            nombre_mag = nombre_defecto if nombre_defecto else sid[:15]
            nombre_gen = sid[:25]

        if es_resistente:
            if modelo_etapa2 is not None:
                i_max = int(np.argmax(probs_frags))
                familia_pred = str(modelo_etapa2.predict(X_frags[i_max:i_max + 1])[0])
            else:
                familia_pred = "Sin clasificar"
            presencia = "Sí"
        else:
            familia_pred, presencia = "no_resist", "No"

        datos.append({
            "mag": nombre_mag,
            "gen": nombre_gen,
            "presencia": presencia,
            "familia": familia_pred,
            "probabilidad": round(prob_maxima, 2),
        })

        if callback_progreso:
            callback_progreso(idx, total)

    return datos


# ============================================================
#  HILO EN SEGUNDO PLANO (Para no congelar la interfaz)
# ============================================================
class HiloPrediccion(QThread):
    progreso = Signal(int, int)       # (procesadas, total)
    finalizado = Signal(list, bool)   # (datos_resultado, fue_cancelado)
    error = Signal(str)

    def __init__(self, secuencias, nombre_mag=None):
        super().__init__()
        self.secuencias = secuencias
        self.nombre_mag = nombre_mag
        self._detener = False

    def solicitar_parada(self):
        self._detener = True

    def run(self):
        try:
            resultados = predecir_mag(
                self.secuencias,
                callback_progreso= lambda act, tot: self.progreso.emit(act, tot),
                esta_detenido= lambda: self._detener,
                nombre_defecto=self.nombre_mag
            )
            self.finalizado.emit(resultados, self._detener)
        except Exception as e:
            self.error.emit(str(e))


def generar_datos_falsos():
    familias = ["Beta-lactam", "Aminoglycoside", "Tetracycline", "Quinolone", "MLS"]
    mags = [f"MAG_{i:02d}" for i in range(1, 6)]
    datos = []
    random.seed(0)
    for mag in mags:
        for _ in range(random.randint(2, 5)):
            if random.random() < 0.85:
                familia = random.choice(familias)
                prob = round(random.uniform(0.6, 0.99), 2)
                datos.append({"mag": mag, "gen": f"{familia[:3].lower()}_{random.randint(1,50)}",
                              "presencia": "Sí", "familia": familia, "probabilidad": prob})
            else:
                datos.append({"mag": mag, "gen": "-", "presencia": "No",
                              "familia": "no_resist", "probabilidad": 0.0})
    return datos


# === HISTORIAL-INICIO ==========================================
# Guarda en disco los últimos MAX_HISTORIAL análisis (un .json por análisis).
# Solo se guardan los RESULTADOS, no las secuencias del archivo cargado.
MAX_HISTORIAL = 5
CARPETA_HISTORIAL = os.path.join(RAIZ, "historial_analisis")

def guardar_analisis(datos, meta, carpeta=CARPETA_HISTORIAL, maximo=MAX_HISTORIAL):
    """Guarda un análisis y borra los más antiguos para conservar solo `maximo`."""
    os.makedirs(carpeta, exist_ok=True)
    # Con microsegundos el nombre crece siempre, así el orden alfabético = orden cronológico
    sello = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ruta = os.path.join(carpeta, f"analisis_{sello}.json")
    tmp = ruta + ".tmp"                      # escritura atómica: no queda a medias
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "datos": datos}, f, ensure_ascii=False)
    os.replace(tmp, ruta)

    archivos = sorted(a for a in os.listdir(carpeta)
                      if a.startswith("analisis_") and a.endswith(".json"))
    for viejo in archivos[:-maximo]:
        try:
            os.remove(os.path.join(carpeta, viejo))
        except OSError:
            pass
    return ruta

def listar_historial(carpeta=CARPETA_HISTORIAL):
    """Devuelve [{'ruta':..., 'meta':...}, ...] del más reciente al más antiguo."""
    if not os.path.isdir(carpeta):
        return []
    archivos = sorted((a for a in os.listdir(carpeta)
                       if a.startswith("analisis_") and a.endswith(".json")), reverse=True)
    entradas = []
    for a in archivos:
        ruta = os.path.join(carpeta, a)
        try:
            with open(ruta, encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("meta"), dict) and isinstance(obj.get("datos"), list):
                entradas.append({"ruta": ruta, "meta": obj["meta"]})
        except Exception:
            continue                          # archivo dañado: se ignora
    return entradas

def cargar_analisis(ruta):
    with open(ruta, encoding="utf-8") as f:
        obj = json.load(f)
    return obj["meta"], obj["datos"]
# === HISTORIAL-FIN =============================================


# === TEMA-INICIO ===============================================
# Paletas de color (modo día / modo noche) y hoja de estilo de la interfaz.
from string import Template
from matplotlib.colors import LinearSegmentedColormap

PALETAS = {
    "claro": dict(
        # --- interfaz ---
        fondo="#F8F9FA", tarjeta="#FFFFFF", alt="#F8F9FA", texto_ui="#212529",
        titulo_ui="#2C3E50", suave_ui="#495057", borde="#DEE2E6",
        tab="#E9ECEF", tab_txt="#495057", acento="#0D6EFD",
        # --- gráficas ---
        lienzo="#FFFFFF", g_texto="#495057", g_titulo="#2C3E50", g_suave="#6C757D",
        g_borde="#DEE2E6", g_barra="#0D6EFD", g_si="#D9534F", g_no="#6C757D",
        cmap="Blues", celda_alta="white", celda_baja="#495057",
        scroll_bg="#F1F3F5", scroll_handle="#CED4DA", scroll_hover="#ADB5BD",
    ),
    "oscuro": dict(
        fondo="#15181C", tarjeta="#1F2329", alt="#252A31", texto_ui="#E9ECEF",
        titulo_ui="#F8F9FA", suave_ui="#ADB5BD", borde="#3A4047",
        tab="#2B3035", tab_txt="#ADB5BD", acento="#6EA8FE",
        lienzo="#1F2329", g_texto="#CED4DA", g_titulo="#F8F9FA", g_suave="#ADB5BD",
        g_borde="#495057", g_barra="#4D9DFF", g_si="#EA868F", g_no="#6C757D",
        cmap=LinearSegmentedColormap.from_list("argus_noche", ["#1F2329", "#6EA8FE"]),
        celda_alta="#0B1B33", celda_baja="#E9ECEF",
        scroll_bg="#1F2329", scroll_handle="#495057", scroll_hover="#6C757D",
    ),
}

_PLANTILLA_ESTILO = Template("""
* { font-family: 'Segoe UI', 'Roboto', 'Inter', sans-serif; }
QMainWindow, QDialog, QMessageBox { background-color: $fondo; }
QMessageBox QLabel { color: $texto_ui; }
QLabel#tituloPrincipal { font-size: 18pt; font-weight: bold; color: $titulo_ui; padding: 10px 0px; }
QLabel#etiquetaControl { font-weight: bold; color: $suave_ui; }

QPushButton { background-color: #0D6EFD; color: white; font-weight: bold; font-size: 10pt; border-radius: 6px; padding: 8px 15px; }
QPushButton:hover { background-color: #0B5ED7; }
QPushButton:disabled { background-color: #CED4DA; color: #6C757D; }

QPushButton#botonIniciar { background-color: #198754; }
QPushButton#botonIniciar:hover { background-color: #157347; }
QPushButton#botonIniciar:disabled { background-color: #A3CFBB; color: #F8F9FA; }

QPushButton#botonDetener { background-color: #DC3545; }
QPushButton#botonDetener:hover { background-color: #BB2D3B; }
QPushButton#botonDetener:disabled { background-color: #F1AEB5; color: #F8F9FA; }

QPushButton#botonSecundario { background-color: #198754; }
QPushButton#botonSecundario:hover { background-color: #157347; }

QPushButton#botonTema { background-color: $tab; color: $tab_txt; border: 1px solid $borde; }
QPushButton#botonTema:hover { background-color: $borde; }

QComboBox { padding: 5px; font-size: 10pt; border: 1px solid $borde; border-radius: 4px; background-color: $tarjeta; color: $texto_ui; }
QComboBox QAbstractItemView { background-color: $tarjeta; color: $texto_ui; selection-background-color: $acento; selection-color: #FFFFFF; }
QTabWidget::pane { border: 1px solid $borde; background: $tarjeta; border-radius: 4px; }
QTabBar::tab { background: $tab; color: $tab_txt; padding: 10px 20px; font-size: 10pt; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; border: 1px solid transparent; }
QTabBar::tab:selected { background: $tarjeta; color: $acento; font-weight: bold; border: 1px solid $borde; border-bottom: 3px solid $acento; }
QTableView, QTableWidget { background-color: $tarjeta; alternate-background-color: $alt; color: $texto_ui; border: none; gridline-color: $borde; font-size: 10pt; selection-background-color: $acento; selection-color: #FFFFFF; }
QHeaderView::section { background-color: $tab; color: $texto_ui; font-weight: bold; padding: 8px; border: none; border-right: 1px solid $borde; border-bottom: 2px solid $borde; }
QHeaderView { background-color: $tab; }
QTableCornerButton::section { background-color: $tab; border: none; }

QScrollBar:vertical { background: $scroll_bg; width: 12px; margin: 0; border: none; }
QScrollBar::handle:vertical { background: $scroll_handle; min-height: 30px; border-radius: 4px; margin: 2px; }
QScrollBar::handle:vertical:hover { background: $scroll_hover; }
QScrollBar:horizontal { background: $scroll_bg; height: 12px; margin: 0; border: none; }
QScrollBar::handle:horizontal { background: $scroll_handle; min-width: 30px; border-radius: 4px; margin: 2px; }
QScrollBar::handle:horizontal:hover { background: $scroll_hover; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; background: none; border: none; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
QAbstractScrollArea::corner { background: $scroll_bg; }

QDialog { background-color: $fondo; border: 1px solid $borde; }
QDialog QLabel { color: $texto_ui; }
QLineEdit { background-color: $tarjeta; color: $texto_ui; border: 1px solid $borde; border-radius: 4px; padding: 5px; selection-background-color: $acento; selection-color: #FFFFFF; }
QTreeView, QListView { background-color: $tarjeta; alternate-background-color: $alt; color: $texto_ui; border: 1px solid $borde; selection-background-color: $acento; selection-color: #FFFFFF; }

QWidget#barraTitulo { background-color: $fondo; }
QLabel#textoBarraTitulo { color: $suave_ui; font-size: 9pt; }
QToolButton#botonVentana, QToolButton#botonCerrarVentana { background: transparent; color: $texto_ui; border: none; border-radius: 0px; min-width: 44px; min-height: 30px; font-size: 11pt; }
QToolButton#botonVentana:hover { background-color: $tab; }
QToolButton#botonCerrarVentana:hover { background-color: #E81123; color: white; }
""")

def generar_estilo(p):
    return _PLANTILLA_ESTILO.substitute(p)
# === TEMA-FIN ==================================================


# === GRAFICAS-INICIO ===========================================
# Funciones de dibujo independientes de la ventana. Reciben la figura,
# los datos, las opciones y la paleta; no tocan ningún widget.
SIN_FAMILIA = "Sin clasificar"
TOP_OPCIONES = [10, 20, 30, 50]
MAX_COLUMNAS_MAPA = 30      # máximo de MAGs (columnas) en el mapa de calor

def _corto(texto, n=32):
    texto = str(texto)
    return texto if len(texto) <= n else texto[:n - 1] + "…"

def _sin_familia(datos):
    """True si hay detecciones pero ninguna tiene familia asignada (Etapa 2 sin modelo)."""
    fams = {d["familia"] for d in datos if d["presencia"] == "Sí"}
    return bool(fams) and fams <= {SIN_FAMILIA}

def _preparar(fig, p):
    fig.clear()
    fig.patch.set_facecolor(p["lienzo"])

def _mensaje(fig, texto, p):
    _preparar(fig, p)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, texto, ha="center", va="center", fontsize=11,
            color=p["g_suave"], transform=ax.transAxes)

def _estilo_ejes(ax, p):
    ax.set_facecolor(p["lienzo"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(p["g_borde"])
    ax.spines["bottom"].set_color(p["g_borde"])
    ax.tick_params(colors=p["g_texto"])

def dibujar_barras(fig, datos, modo, top_n, p=None):
    """modo 0 = por familia, modo 1 = por gen específico (solo los top_n más frecuentes)."""
    p = p or PALETAS["claro"]
    _preparar(fig, p)
    positivos = [d for d in datos if d["presencia"] == "Sí"]
    if not positivos:
        _mensaje(fig, "No se detectaron secuencias con resistencia.", p)
        return

    ax = fig.add_subplot(111)

    # Sin modelo de familias: en vez de una sola barra "Sin clasificar",
    # mostramos el resultado de la Etapa 1 (con / sin resistencia).
    if modo == 0 and _sin_familia(datos):
        n_si = len(positivos)
        n_no = len(datos) - n_si
        barras = ax.bar(["Con resistencia", "Sin resistencia"], [n_si, n_no],
                        color=[p["g_si"], p["g_no"]], width=0.5)
        for b, v in zip(barras, [n_si, n_no]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v}", ha="center", va="bottom",
                    fontsize=10, color=p["g_texto"])
        ax.set_ylabel("Número de secuencias", fontsize=10, color=p["g_texto"])
        ax.set_title("Resultado de la detección\n(la clasificación por familia aún no está disponible)",
                     fontsize=12, fontweight="bold", color=p["g_titulo"], pad=15)
        ax.set_ylim(0, max(n_si, n_no) * 1.12)
        _estilo_ejes(ax, p)
        fig.tight_layout()
        return

    llave = "familia" if modo == 0 else "gen"
    conteo = Counter(d[llave] for d in positivos if d[llave] != "-")
    distintos = len(conteo)
    top = conteo.most_common(top_n)
    nombres = [_corto(k) for k, _ in top]
    valores = [v for _, v in top]
    y = np.arange(len(top))

    ax.barh(y, valores, color=p["g_barra"], edgecolor="none", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(nombres, fontsize=8)
    ax.invert_yaxis()                       # el más frecuente arriba
    for yi, v in zip(y, valores):
        ax.text(v, yi, f" {v}", va="center", ha="left", fontsize=8, color=p["g_texto"])
    ax.set_xlim(0, max(valores) * 1.12)
    ax.set_xlabel("Número de detecciones", fontsize=10, color=p["g_texto"])
    if modo == 0:
        titulo = f"Genes de resistencia por familia (top {len(top)} de {distintos})"
    else:
        titulo = f"Frecuencia por gen específico (top {len(top)} de {distintos} genes distintos)"
    ax.set_title(titulo, fontsize=12, fontweight="bold", color=p["g_titulo"], pad=15)
    _estilo_ejes(ax, p)
    fig.tight_layout()

def dibujar_mapa(fig, datos, modo, top_n, p=None):
    """modo 0 = familias vs MAGs, modo 1 = genes vs MAGs (solo los top_n genes más frecuentes)."""
    p = p or PALETAS["claro"]
    _preparar(fig, p)
    positivos = [d for d in datos if d["presencia"] == "Sí"]
    if not positivos:
        _mensaje(fig, "No se detectaron secuencias con resistencia.", p)
        return
    if modo == 0 and _sin_familia(datos):
        _mensaje(fig, "El mapa por familia necesita el modelo de familias (Etapa 2),\n"
                      "que aún no está entrenado.\n\n"
                      "Usa la opción 'Genes Específicos vs MAGs'.", p)
        return

    llave = "familia" if modo == 0 else "gen"
    pos = [d for d in positivos if d[llave] != "-"]
    if not pos:
        _mensaje(fig, "No hay elementos para graficar.", p)
        return

    cuenta = Counter(d[llave] for d in pos)
    pmax = {}
    for d in pos:
        pmax[d[llave]] = max(pmax.get(d[llave], 0.0), d["probabilidad"])
    todos = sorted(cuenta, key=lambda k: (-cuenta[k], -pmax[k], k))
    elementos = todos[:top_n]
    sel = set(elementos)

    cuenta_mag = Counter(d["mag"] for d in pos if d[llave] in sel)
    todos_mags = sorted(cuenta_mag, key=lambda m: (-cuenta_mag[m], m))
    mags = sorted(todos_mags[:MAX_COLUMNAS_MAPA])

    idx_e = {e: i for i, e in enumerate(elementos)}
    idx_m = {m: j for j, m in enumerate(mags)}
    matriz = np.zeros((len(elementos), len(mags)))
    for d in pos:
        i = idx_e.get(d[llave])
        j = idx_m.get(d["mag"])
        if i is None or j is None:
            continue
        matriz[i, j] = max(matriz[i, j], d["probabilidad"])

    ax = fig.add_subplot(111)
    ax.set_facecolor(p["lienzo"])
    im = ax.imshow(matriz, cmap=p["cmap"], aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(mags)))
    ax.set_xticklabels([_corto(m, 18) for m in mags], rotation=45, ha="right",
                       fontsize=8, color=p["g_texto"])
    ax.set_yticks(range(len(elementos)))
    ax.set_yticklabels([_corto(e) for e in elementos], fontsize=8, color=p["g_texto"])
    for sp in ax.spines.values():
        sp.set_color(p["g_borde"])

    # Valor dentro de cada celda cuando la matriz es pequeña
    if matriz.size <= 200:
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                v = matriz[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color=p["celda_alta"] if v > 0.6 else p["celda_baja"])

    nombre = "Familia" if modo == 0 else "Gen"
    ax.set_title(f"Probabilidad por {nombre} (top {len(elementos)} de {len(todos)})",
                 fontsize=12, fontweight="bold", color=p["g_titulo"], pad=15)
    if len(todos_mags) > len(mags):
        ax.set_xlabel(f"Se muestran {len(mags)} de {len(todos_mags)} MAGs (los de más detecciones)",
                      fontsize=8, color=p["g_suave"])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Probabilidad de predicción", color=p["g_texto"])
    cbar.ax.tick_params(colors=p["g_texto"])
    cbar.outline.set_edgecolor(p["g_borde"])
    fig.tight_layout()
# === GRAFICAS-FIN ==============================================


class BarraTitulo(QWidget):
    """Barra de título propia: respeta el modo día/noche (la del sistema no se puede colorear)."""
    def __init__(self, ventana, simple=False):
        super().__init__(ventana)
        self._ventana = ventana
        self._simple = simple      # True = diálogo: solo botón cerrar
        self._arrastre = None
        self.setObjectName("barraTitulo")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedHeight(34)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 0, 0)
        lay.setSpacing(0)
        self.texto = QLabel(ventana.windowTitle())
        self.texto.setObjectName("textoBarraTitulo")
        lay.addWidget(self.texto)
        lay.addStretch()
        self.btn_min = self._boton("—", "botonVentana", ventana.showMinimized)
        self.btn_max = self._boton("☐", "botonVentana", self.alternar_maximizado)
        self.btn_cerrar = self._boton("✕", "botonCerrarVentana", ventana.close)
        if simple:
            self.btn_min.hide()
            self.btn_max.hide()
        for b in (self.btn_min, self.btn_max, self.btn_cerrar):
            lay.addWidget(b)

    @staticmethod
    def _boton(texto, nombre, accion):
        b = QToolButton()
        b.setText(texto)
        b.setObjectName(nombre)
        b.setFocusPolicy(Qt.NoFocus)
        b.clicked.connect(accion)
        return b

    def alternar_maximizado(self):
        if self._ventana.isMaximized():
            self._ventana.showNormal()
        else:
            self._ventana.showMaximized()

    def actualizar_icono(self, maximizada):
        self.btn_max.setText("❐" if maximizada else "☐")

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            h = self._ventana.windowHandle()
            if h is not None and h.startSystemMove():
                return
            # Respaldo si el sistema no permite mover desde la app
            self._arrastre = e.globalPosition().toPoint() - self._ventana.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self._arrastre is not None and (e.buttons() & Qt.LeftButton) and not self._ventana.isMaximized():
            self._ventana.move(e.globalPosition().toPoint() - self._arrastre)

    def mouseReleaseEvent(self, e):
        self._arrastre = None

    def mouseDoubleClickEvent(self, e):
        if not self._simple and e.button() == Qt.LeftButton:
            self.alternar_maximizado()


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        nombre_sistema = "ARGus (Antibiotic Resistance Genome understanding system)"
        self.setWindowTitle(nombre_sistema)
        self.resize(1100, 700)
        self.tema = self._leer_tema()     # "claro" u "oscuro" (se recuerda entre sesiones)
        if BARRA_TITULO_PROPIA:
            self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
            self.setMouseTracking(True)
            self.setContentsMargins(MARGEN_VENTANA, MARGEN_VENTANA, MARGEN_VENTANA, MARGEN_VENTANA)
            self.barra_titulo = BarraTitulo(self)
            self.setMenuWidget(self.barra_titulo)

        self.datos = generar_datos_falsos()
        self.secuencias_cargadas = []
        self.ruta_actual = ""
        self.hilo = None

        self.setWindowIcon(QIcon("logo.png"))

        self.setStyleSheet(generar_estilo(PALETAS[self.tema]))

        contenedor = QWidget()
        layout = QVBoxLayout(contenedor)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(15)

        # ---------- CABECERA: logo y titulo a la izq, logos institucionales a la der ----------
        cabecera = QHBoxLayout()

        logo_principal = self._cargar_logo("../ARGus.png", "ARGus")
        cabecera.addWidget(logo_principal)

        titulo = QLabel(f"  {nombre_sistema}")
        titulo.setObjectName("tituloPrincipal")
        titulo.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        cabecera.addWidget(titulo)

        cabecera.addStretch()

        # Interruptor de modo día / modo noche
        self.boton_tema = QPushButton()
        self.boton_tema.setObjectName("botonTema")
        self.boton_tema.setCheckable(True)
        self.boton_tema.setChecked(self.tema == "oscuro")
        self.boton_tema.setCursor(Qt.PointingHandCursor)
        self.boton_tema.toggled.connect(self.cambiar_tema)
        self._texto_boton_tema()
        cabecera.addWidget(self.boton_tema)

        cabecera.addWidget(self._cargar_logo("../logo_escuela.png", "TecNM"))
        cabecera.addWidget(self._cargar_logo("../cinvestav.png", "Cinvestav"))

        layout.addLayout(cabecera)

        # ---------- BARRA DE CONTROLES (Cargar, Iniciar, Detener) ----------
        barra = QHBoxLayout()

        self.boton_cargar = QPushButton("📂 Cargar archivo MAG (.fasta)")
        self.boton_cargar.setCursor(Qt.PointingHandCursor)
        self.boton_cargar.clicked.connect(self.cargar_mag)
        barra.addWidget(self.boton_cargar)

        self.boton_iniciar = QPushButton("Iniciar análisis")
        self.boton_iniciar.setObjectName("botonIniciar")
        self.boton_iniciar.setCursor(Qt.PointingHandCursor)
        self.boton_iniciar.setEnabled(False)  # Deshabilitado hasta cargar un archivo
        self.boton_iniciar.clicked.connect(self.iniciar_analisis)
        barra.addWidget(self.boton_iniciar)

        self.boton_detener = QPushButton("Detener")
        self.boton_detener.setObjectName("botonDetener")
        self.boton_detener.setCursor(Qt.PointingHandCursor)
        self.boton_detener.setEnabled(False)  # Solo se habilita durante la ejecución
        self.boton_detener.clicked.connect(self.detener_analisis)
        barra.addWidget(self.boton_detener)

        barra.addStretch()

        self.label_estado = QLabel("Sin archivo cargado — mostrando datos de ejemplo.")
        self.label_estado.setStyleSheet("color: #6C757D; font-style: italic;")
        barra.addWidget(self.label_estado)

        layout.addLayout(barra)

        # ---------- PESTAÑAS ----------
        self.pestañas = QTabWidget()
        self.pestañas.addTab(self.crear_pestana_tabla(), "Tabla de resultados")
        self.pestañas.addTab(self.crear_pestana_barras(), "Gráfica de barras")
        self.pestañas.addTab(self.crear_pestana_mapa(), "Mapa de calor")
        self.pestañas.addTab(self.crear_pestana_historial(), "Historial")
        layout.addWidget(self.pestañas)

        self.setCentralWidget(contenedor)

        # Al abrir la app, mostrar el último análisis guardado (si existe)
        self._cargar_ultimo_al_inicio()

    def _cargar_logo(self, ruta, alt):
        """Carga un logo escalado; si no existe, muestra texto de respaldo."""
        base = os.path.dirname(os.path.abspath(__file__))
        ruta_abs = os.path.join(base, ruta)

        lbl = QLabel()
        pix = QPixmap(ruta_abs)
        if not pix.isNull():
            lbl.setPixmap(pix.scaledToHeight(55, Qt.SmoothTransformation))
        else:
            print(f"[LOGO] No se encontró: {ruta_abs}")
            lbl.setText(alt)
            lbl.setStyleSheet("color:#ADB5BD; border:1px dashed #CED4DA; padding:12px; font-size:9pt;")
            lbl.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
            lbl.setContentsMargins(10, 0, 0, 0)
        return lbl

    # ---------- DIÁLOGOS CON EL MISMO ESTILO (barra oscura, sin marco del sistema) ----------
    def _vestir_dialogo(self, dlg):
        if BARRA_TITULO_PROPIA:
            dlg.setWindowFlag(Qt.FramelessWindowHint, True)
            dlg.layout().setMenuBar(BarraTitulo(dlg, simple=True))

    def _crear_msg(self, icono, titulo, texto, botones=QMessageBox.Ok):
        m = QMessageBox(self)
        m.setIcon(icono)
        m.setWindowTitle(titulo)
        m.setText(texto)
        m.setStandardButtons(botones)
        self._vestir_dialogo(m)
        return m

    def _msg(self, icono, titulo, texto):
        self._crear_msg(icono, titulo, texto).exec()

    def _confirmar(self, titulo, texto):
        m = self._crear_msg(QMessageBox.Question, titulo, texto, QMessageBox.Yes | QMessageBox.No)
        m.setDefaultButton(QMessageBox.No)
        return m.exec() == QMessageBox.Yes

    def _crear_dialogo_archivo(self, titulo, filtros, guardar):
        dlg = QFileDialog(self, titulo, "", filtros)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        if guardar:
            dlg.setAcceptMode(QFileDialog.AcceptSave)
            dlg.setOption(QFileDialog.DontConfirmOverwrite, True)   # lo confirmamos nosotros
        else:
            dlg.setAcceptMode(QFileDialog.AcceptOpen)
            dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setSizeGripEnabled(True)
        self._vestir_dialogo(dlg)
        return dlg

    def _dialogo_archivo(self, titulo, filtros, guardar):
        """Devuelve (ruta, filtro_elegido); ruta = '' si se cancela."""
        dlg = self._crear_dialogo_archivo(titulo, filtros, guardar)
        if dlg.exec() == QDialog.Accepted and dlg.selectedFiles():
            return dlg.selectedFiles()[0], dlg.selectedNameFilter()
        return "", ""

    def _ok_sobrescribir(self, ruta):
        if os.path.exists(ruta):
            return self._confirmar("Sobrescribir", f"El archivo ya existe:\n{ruta}\n\n¿Deseas reemplazarlo?")
        return True

    # ---------- REDIMENSIONAR VENTANA SIN MARCO ----------
    def _bordes(self, pos):
        if not BARRA_TITULO_PROPIA or self.isMaximized() or self.isFullScreen():
            return []
        m = MARGEN_VENTANA
        b = []
        if pos.x() < m:
            b.append(Qt.Edge.LeftEdge)
        if pos.x() >= self.width() - m:
            b.append(Qt.Edge.RightEdge)
        if pos.y() < m:
            b.append(Qt.Edge.TopEdge)
        if pos.y() >= self.height() - m:
            b.append(Qt.Edge.BottomEdge)
        return b

    @staticmethod
    def _cursor_para(bordes):
        izq, der = Qt.Edge.LeftEdge in bordes, Qt.Edge.RightEdge in bordes
        arr, aba = Qt.Edge.TopEdge in bordes, Qt.Edge.BottomEdge in bordes
        if (izq and arr) or (der and aba):
            return Qt.SizeFDiagCursor
        if (der and arr) or (izq and aba):
            return Qt.SizeBDiagCursor
        if izq or der:
            return Qt.SizeHorCursor
        return Qt.SizeVerCursor

    def mouseMoveEvent(self, e):
        bordes = self._bordes(e.position().toPoint())
        if bordes:
            self.setCursor(self._cursor_para(bordes))
        else:
            self.unsetCursor()
        super().mouseMoveEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            bordes = self._bordes(e.position().toPoint())
            h = self.windowHandle()
            if bordes and h is not None:
                flags = bordes[0]
                for b in bordes[1:]:
                    flags |= b
                h.startSystemResize(flags)
                return
        super().mousePressEvent(e)

    def changeEvent(self, e):
        if BARRA_TITULO_PROPIA and e.type() == QEvent.WindowStateChange and hasattr(self, "barra_titulo"):
            maxi = self.isMaximized() or self.isFullScreen()
            m = 0 if maxi else MARGEN_VENTANA
            self.setContentsMargins(m, m, m, m)
            self.barra_titulo.actualizar_icono(maxi)
        super().changeEvent(e)

    # ---------- MODO DÍA / NOCHE ----------
    @staticmethod
    def _leer_tema():
        try:
            v = QSettings("ARGus", "ARGusGUI").value("tema", "claro")
            return v if v in PALETAS else "claro"
        except Exception:
            return "claro"

    def _texto_boton_tema(self):
        self.boton_tema.setText("☀️ Modo día" if self.tema == "oscuro" else "🌙 Modo noche")

    def cambiar_tema(self, oscuro):
        self.tema = "oscuro" if oscuro else "claro"
        self.setStyleSheet(generar_estilo(PALETAS[self.tema]))
        self._texto_boton_tema()
        self.actualizar_grafica_barras()
        self.actualizar_mapa_calor()
        try:
            QSettings("ARGus", "ARGusGUI").setValue("tema", self.tema)
        except Exception:
            pass

    @staticmethod
    def _crear_combo_top():
        """Combo 'Top N' (por defecto 20)."""
        combo = QComboBox()
        combo.addItems([f"Top {n}" for n in TOP_OPCIONES])
        combo.setCurrentIndex(TOP_OPCIONES.index(20))
        return combo

    @staticmethod
    def _valor_top(combo):
        return TOP_OPCIONES[combo.currentIndex()]

    # ---------- PASO 1: CARGAR Y VALIDAR EL ARCHIVO (Sin ejecutar aún) ----------
    def cargar_mag(self):
        ruta, _ = self._dialogo_archivo(
            "Seleccionar archivo MAG",
            "FASTA (*.fasta *.fa *.fna *.fas);;Todos los archivos (*)", guardar=False)
        if not ruta:
            return
        try:
            secuencias = leer_fasta(ruta)
        except ValueError as e:
            self._msg(QMessageBox.Warning, "Archivo no válido", str(e))
            return
        except Exception as e:
            self._msg(QMessageBox.Critical, "Error", f"No se pudo leer el archivo:\n{e}")
            return

        if not secuencias:
            self._msg(QMessageBox.Warning, "Archivo vacío", "No se encontraron secuencias en el archivo.")
            return

        # Guardamos las secuencias listas y habilitamos el botón de "Iniciar"
        self.secuencias_cargadas = secuencias
        self.ruta_actual = ruta
        self.boton_iniciar.setEnabled(True)
        self.boton_detener.setEnabled(False)

        n = len(secuencias)
        nombre_arch = os.path.basename(ruta)
        self.label_estado.setText(f"Archivo listo: {nombre_arch} \nSecuencias: {n}")
        self.label_estado.setStyleSheet("color: #0D6EFD; font-weight: bold;")

    # ---------- PASO 2: INICIAR EL ANÁLISIS ----------
    def iniciar_analisis(self):
        if not self.secuencias_cargadas:
            return

        # Ajustar el estado de los botones
        self.boton_cargar.setEnabled(False)
        self.boton_iniciar.setEnabled(False)
        self.boton_detener.setEnabled(True)

        self.label_estado.setText("⏳ Iniciando predicción...")
        self.label_estado.setStyleSheet("color: #FD7E14; font-weight: bold;")

        # Crear y lanzar el hilo de predicción
        nombre_mag = os.path.splitext(os.path.basename(self.ruta_actual))[0]
        self.hilo = HiloPrediccion(self.secuencias_cargadas, nombre_mag=nombre_mag)
        self.hilo.progreso.connect(self.actualizar_progreso)
        self.hilo.finalizado.connect(self.analisis_terminado)
        self.hilo.error.connect(self.analisis_error)
        self.hilo.start()

    # ---------- PASO 3: DETENER EL ANÁLISIS SI SE REQUIERE ----------
    def detener_analisis(self):
        if self.hilo and self.hilo.isRunning():
            self.label_estado.setText("🛑 Deteniendo el proceso...")
            self.label_estado.setStyleSheet("color: #DC3545; font-weight: bold;")
            self.boton_detener.setEnabled(False)
            self.hilo.solicitar_parada()

    def actualizar_progreso(self, actual, total):
        porcentaje = int((actual / total) * 100)
        self.label_estado.setText(f"⚙️ Analizando: {actual} de {total} secuencias ({porcentaje}%)...")

    def analisis_terminado(self, resultados, fue_cancelado):
        # Restaurar botones
        self.boton_cargar.setEnabled(True)
        self.boton_iniciar.setEnabled(True)
        self.boton_detener.setEnabled(False)

        # Si hay resultados (completos o parciales), actualizamos las tablas y gráficas
        if resultados:
            self.datos = resultados
            self.refrescar_vistas()

        nombre_arch = os.path.basename(self.ruta_actual)
        n_procesadas = len(resultados)
        n_total = len(self.secuencias_cargadas)
        n_res = sum(1 for d in resultados if d["presencia"] == "Sí")

        # Guardar en el historial (también los análisis detenidos, marcados como parciales)
        if resultados:
            try:
                guardar_analisis(resultados, {
                    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "archivo": nombre_arch,
                    "n_total": n_total,
                    "n_procesadas": n_procesadas,
                    "n_resistentes": n_res,
                    "cancelado": bool(fue_cancelado),
                })
                self.refrescar_historial()
            except Exception as e:
                print(f"[HISTORIAL] No se pudo guardar el análisis: {e}")

        if fue_cancelado:
            self.label_estado.setText(
                f"⚠️ Detenido ({nombre_arch}) — Se procesaron {n_procesadas}/{n_total} secuencias ({n_res} con resistencia)."
            )
            self.label_estado.setStyleSheet("color: #DC3545; font-weight: bold;")
        else:
            self.label_estado.setText(
                f"Completado ({nombre_arch}) \n{n_total} secuencias analizadas, {n_res} con resistencia."
            )
            self.label_estado.setStyleSheet("color: #198754; font-weight: bold;")

    def analisis_error(self, mensaje_error):
        self.boton_cargar.setEnabled(True)
        self.boton_iniciar.setEnabled(True)
        self.boton_detener.setEnabled(False)
        self.label_estado.setText("❌ Error durante el análisis.")
        self.label_estado.setStyleSheet("color: #DC3545; font-weight: bold;")
        self._msg(QMessageBox.Critical, "Error en la predicción", f"Ocurrió un fallo al ejecutar el modelo:\n{mensaje_error}")

    def refrescar_vistas(self):
        self._llenar_tabla()
        self.actualizar_grafica_barras()
        self.actualizar_mapa_calor()

    # ---------- EXPORTACION ----------
    def exportar_csv(self):
        ruta, _ = self._dialogo_archivo("Guardar Resultados", "CSV files (*.csv)", guardar=True)
        if ruta:
            if not ruta.lower().endswith(".csv"):
                ruta += ".csv"
            if not self._ok_sobrescribir(ruta):
                return
            try:
                df = pd.DataFrame(self.datos)
                df.columns = ["MAG", "Gen", "Presencia", "Familia", "Probabilidad"]
                df.to_csv(ruta, index=False)
                self._msg(QMessageBox.Information, "Éxito", "Tabla guardada correctamente.")
            except Exception as e:
                self._msg(QMessageBox.Critical, "Error", f"No se pudo guardar el archivo:\n{str(e)}")

    def exportar_grafica(self, cual):
        """cual = 'barras' o 'mapa'. Siempre se exporta con fondo claro (para informes/papers),
        aunque la app esté en modo noche."""
        ruta, filtro = self._dialogo_archivo("Guardar Gráfica", "TIFF files (*.tiff);;PDF files (*.pdf)", guardar=True)
        if ruta:
            if "TIFF" in filtro and not ruta.lower().endswith(('.tif', '.tiff')):
                ruta += ".tiff"
            elif "PDF" in filtro and not ruta.lower().endswith('.pdf'):
                ruta += ".pdf"
            if not self._ok_sobrescribir(ruta):
                return
            try:
                fig = Figure(figsize=(10, 6), facecolor="white")
                if cual == "barras":
                    dibujar_barras(fig, self.datos, self.combo_barras.currentIndex(),
                                   self._valor_top(self.combo_top_barras), PALETAS["claro"])
                else:
                    dibujar_mapa(fig, self.datos, self.combo_mapa.currentIndex(),
                                 self._valor_top(self.combo_top_mapa), PALETAS["claro"])
                fig.savefig(ruta, dpi=300, bbox_inches='tight')
                self._msg(QMessageBox.Information, "Éxito", f"Gráfica guardada en alta resolución (300 DPI):\n{ruta}")
            except Exception as e:
                self._msg(QMessageBox.Critical, "Error", f"No se pudo guardar la gráfica:\n{str(e)}")

    # ---------- PESTAÑA HISTORIAL ----------
    def crear_pestana_historial(self):
        pestana = QWidget()
        layout = QVBoxLayout(pestana)

        nota = QLabel(f"Se guardan automáticamente los últimos {MAX_HISTORIAL} análisis. "
                      "Doble clic sobre uno para abrirlo.")
        nota.setStyleSheet("color: #6C757D; font-style: italic;")
        layout.addWidget(nota)

        columnas = ["Fecha", "Archivo", "Secuencias", "Con resistencia", "Estado"]
        self.tabla_historial = QTableWidget(0, len(columnas))
        self.tabla_historial.setHorizontalHeaderLabels(columnas)
        self.tabla_historial.setAlternatingRowColors(True)
        self.tabla_historial.verticalHeader().setVisible(False)
        self.tabla_historial.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_historial.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tabla_historial.setSelectionBehavior(QTableWidget.SelectRows)
        self.tabla_historial.setSelectionMode(QTableWidget.SingleSelection)
        self.tabla_historial.setShowGrid(False)
        self.tabla_historial.cellDoubleClicked.connect(lambda *_: self.abrir_seleccion_historial())
        layout.addWidget(self.tabla_historial)

        botones = QHBoxLayout()
        boton_abrir = QPushButton("Abrir seleccionado")
        boton_abrir.setObjectName("botonSecundario")
        boton_abrir.setCursor(Qt.PointingHandCursor)
        boton_abrir.clicked.connect(self.abrir_seleccion_historial)
        boton_borrar = QPushButton("Eliminar seleccionado")
        boton_borrar.setObjectName("botonDetener")
        boton_borrar.setCursor(Qt.PointingHandCursor)
        boton_borrar.clicked.connect(self.eliminar_seleccion_historial)
        botones.addStretch()
        botones.addWidget(boton_abrir)
        botones.addWidget(boton_borrar)
        layout.addLayout(botones)

        self._entradas_historial = []
        self.refrescar_historial()
        return pestana

    def refrescar_historial(self):
        self._entradas_historial = listar_historial(CARPETA_HISTORIAL)
        t = self.tabla_historial
        t.setRowCount(len(self._entradas_historial))
        for fila, e in enumerate(self._entradas_historial):
            m = e["meta"]
            procesadas = m.get("n_procesadas", 0)
            total = m.get("n_total", procesadas)
            if m.get("cancelado"):
                secuencias, estado = f"{procesadas}/{total}", "Parcial (detenido)"
            else:
                secuencias, estado = f"{procesadas}", "Completo"
            valores = [m.get("fecha", ""), m.get("archivo", ""), secuencias,
                       str(m.get("n_resistentes", 0)), estado]
            for col, v in enumerate(valores):
                item = QTableWidgetItem(v)
                if col >= 2:
                    item.setTextAlignment(Qt.AlignCenter)
                t.setItem(fila, col, item)

    def _mostrar_entrada_historial(self, entrada, cambiar_pestana=True):
        try:
            meta, datos = cargar_analisis(entrada["ruta"])
        except Exception as e:
            self._msg(QMessageBox.Critical, "Historial", f"No se pudo abrir el análisis guardado:\n{e}")
            return
        self.datos = datos
        self.refrescar_vistas()
        self.label_estado.setText(
            f"🕘 Historial: {meta.get('archivo', '?')}"
            f"\n{meta.get('n_procesadas', len(datos))} secuencias, \n{meta.get('n_resistentes', '?')} con resistencia. \n({meta.get('fecha', '')})"
        )
        self.label_estado.setStyleSheet("color: #6C757D; font-weight: bold;")
        if cambiar_pestana:
            self.pestañas.setCurrentIndex(0)

    def _cargar_ultimo_al_inicio(self):
        if self._entradas_historial:
            self._mostrar_entrada_historial(self._entradas_historial[0], cambiar_pestana=False)

    def _entrada_seleccionada(self):
        fila = self.tabla_historial.currentRow()
        if 0 <= fila < len(self._entradas_historial):
            return self._entradas_historial[fila]
        self._msg(QMessageBox.Information, "Historial", "Selecciona un análisis de la lista.")
        return None

    def abrir_seleccion_historial(self):
        if self.hilo and self.hilo.isRunning():
            self._msg(QMessageBox.Information, "Historial",
                                    "Hay un análisis en curso. Espera a que termine o deténlo.")
            return
        entrada = self._entrada_seleccionada()
        if entrada:
            self._mostrar_entrada_historial(entrada)

    def eliminar_seleccion_historial(self):
        entrada = self._entrada_seleccionada()
        if not entrada:
            return
        if not self._confirmar("Eliminar", "¿Eliminar este análisis del historial?"):
            return
        try:
            os.remove(entrada["ruta"])
        except OSError as e:
            self._msg(QMessageBox.Critical, "Error", f"No se pudo eliminar:\n{e}")
        self.refrescar_historial()

    # ---------- PESTAÑA TABLA ----------
    def crear_pestana_tabla(self):
        pestana = QWidget()
        layout = QVBoxLayout(pestana)
        barra_controles = QHBoxLayout()
        barra_controles.addStretch()
        boton_exportar = QPushButton("Descargar CSV")
        boton_exportar.setObjectName("botonSecundario")
        boton_exportar.clicked.connect(self.exportar_csv)
        boton_exportar.setCursor(Qt.PointingHandCursor)
        barra_controles.addWidget(boton_exportar)
        layout.addLayout(barra_controles)

        columnas = ["MAG", "Gen", "Presencia de resistencia", "Familia", "Probabilidad"]
        self.tabla = QTableWidget(0, len(columnas))
        self.tabla.setHorizontalHeaderLabels(columnas)
        self.tabla.setAlternatingRowColors(True)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.setSelectionBehavior(QTableWidget.SelectRows)
        self.tabla.setShowGrid(False)
        layout.addWidget(self.tabla)
        self._llenar_tabla()
        return pestana

    def _llenar_tabla(self):
        self.tabla.setRowCount(len(self.datos))
        for fila, d in enumerate(self.datos):
            self.tabla.setItem(fila, 0, QTableWidgetItem(d["mag"]))
            self.tabla.setItem(fila, 1, QTableWidgetItem(d["gen"]))
            texto_presencia = f"🔴 {d['presencia']}" if d["presencia"] == "Sí" else f"🟢 {d['presencia']}"
            item_presencia = QTableWidgetItem(texto_presencia)
            item_presencia.setTextAlignment(Qt.AlignCenter)
            item_presencia.setForeground(QBrush(QColor("#D9534F")) if d["presencia"] == "Sí" else QBrush(QColor("#6C757D")))
            self.tabla.setItem(fila, 2, item_presencia)
            self.tabla.setItem(fila, 3, QTableWidgetItem(d["familia"]))
            item_prob = QTableWidgetItem(f"{d['probabilidad']:.2f}")
            item_prob.setTextAlignment(Qt.AlignCenter)
            if d["probabilidad"] > 0.90:
                fnt = item_prob.font(); fnt.setBold(True); item_prob.setFont(fnt)
                item_prob.setForeground(QBrush(QColor("#198754")))
            elif d["probabilidad"] == 0.0:
                item_prob.setForeground(QBrush(QColor("#ADB5BD")))
            self.tabla.setItem(fila, 4, item_prob)

    # ---------- PESTAÑA BARRAS ----------
    def crear_pestana_barras(self):
        pestana = QWidget()
        layout = QVBoxLayout(pestana)
        controles = QHBoxLayout()
        etiqueta = QLabel("Modo de visualización:")
        etiqueta.setObjectName("etiquetaControl")
        self.combo_barras = QComboBox()
        self.combo_barras.addItems(["Agrupado por Familia", "Conteo por Gen Específico"])
        self.combo_top_barras = self._crear_combo_top()
        # Se conectan DESPUÉS de crear ambos combos y la figura (ver abajo)
        boton_exportar = QPushButton("Exportar Gráfica (TIFF/PDF)")
        boton_exportar.setObjectName("botonSecundario")
        boton_exportar.setCursor(Qt.PointingHandCursor)
        boton_exportar.clicked.connect(lambda: self.exportar_grafica("barras"))
        controles.addWidget(etiqueta); controles.addWidget(self.combo_barras)
        controles.addWidget(self.combo_top_barras)
        controles.addStretch(); controles.addWidget(boton_exportar)
        layout.addLayout(controles)
        self.fig_barras = Figure(figsize=(6, 4), facecolor="white")
        self.canvas_barras = FigureCanvasQTAgg(self.fig_barras)
        layout.addWidget(self.canvas_barras)
        self.actualizar_grafica_barras()
        self.combo_barras.currentIndexChanged.connect(self.actualizar_grafica_barras)
        self.combo_top_barras.currentIndexChanged.connect(self.actualizar_grafica_barras)
        return pestana

    def actualizar_grafica_barras(self, *_):
        dibujar_barras(self.fig_barras, self.datos,
                       self.combo_barras.currentIndex(),
                       self._valor_top(self.combo_top_barras),
                       PALETAS[self.tema])
        self.canvas_barras.draw()

    # ---------- PESTAÑA MAPA ----------
    def crear_pestana_mapa(self):
        pestana = QWidget()
        layout = QVBoxLayout(pestana)
        controles = QHBoxLayout()
        etiqueta = QLabel("Eje de variables:")
        etiqueta.setObjectName("etiquetaControl")
        self.combo_mapa = QComboBox()
        self.combo_mapa.addItems(["Familias vs MAGs", "Genes Específicos vs MAGs"])
        self.combo_top_mapa = self._crear_combo_top()
        boton_exportar = QPushButton("Exportar Gráfica (TIFF/PDF)")
        boton_exportar.setObjectName("botonSecundario")
        boton_exportar.setCursor(Qt.PointingHandCursor)
        boton_exportar.clicked.connect(lambda: self.exportar_grafica("mapa"))
        controles.addWidget(etiqueta); controles.addWidget(self.combo_mapa)
        controles.addWidget(self.combo_top_mapa)
        controles.addStretch(); controles.addWidget(boton_exportar)
        layout.addLayout(controles)
        self.fig_mapa = Figure(figsize=(6, 4), facecolor="white")
        self.canvas_mapa = FigureCanvasQTAgg(self.fig_mapa)
        layout.addWidget(self.canvas_mapa)
        self.actualizar_mapa_calor()
        self.combo_mapa.currentIndexChanged.connect(self.actualizar_mapa_calor)
        self.combo_top_mapa.currentIndexChanged.connect(self.actualizar_mapa_calor)
        return pestana

    def actualizar_mapa_calor(self, *_):
        dibujar_mapa(self.fig_mapa, self.datos,
                     self.combo_mapa.currentIndex(),
                     self._valor_top(self.combo_top_mapa),
                     PALETAS[self.tema])
        self.canvas_mapa.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())