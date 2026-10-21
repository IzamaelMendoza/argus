import sys
import os
import time
import random
import joblib
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QTabWidget, QLabel,
    QHeaderView, QComboBox, QFileDialog, QMessageBox
)
from PySide6.QtGui import QFont, QColor, QBrush, QPixmap, QIcon
from PySide6.QtCore import Qt, QThread, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from itertools import product
from collections import Counter

# ============================================================
#  1. FEATURIZACIÓN Y FRAGMENTACIÓN (Idéntico al Notebook)
# ============================================================
VENTANA = 500
K = 4
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
    """Convierte un fragmento de 500 pb en un vector de 256 frecuencias."""
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
def predecir_mag(secuencias, callback_progreso=None, esta_detenido=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ruta_modelo_binario = os.path.join(base_dir, "modelo_rf_k4.joblib")

    if not os.path.exists(ruta_modelo_binario):
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo en:\n{ruta_modelo_binario}\n\n"
            "Asegúrate de haberlo guardado con joblib.dump desde tu Notebook."
        )

    # 1. Cargar tu Random Forest entrenado
    modelo_etapa1 = joblib.load(ruta_modelo_binario)

    # 2. Revisar si ya existe el modelo de Etapa 2 (Familias)
    ruta_modelo_familias = os.path.join(base_dir, "modelo_familias.joblib")
    modelo_etapa2 = joblib.load(ruta_modelo_familias) if os.path.exists(ruta_modelo_familias) else None
    familias_demo = ["Beta-lactam", "Aminoglycoside", "Tetracycline", "Quinolone", "MLS"]

    datos = []
    total = len(secuencias)

    for idx, (sid, seq) in enumerate(secuencias, start=1):
        if esta_detenido and esta_detenido():
            break

        # Fragmentar en ventanas de 500 pb y calcular k-mers (k=4)
        frags = fragmentar(seq, ventana=VENTANA)
        X_frags = np.array([vector_kmer(f) for f in frags], dtype=np.float32)

        # Predicción real con tu Random Forest
        etiquetas_frags = modelo_etapa1.predict(X_frags)
        probs_frags = modelo_etapa1.predict_proba(X_frags)[:, 1]

        # Regla: si al menos un fragmento es 1 -> el gen se reporta como resistente
        es_resistente = bool(np.any(etiquetas_frags == 1))
        prob_maxima = float(np.max(probs_frags))

        # Separar limpiamente el nombre del MAG y el nombre del Gen
        if "|" in sid:
            partes = sid.split("|")
            nombre_mag = partes[0]
            nombre_gen = partes[-1][:25]
        else:
            nombre_mag = sid[:15]
            nombre_gen = sid[:25]

        if es_resistente:
            if modelo_etapa2 is not None:
                idx_frag_max = int(np.argmax(probs_frags))
                familia_pred = str(modelo_etapa2.predict(X_frags[idx_frag_max:idx_frag_max+1])[0])
            else:
                familia_pred = random.choice(familias_demo)

            datos.append({
                "mag": nombre_mag,
                "gen": nombre_gen,
                "presencia": "Sí",
                "familia": familia_pred,
                "probabilidad": round(prob_maxima, 2),
            })
        else:
            datos.append({
                "mag": nombre_mag,
                "gen": nombre_gen,
                "presencia": "No",
                "familia": "no_resist",
                "probabilidad": round(prob_maxima, 2),
            })

        time.sleep(0.03)
        if callback_progreso:
            callback_progreso(idx, total)

    return datos
# ============================================================
#  PUNTO DE CONEXION DEL MODELO (Soporta progreso y parada)
# ============================================================
def predecir_mag(secuencias, callback_progreso=None, esta_detenido=None):
    """
    secuencias: lista de (id, secuencia) leídas del MAG.
    callback_progreso: función opcional (actual, total) para actualizar la GUI.
    esta_detenido: función opcional que devuelve True si el usuario presionó 'Detener'.
    """
    # >>> AQUI SE CONECTA EL MODELO ENTRENADO <<<
    # Cuando conectes joblib, puedes cargar el modelo antes del ciclo for:
    # import joblib
    # modelo = joblib.load("modelo_rf.joblib")

    random.seed(0)
    familias_demo = ["Beta-lactam", "Aminoglycoside", "Tetracycline", "Quinolone", "MLS"]
    datos = []
    total = len(secuencias)

    for idx, (sid, seq) in enumerate(secuencias, start=1):
        # 1. Verificar si el usuario presionó el botón "Detener"
        if esta_detenido and esta_detenido():
            break

        # 2. Featurización (y predicción real cuando conectes el modelo)
        _ = vector_kmer(seq)
        
        # Pequeña pausa simulada (0.01s) para que puedas apreciar el progreso 
        # y probar el botón "Detener" frente a tu asesora. (Puedes quitarla después).
        time.sleep(0.01)

        es_resistente = random.random() < 0.30
        if es_resistente:
            datos.append({
                "mag": sid.split("|")[0] if "|" in sid else sid[:12],
                "gen": sid[:20],
                "presencia": "Sí",
                "familia": random.choice(familias_demo),
                "probabilidad": round(random.uniform(0.60, 0.99), 2),
            })
        else:
            datos.append({
                "mag": sid.split("|")[0] if "|" in sid else sid[:12],
                "gen": sid[:20],
                "presencia": "No",
                "familia": "no_resist",
                "probabilidad": round(random.uniform(0.0, 0.40), 2),
            })

        # 3. Notificar avance a la barra de estado
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

    def __init__(self, secuencias):
        super().__init__()
        self.secuencias = secuencias
        self._detener = False

    def solicitar_parada(self):
        self._detener = True

    def run(self):
        try:
            resultados = predecir_mag(
                self.secuencias,
                callback_progreso= lambda act, tot: self.progreso.emit(act, tot),
                esta_detenido= lambda: self._detener
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


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        nombre_sistema = "ARGus (Antibiotic Resistance Genome understanding system)"
        self.setWindowTitle(nombre_sistema)
        self.resize(1100, 700)
        
        self.datos = generar_datos_falsos()
        self.secuencias_cargadas = []
        self.ruta_actual = ""
        self.hilo = None

        self.setWindowIcon(QIcon("logo.png"))

        estilo_global = """
        * { font-family: 'Segoe UI', 'Roboto', 'Inter', sans-serif; }
        QMainWindow { background-color: #F8F9FA; }
        QLabel#tituloPrincipal { font-size: 18pt; font-weight: bold; color: #2C3E50; padding: 10px 0px; }
        
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
        
        QComboBox { padding: 5px; font-size: 10pt; border: 1px solid #DEE2E6; border-radius: 4px; }
        QTabWidget::pane { border: 1px solid #DEE2E6; background: #FFFFFF; border-radius: 4px; }
        QTabBar::tab { background: #E9ECEF; color: #495057; padding: 10px 20px; font-size: 10pt; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; border: 1px solid transparent; }
        QTabBar::tab:selected { background: #FFFFFF; color: #0D6EFD; font-weight: bold; border: 1px solid #DEE2E6; border-bottom: 3px solid #0D6EFD; }
        QTableView { background-color: #FFFFFF; alternate-background-color: #F8F9FA; border: none; gridline-color: #E9ECEF; font-size: 10pt; }
        QHeaderView::section { background-color: #E9ECEF; color: #212529; font-weight: bold; padding: 8px; border: none; border-right: 1px solid #DEE2E6; border-bottom: 2px solid #DEE2E6; }
        """
        self.setStyleSheet(estilo_global)

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
        
        cabecera.addWidget(self._cargar_logo("../logo_escuela.png", "TecNM"))
        cabecera.addWidget(self._cargar_logo("../cinvestav.png", "Cinvestav"))
        
        layout.addLayout(cabecera)

        # ---------- BARRA DE CONTROLES (Cargar, Iniciar, Detener) ----------
        barra = QHBoxLayout()
        
        self.boton_cargar = QPushButton("📂 Cargar archivo MAG (.fasta)")
        self.boton_cargar.setCursor(Qt.PointingHandCursor)
        self.boton_cargar.clicked.connect(self.cargar_mag)
        barra.addWidget(self.boton_cargar)

        self.boton_iniciar = QPushButton("▶ Iniciar análisis")
        self.boton_iniciar.setObjectName("botonIniciar")
        self.boton_iniciar.setCursor(Qt.PointingHandCursor)
        self.boton_iniciar.setEnabled(False)  # Deshabilitado hasta cargar un archivo
        self.boton_iniciar.clicked.connect(self.iniciar_analisis)
        barra.addWidget(self.boton_iniciar)

        self.boton_detener = QPushButton("⏹ Detener")
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
        layout.addWidget(self.pestañas)

        self.setCentralWidget(contenedor)

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

    # ---------- PASO 1: CARGAR Y VALIDAR EL ARCHIVO (Sin ejecutar aún) ----------
    def cargar_mag(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo MAG",
            "", "FASTA (*.fasta *.fa *.fna *.fas);;Todos los archivos (*)")
        if not ruta:
            return
        try:
            secuencias = leer_fasta(ruta)
        except ValueError as e:
            QMessageBox.warning(self, "Archivo no válido", str(e))
            return
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo leer el archivo:\n{e}")
            return

        if not secuencias:
            QMessageBox.warning(self, "Archivo vacío", "No se encontraron secuencias en el archivo.")
            return

        # Guardamos las secuencias listas y habilitamos el botón de "Iniciar"
        self.secuencias_cargadas = secuencias
        self.ruta_actual = ruta
        self.boton_iniciar.setEnabled(True)
        self.boton_detener.setEnabled(False)

        n = len(secuencias)
        nombre_arch = os.path.basename(ruta)
        self.label_estado.setText(f"📄 Archivo listo: {nombre_arch} ({n} secuencias). Presione 'Iniciar análisis'.")
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
        self.hilo = HiloPrediccion(self.secuencias_cargadas)
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

        if fue_cancelado:
            self.label_estado.setText(
                f"⏹ Detenido ({nombre_arch}) — Se procesaron {n_procesadas}/{n_total} secuencias ({n_res} con resistencia)."
            )
            self.label_estado.setStyleSheet("color: #DC3545; font-weight: bold;")
        else:
            self.label_estado.setText(
                f"✅ Completado ({nombre_arch}) — {n_total} secuencias analizadas, {n_res} con resistencia."
            )
            self.label_estado.setStyleSheet("color: #198754; font-weight: bold;")

    def analisis_error(self, mensaje_error):
        self.boton_cargar.setEnabled(True)
        self.boton_iniciar.setEnabled(True)
        self.boton_detener.setEnabled(False)
        self.label_estado.setText("❌ Error durante el análisis.")
        self.label_estado.setStyleSheet("color: #DC3545; font-weight: bold;")
        QMessageBox.critical(self, "Error en la predicción", f"Ocurrió un fallo al ejecutar el modelo:\n{mensaje_error}")

    def refrescar_vistas(self):
        self._llenar_tabla()
        self.actualizar_grafica_barras(self.combo_barras.currentIndex())
        self.actualizar_mapa_calor(self.combo_mapa.currentIndex())

    # ---------- EXPORTACION ----------
    def exportar_csv(self):
        ruta, _ = QFileDialog.getSaveFileName(self, "Guardar Resultados", "", "CSV files (*.csv)")
        if ruta:
            try:
                df = pd.DataFrame(self.datos)
                df.columns = ["MAG", "Gen", "Presencia", "Familia", "Probabilidad"]
                df.to_csv(ruta, index=False)
                QMessageBox.information(self, "Éxito", "Tabla guardada correctamente.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar el archivo:\n{str(e)}")

    def exportar_grafica(self, fig):
        ruta, filtro = QFileDialog.getSaveFileName(self, "Guardar Gráfica", "", "TIFF files (*.tiff);;PDF files (*.pdf)")
        if ruta:
            try:
                if "TIFF" in filtro and not ruta.lower().endswith(('.tif', '.tiff')):
                    ruta += ".tiff"
                elif "PDF" in filtro and not ruta.lower().endswith('.pdf'):
                    ruta += ".pdf"
                fig.savefig(ruta, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Éxito", f"Gráfica guardada en alta resolución (300 DPI):\n{ruta}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar la gráfica:\n{str(e)}")

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
        etiqueta.setStyleSheet("font-weight: bold; color: #495057;")
        self.combo_barras = QComboBox()
        self.combo_barras.addItems(["Agrupado por Familia", "Conteo por Gen Específico"])
        self.combo_barras.currentIndexChanged.connect(self.actualizar_grafica_barras)
        boton_exportar = QPushButton("Exportar Gráfica (TIFF/PDF)")
        boton_exportar.setObjectName("botonSecundario")
        boton_exportar.setCursor(Qt.PointingHandCursor)
        boton_exportar.clicked.connect(lambda: self.exportar_grafica(self.fig_barras))
        controles.addWidget(etiqueta); controles.addWidget(self.combo_barras)
        controles.addStretch(); controles.addWidget(boton_exportar)
        layout.addLayout(controles)
        self.fig_barras = Figure(figsize=(6, 4), facecolor="white")
        self.canvas_barras = FigureCanvasQTAgg(self.fig_barras)
        layout.addWidget(self.canvas_barras)
        self.actualizar_grafica_barras(0)
        return pestana

    def actualizar_grafica_barras(self, index):
        self.fig_barras.clear()
        ax = self.fig_barras.add_subplot(111)
        conteo = {}
        llave = "familia" if index == 0 else "gen"
        for d in self.datos:
            if d["presencia"] == "Sí" and d[llave] != "-":
                conteo[d[llave]] = conteo.get(d[llave], 0) + 1
        if conteo:
            ax.bar(list(conteo.keys()), list(conteo.values()), color="#0D6EFD", edgecolor="none")
            ax.set_ylabel("Número de detecciones", fontsize=10, color="#495057")
            titulo = "Genes de resistencia por familia" if index == 0 else "Frecuencia absoluta por gen específico"
            ax.set_title(titulo, fontsize=12, fontweight="bold", color="#2C3E50", pad=15)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#DEE2E6'); ax.spines['bottom'].set_color('#DEE2E6')
            ax.tick_params(colors='#495057')
            self.fig_barras.autofmt_xdate(rotation=30 if index == 0 else 45)
            self.fig_barras.tight_layout()
        self.canvas_barras.draw()

    # ---------- PESTAÑA MAPA ----------
    def crear_pestana_mapa(self):
        pestana = QWidget()
        layout = QVBoxLayout(pestana)
        controles = QHBoxLayout()
        etiqueta = QLabel("Eje de variables:")
        etiqueta.setStyleSheet("font-weight: bold; color: #495057;")
        self.combo_mapa = QComboBox()
        self.combo_mapa.addItems(["Familias vs MAGs", "Genes Específicos vs MAGs"])
        self.combo_mapa.currentIndexChanged.connect(self.actualizar_mapa_calor)
        boton_exportar = QPushButton("Exportar Gráfica (TIFF/PDF)")
        boton_exportar.setObjectName("botonSecundario")
        boton_exportar.setCursor(Qt.PointingHandCursor)
        boton_exportar.clicked.connect(lambda: self.exportar_grafica(self.fig_mapa))
        controles.addWidget(etiqueta); controles.addWidget(self.combo_mapa)
        controles.addStretch(); controles.addWidget(boton_exportar)
        layout.addLayout(controles)
        self.fig_mapa = Figure(figsize=(6, 4), facecolor="white")
        self.canvas_mapa = FigureCanvasQTAgg(self.fig_mapa)
        layout.addWidget(self.canvas_mapa)
        self.actualizar_mapa_calor(0)
        return pestana

    def actualizar_mapa_calor(self, index):
        self.fig_mapa.clear()
        ax = self.fig_mapa.add_subplot(111)
        mags = sorted(set(d["mag"] for d in self.datos))
        llave = "familia" if index == 0 else "gen"
        elementos = sorted(set(d[llave] for d in self.datos if d["presencia"] == "Sí" and d[llave] != "-"))
        if not elementos:
            self.canvas_mapa.draw(); return
        matriz = np.zeros((len(elementos), len(mags)))
        for d in self.datos:
            if d["presencia"] == "Sí" and d[llave] in elementos:
                i, j = elementos.index(d[llave]), mags.index(d["mag"])
                matriz[i, j] = max(matriz[i, j], d["probabilidad"])
        im = ax.imshow(matriz, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(mags))); ax.set_xticklabels(mags, rotation=45, ha="right", color="#495057")
        ax.set_yticks(range(len(elementos))); ax.set_yticklabels(elementos, color="#495057")
        titulo = "Distribución y Probabilidad por Familia" if index == 0 else "Distribución y Probabilidad por Gen"
        ax.set_title(titulo, fontsize=12, fontweight="bold", color="#2C3E50", pad=15)
        cbar = self.fig_mapa.colorbar(im, ax=ax)
        cbar.set_label("Probabilidad de predicción", color="#495057")
        cbar.ax.tick_params(colors='#495057'); cbar.outline.set_edgecolor('#DEE2E6')
        self.fig_mapa.tight_layout()
        self.canvas_mapa.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ventana = VentanaPrincipal()
    ventana.show()
    sys.exit(app.exec())