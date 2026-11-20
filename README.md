# ARGus

**A**ntibiotic **R**esistance **G**enome **u**nderstanding **s**ystem

Aplicación de escritorio que detecta genes de resistencia a antibióticos (ARGs) en genomas
ensamblados a partir de metagenomas (MAGs) y los clasifica por familia de resistencia.

ARGus trabaja en dos etapas:

1. **¿Es un gen de resistencia?** Un Random Forest sobre frecuencias de 6-meros responde sí o no
   para cada secuencia del archivo.
2. **¿De qué familia?** Para las que salieron positivas, un modelo basado en DNABERT-6 indica a
   cuál de 12 familias de resistencia pertenece.

Todo corre de forma local: no hay servidor, no se sube nada a internet y no hace falta tarjeta
gráfica.

> **Para qué sirve y para qué no.** ARGus entrega una **lista de candidatos a verificar**, no una
> identificación definitiva. Antes de interpretar resultados conviene leer
> [Limitaciones conocidas](#limitaciones-conocidas).

---

## Índice

- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Cómo ejecutarlo](#cómo-ejecutarlo)
- [El archivo de entrada](#el-archivo-de-entrada)
- [Qué devuelve](#qué-devuelve)
- [La ventana, por partes](#la-ventana-por-partes)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Reproducir los resultados](#reproducir-los-resultados)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Problemas frecuentes](#problemas-frecuentes)
- [Créditos](#créditos)

---

## Requisitos

| | |
|---|---|
| Sistema | Windows 10/11, Linux o macOS |
| Python | 3.10 u 3.11 |
| Memoria | 8 GB recomendado |
| Disco | ~1.5 GB (código, modelos y caché de DNABERT) |
| Internet | Solo al instalar y en el primer análisis |

No hace falta tarjeta gráfica: todo corre en CPU.

---

## Instalación

### 1. Instalar Git LFS — antes de clonar

El modelo de la primera etapa (`argus_rf_k6_id95_v2.joblib`, ~9 MB) se distribuye con
**Git LFS**. Hay que instalarlo *antes* de clonar; si no, ese archivo llega como un puntero de
texto de unos cientos de bytes y la aplicación falla al cargarlo.

```bash
# Ubuntu / Debian / WSL
sudo apt install git-lfs
# macOS
brew install git-lfs
# Windows: descargar de https://git-lfs.com

git lfs install
```

> La cabeza de clasificación de la segunda etapa (`dnabert6_familias_cabeza.pt`, 784 KB) **no**
> usa LFS: es lo bastante pequeña para versionarse de forma normal.

### 2. Clonar

```bash
git clone https://github.com/IzamaelMendoza/argus.git
cd argus
git lfs pull
```

### 3. Crear el entorno e instalar dependencias

Con **conda**:

```bash
conda create -n argus python=3.11
conda activate argus
pip install -r requirements.txt
```

O con **venv**:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Si `pip` empieza a descargar ~2.5 GB de PyTorch, es la variante para GPU, que no se necesita.
> Interrumpe e instala la de CPU:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 4. Comprobar la instalación

```bash
python app/verificar_instalacion.py
```

Revisa la versión de Python, los paquetes, los cuatro archivos del modelo —detectando si alguno
quedó como puntero de LFS— y la caché de DNABERT. Si algo falta, dice exactamente qué hacer.

---

## Cómo ejecutarlo

```bash
conda activate argus          # o: source .venv/bin/activate
python app/argus_gui.py
```

La **primera vez** que analices un archivo, ARGus descarga DNABERT-6 desde Hugging Face (359 MB)
y lo guarda en la caché local. Puede tardar varios minutos y parecer que el programa está
detenido; solo ocurre una vez.

Al abrir, la aplicación carga el último análisis guardado. Si es la primera vez, muestra datos de
ejemplo y la línea de estado dice *«Sin archivo cargado — mostrando datos de ejemplo»*: son
ficticios, están ahí para que la ventana no aparezca vacía.

### Probar sin tener datos propios

El repositorio incluye un archivo de ejemplo con 20 genes reales, diez con resistencia y diez sin
ella, tomados del conjunto de evaluación:

```
data/ejemplo/ejemplo_20_genes.fasta
```

---

## El archivo de entrada

ARGus lee **un archivo FASTA de nucleótidos** con extensión `.fasta`, `.fa`, `.fna` o `.fas`.

### Qué debe contener

- **Nucleótidos (A, C, G, T)**, no proteínas. El modelo trabaja sobre 6-meros de ADN; una
  secuencia de aminoácidos daría resultados sin sentido y **el sistema no puede detectarlo**.
- **Secuencias codificantes (genes u ORFs)**, una por entrada. ARGus clasifica cada entrada por
  separado, así que no se le pasa un genoma completo en un solo registro: hay que predecir los
  marcos de lectura antes, con Prodigal por ejemplo.
- **Al menos 150 pb por secuencia.** Las más cortas se marcan como *No evaluable*.

### El encabezado decide el nombre del MAG

ARGus parte el encabezado por el carácter `|`:

```
>MAG_01|k141_2035_1          ->  MAG: MAG_01                 Gen: k141_2035_1
>k141_2035_1                 ->  MAG: <nombre del archivo>   Gen: k141_2035_1
```

- **Con `|`:** lo que va antes del primer `|` es el MAG; lo que va después del último, el gen.
- **Sin `|`:** el MAG pasa a ser el nombre del archivo.

Esto importa para el **mapa de calor**, que pone un MAG por columna. Si vas a analizar varios
MAGs en un mismo archivo, usa el formato `MAG|gen`.

### Ejemplo válido

```
>MAG_01|k141_2035_1
ATGAAAAAGTTATTTACTGTTTTAGCTGCAGTTCTGTTATTTAGTACAGCTAGCGTTATT
GCTGATCGTAAAACTTTAGAAACTGTAAAAGCTGATTTAGTTAATAAAGCTGGTACTTTA
>MAG_01|k141_2035_2
ATGAGTATTCAACATTTCCGTGTCGCCCTTATTCCCTTTTTTGCGGCATTTTGCCTTCCT
```

### Qué no acepta

- **FASTQ.** Si el archivo empieza con `@`, ARGus lo detecta y avisa.
- **Proteínas.** No hay comprobación automática.

---

## Qué devuelve

### En pantalla

Una fila por secuencia, con cinco columnas:

| columna | contenido |
|---|---|
| **MAG** | de dónde viene la secuencia |
| **Gen** | identificador de la secuencia |
| **Presencia de resistencia** | `Sí`, `No` o `No evaluable` |
| **Familia** | la familia de resistencia y su confianza |
| **Probabilidad** | de 0 a 1, la confianza de la etapa 1 |

Valores posibles en **Familia**:

| valor | significado |
|---|---|
| `OXA beta-lactamasa (87%)` | familia identificada, con su confianza |
| `Sin determinar` | es resistente, pero ninguna familia alcanza el 20 % de confianza |
| `no_resist` | la secuencia no es un gen de resistencia |
| `< 150 pb` | demasiado corta para evaluarla |
| `Sin clasificar` | el modelo de familias no estaba disponible |

> **`No evaluable` no es lo mismo que `No`.** Significa que la secuencia es demasiado corta para
> juzgarla, no que se haya descartado como no resistente. Se muestra en ámbar y no cuenta en las
> gráficas ni en el total de resistentes.

### Archivos que exporta

| desde | formato | contenido |
|---|---|---|
| Tabla de resultados | **CSV** | las cinco columnas, para Excel o análisis posterior |
| Tabla de resultados | **PDF** | la tabla paginada, a 300 DPI y con marca de agua |
| Gráfica de barras | **TIFF o PDF** | la gráfica tal como se ve |
| Mapa de calor | **TIFF o PDF** | ídem |

Las gráficas se exportan siempre con fondo claro, aunque uses el tema oscuro, para que sirvan en
un documento impreso.

---

## La ventana, por partes

### Controles principales

| control | qué hace |
|---|---|
| **Cargar archivo MAG (.fasta)** | abre el explorador para elegir el archivo |
| **Iniciar análisis** | arranca la clasificación (inactivo si no hay archivo) |
| **Detener** | cancela a medio camino; se conservan los resultados obtenidos |
| **Modo claro / Modo oscuro** | cambia el tema; recuerda tu elección entre sesiones |

El análisis corre en un hilo aparte, así que la ventana sigue respondiendo mientras trabaja.

### Pestaña «Tabla de resultados»

La tabla completa, con las filas resistentes en rojo, las no resistentes en gris y las no
evaluables en ámbar. Abajo, **Exportar Tabla (PDF)** y **Descargar CSV**.

### Pestaña «Gráfica de barras»

**Modo de visualización** ofrece *Agrupado por Familia* o *Conteo por Gen Específico*.
**Top N** limita cuántas categorías se dibujan (20 por defecto).

### Pestaña «Mapa de calor»

Compara varios MAGs entre sí. **Eje de variables** ofrece *Familias vs MAGs* o
*Genes Específicos vs MAGs*. Cada celda es la cantidad de genes de esa fila en ese MAG. Requiere
encabezados con formato `MAG|gen`. Se limita a 30 columnas por legibilidad.

### Pestaña «Historial»

Los **últimos 5 análisis** se guardan solos, con fecha, archivo, número de secuencias y cuántas
salieron con resistencia. Doble clic sobre una fila para reabrir ese análisis sin repetirlo.

Se guardan en `historial_analisis/` como JSON. Al ser solo cinco, el historial sirve para
recuperar trabajo reciente, no para archivar: lo que quieras conservar, expórtalo.

---

## Estructura del repositorio

```
.
├── app/                                  la aplicación
│   ├── argus_gui.py                      interfaz y etapa 1
│   ├── familias_dnabert_gui.py           etapa 2 (carga DNABERT y predice familia)
│   └── verificar_instalacion.py          comprueba que todo esté en su sitio
├── data/
│   ├── ejemplo/                          FASTA de ejemplo, 20 genes
│   ├── interim/                          conjunto de evaluación 1k+1k
│   ├── models/                           modelos entrenados
│   └── processed/                        conjuntos positivo y negativo finales
├── results/
│   └── familias/                         metadatos del modelo de familias y resultados
├── notebooks/                            entrenamiento y comparaciones
├── scripts/                              análisis de datos y experimentos
├── docs/                                 protocolo de datasets y bitácoras
├── requirements.txt
└── README.md
```

### Archivos imprescindibles para que el modelo corra

Si alguno falta, ARGus no funciona del todo. `verificar_instalacion.py` los revisa:

| archivo | para qué |
|---|---|
| `data/models/argus_rf_k6_id95_v2.joblib` | el modelo de la etapa 1, con su umbral de decisión |
| `data/models/dnabert6_familias_cabeza.pt` | la cabeza entrenada de la etapa 2 |
| `results/familias/familias_dnabert_meta.json` | las 12 familias **en el orden correcto** |
| `results/familias/dnabert6_norm.npy` | normalización de los vectores de DNABERT |

Los dos últimos son pequeños pero **imprescindibles**: sin el `meta.json` la red produce doce
números sin manera de saber a qué familia corresponde cada uno, y sin la normalización las
predicciones pierden sentido sin que nada falle de forma visible.

### Lo que no está en el repositorio

Los datos crudos e intermedios (bases de datos descargadas, ORFs predichos, pasos previos al
agrupamiento) **no se versionan**: son varios cientos de megabytes y se reconstruyen siguiendo
`docs/Protocolo_datasets_ARGus.docx`. El modelo ya viene entrenado, así que no hacen falta para
usar la aplicación.

---

## Reproducir los resultados

Los notebooks y scripts que respaldan cada resultado del reporte técnico:

| resultado | dónde se produce |
|---|---|
| Entrenamiento del modelo de la etapa 1 | `notebooks/02_rf_k6_undersampling.ipynb` |
| Comparación entre niveles de identidad (95/90/85) | `notebooks/02_argus_k6_id90.ipynb`, `02_argus_k6_id85.ipynb` |
| Random Forest contra XGBoost y ajuste del umbral | `notebooks/argus_k6_rf_vs_xgboost.ipynb` |
| Variante con DNABERT-2 | `notebooks/argus_dnabert2_id95.ipynb` |
| Comparación con DeepARG y ARGNet | `notebooks/deepArg*.ipynb`, `notebooks/argNet*.ipynb` |
| Entrenamiento del modelo de familias (etapa 2) | `scripts/familias_dnabert.py` |
| Caracterización del conjunto positivo | `scripts/tarea1_sets.py` |
| Falsos positivos del conjunto negativo completo | `scripts/tarea1_negativos.py` |
| Sobremuestreo contra ponderación de clases | `scripts/tarea3_sobremuestreo.py` |
| Desempeño por familia y matriz de confusión | `scripts/reporte_familias.py` |
| Homología dentro de cada familia | `scripts/coherencia_familias.py` |
| Conteos por base de datos y efecto de CD-HIT | `scripts/conteo_datasets.py` |
| Construcción de los conjuntos de datos | `scripts/protocolo_datasets.py` |

El conjunto de evaluación con el que se comparan todas las herramientas está en
`data/interim/muestra_genes_completos.fasta` y sus etiquetas en `muestra_genes_labels.tsv`:
1000 genes con resistencia y 1000 sin ella, apartados antes de entrenar.

---

## Limitaciones conocidas

Conviene tenerlas presentes al interpretar los resultados.

**El modelo responde en parte a la taxonomía, no solo a la función.** Su tasa de falsos positivos
pasa de 0.08 a 0.46 según el genoma de origen, y correlaciona con el contenido GC (−0.90). Sobre
un organismo poco representado en el entrenamiento, la precisión cae.

**La precisión depende mucho de cuántos ARGs haya realmente.** Medida sobre un conjunto
balanceado (50 % de ARGs) es de 0.78; sobre un MAG real, donde los ARGs son el 1 % de los genes,
baja bastante. Conviene tomar los resultados como una lista de candidatos a verificar.

**La familia solo cubre 12 clases**, derivadas de la ontología de CARD. Un gen de resistencia de
una familia no incluida se clasificará en la más parecida o quedará como «Sin determinar».

**El etiquetado de familias se entrenó con 1,011 genes.** Las familias con menos ejemplos (APH
fosfotransferasa, regulador de eflujo) aciertan alrededor del 33 % de las veces; las mejor
representadas superan el 89 %. La exactitud balanceada global es de **0.702** sobre las 12
familias.

**El conjunto de evaluación no es independiente.** Salió de las mismas bases de datos que el
entrenamiento, así que es un conjunto retenido. La reducción de redundancia al 95 % evita copias
casi idénticas, pero puede haber secuencias emparentadas repartidas entre ambos lados.

---

## Problemas frecuentes

| síntoma | causa y solución |
|---|---|
| La aplicación no abre y menciona un error al cargar el modelo | Los archivos de LFS llegaron vacíos. `git lfs install && git lfs pull` |
| La columna Familia dice `Sin clasificar` en todas las filas | La etapa 2 no cargó. Ejecuta `python app/verificar_instalacion.py` |
| El primer análisis parece trabado | Está descargando DNABERT (359 MB). Solo ocurre la primera vez |
| Avisa que el archivo es FASTQ | Es otro formato. Hay que convertirlo a FASTA |
| Todas las secuencias salen `No evaluable` | Miden menos de 150 pb. ¿Son lecturas en lugar de genes? |
| El mapa de calor muestra un solo MAG | Los encabezados no traen el separador `|`. Ver [El archivo de entrada](#el-archivo-de-entrada) |
| El análisis se detuvo solo | Si pulsaste «Detener», los resultados son parciales; el historial lo registra |

Ante cualquier otro problema, el primer paso es siempre `python app/verificar_instalacion.py`.

---

## Créditos

Proyecto de residencia profesional — Ingeniería en Sistemas Computacionales.

ARGus usa [DNABERT](https://github.com/jerryji1993/DNABERT) (`zhihan1996/DNA_bert_6`) como modelo
base para la clasificación por familia.

Los genes de resistencia proceden de **CARD**, **ResFinder**, **ResFinderFG**, **MEGARes**,
**ARG-ANNOT** y **AMRFinderPlus**. El protocolo completo de construcción de los conjuntos de
datos, con fechas de descarga, versiones y parámetros, está en
`docs/Protocolo_datasets_ARGus.docx`.
