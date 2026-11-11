"""
protocolo_datasets.py -- reconstruye el protocolo de descarga y preparacion de los
conjuntos de datos (Tarea 2 de la reunion del 10-nov).

La asesora pidio: "fecha de cuando las descargue, las bases de datos, como estandarice
los datos, como use el cd-hit", y lo mismo para el negativo.

Buena parte de eso NO esta escrito en ningun lado, pero SI quedo registrado en el
sistema. Este script lo recupera:

  * FECHAS      la fecha de modificacion de cada archivo en data/raw/ es cuando lo
                bajaste. Es la evidencia mas confiable que queda.
  * VERSIONES   corre --version de cd-hit, prodigal, blastn, amrfinder, cd-hit-est
  * COMANDOS    busca en el historial de bash las lineas de cd-hit, prodigal, blast,
                wget y curl: ahi estan los parametros exactos y las URLs
  * CONTEOS     cuenta secuencias de cada FASTA en raw/, interim/ y processed/
  * CADENA      intenta ordenar los archivos por fecha para reconstruir el flujo

Lo que encuentre se escribe como un borrador de markdown que YA puedes editar y pegar
en el capitulo de Metodologia. Lo que no pueda recuperar lo deja marcado con
[COMPLETAR] para que no se te pase.

NOTA sobre el historial: solo se extraen las lineas que mencionan las herramientas del
proyecto. No se vuelca el historial completo en ningun archivo.

USO
---
    python protocolo_datasets.py
    python protocolo_datasets.py --sin-historial     # si prefieres no tocarlo

SALIDA
------
    results/protocolo/protocolo_datasets.md
    results/protocolo/inventario_archivos.csv
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--raiz", default=None)
ap.add_argument("--sin-historial", action="store_true",
                help="no leer ~/.bash_history ni ~/.zsh_history")
args = ap.parse_args()

if args.raiz:
    RAIZ = Path(args.raiz).expanduser()
else:
    RAIZ = None
    for cand in [Path.cwd(), *Path.cwd().parents]:
        if (cand / "data").is_dir():
            RAIZ = cand
            break
    if RAIZ is None:
        sys.exit('No encontre el proyecto. Usa --raiz "ruta/al/Proyecto_Residencias"')

SAL = RAIZ / "results" / "protocolo"
SAL.mkdir(parents=True, exist_ok=True)

MD = []


def w(*a):
    MD.append(" ".join(str(x) for x in a))


def fecha(p):
    try:
        return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "?"


def tam(p):
    try:
        n = p.stat().st_size
    except OSError:
        return "?"
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def contar_fasta(p):
    """Cuenta '>' sin cargar el archivo en memoria."""
    try:
        n = 0
        with open(p, "rb") as fh:
            for linea in fh:
                if linea.startswith(b">"):
                    n += 1
        return n
    except OSError:
        return None


def version_de(cmd, flags=("--version", "-v", "-version", "-h")):
    """Prueba varias banderas hasta que una devuelva algo que parezca una version."""
    if not any((Path(d) / cmd[0]).exists() for d in os.environ.get("PATH", "").split(os.pathsep)):
        return None
    for fl in flags:
        try:
            r = subprocess.run(list(cmd) + [fl], capture_output=True, text=True, timeout=20)
            txt = (r.stdout + "\n" + r.stderr).strip()
            if not txt:
                continue
            for linea in txt.splitlines():
                if re.search(r"\d+\.\d+", linea) and len(linea) < 200:
                    return linea.strip()
            return txt.splitlines()[0][:150]
        except (OSError, subprocess.SubprocessError):
            continue
    return None


print("=" * 84)
print("RECONSTRUYENDO EL PROTOCOLO DE DATOS")
print("=" * 84)
print(f"proyecto: {RAIZ}\n")

hoy = datetime.now().strftime("%Y-%m-%d")
w(f"# Protocolo de construccion de los conjuntos de datos — ARGus")
w("")
w(f"> Borrador generado automaticamente el {hoy} a partir de los archivos y el historial")
w("> del sistema. **Revisa cada dato antes de pasarlo al documento final**: las fechas son")
w("> de modificacion del archivo, que normalmente es la de descarga, pero puede cambiar si")
w("> el archivo se copio o se movio despues.")
w("")
w("Lo marcado con **[COMPLETAR]** no se pudo recuperar y lo tienes que llenar tu.")
w("")

# =============================================================================
#  1. VERSIONES DE LAS HERRAMIENTAS
# =============================================================================
print("Buscando versiones de las herramientas...")
w("---")
w("")
w("## 1. Software utilizado")
w("")
w("| herramienta | version | para que se uso |")
w("|---|---|---|")

HERR = [
    (("cd-hit-est",), "reduccion de redundancia del conjunto positivo"),
    (("cd-hit",), "(variante del anterior)"),
    (("prodigal",), "prediccion de ORFs en los genomas del conjunto negativo"),
    (("blastn",), "descarte de ORFs homologos a ARG"),
    (("makeblastdb",), "construccion de la base BLAST"),
    (("amrfinder",), "auditoria final del conjunto negativo"),
    (("python3",), "procesamiento, entrenamiento y evaluacion"),
]
encontradas = {}
for cmd, uso in HERR:
    v = version_de(cmd)
    if v:
        encontradas[cmd[0]] = v
        w(f"| `{cmd[0]}` | {v} | {uso} |")
        print(f"  {cmd[0]:<14} {v[:60]}")
    else:
        w(f"| `{cmd[0]}` | **[COMPLETAR]** no esta en el PATH de este ambiente | {uso} |")
        print(f"  {cmd[0]:<14} no encontrado en el PATH")

# versiones de paquetes de python
w("")
w("Paquetes de Python:")
w("")
w("| paquete | version |")
w("|---|---|")
for mod in ("numpy", "pandas", "scikit-learn", "xgboost", "torch", "transformers",
            "matplotlib", "joblib", "PySide6"):
    try:
        import importlib.metadata as im
        w(f"| {mod} | {im.version(mod)} |")
    except Exception:
        w(f"| {mod} | no instalado en este ambiente |")
w("")
w(f"Ambiente de conda activo al generar esto: `{os.environ.get('CONDA_DEFAULT_ENV', '[COMPLETAR]')}`")
w("")

# =============================================================================
#  2. INVENTARIO DE ARCHIVOS CON FECHAS
# =============================================================================
print("\nInventario de archivos...")
w("---")
w("")
w("## 2. Archivos y fechas")
w("")
w("La fecha es la de ultima modificacion del archivo en disco.")
w("")

inventario = []
EXT = (".fasta", ".fa", ".fna", ".tsv", ".csv", ".txt", ".json", ".gz", ".zip")
for sub in ("data/raw", "data/interim", "data/processed"):
    d = RAIZ / sub
    if not d.is_dir():
        continue
    archivos = sorted((p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXT),
                      key=lambda p: p.stat().st_mtime)
    if not archivos:
        continue
    w(f"### `{sub}/`")
    w("")
    w("| archivo | fecha | tamano | secuencias |")
    w("|---|---|---|---|")
    print(f"\n  {sub}/  ({len(archivos)} archivos)")
    for p in archivos:
        n = contar_fasta(p) if p.suffix.lower() in (".fasta", ".fa", ".fna") else None
        rel = p.relative_to(d)
        w(f"| `{rel}` | {fecha(p)} | {tam(p)} | {f'{n:,}' if n is not None else '—'} |")
        inventario.append({"carpeta": sub, "archivo": str(rel), "fecha": fecha(p),
                           "tamano_bytes": p.stat().st_size, "secuencias": n if n is not None else ""})
        if n:
            print(f"    {fecha(p)}  {n:>8,} seq  {rel}")
    w("")

with open(SAL / "inventario_archivos.csv", "w", newline="", encoding="utf-8") as fh:
    ww = csv.DictWriter(fh, fieldnames=["carpeta", "archivo", "fecha", "tamano_bytes", "secuencias"])
    ww.writeheader()
    ww.writerows(inventario)

# --- la primera y la ultima fecha de data/raw es la ventana de descarga ---
raws = [i for i in inventario if i["carpeta"] == "data/raw"]
if raws:
    w(f"**Ventana de descarga de las bases:** del {raws[0]['fecha']} al {raws[-1]['fecha']} "
      f"(primera y ultima modificacion en `data/raw/`).")
    w("")

# =============================================================================
#  3. COMANDOS DEL HISTORIAL
# =============================================================================
w("---")
w("")
w("## 3. Comandos ejecutados")
w("")

if args.sin_historial:
    w("*(no se reviso el historial; corre sin `--sin-historial` para recuperarlo)*")
    w("")
else:
    print("\nBuscando comandos en el historial...")
    # extensiones de secuencia que aparecen en este proyecto (ResFinder usa .fsa)
    SEQ = r"(fasta|fa|fna|fsa|fas|ffn|faa|seq|tsv|txt)\b"
    PATRONES = {
        "descarga": r"\b(wget|curl)\b|\bdatasets\s+download\b|\bgit\s+clone\b",
        "cd-hit": r"\bcd-hit",
        "prodigal": r"\bprodigal\b",
        "blast": r"\b(blastn|blastp|tblastn|tblastx|makeblastdb|diamond)\b",
        "amrfinder": r"\bamrfinder",
        "union de archivos": r"\bcat\b[^|]*>\s*\S*" + SEQ,
        "estandarizacion": r"\b(sed|awk|dos2unix|tr|seqkit|grep\s+-v)\b.*(\." + SEQ + r"|/data/|\bdata/)",
        "muestreo / scripts": r"\bpython3?\b.*\.py\b",
    }
    hallados = {k: [] for k in PATRONES}
    vistos = set()
    for hp in (Path.home() / ".bash_history", Path.home() / ".zsh_history"):
        if not hp.exists():
            continue
        try:
            texto = hp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for linea in texto.splitlines():
            linea = linea.strip()
            # zsh guarda ': 1699999999:0;comando'
            if linea.startswith(":") and ";" in linea:
                linea = linea.split(";", 1)[1].strip()
            if not linea or linea in vistos:
                continue
            for k, pat in PATRONES.items():
                if re.search(pat, linea):
                    hallados[k].append(linea)
                    vistos.add(linea)
                    break

    total = sum(len(v) for v in hallados.values())
    print(f"  {total} comandos relevantes encontrados")
    if total == 0:
        w("No encontre comandos relevantes en el historial. Puede ser que:")
        w("")
        w("- el historial se haya limpiado,")
        w("- los comandos se corrieran en otra terminal o en otra maquina,")
        w("- o que el historial no se haya escrito todavia (corre `history -a` y repite).")
        w("")
        w("**[COMPLETAR]** los comandos exactos de descarga, cd-hit, Prodigal y BLAST.")
        w("")
    else:
        for k, lineas in hallados.items():
            if not lineas:
                continue
            w(f"### {k}")
            w("")
            w("```bash")
            for l in lineas[-25:]:
                w(l)
            w("```")
            if len(lineas) > 25:
                w(f"*(se muestran los ultimos 25 de {len(lineas)})*")
            w("")
            print(f"    {k:<18} {len(lineas)}")

# =============================================================================
#  4. PLANTILLA CON LO QUE YA SE SABE
# =============================================================================
w("---")
w("")
w("## 4. Conjunto POSITIVO")
w("")
w("### 4.1 Bases de datos de origen")
w("")
w("| base | version | fecha de descarga | URL | secuencias aportadas |")
w("|---|---|---|---|---|")
for b, url in [
    ("CARD", "https://card.mcmaster.ca/download"),
    ("ResFinder", "https://bitbucket.org/genomicepidemiology/resfinder_db"),
    ("ResFinderFG", "https://research.pasteur.fr/en/tool/resfinderfg/"),
    ("MEGARes", "https://www.meglab.org/megares/"),
    ("ARG-ANNOT", "http://backup.mediterranee-infection.com/"),
    ("AMRFinderPlus", "https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/"),
]:
    w(f"| {b} | **[COMPLETAR]** | **[COMPLETAR]** | {url} | **[COMPLETAR]** |")
w("")
w("> Las URL son las oficiales de cada base, puestas aqui como punto de partida:")
w("> **verifica cual usaste tu** y con que version. En el historial de la seccion 3")
w("> deberian estar las lineas de `wget`/`curl` con la URL exacta.")
w("")
w("**Excluida: BacMet.** Por indicacion de la asesora: contiene resistencia a metales y")
w("biocidas, no a antibioticos. Se confirmo al revisar los hits de BLAST, donde aparecieron")
w("genes como `copA` y `cueR` (transportadores de cobre).")
w("")
w("**ResFinderFG se incorporo en una segunda ronda**, tambien por indicacion de la asesora.")
w("Aporta el 47.5 % de los positivos y proviene de metagenomica funcional de intestino.")
w("")

w("### 4.2 Estandarizacion")
w("")
w("1. **Saltos de linea.** ResFinder venia con terminaciones de Windows (CRLF). Se detecto")
w("   porque la verificacion de caracteres de nucleotido reportaba `TT` como caracter")
w("   anomalo. Correccion:")
w("")
w("   ```bash")
w("   sed -i 's/\\r$//' archivo.fasta")
w("   ```")
w("")
w("2. **Filtros de contenido.** Se eliminaron las entradas que no corresponden a genes de")
w("   resistencia adquirida:")
w("")
w("   - entradas dependientes de SNP (resistencia por mutacion puntual, no por el gen)")
w("   - genes de virulencia")
w("   - resistencia a metales y biocidas. En MEGARes el desglose fue: Drugs 7,570,")
w("     Metals 529, Multi-compound 482, Biocides 152. Se conservo solo **Drugs**")
w("   - se dejo fuera la sobreexpresion como mecanismo")
w("")
w("   **[COMPLETAR]** el script o los comandos con los que se aplico cada filtro.")
w("")
w("3. **Union.** Los seis FASTA filtrados se concatenaron en un solo archivo.")
w("   **[COMPLETAR]** nombre del archivo combinado y comando usado.")
w("")

w("### 4.3 Reduccion de redundancia con CD-HIT")
w("")
w("Herramienta: **`cd-hit-est`** (la variante para nucleotidos).")
w("")
w("Se evaluo `cd-hit-dup` y se descarto: `-est` es el correcto para secuencias de")
w("nucleotidos con tolerancia a diferencias, mientras que `-dup` solo quita duplicados exactos.")
w("")
w("Comando:")
w("")
w("```bash")
w("cd-hit-est -i argdb_combinado.fasta -o argdb_POSITIVO_final.fasta \\")
w("           -c 0.95 -aS 0.9 -d 0")
w("```")
w("")
w("| parametro | valor | por que |")
w("|---|---|---|")
w("| `-c` | 0.95 | umbral de identidad: dos secuencias con 95 % o mas se colapsan en una |")
w("| `-aS` | 0.9 | cobertura minima sobre la secuencia mas corta: evita que un fragmento corto colapse con un gen largo |")
w("| `-d` | 0 | conserva el encabezado completo. Sin esto CD-HIT trunca los identificadores (fue un error real: se perdia la trazabilidad de AMRFinderPlus) |")
w("| `-n` | **[COMPLETAR]** | tamano de palabra; para `-c 0.95` lo normal es 10 |")
w("| `-T` / `-M` | **[COMPLETAR]** | hilos y memoria |")
w("")
w("**Resultado: 30,014 secuencias de entrada → 4,228 de salida** (`argdb_POSITIVO_final.fasta`).")
w("")
w("Variantes adicionales generadas para experimentacion:")
w("")
w("| conjunto | identidad | secuencias | archivo |")
w("|---|---|---|---|")
w("| base | 95 % | 4,228 | `argdb_POSITIVO_final.fasta` |")
w("| id90 | 90 % | 3,762 | `positivo_id90_limpio.fasta` |")
w("| id85 | 85 % | 3,477 | `positivo_id85_limpio.fasta` |")
w("")
w("> Resultado negativo relevante: **bajar la identidad no ayudo.** 95 % dio mejores")
w("> metricas que 90 % y que 85 %, y la ventaja se mantuvo despues de corregir la")
w("> fragmentacion. La hipotesis (menos redundancia deberia reducir la fuga por homologia)")
w("> era razonable y resulto falsa.")
w("")

w("---")
w("")
w("## 5. Conjunto NEGATIVO")
w("")
w("### 5.1 Metodologia")
w("")
w("Se siguio la metodologia de **ARGNet**: tomar genomas bacterianos completos, predecir sus")
w("marcos abiertos de lectura y quedarse con los que **no** son genes de resistencia.")
w("")
w("### 5.2 Genomas de origen")
w("")
w("| ensamble | organismo | %GC | fecha de descarga | URL |")
w("|---|---|---|---|---|")
for acc, gcv in [("GCF_000412675.1", 62.7), ("GCF_000008865.2", 49.8),
                 ("GCF_000009045.1", 43.4), ("GCF_000025985.1", 43.8),
                 ("GCF_000005845.2", 50.8), ("GCF_009913655.1", 45.5)]:
    w(f"| {acc} | **[COMPLETAR]** | {gcv} | **[COMPLETAR]** | "
      f"https://www.ncbi.nlm.nih.gov/datasets/genome/{acc}/ |")
w("")
w("> **[COMPLETAR] los nombres de especie.** Importan para el documento: en la FASE 7 se")
w("> detecto que dos de los seis parecen ser cepas de *E. coli*, y esa redundancia")
w("> taxonomica es parte del hallazgo del atajo taxonomico. Confirmalo en NCBI.")
w("")
w("### 5.3 Tuberia")
w("")
w("```")
w("genomas (6)")
w("    |")
w("    v  Prodigal          prediccion de ORFs")
w("  ORFs")
w("    |")
w("    v  BLAST             descartar los que pegan contra la base de positivos")
w("  ORFs no homologos")
w("    |")
w("    v  AMRFinderPlus     auditoria: atrapar los que BLAST dejo pasar")
w("  argdb_NEGATIVO_final.fasta   (26,513 secuencias)")
w("```")
w("")
w("**[COMPLETAR]** los comandos exactos de cada paso, con sus parametros:")
w("")
w("```bash")
w("# Prodigal")
w("prodigal -i genoma.fna -d orfs.fna -p single        # [COMPLETAR] confirmar banderas")
w("")
w("# BLAST")
w("makeblastdb -in argdb_POSITIVO_final.fasta -dbtype nucl -out db_positivos")
w("blastn -query orfs.fna -db db_positivos -evalue ___ -outfmt 6 > hits.tsv   # [COMPLETAR] e-valor")
w("")
w("# AMRFinderPlus")
w("amrfinder -n orfs_filtrados.fna ___                 # [COMPLETAR]")
w("```")
w("")
w("Conteos intermedios: 32,055 → 4,887 segun la bitacora, **marcados como por verificar**.")
w("Conviene reconfirmarlos contra los logs antes de ponerlos en el documento.")
w("")
w("### 5.4 Limitacion conocida de este conjunto")
w("")
w("Los positivos vienen de seis bases con organismos de todo el mundo; los negativos, de")
w("solo seis genomas. Las frecuencias de k-meros codifican uso de codones, que es un rasgo")
w("taxonomico, asi que el clasificador puede separar los dos grupos por su **origen** en vez")
w("de por su **funcion**.")
w("")
w("Medido con validacion dejando un genoma fuera:")
w("")
w("| | tasa de falsos positivos |")
w("|---|---|")
w("| genoma dentro del entrenamiento | 0.323 |")
w("| genoma fuera del entrenamiento | **0.624** |")
w("")
w("Es la limitacion mas importante del proyecto y esta documentada en")
w("`claude/hallazgo_atajo_taxonomico_id90.md`.")
w("")

w("---")
w("")
w("## 6. Conjunto de evaluacion (1k + 1k)")
w("")
w("Se apartaron **1,000 genes positivos y 1,000 negativos al azar**, como genes completos:")
w("`muestra_genes_completos.fasta` + `muestra_genes_labels.tsv`.")
w("")
w("Estos 2,000 genes se **excluyen del entrenamiento y del test**, con un `assert` que")
w("detiene la ejecucion si falta alguno.")
w("")
w("**Salvedad metodologica que conviene declarar en el documento:** salio de las mismas")
w("bases que el entrenamiento, asi que es un conjunto retenido, no una fuente independiente.")
w("CD-HIT al 95 % evita copias casi identicas, pero puede haber parientes con 90-95 % de")
w("identidad en el entrenamiento. La verificacion estricta (`cd-hit-est-2d` o BLAST entre")
w("ambos) sigue pendiente.")
w("")
w("**[COMPLETAR]** la semilla y el script con el que se hizo el muestreo.")
w("")

w("---")
w("")
w("## 7. Lo que falta completar")
w("")
w("Lista para ir tachando:")
w("")
for item in [
    "Version y fecha de descarga de cada una de las seis bases de datos",
    "URL exacta de descarga de cada base (revisa la seccion 3 del historial)",
    "Nombres de especie de los seis genomas del conjunto negativo",
    "Fecha de descarga de los genomas",
    "Comando y parametros completos de cd-hit-est (`-n`, `-T`, `-M`)",
    "Nombre del archivo combinado de antes de cd-hit",
    "Script o comandos de los filtros (SNP, virulencia, metales, biocidas)",
    "Comandos completos de Prodigal, BLAST y AMRFinderPlus con sus parametros",
    "E-valor usado en el BLAST de descarte",
    "Verificar los conteos intermedios del negativo (32,055 → 4,887)",
    "Semilla del muestreo del 1k+1k",
]:
    w(f"- [ ] {item}")
w("")

ruta = SAL / "protocolo_datasets.md"
ruta.write_text("\n".join(MD) + "\n", encoding="utf-8")
print("\n" + "=" * 84)
print(f"Borrador escrito en: {ruta}")
print(f"Inventario en:       {SAL / 'inventario_archivos.csv'}")
print("=" * 84)
print("\nAbrelo, revisa las fechas y llena los [COMPLETAR].")
print("Lo que mas valor tiene es la seccion 3 (comandos del historial): ahi suelen")
print("estar las URL de descarga y los parametros exactos de cd-hit.")
