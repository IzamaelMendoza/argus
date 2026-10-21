# Bitácora — Fusión y reducción de bases de datos ARG con CD-HIT
**Fecha:** 26 agosto 2026

## Bases de datos utilizadas
Según el anteproyecto oficial: CARD, ResFinder, MEGARes, ARG-ANNOT, AMRFinderPlus.
(BacMet y ResFinderFG se descargaron pero se excluyeron por no estar en el documento oficial;
DeepARG-DB se descargó como extra, sin uso por ahora.)
Decisión tomada por criterio propio ante falta de respuesta oportuna de la asesora,
pendiente de confirmar en la siguiente reunión.

## Archivos exactos usados por base (solo nucleótidos)
| Base | Archivo | Notas |
|---|---|---|
| CARD | nucleotide_fasta_protein_homolog_model.fasta | excluidos: variant, rRNA_variant, knockout, overexpression models |
| ResFinder | all.fsa | confirmado como suma exacta de las 15 clases individuales (3212 secuencias) |
| MEGARes | megares_database_v3.00.fasta | filtro SNP aplicado (ver abajo) |
| ARG-ANNOT | argannot_sequences.fasta | |
| AMRFinderPlus | AMR_CDS.fa | excluido AMRProt.fa (proteína); filtro type/subtype aplicado (ver abajo) |

## Preprocesamiento aplicado a las 5 bases
- Corrección de saltos de línea CRLF (detectado en ResFinder, corregido en las 5 por precaución)
- Normalización a mayúsculas (AMRFinderPlus venía en minúsculas)
- Etiquetado de procedencia en cada header (prefijo CARD| / ResFinder| / etc.)

## Filtro SNP — MEGARes
El campo "header" del CSV de anotaciones trae un 6to segmento opcional (separado por "|")
con el valor "RequiresSNPConfirmation" cuando el gen requiere una mutación puntual para
conferir resistencia. Se excluyeron todas las secuencias con ese marcador.
Resultado: 8733 -> 8161 secuencias (572 excluidas).

## Filtro type/subtype — AMRFinderPlus
AMR_CDS.fa mezcla genes de resistencia con genes de virulencia y estrés (metal/biocida/
calor/ácido), confirmado cruzando contra ReferenceGeneCatalog.txt (columnas 'type' y
'subtype'). Se detectaron ejemplos claros de contaminación: toxinas de Shiga (stxA2b,
stxA2c, stxA2e), enterotoxinas (seh, ystA, astA), adhesinas de virulencia (air), y una
policétido sintasa de toxina (bonA) -- ninguno es un gen de resistencia a antibióticos.
Se retuvieron únicamente las entradas con type=AMR Y subtype=AMR (excluyendo también
POINT, POINT_DISRUPT y AMR-SUSCEPTIBLE, consistente con el criterio de excluir
resistencia dependiente de SNP ya aplicado a CARD y MEGARes).
Resultado: 9786 -> 8502 secuencias (1284 excluidas).

## Combinación final antes de CD-HIT
30014 (original) - 572 (SNP MEGARes) - 1284 (no-AMR AMRFinderPlus) = 28158 secuencias

## CD-HIT — parámetros y resultado final
- Herramienta: cd-hit-est v4.8.1
- Parámetros: -c 0.95 (identidad) -aS 0.90 -aL 0.90 (coverage) -n 9 -d 0
- Resultado: 28158 -> 2896 secuencias representantes (89.7% de reducción)

## Historial de corridas (para trazabilidad)
1. Sin ningún filtro: 30014 -> 3851 clusters (rechazada, contenía SNP-dependientes y virulencia)
2. Con filtro SNP de MEGARes únicamente: 29442 -> 3507 clusters (rechazada, AMRFinderPlus
   aún contenía virulencia/estrés -- detectado al inspeccionar headers completos tras -d 0)
3. Con ambos filtros (SNP MEGARes + type/subtype AMRFinderPlus): 28158 -> 2896 clusters (FINAL)

## Validación del resultado final
Se inspeccionaron los 5 clusters más grandes de la corrida final, confirmando que
corresponden a familias alélicas con diversidad natural bien documentada:
PDC (1740), OXA-51-like (1396), SHV (1000), TEM (1000), ADC (723) -- todas beta-lactamasas
de diversidad masiva conocida en la literatura. Sin rastro de contaminación por virulencia.

## Archivo final
data/processed/argdb_5fuentes_c95_cov90_filtrado.fasta (+ su .clstr)

## Pendiente antes del entrenamiento del modelo
1. Armonización de categorías de resistencia entre las 5 bases
2. Confirmar con la asesora si BacMet/ResFinderFG deben incluirse

## Decisiones confirmadas con la asesora
- Usar solo las 5 bases del documento oficial (no BacMet/ResFinderFG) -- pendiente confirmación final
- Parámetros CD-HIT: identidad 95%, coverage 90%

## Actualización — respuesta de la asesora
- BacMet: EXCLUIDA definitivamente (co-resistencia a metales pesados, fuera del alcance)
- ResFinderFG: SE INCLUYE (genes identificados por vía funcional, útil para genes nuevos)
- Se requiere generar grupo negativo (genes de no-resistencia) para entrenamiento supervisado

## Actualización — 27 agosto 2026: ResFinderFG y grupo negativo

### Inclusión de ResFinderFG
Se identificó el archivo final correcto (ResFinder_FG.fasta, 3897 secuencias) revisando
el script maestro del repositorio de construcción (RFG_db_construction.sh), confirmando
que pasa por curación + deduplicación (cd-hit -c 1) antes de ser el producto final.
Se verificó ausencia de contaminación por virulencia (búsqueda de palabras clave: 0 hits).
CRLF detectado y corregido (mismo patrón que ResFinder, ambos clonados via git).

### Re-fusión con 6 bases
32055 secuencias antes de CD-HIT (28158 + 3897 ResFinderFG)
-> 4887 clusters tras CD-HIT (c=0.95, aS=aL=0.90) -- este es el GRUPO POSITIVO final.

### Construcción del grupo negativo (metodología de ARGNet)
1. Se descargaron 6 genomas bacterianos completos de referencia (NCBI): E. coli K-12,
   E. coli O157:H7, B. subtilis, B. fragilis, P. putida, L. plantarum.
2. Predicción de ORFs con Prodigal (modo single): 26684 genes candidatos.
3. Filtro de completitud (excluir ORFs parciales en los bordes del contig): 26677 conservados.
4. BLASTn contra la base de ARGs positiva (evalue 1e-5, permisivo a propósito):
   428 ORFs con coincidencia -> excluidos.
5. GRUPO NEGATIVO final: 26249 secuencias sin ningún parecido a ARGs conocidos.

### Balance de clases resultante
Positivo: 4887 | Negativo: 26249 (proporción ~1:5.4)
Pendiente de discutir con la asesora: estrategia de balanceo para el entrenamiento
(submuestreo, sobremuestreo, o ponderación de clases).

### Archivos finales
- data/processed/argdb_6fuentes_c95_cov90_POSITIVO.fasta
- data/processed/argdb_NEGATIVO.fasta

## Corrección — 27 agosto 2026: contaminación de metales/biocidas en MEGARes
Al validar biológicamente los hits del grupo negativo (script validar_hits_excluidos.py),
se detectaron genes de resistencia a METALES (copA, cueR) entre los "genes de resistencia
a antibióticos" que estaban removiendo ORFs del grupo negativo. Investigación reveló que
MEGARes trae un campo 'type' con 4 valores (Drugs: 7570, Metals: 529, Multi-compound: 482,
Biocides: 152) que nunca se había filtrado -- solo se había aplicado el filtro de SNP.

Esto es consistente con la instrucción explícita de la asesora de excluir genes de
resistencia a metales pesados (motivo por el cual se excluyó BacMet).

### Corrección aplicada
Filtro de MEGARes actualizado: type == 'Drugs' AND sin marcador de SNP.
8733 -> 7018 secuencias (1715 excluidas: 572 SNP + 1143 no-Drugs).

### Re-ejecución completa del pipeline (v4, versión definitiva)
- Positivo: 33911 (6 bases sin filtrar) -> 30912 (tras ambos filtros) -> 4228 (tras CD-HIT)
- Negativo: BLAST contra base v4 corregida -> hits bajaron de 428 a 162 (confirma que
  ~266 exclusiones anteriores eran por genes de metales, no de antibióticos)
- Negativo final: 26515 secuencias

### Archivos finales (v4, DEFINITIVOS)
- data/processed/argdb_POSITIVO_final.fasta (4228 secuencias)
- data/processed/argdb_NEGATIVO_final.fasta (26515 secuencias)
Versiones anteriores conservadas con sufijo _DEPRECADO para trazabilidad.

## Validación del grupo negativo con AMRFinderPlus — 16 septiembre 2026
Instrucción de la asesora: validar el grupo negativo con una herramienta externa e
independiente (AMRFinderPlus en modo nucleótido) para descartar contaminación por genes
de resistencia no detectados al construirlo.

Método: se separó el negativo por genoma de origen (6 genomas) y se corrió AMRFinderPlus
(--plus). Para las 2 cepas de E. coli se usó --organism Escherichia; las otras 4 especies
(B. subtilis, B. fragilis, P. putida, L. plantarum) no tienen panel de organismo disponible
en AMRFinderPlus, por lo que su validación fue parcial (sin búsqueda de mutación puntual).

Resultado: 39 hits totales sobre 26,515 secuencias (0.15%). De estos:
- 37 eran VIRULENCE o STRESS/METAL (fuera del alcance del proyecto, no contradicen no_resist)
- 2 eran AMR genuino por mutación puntual (POINT), ambos en E. coli O157:H7:
    * glpT_E448K (resistencia a fosfomicina)
    * pmrB_Y358N (resistencia a colistina)

Decisión: remover esas 2 secuencias del negativo (consistente con la exclusión de genes
SNP-dependientes aplicada en CARD, MEGARes y AMRFinderPlus durante la construcción del positivo).

Limitación documentada: la validación fue completa solo en 2 de 6 genomas (los de E. coli);
en los otros 4 no se pudo verificar resistencia por mutación puntual por falta de panel.

## ESTADO FINAL DEL DATASET
- Positivo: 4,228 secuencias (data/processed/argdb_POSITIVO_final.fasta)
- Negativo: 26,513 secuencias (data/processed/argdb_NEGATIVO_final.fasta)
- Dataset etiquetado: 30,741 secuencias + encabezado (dataset_etiquetado_completo.tsv)
