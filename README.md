# ARGus

Modelo supervisado que predice la presencia de genes de resistencia a antibioticos (ARGs) en archivos FASTA de MAGs, con una interfaz de escritorio (PySide6).

## Resultados principales (1,000 genes ARG + 1,000 no ARG, nivel gen)

| Modelo | Recall | Precision | F1 |
|---|---|---|---|
| ARGus v1 (Random Forest, k=4) | 0.893 | 0.745 | 0.813 |
| ARGus v2 (Random Forest, k=6, positivos al 95 %) | 0.893 | 0.752 | 0.817 |
| DNABERT-2 congelado + cabeza | 0.913 | 0.719 | 0.805 |
| ARGNet (argnet-s, lecturas de 150 nt, regla OR) | 0.589 | 0.882 | 0.706 |
| DeepARG (identidad 30 %) | 0.720 | 0.978 | 0.829 |

## Estructura (ajusta los nombres a tu carpeta)

- `data/interim/`, `data/processed/`: datos de entrada (positivos, negativos, fragmentos, muestra 1k+1k).
- `data/models/`: modelos guardados (`argus_rf_k6_id95.joblib`, etc.).
- `results/`: salidas de ARGNet y DeepARG sobre la muestra de comparacion.
- cuadernos `argus_k6_id95.ipynb`, `argus_k6_id90.ipynb`, `argus_k6_id85.ipynb` (Random Forest) y `argus_dnabert2_id95.ipynb` (DNABERT-2).
- `app/`: interfaz grafica.
- `verificar_modelo.py`: comprueba que el modelo guardado carga y predice en esta maquina.

## Puesta en marcha

1. Clonar y bajar los archivos grandes (Git LFS):

   ```bash
   git clone https://github.com/IzamaelMendoza/argus.git
   cd argus
   git lfs pull
   ```

2. Crear los ambientes (los modelos .joblib dependen de la version de scikit-learn, por eso estan fijadas):

   ```bash
   conda env create -f environment.yml          # analisis, Random Forest y GUI
   conda create -n dnabert python=3.10 -y       # solo para DNABERT-2
   conda activate dnabert
   pip install -r requirements-dnabert.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

3. Verificar que el modelo funciona:

   ```bash
   python verificar_modelo.py --modelo data/models/argus_rf_k6_id95.joblib \
     --fasta data/interim/muestra_genes_completos.fasta \
     --etiquetas data/interim/muestra_genes_labels.tsv --n 40
   ```

4. En los cuadernos, ajusta `INTERIM_MANUAL` (celda 1) a la ruta de `data/interim` de esta maquina.
