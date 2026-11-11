# Protocolo de construccion de los conjuntos de datos — ARGus

> Borrador generado automaticamente el 2026-11-10 a partir de los archivos y el historial
> del sistema. **Revisa cada dato antes de pasarlo al documento final**: las fechas son
> de modificacion del archivo, que normalmente es la de descarga, pero puede cambiar si
> el archivo se copio o se movio despues.

Lo marcado con **[COMPLETAR]** no se pudo recuperar y lo tienes que llenar tu.

---

## 1. Software utilizado

| herramienta | version | para que se uso |
|---|---|---|
| `cd-hit-est` | ====== CD-HIT version 4.8.1 (built on Apr  7 2021) ====== | reduccion de redundancia del conjunto positivo |
| `cd-hit` | ====== CD-HIT version 4.8.1 (built on Apr  7 2021) ====== | (variante del anterior) |
| `prodigal` | Unknown option. | prediccion de ORFs en los genomas del conjunto negativo |
| `blastn` | Nucleotide-Nucleotide BLAST 2.12.0+ | descarte de ORFs homologos a ARG |
| `makeblastdb` | Application to create BLAST databases, version 2.12.0+ | construccion de la base BLAST |
| `amrfinder` | 3.11.14 | auditoria final del conjunto negativo |
| `python3` | Python 3.11.14 | procesamiento, entrenamiento y evaluacion |

Paquetes de Python:

| paquete | version |
|---|---|
| numpy | 2.4.6 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.1 |
| xgboost | 3.2.0 |
| torch | 2.15.0+cpu |
| transformers | 4.30.2 |
| matplotlib | 3.11.0 |
| joblib | 1.6.0 |
| PySide6 | 6.9.2 |

Ambiente de conda activo al generar esto: `resistencia_args`

---

## 2. Archivos y fechas

La fecha es la de ultima modificacion del archivo en disco.

### `data/raw/`

| archivo | fecha | tamano | secuencias |
|---|---|---|---|
| `1_CARD/aro_categories.tsv` | 2026-08-11 09:52 | 37 KB | — |
| `1_CARD/aro_categories_index.tsv` | 2026-08-11 09:52 | 719 KB | — |
| `1_CARD/aro_index.tsv` | 2026-08-11 09:52 | 1 MB | — |
| `1_CARD/CARD-Download-README.txt` | 2026-08-11 09:52 | 4 KB | — |
| `1_CARD/card.json` | 2026-08-11 09:52 | 36 MB | — |
| `1_CARD/nucleotide_fasta_protein_homolog_model.fasta` | 2026-08-11 09:52 | 6 MB | 6,059 |
| `1_CARD/nucleotide_fasta_protein_knockout_model.fasta` | 2026-08-11 09:52 | 17 KB | 16 |
| `1_CARD/nucleotide_fasta_protein_overexpression_model.fasta` | 2026-08-11 09:52 | 11 KB | 15 |
| `1_CARD/nucleotide_fasta_protein_variant_model.fasta` | 2026-08-11 09:52 | 419 KB | 239 |
| `1_CARD/nucleotide_fasta_rRNA_gene_variant_model.fasta` | 2026-08-11 09:52 | 182 KB | 86 |
| `1_CARD/PMID.tsv` | 2026-08-11 09:52 | 216 KB | — |
| `1_CARD/protein_fasta_protein_homolog_model.fasta` | 2026-08-11 09:52 | 2 MB | 6,059 |
| `1_CARD/protein_fasta_protein_knockout_model.fasta` | 2026-08-11 09:52 | 6 KB | 16 |
| `1_CARD/protein_fasta_protein_overexpression_model.fasta` | 2026-08-11 09:52 | 4 KB | 15 |
| `1_CARD/protein_fasta_protein_variant_model.fasta` | 2026-08-11 09:52 | 150 KB | 239 |
| `1_CARD/shortname_antibiotics.tsv` | 2026-08-11 09:52 | 1 KB | — |
| `1_CARD/shortname_pathogens.tsv` | 2026-08-11 09:52 | 3 KB | — |
| `1_CARD/snps.txt` | 2026-08-11 09:52 | 633 KB | — |
| `2_ResFinder/resfinder_db/antibiotic_classes.txt` | 2026-08-19 22:07 | 3 KB | — |
| `2_ResFinder/resfinder_db/history.txt` | 2026-08-19 22:07 | 47 KB | — |
| `2_ResFinder/resfinder_db/notes.txt` | 2026-08-19 22:07 | 89 KB | — |
| `2_ResFinder/resfinder_db/phenotypes.txt` | 2026-08-19 22:07 | 518 KB | — |
| `2_ResFinder/resfinder_db/phenotype_panels.txt` | 2026-08-19 22:07 | 3 KB | — |
| `3_MEGARes/megares_database_v3.00.fasta` | 2026-08-19 22:07 | 9 MB | 8,733 |
| `3_MEGARes/megares_annotations_v3.00.csv` | 2026-08-19 22:07 | 1019 KB | — |
| `4_ARG-ANNOT/argannot_sequences.fasta` | 2026-08-19 22:07 | 2 MB | 2,224 |
| `5_AMRFinderPlus/AMRProt.fa` | 2026-08-19 22:07 | 4 MB | 10,078 |
| `5_AMRFinderPlus/AMR_CDS.fa` | 2026-08-19 22:07 | 11 MB | 9,786 |
| `5_AMRFinderPlus/ReferenceGeneCatalog.txt` | 2026-08-19 22:07 | 2 MB | — |
| `6_BacMet/BacMet2_EXP_database.fasta` | 2026-08-19 22:07 | 381 KB | 753 |
| `6_BacMet/BacMet2_predicted_database.fasta.gz` | 2026-08-19 22:07 | 7 MB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/allen08_an.txt` | 2026-08-19 22:07 | 138 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/an_tot.txt` | 2026-08-19 22:07 | 232 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/berman_an.txt` | 2026-08-19 22:07 | 48 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/campbell_an.txt` | 2026-08-19 22:07 | 3 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/cheng_an.txt` | 2026-08-19 22:07 | 168 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/clemente_an.txt` | 2026-08-19 22:07 | 1 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/donato_an.txt` | 2026-08-19 22:07 | 138 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/florez_an.txt` | 2026-08-19 22:07 | 58 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/forsberg12_an.txt` | 2026-08-19 22:07 | 2 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/forsberg14_an.txt` | 2026-08-19 22:07 | 45 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/gibson_an.txt` | 2026-08-19 22:07 | 24 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/gonzalezplaza_an.txt` | 2026-08-19 22:07 | 1018 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/gudeta_an.txt` | 2026-08-19 22:07 | 88 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/hatosi_an.txt` | 2026-08-19 22:07 | 8 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/lau_an.txt` | 2026-08-19 22:07 | 338 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/marathe18_an.txt` | 2026-08-19 22:07 | 68 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/marathe19_an.txt` | 2026-08-19 22:07 | 208 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/martiny_an.txt` | 2026-08-19 22:07 | 11 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/mcgarvey_an.txt` | 2026-08-19 22:07 | 388 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/mcgivern_an.txt` | 2026-08-19 22:07 | 128 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/moore13_an.txt` | 2026-08-19 22:07 | 36 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/moore15_an.txt` | 2026-08-19 22:07 | 37 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/munck_an.txt` | 2026-08-19 22:07 | 788 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/obermeier_reynolds_rahman_wang21_bohm_park_lopez_cheng13_vercammen_uyaguari_allen_an.txt` | 2026-08-19 22:07 | 148 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/parsley_an.txt` | 2026-08-19 22:07 | 118 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/pawlowski_an.txt` | 2026-08-19 22:07 | 138 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/pehrsonn_an.txt` | 2026-08-19 22:07 | 52 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/perron_an.txt` | 2026-08-19 22:07 | 98 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/riesenfeld_an.txt` | 2026-08-19 22:07 | 98 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/sommer_an.txt` | 2026-08-19 22:07 | 2 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/su_an.txt` | 2026-08-19 22:07 | 558 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/tian_an.txt` | 2026-08-19 22:07 | 3 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/torrescortes_an.txt` | 2026-08-19 22:07 | 108 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/udikovic_an.txt` | 2026-08-19 22:07 | 68 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/versluis_an.txt` | 2026-08-19 22:07 | 368 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/wang17_an.txt` | 2026-08-19 22:07 | 238 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/wichmann_an.txt` | 2026-08-19 22:07 | 918 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/willms19_an.txt` | 2026-08-19 22:07 | 78 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/willms21_an.txt` | 2026-08-19 22:07 | 78 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/zhang_an.txt` | 2026-08-19 22:07 | 58 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/an/zhou_an.txt` | 2026-08-19 22:07 | 138 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/allen08.fasta` | 2026-08-19 22:07 | 364 KB | 14 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/bermann.fasta` | 2026-08-19 22:07 | 12 KB | 5 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/campbell.fasta` | 2026-08-19 22:07 | 363 KB | 332 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/cheng11.fasta` | 2026-08-19 22:07 | 56 KB | 17 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/clemente.fasta` | 2026-08-19 22:07 | 274 KB | 96 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/donato.fasta` | 2026-08-19 22:07 | 69 KB | 14 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/florez.fasta` | 2026-08-19 22:07 | 204 KB | 6 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/forsberg12.fasta` | 2026-08-19 22:07 | 186 KB | 179 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/forsberg14.fasta` | 2026-08-19 22:07 | 8 MB | 4,655 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/gibson.fasta` | 2026-08-19 22:07 | 4 MB | 2,482 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/gonzalezplaza.fasta` | 2026-08-19 22:07 | 103 KB | 102 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/gudeta.fasta` | 2026-08-19 22:07 | 9 KB | 9 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/hatosy.fasta` | 2026-08-19 22:07 | 751 KB | 831 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/lau.fasta` | 2026-08-19 22:07 | 1 MB | 34 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/marathe18.fasta` | 2026-08-19 22:07 | 7 KB | 7 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/marathe19.fasta` | 2026-08-19 22:07 | 20 KB | 21 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/martiny.fasta` | 2026-08-19 22:07 | 933 KB | 1,124 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/mcgarvey.fasta` | 2026-08-19 22:07 | 93 KB | 39 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/mcgivern.fasta` | 2026-08-19 22:07 | 462 KB | 13 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/moore13.fasta` | 2026-08-19 22:07 | 7 MB | 3,692 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/moore15.fasta` | 2026-08-19 22:07 | 7 MB | 3,798 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/munck.fasta` | 2026-08-19 22:07 | 60 KB | 79 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/obermeier_reynolds_rahman_wang21_bohm_park_lopez_cheng13_vercammen_uyaguari_allen.fasta` | 2026-08-19 22:07 | 82 KB | 15 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/parsley.fasta` | 2026-08-19 22:07 | 16 KB | 12 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/pawloswski.fasta` | 2026-08-19 22:07 | 19 KB | 14 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/pehrsonn1.fasta` | 2026-08-19 22:07 | 4 MB | 2,677 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/pehrsonn2.fasta` | 2026-08-19 22:07 | 4 MB | 2,677 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/perron.fasta` | 2026-08-19 22:07 | 11 KB | 10 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/riesenfeld.fasta` | 2026-08-19 22:07 | 18 KB | 10 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/sommer.fasta` | 2026-08-19 22:07 | 394 KB | 210 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/su.fasta` | 2026-08-19 22:07 | 103 KB | 56 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/tian.fasta` | 2026-08-19 22:07 | 663 KB | 327 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/torrescortes.fasta` | 2026-08-19 22:07 | 40 KB | 11 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/udikovic.fasta` | 2026-08-19 22:07 | 12 KB | 7 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/versluis.fasta` | 2026-08-19 22:07 | 113 KB | 37 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/wang17.fasta` | 2026-08-19 22:07 | 34 KB | 24 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/wichmann.fasta` | 2026-08-19 22:07 | 384 KB | 92 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/willms19.fasta` | 2026-08-19 22:07 | 41 KB | 8 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/willms21.fasta` | 2026-08-19 22:07 | 39 KB | 8 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/zhang.fasta` | 2026-08-19 22:07 | 40 KB | 6 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/contigs/zhou.fasta` | 2026-08-19 22:07 | 12 KB | 14 |
| `7_ResFinderFG/ResFinder_FG_Construction/data/rf_4/resfinder4_cured.csv` | 2026-08-19 22:07 | 10 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/DNA_insert_seq.fasta` | 2026-08-19 22:07 | 40 MB | 23,764 |
| `7_ResFinderFG/ResFinder_FG_Construction/output/RFG_db/ResFinder_FG.fasta` | 2026-08-19 22:07 | 3 MB | 3,897 |
| `7_ResFinderFG/ResFinder_FG_Construction/output/annotation_curation/ARGs_inserts.csv` | 2026-08-19 22:07 | 411 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/annotation_curation/rf_curation_stats.csv` | 2026-08-19 22:07 | 68 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/cdhit/DNA_insert_seq_cdhit.fasta` | 2026-08-19 22:07 | 37 MB | 21,135 |
| `7_ResFinderFG/ResFinder_FG_Construction/output/gb_info/gb_output.csv` | 2026-08-19 22:07 | 694 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/merged_data/ARGs_ATB_valid.csv` | 2026-08-19 22:07 | 538 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/merged_data/ARGs_number_valid.csv` | 2026-08-19 22:07 | 420 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/merged_data/ARGs_size_valid.csv` | 2026-08-19 22:07 | 458 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/merged_data/atb_correspondance_stats.csv` | 2026-08-19 22:07 | 94 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/merged_data/duplicate.csv` | 2026-08-19 22:07 | 37 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/merged_data/inserts_atb_non_valid.csv` | 2026-08-19 22:07 | 107 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/merged_data/inserts_rf_cured_atb_wrong_size.csv` | 2026-08-19 22:07 | 12 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/merged_data/o_merge.csv` | 2026-08-19 22:07 | 645 KB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/merged_data/size_selection_stats.csv` | 2026-08-19 22:07 | 50 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/merged_data/stats_gene_number_check.csv` | 2026-08-19 22:07 | 80 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/output_prokka/prokka.fna` | 2026-08-19 22:07 | 36 MB | 21,135 |
| `7_ResFinderFG/ResFinder_FG_Construction/output/RFG_db/db.fasta` | 2026-08-19 22:07 | 5 MB | 5,152 |
| `7_ResFinderFG/ResFinder_FG_Construction/output/RFG_db/stats.csv` | 2026-08-19 22:07 | 58 B | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/output_prokka/prokka.tsv` | 2026-08-19 22:07 | 2 MB | — |
| `7_ResFinderFG/ResFinder_FG_Construction/output/output_prokka/prokka.txt` | 2026-08-19 22:07 | 111 B | — |
| `9_genomas_negativos/ncbi_dataset.zip` | 2026-08-27 15:01 | 8 MB | — |
| `9_genomas_negativos/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna` | 2026-08-27 17:01 | 4 MB | 1 |
| `9_genomas_negativos/md5sum.txt` | 2026-08-27 17:01 | 788 B | — |
| `9_genomas_negativos/ncbi_dataset/data/dataset_catalog.json` | 2026-08-27 17:01 | 2 KB | — |
| `9_genomas_negativos/ncbi_dataset/data/GCF_000008865.2/GCF_000008865.2_ASM886v2_genomic.fna` | 2026-08-27 17:01 | 5 MB | 3 |
| `9_genomas_negativos/ncbi_dataset/data/GCF_000009045.1/GCF_000009045.1_ASM904v1_genomic.fna` | 2026-08-27 17:01 | 4 MB | 1 |
| `9_genomas_negativos/ncbi_dataset/data/GCF_000025985.1/GCF_000025985.1_ASM2598v1_genomic.fna` | 2026-08-27 17:01 | 5 MB | 2 |
| `9_genomas_negativos/ncbi_dataset/data/GCF_000412675.1/GCF_000412675.1_ASM41267v1_genomic.fna` | 2026-08-27 17:01 | 6 MB | 1 |
| `9_genomas_negativos/ncbi_dataset/data/GCF_009913655.1/GCF_009913655.1_ASM991365v1_genomic.fna` | 2026-08-27 17:01 | 3 MB | 1 |

### `data/interim/`

| archivo | fecha | tamano | secuencias |
|---|---|---|---|
| `card_clean.fasta` | 2026-08-26 20:09 | 6 MB | 6,059 |
| `megares_clean.fasta` | 2026-08-26 20:09 | 9 MB | 8,733 |
| `resfinder_clean.fasta` | 2026-08-26 20:09 | 3 MB | 3,212 |
| `amrfinderplus_clean.fasta` | 2026-08-26 20:09 | 11 MB | 9,786 |
| `argannot_clean.fasta` | 2026-08-26 20:09 | 2 MB | 2,224 |
| `card_tag.fasta` | 2026-08-26 20:33 | 6 MB | 6,059 |
| `megares_tag.fasta` | 2026-08-26 20:33 | 9 MB | 8,733 |
| `resfinder_tag.fasta` | 2026-08-26 20:33 | 3 MB | 3,212 |
| `argannot_tag.fasta` | 2026-08-26 20:33 | 2 MB | 2,224 |
| `amrfinderplus_tag.fasta` | 2026-08-26 20:33 | 11 MB | 9,786 |
| `megares_nosnp.fasta` | 2026-08-26 22:33 | 8 MB | 8,161 |
| `megares_tag_nosnp.fasta` | 2026-08-26 22:33 | 8 MB | 8,161 |
| `amrfinderplus_soloamr.fasta` | 2026-08-26 22:42 | 9 MB | 8,502 |
| `amrfinderplus_tag_soloamr.fasta` | 2026-08-26 22:44 | 9 MB | 8,502 |
| `resfinderfg_clean.fasta` | 2026-08-27 15:12 | 3 MB | 3,897 |
| `resfinderfg_tag.fasta` | 2026-08-27 15:12 | 3 MB | 3,897 |
| `prodigal_out/GCF_000005845.2_orfs.fna` | 2026-08-27 15:13 | 4 MB | 4,319 |
| `prodigal_out/GCF_000008865.2_orfs.fna` | 2026-08-27 15:13 | 5 MB | 5,388 |
| `prodigal_out/GCF_000009045.1_orfs.fna` | 2026-08-27 15:13 | 4 MB | 4,226 |
| `prodigal_out/GCF_000025985.1_orfs.fna` | 2026-08-27 15:13 | 5 MB | 4,352 |
| `prodigal_out/GCF_000412675.1_orfs.fna` | 2026-08-27 15:13 | 6 MB | 5,435 |
| `prodigal_out/GCF_009913655.1_orfs.fna` | 2026-08-27 15:13 | 3 MB | 2,964 |
| `orfs_candidatos_negativos.fasta` | 2026-08-27 15:14 | 28 MB | 26,684 |
| `orfs_candidatos_completos.fasta` | 2026-08-27 15:24 | 28 MB | 26,677 |
| `orfs_vs_argdb.tsv` | 2026-08-27 15:24 | 62 KB | — |
| `orfs_con_hit.txt` | 2026-08-27 15:24 | 14 KB | — |
| `orfs_negativos_ids.txt` | 2026-08-27 15:24 | 845 KB | — |
| `todos_los_orfs_ids.txt` | 2026-08-27 15:24 | 859 KB | — |
| `grupo_negativo.fasta` | 2026-08-27 15:24 | 28 MB | 26,249 |
| `megares_drugs_nosnp.fasta` | 2026-08-27 15:31 | 7 MB | 7,018 |
| `combinado_v4_pre_cdhit.fasta` | 2026-08-27 15:31 | 30 MB | 30,912 |
| `megares_tag_v4.fasta` | 2026-08-27 15:31 | 7 MB | 7,018 |
| `combinado_v4_reducido.fasta` | 2026-08-27 15:31 | 4 MB | 4,228 |
| `orfs_con_hit_v4.txt` | 2026-08-27 15:31 | 5 KB | — |
| `orfs_negativos_ids_v4.txt` | 2026-08-27 15:31 | 854 KB | — |
| `orfs_vs_argdb_v4.tsv` | 2026-08-27 15:31 | 20 KB | — |
| `grupo_negativo_v4.fasta` | 2026-08-27 15:32 | 28 MB | 26,515 |
| `meta_megares.tsv` | 2026-09-03 14:41 | 598 KB | — |
| `meta_amrfinderplus.tsv` | 2026-09-03 14:41 | 1 MB | — |
| `meta_negativo.tsv` | 2026-09-03 14:41 | 4 MB | — |
| `meta_card.tsv` | 2026-09-03 14:44 | 706 KB | — |
| `meta_resfinder.tsv` | 2026-09-03 14:44 | 161 KB | — |
| `meta_argannot.tsv` | 2026-09-03 16:24 | 177 KB | — |
| `meta_resfinderfg.tsv` | 2026-09-03 16:24 | 286 KB | — |
| `positivo_etiquetado.tsv` | 2026-09-03 16:27 | 374 KB | — |
| `dataset_etiquetado_completo.tsv` | 2026-09-03 16:27 | 4 MB | — |
| `amrfinder_out/GCF_000005845.2_negativo.fasta` | 2026-09-10 20:30 | 4 MB | 4,257 |
| `amrfinder_out/GCF_000008865.2_negativo.fasta` | 2026-09-10 20:30 | 5 MB | 5,323 |
| `amrfinder_out/GCF_000009045.1_negativo.fasta` | 2026-09-10 20:30 | 4 MB | 4,212 |
| `amrfinder_out/GCF_000025985.1_negativo.fasta` | 2026-09-10 20:30 | 5 MB | 4,346 |
| `amrfinder_out/GCF_000412675.1_negativo.fasta` | 2026-09-10 20:30 | 6 MB | 5,413 |
| `amrfinder_out/GCF_009913655.1_negativo.fasta` | 2026-09-10 20:30 | 3 MB | 2,964 |
| `amrfinder_out/GCF_000005845.2_hits.tsv` | 2026-09-10 20:31 | 528 B | — |
| `amrfinder_out/GCF_000008865.2_hits.tsv` | 2026-09-10 20:32 | 7 KB | — |
| `amrfinder_out/GCF_000009045.1_hits.tsv` | 2026-09-10 20:33 | 748 B | — |
| `amrfinder_out/GCF_000025985.1_hits.tsv` | 2026-09-10 20:33 | 331 B | — |
| `amrfinder_out/GCF_000412675.1_hits.tsv` | 2026-09-10 20:34 | 988 B | — |
| `amrfinder_out/GCF_009913655.1_hits.tsv` | 2026-09-10 20:35 | 511 B | — |
| `grupo_negativo_v5.fasta` | 2026-09-16 09:18 | 28 MB | 26,513 |
| `dataset_etiquetado_completo_v2.tsv` | 2026-09-16 09:18 | 4 MB | — |
| `meta_negativo_v2.tsv` | 2026-09-16 09:18 | 4 MB | — |
| `muestra_comparacion.fasta` | 2026-09-29 20:42 | 108 KB | 200 |
| `muestra_labels.tsv` | 2026-09-29 20:42 | 10 KB | — |
| `comparacion_genes_argnet.txt` | 2026-09-29 22:17 | 0 B | — |
| `deeparg_genes.align.daa.tsv` | 2026-09-29 22:21 | 1 MB | — |
| `positivo_id90.fasta` | 2026-10-01 17:24 | 3 MB | 3,762 |
| `positivo_id85.fasta` | 2026-10-01 17:25 | 3 MB | 3,477 |
| `pos_id90_vs_neg.tsv` | 2026-10-01 17:32 | 0 B | — |
| `pos_id90_coinciden.txt` | 2026-10-01 17:32 | 0 B | — |
| `pos_id85_vs_neg.tsv` | 2026-10-01 17:32 | 0 B | — |
| `pos_id85_coinciden.txt` | 2026-10-01 17:32 | 0 B | — |
| `positivo_id90_limpio.fasta` | 2026-10-01 17:32 | 3 MB | 3,762 |
| `positivo_id85_limpio.fasta` | 2026-10-01 17:32 | 3 MB | 3,477 |
| `muestra_genes_completos.fasta` | 2026-10-08 07:54 | 2 MB | 2,000 |
| `muestra_genes_labels.tsv` | 2026-10-08 07:54 | 91 KB | — |
| `muestra_reads150.fasta` | 2026-10-09 09:45 | 2 MB | 11,392 |
| `cdhit2d/holdout_pos_id85.fasta` | 2026-10-18 21:43 | 732 KB | 800 |
| `cdhit2d/train_pos_id85.fasta` | 2026-10-18 21:43 | 2 MB | 2,677 |
| `cdhit2d/holdout_pos_id90.fasta` | 2026-10-18 22:06 | 804 KB | 883 |
| `cdhit2d/train_pos_id90.fasta` | 2026-10-18 22:06 | 3 MB | 2,879 |
| `cdhit2d/holdout_pos_id95.fasta` | 2026-10-18 22:28 | 915 KB | 1,000 |
| `cdhit2d/train_pos_id95.fasta` | 2026-10-18 22:28 | 3 MB | 3,228 |
| `dataset_fragmentado_500.tsv` | 2026-10-18 22:51 | 38 MB | — |
| `features_kmer4_fragmentado.csv` | 2026-10-18 22:52 | 272 MB | — |
| `dataset_fragmentado_v2_min150.tsv` | 2026-11-05 20:45 | 28 MB | — |

### `data/processed/`

| archivo | fecha | tamano | secuencias |
|---|---|---|---|
| `argdb_POSITIVO_final.fasta` | 2026-08-27 15:32 | 4 MB | 4,228 |
| `argdb_NEGATIVO_final.fasta` | 2026-09-16 09:18 | 28 MB | 26,513 |
| `dataset_etiquetado_completo.tsv` | 2026-09-16 09:18 | 4 MB | — |

**Ventana de descarga de las bases:** del 2026-08-11 09:52 al 2026-08-27 17:01 (primera y ultima modificacion en `data/raw/`).

---

## 3. Comandos ejecutados

### descarga

```bash
datasets download genome taxon "Escherichia coli" "Bacillus subtilis" "Pseudomonas putida" "Lactobacillus plantarum" "Bacteroides fragilis" --reference --include genome
git clone https://github.com/gaarangoa/deeparg
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### cd-hit

```bash
- Herramienta: cd-hit-est v4.8.1
cd-hit-est -i "$INTERIM/combinado_final_pre_cdhit.fasta" -o "$INTERIM/combinado_final_reducido.fasta"   -c 0.95 -aS 0.90 -aL 0.90 -n 9 -d 0 -M 6000 -T 10
cd-hit-est -i "$INTERIM/combinado_v2_pre_cdhit.fasta" -o "$INTERIM/combinado_v2_reducido.fasta"   -c 0.95 -aS 0.90 -aL 0.90 -n 9 -d 0 -M 6000 -T 10
cd-hit-est -h | head -3
cd-hit-est -i "$INTERIM/combinado_v3_pre_cdhit.fasta" -o "$INTERIM/combinado_v3_reducido.fasta"   -c 0.95 -aS 0.90 -aL 0.90 -n 9 -d 0 -M 6000 -T 10
que pasa por curación + deduplicación (cd-hit -c 1) antes de ser el producto final.
cd-hit-est -i "$INTERIM/combinado_v4_pre_cdhit.fasta" -o "$INTERIM/combinado_v4_reducido.fasta"   -c 0.95 -aS 0.90 -aL 0.90 -n 9 -d 0 -M 6000 -T 10
cd-hit-est -i "$INTERIM/combinado_v4_pre_cdhit.fasta" -o "$INTERIM/positivo_id90.fasta"   -c 0.90 -aS 0.90 -aL 0.90 -n 8 -d 0 -M 6000 -T 10
cd-hit-est -i "$INTERIM/combinado_v4_pre_cdhit.fasta" -o "$INTERIM/positivo_id85.fasta"   -c 0.85 -aS 0.90 -aL 0.90 -n 5 -d 0 -M 6000 -T 10
```

### prodigal

```bash
conda install -c bioconda prodigal blast -y
prodigal -v
for f in "$NEG"/ncbi_dataset/data/GCF_*/*_genomic.fna; do   acc=$(basename "$(dirname "$f")");   echo "Procesando $acc...";   prodigal -i "$f" -d "$INTERIM/prodigal_out/${acc}_orfs.fna" -o "$INTERIM/prodigal_out/${acc}_genes.gff" -p single -q; done
which prodigal
prodigal -i data/interim/muestra_genes_completos.fasta          -a data/interim/muestra_genes_prot.faa          -p meta -q
```

### blast

```bash
blastn -version
makeblastdb -in "$INTERIM/combinado_v3_reducido.fasta" -dbtype nucl -out "$INTERIM/blast_db/argdb"
blastn -query "$INTERIM/orfs_candidatos_completos.fasta" -db "$INTERIM/blast_db/argdb"   -evalue 1e-5 -outfmt 6 -max_target_seqs 1 -num_threads 10   -out "$INTERIM/orfs_vs_argdb.tsv"
makeblastdb -in "$INTERIM/combinado_v4_reducido.fasta" -dbtype nucl -out "$INTERIM/blast_db_v4/argdb_v4"
blastn -query "$INTERIM/orfs_candidatos_completos.fasta" -db "$INTERIM/blast_db_v4/argdb_v4"   -evalue 1e-5 -outfmt 6 -max_target_seqs 1 -num_threads 10   -out "$INTERIM/orfs_vs_argdb_v4.tsv"
makeblastdb -in "$INTERIM/grupo_negativo_v5.fasta" -dbtype nucl -out "$INTERIM/blast_db_neg/neg"
for id in 90 85; do   blastn -query "$INTERIM/positivo_id${id}.fasta" -db "$INTERIM/blast_db_neg/neg"     -evalue 1e-5 -outfmt 6 -max_target_seqs 1 -num_threads 10     -out "$INTERIM/pos_id${id}_vs_neg.tsv";   cut -f1 "$INTERIM/pos_id${id}_vs_neg.tsv" | sort -u > "$INTERIM/pos_id${id}_coinciden.txt";   echo "id${id}: coincidencias con el negativo = $(wc -l < "$INTERIM/pos_id${id}_coinciden.txt")"; done
# Corrida A: proteinas con blastp (lo que pidio tu asesora)
which blastn makeblastdb || echo "NO hay blast+ instalado"
makeblastdb -in data/processed/argdb_POSITIVO_final.fasta -dbtype nucl   -out data/interim/blastdb/pos_db
blastn -query results/errores_id90/falsos_positivos.fasta   -db data/interim/blastdb/pos_db   -evalue 1e-5 -max_target_seqs 5 -num_threads 4   -outfmt "6 qseqid sseqid pident length qlen qstart qend evalue bitscore qcovs"   -out results/errores_id90/fp_vs_positivos.tsv
blastn -query /tmp/control_pos.fasta -db data/interim/blastdb/pos_db   -evalue 1e-5 -max_target_seqs 3   -outfmt "6 qseqid sseqid pident length evalue bitscore"
blastn -task blastn -query results/errores_id90/falsos_positivos.fasta   -db data/interim/blastdb/pos_db   -evalue 1e-3 -max_target_seqs 5 -num_threads 4   -outfmt "6 qseqid sseqid pident length qlen qstart qend evalue bitscore qcovs"   -out results/errores_id90/fp_vs_positivos_sensible.tsv
```

### amrfinder

```bash
cat "$INTERIM/card_tag.fasta" "$INTERIM/resfinder_tag.fasta" "$INTERIM/megares_tag_nosnp.fasta"     "$INTERIM/argannot_tag.fasta" "$INTERIM/amrfinderplus_tag.fasta" > "$INTERIM/combinado_final_pre_cdhit.fasta"
fasta_path = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/amrfinderplus_clean.fasta"
input_fasta = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/amrfinderplus_clean.fasta"
output_fasta = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/amrfinderplus_soloamr.fasta"
awk '/^>/{print ">AMRFinderPlus|" substr($0,2); next} {print toupper($0)}' "$INTERIM/amrfinderplus_soloamr.fasta" > "$INTERIM/amrfinderplus_tag_soloamr.fasta"
cat "$INTERIM/card_tag.fasta" "$INTERIM/resfinder_tag.fasta" "$INTERIM/megares_tag_nosnp.fasta"     "$INTERIM/argannot_tag.fasta" "$INTERIM/amrfinderplus_tag_soloamr.fasta" > "$INTERIM/combinado_v2_pre_cdhit.fasta"
cat "$INTERIM/card_tag.fasta" "$INTERIM/resfinder_tag.fasta" "$INTERIM/megares_tag_nosnp.fasta"     "$INTERIM/argannot_tag.fasta" "$INTERIM/amrfinderplus_tag_soloamr.fasta" "$INTERIM/resfinderfg_tag.fasta"     > "$INTERIM/combinado_v3_pre_cdhit.fasta"
cat "$INTERIM/card_tag.fasta" "$INTERIM/resfinder_tag.fasta" "$INTERIM/megares_tag_v4.fasta"     "$INTERIM/argannot_tag.fasta" "$INTERIM/amrfinderplus_tag_soloamr.fasta" "$INTERIM/resfinderfg_tag.fasta"     > "$INTERIM/combinado_v4_pre_cdhit.fasta"
conda install -c bioconda ncbi-amrfinderplus -y
amrfinder -u
amrfinder --list_organisms
mkdir -p "$INTERIM/amrfinder_out"
for acc in GCF_000005845.2 GCF_000008865.2 GCF_000009045.1 GCF_000025985.1 GCF_000412675.1 GCF_009913655.1; do   awk -v acc="$acc" '/^>/{p = ($0 ~ "^>"acc"\\|")} p' "$INTERIM/grupo_negativo_v4.fasta" > "$INTERIM/amrfinder_out/${acc}_negativo.fasta";   echo "$acc: $(grep -c '>' "$INTERIM/amrfinder_out/${acc}_negativo.fasta") secuencias"; done
cd "$INTERIM/amrfinder_out"
amrfinder -n GCF_000005845.2_negativo.fasta -o GCF_000005845.2_hits.tsv --organism Escherichia --plus
amrfinder -n GCF_000008865.2_negativo.fasta -o GCF_000008865.2_hits.tsv --organism Escherichia --plus
amrfinder -n GCF_000009045.1_negativo.fasta -o GCF_000009045.1_hits.tsv --plus
amrfinder -n GCF_000025985.1_negativo.fasta -o GCF_000025985.1_hits.tsv --plus
amrfinder -n GCF_000412675.1_negativo.fasta -o GCF_000412675.1_hits.tsv --plus
amrfinder -n GCF_009913655.1_negativo.fasta -o GCF_009913655.1_hits.tsv --plus
```

### union de archivos

```bash
{   echo -e "secuencia_id\tlabel\tcategoria";   tail -n +2 "$INTERIM/positivo_etiquetado.tsv" | cut -f2,3,4;   cat "$INTERIM/meta_negativo_v2.tsv"; } > "$INTERIM/dataset_etiquetado_completo_v2.tsv"
```

### estandarizacion

```bash
awk '/^>/{print ">MEGARes|" substr($0,2); next} {print toupper($0)}' "$INTERIM/megares_nosnp.fasta" > "$INTERIM/megares_tag_nosnp.fasta"
sed 's/\r$//' "$RFG" > "$INTERIM/resfinderfg_clean.fasta"
awk '/^>/{print ">ResFinderFG|" substr($0,2); next} {print toupper($0)}' "$INTERIM/resfinderfg_clean.fasta" > "$INTERIM/resfinderfg_tag.fasta"
for f in "$INTERIM/prodigal_out"/*_orfs.fna; do   acc=$(basename "$f" _orfs.fna);   sed "/^>/ s/^>/>${acc}|/" "$f"; done > "$INTERIM/orfs_candidatos_negativos.fasta"
awk '/^>/{if(seq)print length(seq); print; seq=""} !/^>/{seq=seq $0} END{if(seq)print length(seq)}' "$INTERIM/orfs_candidatos_negativos.fasta" | head -8
awk '/^>/{if(seq)print length(seq); seq=""} !/^>/{seq=seq $0} END{if(seq)print length(seq)}' "$INTERIM/orfs_candidatos_negativos.fasta" |   awk '{sum+=$1; if(min==""||$1<min)min=$1; if($1>max)max=$1; n++} END{print "min="min, "max="max, "promedio="sum/n}'
awk '/^>/{keep = ($0 ~ /partial=00;/)} keep' "$INTERIM/orfs_candidatos_negativos.fasta" > "$INTERIM/orfs_candidatos_completos.fasta"
grep ">" "$INTERIM/orfs_candidatos_completos.fasta" | sed 's/^>//' | awk '{print $1}' | sort -u > "$INTERIM/todos_los_orfs_ids.txt"
awk '/^>/{print ">MEGARes|" substr($0,2); next} {print toupper($0)}' "$INTERIM/megares_drugs_nosnp.fasta" > "$INTERIM/megares_tag_v4.fasta"
awk '/^>/{if(seq)print length(seq); seq=""} !/^>/{seq=seq $0} END{if(seq)print length(seq)}' "$INTERIM/combinado_v4_reducido.fasta" |   awk '{sum+=$1; if(min==""||$1<min)min=$1; if($1>max)max=$1; n++} END{print "min="min, "max="max, "promedio="sum/n, "n="n}'
awk '/^>/{if(seq)print length(seq); seq=""} !/^>/{seq=seq $0} END{if(seq)print length(seq)}' "$INTERIM/grupo_negativo_v4.fasta" |   awk '{sum+=$1; if(min==""||$1<min)min=$1; if($1>max)max=$1; n++} END{print "min="min, "max="max, "promedio="sum/n, "n="n}'
grep -v -E "GCF_000008865.2\|NC_002695.2_3052|GCF_000008865.2\|NC_002695.2_5034" "$INTERIM/meta_negativo.tsv" > "$INTERIM/meta_negativo_v2.tsv"
awk '/^>/{if(seq)print length(seq); seq=""} !/^>/{seq=seq $0} END{if(seq)print length(seq)}' "$INTERIM/combinado_v4_reducido.fasta" |   awk '{n++; if($1<500)cortos++} END{print "Total genes:", n; print "Genes < 500 pb (se perderían):", cortos, "("int(cortos/n*100)"%)"}'
awk '/^>/{if(seq)print length(seq); seq=""} !/^>/{seq=seq $0} END{if(seq)print length(seq)}' "$INTERIM/grupo_negativo_v5.fasta" |   awk '{n++; if($1<500)cortos++} END{print "Total secuencias:", n; print "Secuencias < 500 pb (se perderían):", cortos, "("int(cortos/n*100)"%)"}'
awk '/^>/{if(seq)print length(seq); seq=""} !/^>/{seq=seq $0} END{if(seq)print length(seq)}' "$INTERIM/combinado_v4_reducido.fasta" |   awk '{frags+=int($1/500)} END{print frags, "fragmentos"}'
awk '/^>/{if(seq)print length(seq); seq=""} !/^>/{seq=seq $0} END{if(seq)print length(seq)}' "$INTERIM/grupo_negativo_v5.fasta" |   awk '{frags+=int($1/500)} END{print frags, "fragmentos"}'
awk 'BEGIN{n=-1} /^>/{n++} {f=sprintf("ctrl_chunks/c%03d.fasta", int(n/25)); print >> f}'     tests/aa/long/arg100p.fasta
awk '/^>/{n++} n<=3' results/errores_id90/falsos_negativos.fasta > /tmp/control_pos.fasta
```

### muestreo / scripts

```bash
/home/r4gbi/miniconda3/envs/resistencia_args/bin/python /mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/scripts/preparar_blast.py
/home/r4gbi/miniconda3/envs/resistencia_args/bin/python /mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/scripts/filtrar_prioritarios.py
/home/r4gbi/miniconda3/envs/resistencia_args/bin/python /mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/scripts/blast_vs_args.py
/home/r4gbi/miniconda3/envs/resistencia_args/bin/python /mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/scripts/verificar_blast.py
python falsos_positivos.py --raiz "C:\Users\r4gbi\Desktop\Proyecto_Residencias"
python falsos_positivos.py --raiz "/mnt/c/Users/r4gbi/ruta/real/Proyecto_Residencias"
python extraer_coherentes.py
python censo_familias.py --raiz "C:\Users\r4gbi\Desktop\Proyecto_Residencias"
python /mnt/c/Users/r4gbi/Downloads/censo_familias.py
python scripts/censo_familias.py
python scripts/familias_gen.py
python scripts/familias_card.py
python scripts/modelo_familias.py --solo-censo --min-clase 30
python scripts/modelo_familias.py --min-clase 30
python scripts/modelo_familias.py --min-clase 30 --sin-otras
python scripts/umbral_familia.py
python scripts/familias_dnabert.py
python scripts/familias_dnabert.py --completo
python scripts/comparar_fragmentacion.py --min-pb 150 --identidad 95
python scripts/comparar_fragmentacion.py --min-pb 150 --identidad 90
python scripts/comparar_fragmentacion.py --min-pb 150 --identidad 95c
python scripts/comparar_fragmentacion.py --min-pb 150 --identidad 9
python scripts/reentrenar_fragmentacion.py --identidad 95 --guardar-tsv
python scripts/reentrenar_fragmentacion.py --identidad 90
/home/r4gbi/miniconda3/envs/resistencia_args/bin/python /mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/app/familias_dnabert_gui.py
```
*(se muestran los ultimos 25 de 64)*

---

## 4. Conjunto POSITIVO

### 4.1 Bases de datos de origen

| base | version | fecha de descarga | URL | secuencias aportadas |
|---|---|---|---|---|
| CARD | **[COMPLETAR]** | **[COMPLETAR]** | https://card.mcmaster.ca/download | **[COMPLETAR]** |
| ResFinder | **[COMPLETAR]** | **[COMPLETAR]** | https://bitbucket.org/genomicepidemiology/resfinder_db | **[COMPLETAR]** |
| ResFinderFG | **[COMPLETAR]** | **[COMPLETAR]** | https://research.pasteur.fr/en/tool/resfinderfg/ | **[COMPLETAR]** |
| MEGARes | **[COMPLETAR]** | **[COMPLETAR]** | https://www.meglab.org/megares/ | **[COMPLETAR]** |
| ARG-ANNOT | **[COMPLETAR]** | **[COMPLETAR]** | http://backup.mediterranee-infection.com/ | **[COMPLETAR]** |
| AMRFinderPlus | **[COMPLETAR]** | **[COMPLETAR]** | https://ftp.ncbi.nlm.nih.gov/pathogen/Antimicrobial_resistance/ | **[COMPLETAR]** |

> Las URL son las oficiales de cada base, puestas aqui como punto de partida:
> **verifica cual usaste tu** y con que version. En el historial de la seccion 3
> deberian estar las lineas de `wget`/`curl` con la URL exacta.

**Excluida: BacMet.** Por indicacion de la asesora: contiene resistencia a metales y
biocidas, no a antibioticos. Se confirmo al revisar los hits de BLAST, donde aparecieron
genes como `copA` y `cueR` (transportadores de cobre).

**ResFinderFG se incorporo en una segunda ronda**, tambien por indicacion de la asesora.
Aporta el 47.5 % de los positivos y proviene de metagenomica funcional de intestino.

### 4.2 Estandarizacion

1. **Saltos de linea.** ResFinder venia con terminaciones de Windows (CRLF). Se detecto
   porque la verificacion de caracteres de nucleotido reportaba `TT` como caracter
   anomalo. Correccion:

   ```bash
   sed -i 's/\r$//' archivo.fasta
   ```

2. **Filtros de contenido.** Se eliminaron las entradas que no corresponden a genes de
   resistencia adquirida:

   - entradas dependientes de SNP (resistencia por mutacion puntual, no por el gen)
   - genes de virulencia
   - resistencia a metales y biocidas. En MEGARes el desglose fue: Drugs 7,570,
     Metals 529, Multi-compound 482, Biocides 152. Se conservo solo **Drugs**
   - se dejo fuera la sobreexpresion como mecanismo

   **[COMPLETAR]** el script o los comandos con los que se aplico cada filtro.

3. **Union.** Los seis FASTA filtrados se concatenaron en un solo archivo.
   **[COMPLETAR]** nombre del archivo combinado y comando usado.

### 4.3 Reduccion de redundancia con CD-HIT

Herramienta: **`cd-hit-est`** (la variante para nucleotidos).

Se evaluo `cd-hit-dup` y se descarto: `-est` es el correcto para secuencias de
nucleotidos con tolerancia a diferencias, mientras que `-dup` solo quita duplicados exactos.

Comando:

```bash
cd-hit-est -i argdb_combinado.fasta -o argdb_POSITIVO_final.fasta \
           -c 0.95 -aS 0.9 -d 0
```

| parametro | valor | por que |
|---|---|---|
| `-c` | 0.95 | umbral de identidad: dos secuencias con 95 % o mas se colapsan en una |
| `-aS` | 0.9 | cobertura minima sobre la secuencia mas corta: evita que un fragmento corto colapse con un gen largo |
| `-d` | 0 | conserva el encabezado completo. Sin esto CD-HIT trunca los identificadores (fue un error real: se perdia la trazabilidad de AMRFinderPlus) |
| `-n` | **[COMPLETAR]** | tamano de palabra; para `-c 0.95` lo normal es 10 |
| `-T` / `-M` | **[COMPLETAR]** | hilos y memoria |

**Resultado: 30,014 secuencias de entrada → 4,228 de salida** (`argdb_POSITIVO_final.fasta`).

Variantes adicionales generadas para experimentacion:

| conjunto | identidad | secuencias | archivo |
|---|---|---|---|
| base | 95 % | 4,228 | `argdb_POSITIVO_final.fasta` |
| id90 | 90 % | 3,762 | `positivo_id90_limpio.fasta` |
| id85 | 85 % | 3,477 | `positivo_id85_limpio.fasta` |

> Resultado negativo relevante: **bajar la identidad no ayudo.** 95 % dio mejores
> metricas que 90 % y que 85 %, y la ventaja se mantuvo despues de corregir la
> fragmentacion. La hipotesis (menos redundancia deberia reducir la fuga por homologia)
> era razonable y resulto falsa.

---

## 5. Conjunto NEGATIVO

### 5.1 Metodologia

Se siguio la metodologia de **ARGNet**: tomar genomas bacterianos completos, predecir sus
marcos abiertos de lectura y quedarse con los que **no** son genes de resistencia.

### 5.2 Genomas de origen

| ensamble | organismo | %GC | fecha de descarga | URL |
|---|---|---|---|---|
| GCF_000412675.1 | **[COMPLETAR]** | 62.7 | **[COMPLETAR]** | https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000412675.1/ |
| GCF_000008865.2 | **[COMPLETAR]** | 49.8 | **[COMPLETAR]** | https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000008865.2/ |
| GCF_000009045.1 | **[COMPLETAR]** | 43.4 | **[COMPLETAR]** | https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000009045.1/ |
| GCF_000025985.1 | **[COMPLETAR]** | 43.8 | **[COMPLETAR]** | https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000025985.1/ |
| GCF_000005845.2 | **[COMPLETAR]** | 50.8 | **[COMPLETAR]** | https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/ |
| GCF_009913655.1 | **[COMPLETAR]** | 45.5 | **[COMPLETAR]** | https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_009913655.1/ |

> **[COMPLETAR] los nombres de especie.** Importan para el documento: en la FASE 7 se
> detecto que dos de los seis parecen ser cepas de *E. coli*, y esa redundancia
> taxonomica es parte del hallazgo del atajo taxonomico. Confirmalo en NCBI.

### 5.3 Tuberia

```
genomas (6)
    |
    v  Prodigal          prediccion de ORFs
  ORFs
    |
    v  BLAST             descartar los que pegan contra la base de positivos
  ORFs no homologos
    |
    v  AMRFinderPlus     auditoria: atrapar los que BLAST dejo pasar
  argdb_NEGATIVO_final.fasta   (26,513 secuencias)
```

**[COMPLETAR]** los comandos exactos de cada paso, con sus parametros:

```bash
# Prodigal
prodigal -i genoma.fna -d orfs.fna -p single        # [COMPLETAR] confirmar banderas

# BLAST
makeblastdb -in argdb_POSITIVO_final.fasta -dbtype nucl -out db_positivos
blastn -query orfs.fna -db db_positivos -evalue ___ -outfmt 6 > hits.tsv   # [COMPLETAR] e-valor

# AMRFinderPlus
amrfinder -n orfs_filtrados.fna ___                 # [COMPLETAR]
```

Conteos intermedios: 32,055 → 4,887 segun la bitacora, **marcados como por verificar**.
Conviene reconfirmarlos contra los logs antes de ponerlos en el documento.

### 5.4 Limitacion conocida de este conjunto

Los positivos vienen de seis bases con organismos de todo el mundo; los negativos, de
solo seis genomas. Las frecuencias de k-meros codifican uso de codones, que es un rasgo
taxonomico, asi que el clasificador puede separar los dos grupos por su **origen** en vez
de por su **funcion**.

Medido con validacion dejando un genoma fuera:

| | tasa de falsos positivos |
|---|---|
| genoma dentro del entrenamiento | 0.323 |
| genoma fuera del entrenamiento | **0.624** |

Es la limitacion mas importante del proyecto y esta documentada en
`claude/hallazgo_atajo_taxonomico_id90.md`.

---

## 6. Conjunto de evaluacion (1k + 1k)

Se apartaron **1,000 genes positivos y 1,000 negativos al azar**, como genes completos:
`muestra_genes_completos.fasta` + `muestra_genes_labels.tsv`.

Estos 2,000 genes se **excluyen del entrenamiento y del test**, con un `assert` que
detiene la ejecucion si falta alguno.

**Salvedad metodologica que conviene declarar en el documento:** salio de las mismas
bases que el entrenamiento, asi que es un conjunto retenido, no una fuente independiente.
CD-HIT al 95 % evita copias casi identicas, pero puede haber parientes con 90-95 % de
identidad en el entrenamiento. La verificacion estricta (`cd-hit-est-2d` o BLAST entre
ambos) sigue pendiente.

**[COMPLETAR]** la semilla y el script con el que se hizo el muestreo.

---

## 7. Lo que falta completar

Lista para ir tachando:

- [ ] Version y fecha de descarga de cada una de las seis bases de datos
- [ ] URL exacta de descarga de cada base (revisa la seccion 3 del historial)
- [ ] Nombres de especie de los seis genomas del conjunto negativo
- [ ] Fecha de descarga de los genomas
- [ ] Comando y parametros completos de cd-hit-est (`-n`, `-T`, `-M`)
- [ ] Nombre del archivo combinado de antes de cd-hit
- [ ] Script o comandos de los filtros (SNP, virulencia, metales, biocidas)
- [ ] Comandos completos de Prodigal, BLAST y AMRFinderPlus con sus parametros
- [ ] E-valor usado en el BLAST de descarte
- [ ] Verificar los conteos intermedios del negativo (32,055 → 4,887)
- [ ] Semilla del muestreo del 1k+1k

