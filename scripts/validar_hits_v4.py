import csv
from collections import Counter

blast_tsv = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/orfs_vs_argdb_v4.tsv"

genomas = Counter()
genes_arg = Counter()

with open(blast_tsv) as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        genoma = row[0].split("|")[0]
        genomas[genoma] += 1
        gen_nombre = row[1].split("|")[-1][:50]
        genes_arg[gen_nombre] += 1

print("Hits por genoma (comparar con la corrida anterior: Pseudomonas putida = GCF_000412675.1, antes 40):")
for g, c in genomas.most_common():
    print(f"  {g}: {c}")
print()
print("Genes de la base ARG mas encontrados ahora:")
for gen, c in genes_arg.most_common(15):
    print(f"  {gen}: {c}")
