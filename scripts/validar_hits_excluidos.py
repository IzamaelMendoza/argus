import csv
from collections import Counter

blast_tsv = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/orfs_vs_argdb.tsv"

genomas = Counter()
genes_arg = Counter()

with open(blast_tsv) as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        query_id = row[0]
        subject_id = row[1]
        genoma = query_id.split("|")[0]
        genomas[genoma] += 1
        gen_nombre = subject_id.split("|")[-1][:50]
        genes_arg[gen_nombre] += 1

print("Hits por genoma de origen:")
for g, c in genomas.most_common():
    print(f"  {g}: {c}")
print()
print("Genes de tu base de ARGs mas frecuentemente encontrados (posibles genes intrinsecos):")
for gen, c in genes_arg.most_common(15):
    print(f"  {gen}: {c}")
