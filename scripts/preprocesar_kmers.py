from itertools import product
from collections import Counter
import csv

INTERIM = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim"
K = 4
kmers = ["".join(p) for p in product("ACGT", repeat=K)]
kmer_idx = {km: i for i, km in enumerate(kmers)}

def vector_kmer(seq):
    seq = seq.upper()
    counts = Counter(seq[i:i+K] for i in range(len(seq) - K + 1))
    total = sum(counts.values())
    vec = [0.0] * len(kmers)
    if total > 0:
        for km, c in counts.items():
            if km in kmer_idx:   # ignora k-mers con N u otras letras raras
                vec[kmer_idx[km]] = c / total
    return vec

inp = f"{INTERIM}/dataset_fragmentado_500.tsv"
out = f"{INTERIM}/features_kmer4_fragmentado.csv"

n = 0
with open(inp) as fin, open(out, "w", newline="") as fout:
    reader = csv.DictReader(fin, delimiter="\t")
    writer = csv.writer(fout)
    writer.writerow(["id", "etiqueta"] + kmers)
    for row in reader:
        vec = vector_kmer(row["lectura"])
        writer.writerow([row["id"], row["etiqueta"]] + [f"{x:.6f}" for x in vec])
        n += 1

print(f"Fragmentos procesados: {n}")
print(f"Columnas por fragmento: {len(kmers)} k-mers + id + etiqueta")
print(f"Guardado en: features_kmer4_fragmentado.csv")
