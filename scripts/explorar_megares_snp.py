import csv
from collections import Counter

csv_path = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/raw/3_MEGARes/megares_annotations_v3.00.csv"

total = 0
n_segmentos = Counter()
sexto_segmento = Counter()

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        partes = row["header"].split("|")
        n_segmentos[len(partes)] += 1
        if len(partes) >= 6:
            sexto_segmento[partes[5]] += 1

print(f"Total de filas (genes) en el CSV: {total}")
print()
print("Cuántos segmentos '|' tiene el campo 'header', y cuántas filas caen en cada caso:")
for n, count in sorted(n_segmentos.items()):
    print(f"  {n} segmentos -> {count} filas")
print()
print("Valores unicos que aparecen en el 6to segmento (cuando existe):")
for valor, count in sexto_segmento.most_common():
    print(f"  '{valor}': {count} filas")
