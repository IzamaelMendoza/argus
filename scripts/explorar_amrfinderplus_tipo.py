import csv
from collections import Counter

catalog_path = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/raw/5_AMRFinderPlus/ReferenceGeneCatalog.txt"
fasta_path = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/amrfinderplus_clean.fasta"

# 1. Distribución completa de type / subtype
combo = Counter()
accesiones_amr = set()

with open(catalog_path, newline='') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        combo[(row['type'], row['subtype'])] += 1
        if row['type'] == 'AMR':
            for col in ('refseq_protein_accession', 'genbank_protein_accession'):
                val = row[col].strip()
                if val:
                    accesiones_amr.add(val)

print("Distribución completa de (type, subtype) en el catálogo:")
for (t, s), c in sorted(combo.items(), key=lambda x: -x[1]):
    print(f"  {t} / {s}: {c}")
print()
print(f"Total de accesiones de proteína únicas con type=AMR: {len(accesiones_amr)}")
print()

# 2. Probar el cruce contra los headers reales del FASTA
total = 0
coinciden = 0
no_coinciden_ejemplos = []

with open(fasta_path) as f:
    for line in f:
        if line.startswith(">"):
            total += 1
            campos = line[1:].strip().split("|")
            accesion = campos[0] if campos else ""
            if accesion in accesiones_amr:
                coinciden += 1
            else:
                if len(no_coinciden_ejemplos) < 10:
                    no_coinciden_ejemplos.append(line.strip())

print(f"Total secuencias en AMR_CDS.fa: {total}")
print(f"Coinciden con accesion tipo=AMR: {coinciden}")
print(f"NO coinciden (posible virulencia/estres/o error de cruce): {total - coinciden}")
print()
print("Muestra de 10 headers que NO coincidieron:")
for h in no_coincidan_ejemplos if False else no_coinciden_ejemplos:
    print(f"  {h}")
