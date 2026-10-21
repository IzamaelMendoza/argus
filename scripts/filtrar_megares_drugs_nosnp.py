input_fasta = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/megares_clean.fasta"
csv_path = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/raw/3_MEGARes/megares_annotations_v3.00.csv"
output_fasta = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/megares_drugs_nosnp.fasta"

import csv
headers_drugs = set()
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['type'] == 'Drugs':
            headers_drugs.add(row['header'])

n_total = 0
n_conservados = 0
with open(input_fasta) as fin, open(output_fasta, "w") as fout:
    escribir = False
    for line in fin:
        if line.startswith(">"):
            n_total += 1
            header = line[1:].rstrip("\n")
            es_snp = header.endswith("|RequiresSNPConfirmation")
            es_drugs = header in headers_drugs
            escribir = es_drugs and not es_snp
            if escribir:
                n_conservados += 1
        if escribir:
            fout.write(line)

print(f"Total secuencias MEGARes: {n_total}")
print(f"Conservadas (type=Drugs, sin SNP): {n_conservados}")
