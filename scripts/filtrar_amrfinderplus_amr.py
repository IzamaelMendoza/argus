import csv

catalog_path = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/raw/5_AMRFinderPlus/ReferenceGeneCatalog.txt"
input_fasta = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/amrfinderplus_clean.fasta"
output_fasta = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/amrfinderplus_soloamr.fasta"

# Construir el set de accesiones que son AMR puro (sin POINT, POINT_DISRUPT, AMR-SUSCEPTIBLE, VIRULENCE, STRESS)
accesiones_amr = set()
with open(catalog_path, newline='') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        if row['type'] == 'AMR' and row['subtype'] == 'AMR':
            for col in ('refseq_protein_accession', 'genbank_protein_accession'):
                val = row[col].strip()
                if val:
                    accesiones_amr.add(val)

print(f"Accesiones unicas con type=AMR y subtype=AMR: {len(accesiones_amr)}")

# Filtrar el FASTA
n_total = 0
n_conservados = 0
n_excluidos = 0

with open(input_fasta) as fin, open(output_fasta, "w") as fout:
    escribir = False
    for line in fin:
        if line.startswith(">"):
            n_total += 1
            accesion = line[1:].strip().split("|")[0]
            if accesion in accesiones_amr:
                escribir = True
                n_conservados += 1
            else:
                escribir = False
                n_excluidos += 1
        if escribir:
            fout.write(line)

print(f"Total secuencias en AMR_CDS.fa: {n_total}")
print(f"Conservadas (AMR puro): {n_conservados}")
print(f"Excluidas (POINT, POINT_DISRUPT, AMR-SUSCEPTIBLE, VIRULENCE, STRESS, etc): {n_excluidos}")
