ids_path = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/orfs_negativos_ids.txt"
fasta_in = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/orfs_candidatos_completos.fasta"
fasta_out = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/grupo_negativo.fasta"

with open(ids_path) as f:
    ids = set(line.strip() for line in f)

n_escritas = 0
with open(fasta_in) as fin, open(fasta_out, "w") as fout:
    escribir = False
    for line in fin:
        if line.startswith(">"):
            header_id = line[1:].split()[0]
            escribir = header_id in ids
            if escribir:
                n_escritas += 1
        if escribir:
            fout.write(line)

print(f"Secuencias escritas al grupo negativo: {n_escritas}")
print(f"IDs esperados: {len(ids)}")
