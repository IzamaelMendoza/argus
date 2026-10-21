input_fasta = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/megares_clean.fasta"
output_fasta = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/megares_nosnp.fasta"

n_total = 0
n_excluidos = 0
n_conservados = 0

with open(input_fasta) as fin, open(output_fasta, "w") as fout:
    escribir = False
    for line in fin:
        if line.startswith(">"):
            n_total += 1
            header = line.rstrip("\n")
            if header.endswith("|RequiresSNPConfirmation"):
                escribir = False
                n_excluidos += 1
            else:
                escribir = True
                n_conservados += 1
        if escribir:
            fout.write(line)

print(f"Total secuencias MEGARes: {n_total}")
print(f"Excluidas (SNP-dependientes): {n_excluidos}")
print(f"Conservadas: {n_conservados}")
