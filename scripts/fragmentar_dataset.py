INTERIM = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim"
VENTANA = 500

def leer_fasta(path):
    seqs, header, chunks = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header: seqs[header] = "".join(chunks)
                header = line[1:].split()[0]; chunks = []
            else:
                chunks.append(line.strip())
    if header: seqs[header] = "".join(chunks)
    return seqs

def fragmentar(seq, ventana=VENTANA):
    L = len(seq)
    if L < ventana:
        return [seq]
    return [seq[i*ventana:(i+1)*ventana] for i in range(L // ventana)]

positivas = leer_fasta(f"{INTERIM}/combinado_v4_reducido.fasta")
negativas = leer_fasta(f"{INTERIM}/grupo_negativo_v5.fasta")

filas = []
for grupo, seqs, etiqueta in [("pos", positivas, 1), ("neg", negativas, 0)]:
    for header, seq in seqs.items():
        for idx, frag in enumerate(fragmentar(seq)):
            frag_id = f"{header}__frag{idx}"
            filas.append((frag_id, frag, etiqueta))

# Archivo con identificador, lectura y etiqueta (formato pedido por la asesora)
out = f"{INTERIM}/dataset_fragmentado_500.tsv"
with open(out, "w") as fout:
    fout.write("id\tlectura\tetiqueta\n")
    for fid, frag, et in filas:
        fout.write(f"{fid}\t{frag}\t{et}\n")

# Resumen
n_pos = sum(1 for _, _, e in filas if e == 1)
n_neg = sum(1 for _, _, e in filas if e == 0)
longs = [len(frag) for _, frag, _ in filas]
n_cortos = sum(1 for L in longs if L < VENTANA)

print(f"Total de fragmentos: {len(filas)}")
print(f"  Positivos (1): {n_pos}")
print(f"  Negativos (0): {n_neg}")
print(f"  Fragmentos de longitud completa (500 pb): {len(filas) - n_cortos}")
print(f"  Fragmentos cortos conservados (< 500 pb): {n_cortos}")
print(f"Guardado en: dataset_fragmentado_500.tsv")
