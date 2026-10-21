import re
import random
import sys

clstr_path = "combinado_c95_cov90.fasta.clstr"
clusters_a_revisar = ["1080", "2102", "1880", "1673"]

random.seed(1)

with open(clstr_path) as f:
    content = f.read()

blocks = content.split(">Cluster ")
for block in blocks[1:]:
    lines = block.strip().split("\n")
    cid = lines[0].strip()
    if cid not in clusters_a_revisar:
        continue

    members = []
    for line in lines[1:]:
        m = re.search(r"(\d+)nt, >(.+?)\.\.\.", line)
        if m:
            length = int(m.group(1))
            header = m.group(2)
            is_rep = line.strip().endswith("*")
            members.append((length, header, is_rep))

    lengths = [m[0] for m in members]
    print(f"===== Cluster {cid} =====")
    print(f"Total de miembros: {len(members)}")
    print(f"Longitud: min={min(lengths)} max={max(lengths)} promedio={sum(lengths)/len(lengths):.0f}")

    rep = [m for m in members if m[2]][0]
    print(f"REPRESENTANTE: {rep[0]}nt | {rep[1]}")
    print()
    print("Muestra aleatoria de 15 miembros:")
    muestra = random.sample(members, min(15, len(members)))
    for length, header, is_rep in muestra:
        marca = " <- representante" if is_rep else ""
        print(f"  {length}nt | {header}{marca}")
    print()
