import re

clstr_path = "combinado_c95_cov90.fasta.clstr"
fasta_path = "combinado_raw.fasta"
cluster_objetivo = "1080"

# Cargar todos los headers completos del archivo original
headers_completos = []
with open(fasta_path) as f:
    for line in f:
        if line.startswith(">"):
            headers_completos.append(line[1:].rstrip("\n"))

# Extraer los miembros truncados del cluster objetivo
with open(clstr_path) as f:
    content = f.read()

blocks = content.split(">Cluster ")
miembros_truncados = []
for block in blocks[1:]:
    lines = block.strip().split("\n")
    if lines[0].strip() != cluster_objetivo:
        continue
    for line in lines[1:]:
        m = re.search(r">(.+?)\.\.\.", line)
        if m:
            miembros_truncados.append(m.group(1))

print(f"Miembros truncados encontrados en Cluster {cluster_objetivo}: {len(miembros_truncados)}")
print()
print("Cruzando contra headers completos (mostrando 20 al azar):")
import random
random.seed(2)
muestra = random.sample(miembros_truncados, min(20, len(miembros_truncados)))

for trunc in muestra:
    coincidencias = [h for h in headers_completos if h.startswith(trunc)]
    if len(coincidencias) == 1:
        print(f"  -> {coincidencias[0]}")
    elif len(coincidencias) > 1:
        print(f"  -> AMBIGUO ({len(coincidencias)} coincidencias) para prefijo: {trunc}")
    else:
        print(f"  -> SIN COINCIDENCIA para prefijo: {trunc}")
