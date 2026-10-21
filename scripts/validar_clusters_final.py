import re
import random
from collections import defaultdict

clstr_path = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim/combinado_v2_reducido.fasta.clstr"
random.seed(3)

clusters = defaultdict(list)
current = None

with open(clstr_path) as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith(">Cluster"):
            current = line.replace(">Cluster ", "")
        else:
            m = re.search(r">(.+?)\.\.\.", line)
            if m:
                clusters[current].append(m.group(1))

sizes = sorted(((len(v), k) for k, v in clusters.items()), reverse=True)

print(f"Total de clusters: {len(clusters)}")
print()
print("Los 5 clusters mas grandes:")
for size, cid in sizes[:5]:
    print(f"\n== Cluster {cid} ({size} secuencias) ==")
    miembros = clusters[cid]
    muestra = random.sample(miembros, min(8, len(miembros)))
    for h in muestra:
        print(f"  {h}")
