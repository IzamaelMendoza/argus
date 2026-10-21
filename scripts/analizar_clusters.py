import re
from collections import defaultdict, Counter

clstr_path = "combinado_c95_cov90.fasta.clstr"

clusters = defaultdict(list)
current = None

with open(clstr_path) as f:
    for line in f:
        line = line.rstrip("\n")
        if line.startswith(">Cluster"):
            current = line
        else:
            m = re.search(r">([A-Za-z]+)\|", line)
            source = m.group(1) if m else "DESCONOCIDO"
            is_rep = line.strip().endswith("*")
            clusters[current].append((source, is_rep, line))

total_clusters = len(clusters)
pure = 0
mixed = 0
source_rep_counts = Counter()
cluster_sizes = []

for cid, members in clusters.items():
    sources_in_cluster = set(m[0] for m in members)
    if len(sources_in_cluster) == 1:
        pure += 1
    else:
        mixed += 1
    for source, is_rep, _ in members:
        if is_rep:
            source_rep_counts[source] += 1
    cluster_sizes.append((len(members), cid, sources_in_cluster))

print(f"Total de clusters: {total_clusters}")
print(f"Clusters de una sola fuente (puros): {pure} ({pure/total_clusters:.1%})")
print(f"Clusters con varias fuentes mezcladas: {mixed} ({mixed/total_clusters:.1%})")
print()
print("Representantes finales que sobrevivieron, por fuente:")
for source, count in source_rep_counts.most_common():
    print(f"  {source}: {count}")
print()

cluster_sizes.sort(reverse=True)
print("Los 15 clusters mas grandes (posible redundancia concentrada):")
for size, cid, sources in cluster_sizes[:15]:
    print(f"  {cid}: {size} secuencias, fuentes involucradas: {sources}")
