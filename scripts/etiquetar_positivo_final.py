import re

INTERIM = "/mnt/c/Users/r4gbi/Desktop/Proyecto_Residencias/data/interim"

# 1. Cargar las 6 tablas de metadata en un diccionario único
meta = {}
archivos = ["meta_card.tsv", "meta_resfinder.tsv", "meta_megares.tsv",
            "meta_argannot.tsv", "meta_amrfinderplus.tsv", "meta_resfinderfg.tsv"]

for nombre in archivos:
    with open(f"{INTERIM}/{nombre}") as f:
        for line in f:
            partes = line.rstrip("\n").split("\t")
            if len(partes) != 3:
                continue
            header_completo, label, categoria = partes
            clave = header_completo.split()[0]  # truncar al primer espacio, como hace -d 0
            meta[clave] = (label, categoria)

print(f"Total de entradas cargadas en metadata: {len(meta)}")

# 2. Parsear el .clstr y quedarnos con el representante de cada cluster
clstr_path = f"{INTERIM}/combinado_v4_reducido.fasta.clstr"
resultados = []
sin_match = 0

with open(clstr_path) as f:
    cluster_id = None
    representante = None
    for line in f:
        line = line.rstrip("\n")
        if line.startswith(">Cluster"):
            cluster_id = line.replace(">Cluster ", "")
        else:
            m = re.search(r">(.+?)\.\.\.", line)
            if m and line.strip().endswith("*"):
                representante = m.group(1)
                label, categoria = meta.get(representante, (None, None))
                if label is None:
                    sin_match += 1
                    label, categoria = "SIN_MATCH", "SIN_MATCH"
                resultados.append((cluster_id, representante, label, categoria))

print(f"Total de clusters procesados: {len(resultados)}")
print(f"Representantes SIN coincidencia en metadata: {sin_match}")

# 3. Escribir la tabla final de positivos
with open(f"{INTERIM}/positivo_etiquetado.tsv", "w") as fout:
    fout.write("cluster_id\tsecuencia_id\tlabel\tcategoria_original\n")
    for cid, rep, label, cat in resultados:
        fout.write(f"{cid}\t{rep}\t{label}\t{cat}\n")

print("Guardado en positivo_etiquetado.tsv")
