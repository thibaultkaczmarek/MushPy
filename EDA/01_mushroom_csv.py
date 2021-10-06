
"""Fernando Gonçalves

   Création d'un csv à partir des json "dropbox"

    A vérifier/A modifier :
    -----------------------
    - path_to_json : chemin vers les json
    - csv = open(r".\Dataset\Observations.csv","w") : chemin du fichier généré
   
"""


import os, json
import pandas as pd
import numpy as np
import glob
import json


pd.set_option('display.max_columns', None)

path_to_json = './Dataset/json_files/' 

# Propriétés à extraire des json
properties = ['observation','label','image_id','image_url','date','location','thumbnail']
properties_gbif = ['kingdom', 'family', 'speciesKey', 'rank', 'phylum', 'orderKey', 'species', 'confidence', 'classKey', 'familyKey', 'usageKey', 'kingdomKey','genusKey', 'phylumKey', 'class', 'scientificName', 'genus', 'order', 'genusKey', 'classKey']
properties_gbif_empty = ';;;;;;;;;;;;;;;;;;;;'

# Création du csv de sortie avec entête
csv = open(r".\Dataset\Observations.csv","w")
csv.write('file;' + (str(properties).strip('[]') + ";" + str(properties_gbif).strip('[]')).replace(" ", "").replace("'", "").replace(",", ";") + ";")
csv.write('\n')

# Liste des fichier json
json_pattern = os.path.join(path_to_json,'*.json')
file_list = glob.glob(json_pattern)


for file in file_list:
    print(file)
    
    f  = open(file, encoding="utf8")
    data = json.load(f)

    for m in data:
        #print(f"\t\t{m['label']}")

        # Le fichier json est ajouté au csv afin de retrouver plus simplement les données si besoin
        csv.write(file[file.index('\\')+1:] + ';')

        # Les propriétés à la racine de chaque document mushroom
        for p in properties:
            csv.write(str(m.get(p, '')) + ';')

        # liste de propriétés pour gbif_info
        if 'gbif_info' in m :
            if m['gbif_info'] is not None:

                for p_gbif in properties_gbif:
                    if p_gbif in m['gbif_info']:
                        #print(str(m['gbif_info'][p_gbif]))
                        csv.write(str(m['gbif_info'][p_gbif]).replace('\u011b', '') + ';')
                    else:
                        csv.write(';')
            else:
                csv.write(properties_gbif_empty)
        else:
                csv.write(properties_gbif_empty)
            

        csv.write('\n')


csv.close()



