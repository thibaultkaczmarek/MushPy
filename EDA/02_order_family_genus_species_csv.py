import numpy as np
import pandas as pd

def set_order(df, row):
    
    if pd.isnull(row['order']):
        if pd.notnull(row['family']):
            row['order'] = df[(pd.notnull(df['order']) &
                               df['family']== row['family'])]['order'].head(1)
        elif pd.notnull(row['genus']):
            row['order'] = df[(pd.notnull(df['order']) &
                               df['genus']== row['genus'])]['order'].head(1)
        elif pd.notnull(row['species']):
            row['order'] = df[(pd.notnull(df['order']) &
                               df['species']== row['species'])]['order'].head(1)
        


df = pd.read_csv(".\dataset\Observations.csv", sep=';', usecols=['label', 'rank', 'kingdom', 'family', 'phylum', 'species', 'class', 'genus', 'order', 'image_url'])

df_order_family_genus_species = df[(
                                        pd.notnull(df['order'])|
                                        pd.notnull(df['family']) |
                                        pd.notnull(df['genus']) |
                                        pd.notnull(df['species'])
                                    )]

df_order_family_genus_species = df_order_family_genus_species[
                        ['order', 'family', 'genus', 'species', 'image_url']].groupby(
                        ['order', 'family', 'genus', 'species']).apply(lambda row: row)

df_order_family_genus_species.apply(lambda row: set_order(df_order_family_genus_species, row), axis=1)
#print(df_order_family_genus_species.head())

#print(df_order_family_genus_species['image_url'].str.split('/')[5][5])
# 'C:\Projets\MushroomRecognition\images' + row.str.split('/')[5] + '.jpg')
#df_order_family_genus_species.dropna()

df_order_family_genus_species['image_path'] = df_order_family_genus_species['image_url'].apply(lambda row: ('C:\Projets\MushroomRecognition\images\\' + row.split('/')[5] + '.jpg') if '/' in row else '¤')
df_order_family_genus_species['image_id'] = df_order_family_genus_species['image_url'].apply(lambda row: (row.split('/')[5]) if '/' in row else '¤')

df_order_family_genus_species[['image_id', 'order', 'family', 'genus', 'species', 'image_url', 'image_path']].to_csv(r'./dataset/order_family_genus_species.csv')

