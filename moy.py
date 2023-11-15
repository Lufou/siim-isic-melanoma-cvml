import pandas as pd

# Charger les fichiers CSV
file_path1 = "hair_16000nomel_3500mel_resnet18_20ep_64batch.csv"
file_path2 = "hair_16000nomel_3500mel_resnet34_15ep_64batch.csv"
file_path3 = "hair_efficientnet_b0_20ep_64batch.csv"
file_path4 = "hair_seresnext26d_15ep_64batch.csv"
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)
df3 = pd.read_csv(file_path3)
df4 = pd.read_csv(file_path4)

# Assurez-vous que les deux dataframes ont la même structure
if not all(df1.columns == df2.columns) or not all(df2.columns == df3.columns) or not all(df3.columns == df4.columns):
    raise ValueError("Les colonnes des deux fichiers CSV ne correspondent pas.")

# Calculer la moyenne de la colonne "target"
df1['target'] = (df1['target'] + df2['target'] +df3['target']+ df4['target'])/ 4

# Sauvegarder le résultat dans un nouveau fichier CSV
output_file_path = "testmoy3.csv"
df1.to_csv(output_file_path, index=False)

print(f"La moyenne des colonnes 'target' a été sauvegardée dans {output_file_path}")
