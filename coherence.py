import os
import pandas as pd

# Chemin vers le dossier train-resized
train_resized_dir = "train-resized"

# Charger le fichier CSV existant
df = pd.read_csv("train-labels.csv")

# Liste des fichiers réels dans le dossier train-resized (sans extension)
existing_files = [os.path.splitext(file)[0] for file in os.listdir(train_resized_dir)]

# Liste des fichiers mentionnés dans le CSV (sans extension)
csv_files = [os.path.splitext(file)[0] for file in df["image_name"].tolist()]

# Vérifier la cohérence entre les fichiers CSV et les fichiers réels
inconsistencies = [file for file in csv_files if file not in existing_files]

if not inconsistencies:
    print("Tous les fichiers CSV sont cohérents avec les fichiers réels dans le dossier train-resized.")
else:
    print("Les fichiers CSV suivants ne correspondent pas aux fichiers réels :")
    for file in inconsistencies:
        print(file)

