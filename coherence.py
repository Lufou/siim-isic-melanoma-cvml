import os
import pandas as pd

train_resized_dir = "train-resized"

df = pd.read_csv("train-labels.csv")

existing_files = [os.path.splitext(file)[0] for file in os.listdir(train_resized_dir)]

csv_files = [os.path.splitext(file)[0] for file in df["image_name"].tolist()]

# Vérifier la cohérence entre les fichiers csv et les fichiers existants
inconsistencies = [file for file in csv_files if file not in existing_files]

if not inconsistencies:
    print("Tous les fichiers CSV sont cohérents avec les fichiers existants dans le dossier train-resized.")
else:
    print("Les fichiers CSV suivants ne correspondent pas aux fichiers existants :")
    for file in inconsistencies:
        print(file)

