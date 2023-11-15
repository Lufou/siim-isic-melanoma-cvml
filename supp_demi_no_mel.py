import os
import shutil
import random
import pandas as pd

# Chemin du dossier contenant les images
image_folder = "train-resized"

# Charger le fichier train.csv
df = pd.read_csv("train-labels.csv")

# Comptez combien d'images non mélanome (target == 0) vous avez
non_melanoma_images = df[df['target'] == 0]

# Calculez combien d'images non mélanome vous souhaitez conserver (un tiers)
num_to_keep = len(non_melanoma_images) // 2

# Créez une liste d'indices d'images non mélanome à supprimer
indices_to_delete = random.sample(list(non_melanoma_images.index), len(non_melanoma_images) - num_to_keep)

# Supprimez les images non mélanome correspondant aux indices
for index in indices_to_delete:
    image_name = df.at[index, "image_name"]
    image_path = os.path.join(image_folder, f"{image_name}.jpg")
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Suppression de l'image sans mélanome : {image_name}")

# Mettez à jour le fichier CSV
df = df.drop(indices_to_delete)
df.to_csv("train_updated.csv", index=False)

# Facultatif : Renommez le fichier CSV si vous le souhaitez
# shutil.move("train_updated.csv", "train.csv")

