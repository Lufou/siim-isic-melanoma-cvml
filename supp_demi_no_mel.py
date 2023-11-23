import os
import shutil
import random
import pandas as pd

image_folder = "train-resized"
df = pd.read_csv("train-labels.csv")
# Nombre d'images non-mélanome
non_melanoma_images = df[df['target'] == 0]
# Nombre d'images non-mélanome qu'on souhaite garder
num_to_keep = len(non_melanoma_images) // 2
# Liste des indices d'images non mélanome à supprimer
indices_to_delete = random.sample(list(non_melanoma_images.index), len(non_melanoma_images) - num_to_keep)
# Supprimer les images non-mélanome correspondantes
for index in indices_to_delete:
    image_name = df.at[index, "image_name"]
    image_path = os.path.join(image_folder, f"{image_name}.jpg")
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Suppression de l'image sans mélanome : {image_name}")
df = df.drop(indices_to_delete)
df.to_csv("train_updated.csv", index=False)
