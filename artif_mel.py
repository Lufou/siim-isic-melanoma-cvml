import os
import pandas as pd
from shutil import copyfile
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter
import random
import math
import random
import numpy as np

# Chemin vers le dossier train-resized
augmented_melanoma_images_dir = "train-resized"

# Charger le fichier CSV existant
df = pd.read_csv("train-labels.csv")

# Créez un compteur pour suivre le nombre total d'images générées
image_counter = 0


def draw_hairs(image):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for _ in range(10):  # Vous pouvez ajuster le nombre de cheveux générés
        # Position aléatoire pour le début du cheveu
        start_x = random.randint(0, width)
        start_y = random.randint(0, height)

        # Longueur, angle et ondulation aléatoires pour le cheveu
        length = random.randint(10, 100)   # Ajustez la valeur pour déterminer la longueur du cheveu
        angle = random.uniform(20, 60)    # Ajustez la valeur pour déterminer l'inclinaison du cheveu
        wave_amplitude = random.randint(5, 10)  # Ajustez la valeur pour déterminer l'amplitude de l'ondulation
        wave_frequency = random.uniform(0.01, 0.05)  # Ajustez la valeur pour déterminer la fréquence de l'ondulation

        # Calcul des coordonnées de fin du cheveu
        end_x = start_x + int(length * math.cos(math.radians(angle)))
        end_y = start_y - int(length * math.sin(math.radians(angle)))

        # Dessiner le cheveu sous forme de courbe Bézier
        points = [(start_x, start_y)]
        for t in range(1, 101):
            x = int(start_x + t / 100 * (end_x - start_x))
            y = int(start_y + t / 100 * (end_y - start_y) + wave_amplitude * math.sin(wave_frequency * t))
            points.append((x, y))

        # Couleur aléatoire pour les cheveux noir
        hair_color = (0, 0, 0)  # Noir 

        # Épaisseur aléatoire pour les cheveux (peut être ajustée)
        hair_thickness = random.uniform(0.5, 0.9)

        # Dessiner la courbe Bézier
        draw.line(points, fill=hair_color, width=int(hair_thickness))

    return image



# Parcourir les fichiers dans le dossier des nouvelles images mélanomes
for root, _, files in os.walk(augmented_melanoma_images_dir):
    for filename in files:
        # Vérifier si l'image est un mélanome (target==1 dans le fichier CSV)
        image_name = filename.replace(".jpg", "")
        if df[df["image_name"] == image_name]["target"].values[0] == 1:
            # Créer un nouveau nom d'image unique
            for _ in range(5):  # Répétez la rotation 5 fois
                new_image_name = f"ISIC-{len(df) + image_counter + 1:07d}"
                image_counter += 1

                # Charger l'image
                image = Image.open(os.path.join(root, filename))

                # Appliquer une rotation de 60 degrés
                image = transforms.functional.rotate(image, 60)

                # Dessiner des cheveux artificiels sur l'image
                image = draw_hairs(image)

                # Enregistrez l'image dans le dossier train-resized
                image.save(os.path.join(augmented_melanoma_images_dir, new_image_name + ".jpg"))

                # Ajouter une entrée dans le fichier CSV pour la nouvelle image
                new_entry = {"image_name": new_image_name, "target": 1}
                df = df.append(new_entry, ignore_index=True)

                print(f"Image ajoutée : {new_image_name}")

# Enregistrer le fichier CSV mis à jour
df.to_csv("train-labels.csv", index=False)

print("Mise à jour terminée.")

