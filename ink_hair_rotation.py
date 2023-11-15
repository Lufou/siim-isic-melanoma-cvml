import os
import pandas as pd
from shutil import copyfile
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter
import random
import math
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


def add_ink_drops(image, num_drops):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for _ in range(num_drops):
        # Position aléatoire pour la goutte d'encre
        drop_x = random.randint(0, width)
        drop_y = random.randint(0, height)

        # Taille et couleur de goutte d'encre
        drop_size = random.randint(5, 20)  # Ajustez la valeur pour déterminer la taille de la goutte
        ink_color = (0, 0, 255)  # Bleu

        # Dessiner la goutte d'encre sous forme de cercle
        draw.ellipse([drop_x, drop_y, drop_x + drop_size, drop_y + drop_size], fill=ink_color)

    return image


# Parcourir les fichiers dans le dossier des nouvelles images mélanomes
for root, _, files in os.walk(augmented_melanoma_images_dir):
    for filename in files:
        # Vérifier si l'image est un mélanome (target==1 dans le fichier CSV)
        image_name = filename.replace(".jpg", "")
        if df[df["image_name"] == image_name]["target"].values[0] == 1:
            # Créer un nouveau nom d'image unique
            original_image_name_base = f"ISIC-{len(df) + image_counter + 1:07d}_original"
            image_counter += 1

            # Charger l'image originale
            original_image = Image.open(os.path.join(root, filename))

            # Créer une copie de l'image originale sans rotation avec encre
            original_image_with_ink = add_ink_drops(original_image.copy(), num_drops=random.randint(2, 4))

            # Enregistrer l'image originale avec encre dans le dossier train-resized
            original_image_with_ink.save(os.path.join(augmented_melanoma_images_dir, original_image_name_base + ".jpg"))

            # Ajouter une entrée dans le fichier CSV pour la nouvelle image originale avec encre
            new_entry_original_with_ink = {"image_name": original_image_name_base, "target": 1}
            df = df.append(new_entry_original_with_ink, ignore_index=True)

            print(f"Image ajoutée : {original_image_name_base}")

            # Appliquer 3 fois la rotation de 90 degrés
            for i in range(3):
                # Créer un nouveau nom d'image unique pour les images avec rotation
                new_image_name_base = f"ISIC-{len(df) + image_counter + 1:07d}_rotated_{i+1}"

                # Appliquer une rotation de 90 degrés
                rotated_image = transforms.functional.rotate(original_image.copy(), 90 * (i + 1))

                # Enregistrer l'image avec rotation dans le dossier train-resized
                rotated_image.save(os.path.join(augmented_melanoma_images_dir, new_image_name_base + ".jpg"))

                # Ajouter une entrée dans le fichier CSV pour la nouvelle image avec rotation
                new_entry_rotated = {"image_name": new_image_name_base, "target": 1}
                df = df.append(new_entry_rotated, ignore_index=True)

                print(f"Image ajoutée : {new_image_name_base}")

            # Appliquer 5 fois la rotation de 60 degrés et ajouter des cheveux
            for i in range(5):
                # Créer un nouveau nom d'image unique pour les images avec rotation et cheveux
                new_image_name_base = f"ISIC-{len(df) + image_counter + 1:07d}_rotated_with_hairs_{i+1}"

                # Appliquer une rotation de 60 degrés
                rotated_image = transforms.functional.rotate(original_image.copy(), 60 * (i + 1))

                # Dessiner des cheveux artificiels sur l'image avec rotation
                rotated_image_with_hairs = draw_hairs(rotated_image.copy())

                # Enregistrer l'image avec rotation et cheveux dans le dossier train-resized
                rotated_image_with_hairs.save(os.path.join(augmented_melanoma_images_dir, new_image_name_base + ".jpg"))

                # Ajouter une entrée dans le fichier CSV pour la nouvelle image avec rotation et cheveux
                new_entry_rotated_with_hairs = {"image_name": new_image_name_base, "target": 1}
                df = df.append(new_entry_rotated_with_hairs, ignore_index=True)

                print(f"Image ajoutée : {new_image_name_base}")

            # Appliquer 5 fois la rotation de 60 degrés et ajouter de l'encre
            for i in range(5):
                # Créer un nouveau nom d'image unique pour les images avec rotation et encre
                new_image_name_base = f"ISIC-{len(df) + image_counter + 1:07d}_rotated_with_ink_{i+1}"

                # Appliquer une rotation de 60 degrés
                rotated_image = transforms.functional.rotate(original_image.copy(), 60 * (i + 1))

                # Ajouter entre 2 et 4 gouttes d'encre artificielles sur l'image avec rotation
                rotated_image_with_ink = add_ink_drops(rotated_image.copy(), num_drops=random.randint(2, 4))

                # Enregistrer l'image avec rotation et encre dans le dossier train-resized
                rotated_image_with_ink.save(os.path.join(augmented_melanoma_images_dir, new_image_name_base + ".jpg"))

                # Ajouter une entrée dans le fichier CSV pour la nouvelle image avec rotation et encre
                new_entry_rotated_with_ink = {"image_name": new_image_name_base, "target": 1}
                df = df.append(new_entry_rotated_with_ink, ignore_index=True)

                print(f"Image ajoutée : {new_image_name_base}")

            # Appliquer 5 fois la rotation de 60 degrés, ajouter des cheveux et ajouter de l'encre
            for i in range(5):
                # Créer un nouveau nom d'image unique pour les images avec rotation, cheveux et encre
                new_image_name_base = f"ISIC-{len(df) + image_counter + 1:07d}_rotated_with_hairs_and_ink_{i+1}"

                # Appliquer une rotation de 60 degrés
                rotated_image = transforms.functional.rotate(original_image.copy(), 60 * (i + 1))

                # Dessiner des cheveux artificiels sur l'image avec rotation
                rotated_image_with_hairs = draw_hairs(rotated_image.copy())

                # Ajouter entre 2 et 4 gouttes d'encre artificielles sur l'image avec rotation et cheveux
                rotated_image_with_hairs_and_ink = add_ink_drops(rotated_image_with_hairs.copy(), num_drops=random.randint(2, 4))

                # Enregistrer l'image avec rotation, cheveux et encre dans le dossier train-resized
                rotated_image_with_hairs_and_ink.save(os.path.join(augmented_melanoma_images_dir, new_image_name_base + ".jpg"))

                # Ajouter une entrée dans le fichier CSV pour la nouvelle image avec rotation, cheveux et encre
                new_entry_rotated_with_hairs_and_ink = {"image_name": new_image_name_base, "target": 1}
                df = df.append(new_entry_rotated_with_hairs_and_ink, ignore_index=True)

                print(f"Image ajoutée : {new_image_name_base}")

# Enregistrer le fichier CSV mis à jour
df.to_csv("train-labels.csv", index=False)

print("Mise à jour terminée.")

