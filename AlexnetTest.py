import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# Chargez le modèle pré-entraîné depuis le fichier (assurez-vous d'ajuster le chemin du fichier)
checkpoint = torch.load("mon_modele1.pth")

# Définissez des transformations similaires à celles que vous avez utilisées pour l'entraînement
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Répertoire contenant les images de test
test_directory = "test-resized"

# Liste des noms d'images de test
test_image_names = [f.replace(".jpg", "") for f in os.listdir(test_directory) if f.endswith(".jpg")]

# Créez un ensemble de données personnalisé pour les données de test
class TestDataset(Dataset):
    def __init__(self, image_names, transform=None):
        self.image_names = image_names
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(test_directory, self.image_names[idx] + ".jpg")
        image = default_loader(img_name)
        if self.transform:
            image = self.transform(image)
        return image

# Créez l'ensemble de données pour les données de test
test_dataset = TestDataset(test_image_names, transform=transform)

# Chargez vos nouvelles données de test en utilisant des DataLoader
batch_size = 32  # Choisissez la taille du batch appropriée
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Chargez le modèle AlexNet
alexnet = models.alexnet()
num_classes = 2  # Deux classes
alexnet.classifier[6] = nn.Linear(4096, num_classes)

# Chargez les poids du modèle depuis le checkpoint
alexnet.load_state_dict(checkpoint['model'])
alexnet.eval()

# Créez une liste pour stocker les prédictions
predictions = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utilisez une boucle pour effectuer des prédictions sur les données de test
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = alexnet(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

# Maintenant, la liste "predictions" contient les étiquettes prédites (0 ou 1) pour chaque image de test

# Créez un DataFrame pandas avec les noms d'images et les prédictions
results = pd.DataFrame({'image_name': test_image_names, 'target': predictions})

# Enregistrez le DataFrame dans un fichier CSV
results.to_csv('test_results.csv', index=False)
