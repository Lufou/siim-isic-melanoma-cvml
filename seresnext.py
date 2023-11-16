import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import warnings
from torchvision.datasets.folder import default_loader

# Ignorer les avertissements
warnings.filterwarnings("ignore", category=UserWarning)

# Vérifier la disponibilité de la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vérifiez la disponibilité des GPU
if torch.cuda.is_available():
    # Utilisez le GPU par défaut pour le calcul
    device = torch.device("cuda")
    print("Utilisation du GPU pour le calcul.")
else:
    device = torch.device("cpu")
    print("Pas de GPU.")

# Charger le fichier train-labels.csv
df = pd.read_csv("train-labels.csv")

# Diviser les données en ensembles d'entraînement et de validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Définir les transformations d'images pour l'augmentation de données
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),  # Rotation aléatoire de l'image jusqu'à 30 degrés
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Créer des ensembles de données personnalisés
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = "train-resized/" + self.dataframe.iloc[idx, 0] + ".jpg"
        image = default_loader(img_name)
        label = int(self.dataframe.iloc[idx, 1])  # Utilisation de la colonne "target" comme étiquette

        if self.transform:
            image = self.transform(image)

        return image, label

# Créer des ensembles de données pour l'entraînement et la validation
train_dataset = CustomDataset(train_df, transform=train_transform)
val_dataset = CustomDataset(val_df, transform=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))

# Calculer les poids de classe pour la fonction de perte pondérée
class_labels = train_df['target'].values
class_weights = compute_class_weight('balanced', classes=[0, 1], y=class_labels)
class_weights = torch.FloatTensor(class_weights)
class_weights = class_weights.to(device)

# Créer des chargeurs de données avec un échantillonnage équilibré
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Charger le modèle SEResNeXt26d_32x4d pré-entraîné
seresnext = timm.create_model('seresnext26d_32x4d', pretrained=True)

# Modifier la dernière couche de classification
num_classes = 2  # Deux classes
# Récupérer la taille de la dernière couche de caractéristiques
num_ftrs = seresnext.fc.in_features
seresnext.fc = nn.Linear(num_ftrs, num_classes)

# Déplacer le modèle sur la GPU (si disponible)
seresnext = seresnext.to(device)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Utilisation de la fonction de perte pondérée
optimizer = optim.Adam(seresnext.parameters(), lr=0.001)

# Entraînement du modèle
num_epochs = 15  # Nombre d'époques d'entraînement

# Mettre le modèle en mode d'entraînement
seresnext.train()

for epoch in range(num_epochs):
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Remettre à zéro les gradients
        optimizer.zero_grad()

        # Propagation avant (forward pass)
        outputs = seresnext(inputs)

        # Calcul de la perte
        loss = criterion(outputs, labels)

        # Rétropropagation et mise à jour des poids
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calcul de la perte moyenne sur cette époque
    epoch_loss = running_loss / len(train_loader)
    print(f"Époque [{epoch + 1}/{num_epochs}] - Perte : {epoch_loss:.4f}")

print("Entraînement terminé.")

# Évaluation sur les données de validation
seresnext.eval()  # Mettre le modèle en mode d'évaluation

correct = 0
total = 0

# Ne pas calculer de gradients lors de l'évaluation
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = seresnext(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Précision sur les données de validation : {accuracy:.2f}%")

# Pour sauvegarder le modèle
torch.save({
    'model': seresnext.state_dict(),  # Enregistrez les poids du modèle
    'optimizer': optimizer.state_dict(),  # Enregistrez l'état de l'optimiseur si nécessaire
}, 'hair_seresnext26d_15ep_64batch.pth')

