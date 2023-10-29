import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

# Ignorer les avertissements
warnings.filterwarnings("ignore", category=UserWarning)

# Charger le fichier train.csv
df = pd.read_csv("train.csv")

# Diviser les données en ensembles d'entraînement et de validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Définir les transformations d'images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionner les images à la taille d'entrée d'AlexNet
    transforms.ToTensor(),  # Convertir les images en tenseurs
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation
])

# Créer des ensembles de données personnalisés
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name ="train-resized/" + self.dataframe.iloc[idx, 0] + ".jpg"
        image = default_loader(img_name)
        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# Créer des ensembles de données pour l'entraînement et la validation
train_dataset = CustomDataset(train_df, transform)
val_dataset = CustomDataset(val_df, transform)

# Créer des chargeurs de données
batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Charger le modèle AlexNet pré-entraîné
alexnet = models.alexnet(pretrained=True)

# Modifier la dernière couche de classification
num_classes = 2  # Deux classes 
alexnet.classifier[6] = nn.Linear(4096, num_classes)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=0.001)



# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.parameters(), lr=0.001)

# Entraînement du modèle
num_epochs = 2  # Nombre d'époques d'entraînement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mettre le modèle en mode d'entraînement
alexnet.train()

for epoch in range(num_epochs):
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Remettre à zéro les gradients
        optimizer.zero_grad()

        # Propagation avant (forward pass)
        outputs = alexnet(inputs)
        
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
alexnet.eval()  # Mettre le modèle en mode d'évaluation

correct = 0
total = 0

# Ne pas calculer de gradients lors de l'évaluation
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = alexnet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Précision sur les données de validation : {accuracy:.2f}%")

# Après l'entraînement, sauvegardez le modèle dans un fichier
torch.save(alexnet.state_dict(), "mon_modele1.pth")
