import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import warnings
import numpy as np

# Ignorer les avertissements
warnings.filterwarnings("ignore", category=UserWarning)

#Vérifier la disponibilité de la GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le fichier train-labels.csv
df = pd.read_csv("train-labels_19K.csv")

# Diviser les données en ensembles d'entraînement et de validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Définir les transformations d'images pour l'augmentation de données
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.RandomRotation(90),  # Rotation aléatoire de l'image jusqu'à 90 degrés
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
        img_name = "train-resized_19K/" + self.dataframe.iloc[idx, 0] + ".jpg"
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

# Charger le modèle Resnet pré-entraîné
resnet = models.resnet18(pretrained=True)

# Modifier la dernière couche de classification
num_classes = 2  # Deux classes
# Récupérer la taille de la dernière couche de caractéristiques
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, num_classes)

# Déplacer le modèle sur la GPU (si disponible)
resnet = resnet.to(device)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss(weight=class_weights)  # Utilisation de la fonction de perte pondérée
optimizer = optim.Adam(resnet.parameters(), lr=0.001)

# Initialiser les listes pour stocker les données
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

class0_precision = []
class0_recall = []
class0_f1score = []
class0_score = []
class1_precision = []
class1_recall = []
class1_f1score = []
class1_score = []


# Entraînement du modèle
num_epochs = 50 # Nombre d'époques d'entraînement

# Mettre le modèle en mode d'entraînement
resnet.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Remettre à zéro les gradients
        optimizer.zero_grad()

        # Propagation avant (forward pass)
        outputs = resnet(inputs)

        # Calcul de la perte
        loss = criterion(outputs, labels)

        # Rétropropagation et mise à jour des poids
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calcul de la précision sur les données d'entraînement
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Calcul de la perte moyenne sur cette époque pour les données d'entraînement
    epoch_loss_train = running_loss / len(train_loader)
    train_losses.append(epoch_loss_train)

    # Calcul de la précision sur les données d'entraînement
    accuracy_train = 100 * correct_train / total_train
    train_accuracies.append(accuracy_train)

    # Afficher ou sauvegarder les résultats
    print(f"Époque [{epoch + 1}/{num_epochs}] - Perte (entraînement) : {epoch_loss_train:.4f} - Précision (entraînement) : {accuracy_train:.2f}%")

    # Évaluation sur les données de validation
    resnet.eval()  # Mettre le modèle en mode d'évaluation

    correct_val = 0
    total_val = 0
    val_loss = 0.0

    all_predicted = []
    all_labels = []

    # Ne pas calculer de gradients lors de l'évaluation
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Calcul de la perte moyenne sur cette époque pour les données de validation
    epoch_loss_val = val_loss / len(val_loader)
    val_losses.append(epoch_loss_val)

    # Calcul de la précision sur les données de validation
    accuracy_val = 100 * correct_val / total_val
    val_accuracies.append(accuracy_val)

    # Afficher ou sauvegarder les résultats
    print(f"Époque [{epoch + 1}/{num_epochs}] - Perte (validation) : {epoch_loss_val:.4f} - Précision (validation) : {accuracy_val:.2f}%")

    # Revenir en mode d'entraînement
    resnet.train()

    target_names = ['Non-mélanome', 'Mélanome']  # Remplacez par les noms de vos classes
    classification_rep = classification_report(all_labels, all_predicted, target_names=target_names, output_dict=True)

    # Afficher les métriques pour chaque classe
    for i, class_name in enumerate(target_names):
        metrics = classification_rep[class_name]
        precision = metrics['precision']
        recall = metrics['recall']
        f1_score = metrics['f1-score']
        support = metrics['support']
        
        print(f"{class_name} : Precision = {precision} ; Recall = {recall} ; F1-score = {f1_score} ; Support = {support}")
        if (class_name == "Non-mélanome"):
          class0_precision.append(precision)
          class0_recall.append(recall)
          class0_f1score.append(f1_score)
          class0_score.append(support)
        else:
          class1_precision.append(precision)
          class1_recall.append(recall)
          class1_f1score.append(f1_score)
          class1_score.append(support)

# Sauvegarder les données
result_data = pd.DataFrame({
    'train_loss': train_losses,
    'train_accuracy': train_accuracies,
    'val_loss': val_losses,
    'val_accuracy': val_accuracies
})
result_data.to_csv('training_results.csv', index=False)

result_data = pd.DataFrame({
    'precision': class0_precision,
    'recall': class0_recall,
    'f1_score': class0_f1score,
    'support': class0_score
})
result_data.to_csv('no-melanoms_metrics.csv', index=False)
result_data = pd.DataFrame({
    'precision': class1_precision,
    'recall': class1_recall,
    'f1_score': class1_f1score,
    'support': class1_score
})
result_data.to_csv('melanoms_metrics.csv', index=False)

# Sauvegarder le modèle
torch.save({
    'model': resnet.state_dict(),
    'optimizer': optimizer.state_dict(),
    'result_data': result_data
}, 'test_accuracy_perte.pth')

print("Entraînement terminé.")
