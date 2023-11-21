import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import timm
from sklearn.metrics import classification_report
import warnings
from torchvision.datasets.folder import default_loader


warnings.filterwarnings("ignore", category=UserWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("train-labels.csv")

# Diviser les ensembles d'entraînement et de validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Ensemble des fonctions de transformation d'images pour augmenter les données
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #transforms.RandomRotation(90),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = "train-resized/" + self.dataframe.iloc[idx, 0] + ".jpg"
        image = default_loader(img_name)
        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# Créer les ensembles de données pour l'entraînement et la validation grâce aux fonctions de transformation
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

# Créer les chargeurs de données
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Charger le modèle EfficientNet pré-entraîné
effnet = timm.create_model('efficientnet_b6', pretrained=False)

num_classes = 2

num_ftrs = effnet.classifier.in_features
effnet.classifier = nn.Linear(num_ftrs, num_classes)

effnet = effnet.to(device)

# Fonctions de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(effnet.parameters(), lr=0.001)

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


num_epochs = 20

# Passer en mode entraînement
effnet.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = effnet(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    # Calcul de la perte moyenne sur l'époque
    epoch_loss_train = running_loss / len(train_loader)
    train_losses.append(epoch_loss_train)

    # Calcul de la précision
    accuracy_train = 100 * correct_train / total_train
    train_accuracies.append(accuracy_train)

    print(f"Époque [{epoch + 1}/{num_epochs}] - Perte (entraînement) : {epoch_loss_train:.4f} - Précision (entraînement) : {accuracy_train:.2f}%")

    # Passer en mode validation
    effnet.eval()

    correct_val = 0
    total_val = 0
    val_loss = 0.0

    all_predicted = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = effnet(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            val_loss += criterion(outputs, labels).item()

            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcul de la perte moyenne sur l'époque
    epoch_loss_val = val_loss / len(val_loader)
    val_losses.append(epoch_loss_val)

    # Calcul de la précision
    accuracy_val = 100 * correct_val / total_val
    val_accuracies.append(accuracy_val)

    print(f"Époque [{epoch + 1}/{num_epochs}] - Perte (validation) : {epoch_loss_val:.4f} - Précision (validation) : {accuracy_val:.2f}%")

    # Revenir en mode entraînement
    effnet.train()

    target_names = ['Non-mélanome', 'Mélanome']
    classification_rep = classification_report(all_labels, all_predicted, target_names=target_names, output_dict=True)

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

print("Entraînement terminé.")

# Sauvegarder le modèle
torch.save({
    'model': effnet.state_dict(),
    'optimizer': optimizer.state_dict(),
}, 'hair_efficientnet_b0_20ep_64batch.pth')



