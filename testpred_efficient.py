import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import torch.nn.functional as F

checkpoint = torch.load("hair_efficientnet_b0_20ep_64batch.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_directory = "test-resized"

test_image_names = [f.replace(".jpg", "") for f in os.listdir(test_directory) if f.endswith(".jpg")]

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

test_dataset = TestDataset(test_image_names, transform=transform)

batch_size = 64 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Charger le modèle EfficientNet
efficientnet_b0 = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
efficientnet_b0.load_state_dict(checkpoint['model'])
efficientnet_b0 = efficientnet_b0.eval()

probabilities = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prédictions sur les données
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = efficientnet_b0(inputs)
        probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())


results = pd.DataFrame({'image_name': test_image_names, 'probability': probabilities})

results.to_csv('hair_efficientnet_b0_20ep_64batch.csv', index=False)

