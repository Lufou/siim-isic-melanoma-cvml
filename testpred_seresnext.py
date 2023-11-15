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

# Load the pretrained model from the file (make sure to adjust the file path)
checkpoint = torch.load("hair_seresnext26d_15ep_64batch.pth")

# Set up transformations similar to those used for training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Directory containing test images
test_directory = "test-resized"

# List of test image names
test_image_names = [f.replace(".jpg", "") for f in os.listdir(test_directory) if f.endswith(".jpg")]

# Create a custom dataset for test data
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

# Create the dataset for test data
test_dataset = TestDataset(test_image_names, transform=transform)

# Load your new test data using DataLoaders
batch_size = 64  # Choose the appropriate batch size
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the SE-ResNeXt model using timm
seresnext26d = timm.create_model('seresnext26d_32x4d', pretrained=False, num_classes=2)  # Set pretrained=False
seresnext26d.load_state_dict(checkpoint['model'])
seresnext26d = seresnext26d.eval()

# Create a list to store predictions
probabilities = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use a loop to make predictions on test data
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = seresnext26d(inputs)
        probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

# Now, the list "probabilities" contains the predicted probabilities for each test image

# Create a pandas DataFrame with image names and predictions
results = pd.DataFrame({'image_name': test_image_names, 'target': probabilities})

# Save the DataFrame to a CSV file
results.to_csv('hair_seresnext26d_15ep_64batch.csv', index=False)

