import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json

from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch

# Load pre-trained Vision Transformer
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k', 
    num_labels=41  # 41 characters in Gurmukhi
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# Load the label mapping
with open('label_map.json', 'r', encoding='utf-8') as f:
    label_map = json.load(f)

class GurmukhiDataset(Dataset):
    def __init__(self, image_dir, label_map, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.label_map = label_map
        self.image_paths = []
        self.labels = []

        # Load images and labels
        for folder_name, label in self.label_map.items():
            folder_path = os.path.join(self.image_dir, folder_name)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)  # This should be an integer from label_map

        # Debugging output
        print(f"Number of images: {len(self.image_paths)}")
        print(f"Number of labels: {len(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')  # Convert to RGB

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]  # This will now be an integer
        return img, label
# Define transformations (resize, to tensor, normalize)
# Define transformations (resize, to tensor, normalize)
# Define transformations (resize, to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 to match ViT input size
    transforms.ToTensor(),           # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for RGB
])
output_dir = r"D:\Gurmukhi dataset\Gurmukhi dataset\archive\prepossesed_data"
# Create the dataset
dataset = GurmukhiDataset(image_dir=output_dir, label_map=label_map, transform=transform)

# Create dataloader (batch size = 32)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

num_epochs=3
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()

        pixel_values, labels = batch  # Unpack the batch
        pixel_values = pixel_values.to(device)

        # Convert labels to tensor using clone().detach() to avoid the warning
        labels = torch.tensor(labels).clone().detach().to(device).long()  # Ensure labels are long integers

        # Forward pass
        outputs = model(pixel_values=pixel_values)  # Get model predictions
        loss = criterion(outputs.logits, labels)  # Calculate loss

        # Backward pass and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        running_loss += loss.item()  # Accumulate loss

    # Print average loss for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

print("Training complete!")

