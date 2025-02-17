import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import pandas as pd
from transformers import ViTForImageClassification
import torchvision.transforms as transforms

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom dataset class
class MyCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                if label.startswith("character_") and len(label.split('_')) > 1:
                    try:
                        label_id = int(label.split('_')[1]) - 1
                        if label_id < 0:  # Ensure label ID is non-negative
                            continue
                    except ValueError:
                        print(f"Skipping label '{label}' due to ValueError.")
                        continue

                    for img_name in os.listdir(label_path):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(label_path, img_name)
                            self.images.append(img_path)
                            self.labels.append(label_id)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label

# Define the path to your test dataset
test_root_dir = 'D:\\Gurmukhi dataset\\Gurmukhi dataset\\archive\\test_data'  # Replace with your actual test data path

# Define transforms for test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size for ViT
    transforms.ToTensor(),            # Convert PIL Image to tensor
])

# Initialize the test dataset and DataLoader
test_dataset = MyCustomDataset(root_dir=test_root_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define class names for the Gurmukhi characters
class_names = [
    'Character_1', 'Character_2', 'Character_3', 'Character_4', 'Character_5',
    'Character_6', 'Character_7', 'Character_8', 'Character_9', 'Character_10',
    'Character_11', 'Character_12', 'Character_13', 'Character_14', 'Character_15',
    'Character_16', 'Character_17', 'Character_18', 'Character_19', 'Character_20',
    'Character_21', 'Character_22', 'Character_23', 'Character_24', 'Character_25',
    'Character_26', 'Character_27', 'Character_28', 'Character_29', 'Character_30',
    'Character_31', 'Character_32', 'Character_33', 'Character_34', 'Character_35',
    'Character_36', 'Character_37', 'Character_38', 'Character_39', 'Character_40',
    'Character_41'
]

true_labels = []
predicted_labels = []

# Load your pre-trained model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=len(class_names))
model.to(device)
model.eval()

with torch.no_grad():
    for pixel_values, labels in test_loader:
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        # Get model predictions
        outputs = model(pixel_values)
        _, preds = torch.max(outputs.logits, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())

# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)

# Create a DataFrame from the report
report_df = pd.DataFrame(report).transpose()

# Display the DataFrame
print(report_df)
