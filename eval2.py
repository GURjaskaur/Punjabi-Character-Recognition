import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification
import torchvision.transforms as transforms

# Custom dataset class (reuse from training script)
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
                        if label_id < 0:
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

# Load the trained model and evaluate
def evaluate_model(root_dir, model_path, num_classes):
    # Define the transforms for the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Initialize dataset and dataloader
    dataset = MyCustomDataset(root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Load the model
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_classes)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set the model to evaluation mode
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for inference
        for pixel_values, labels in dataloader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            outputs = model(pixel_values)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f'Accuracy of the model: {accuracy:.2f}%')

if __name__ == "__main__":
    root_dir = 'D:\\Gurmukhi dataset\\Gurmukhi dataset\\archive\\prepossesed_data'  # Path to dataset
    model_path = r"D:\Gurmukhi dataset\Gurmukhi dataset\archive\codes\vit_gurmukhi_model.pth"  # Path to saved model
    num_classes = 41  # Update based on your dataset
    evaluate_model(root_dir, model_path, num_classes)
