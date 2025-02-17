import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification
import torchvision.transforms as transforms
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation for inference
        for pixel_values, labels in dataloader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            outputs = model(pixel_values)
            _, predicted = torch.max(outputs.logits, 1)

            all_preds.extend(predicted.cpu().numpy())  # Collect predictions
            all_labels.extend(labels.cpu().numpy())  # Collect true labels

    # Calculate precision, recall, and F1 score
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100

    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    # Plotting the results
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]

    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
    plt.ylim(0, 100)
    plt.ylabel('Score (%)')
    plt.title('Model Evaluation Metrics')
    plt.grid(axis='y')
    plt.show()

if __name__ == "__main__":
    root_dir = 'D:\\Gurmukhi dataset\\Gurmukhi dataset\\archive\\prepossesed_data'  # Path to dataset
    model_path = r"D:\Gurmukhi dataset\Gurmukhi dataset\archive\codes\vit_gurmukhi_model.pth"  # Path to saved model
    num_classes = 41  # Update based on your dataset
    evaluate_model(root_dir, model_path, num_classes)
