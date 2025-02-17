import torch
from transformers import ViTForImageClassification

# Load or create your model (if it's already trained, you can skip training)
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=41)

# Load the trained weights into your model (if you've saved them previously)
# Uncomment if you want to load previously trained weights
# model.load_state_dict(torch.load('path_to_previous_weights.pth'))

# Save the current model state (after training)
model_path = r'D:\Gurmukhi dataset\Gurmukhi dataset\archive\codes\vit_gurmukhi_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved at: {model_path}")
