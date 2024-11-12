from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import timm
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import random_split


# Load datasets
afhq_dataset = load_dataset("bitmind/AFHQ")
afhq_realvisxl_dataset = load_dataset("bitmind/AFHQ___RealVisXL_V4.0")

# # Explore and visualize some samples
# def show_sample_images(dataset, title, num_images=3):
#     plt.figure(figsize=(10, 5))
#     for i in range(num_images):
#         example = dataset[i]
#         plt.subplot(1, num_images, i + 1)
#         plt.imshow(example["image"])
#         plt.axis('off')
#         plt.title(title)
#     plt.show()

# show_sample_images(afhq_dataset["train"], "Real Images")
# show_sample_images(afhq_realvisxl_dataset["train"], "Synthetic Images")

# Custom Dataset Class
class ImageDataset(Dataset):
    def __init__(self, dataset, is_synthetic=False):
        self.dataset = dataset
        self.is_synthetic = is_synthetic
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        
        # Ensure image is in correct format
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = self.transform(image)
        label = torch.tensor(1 if self.is_synthetic else 0, dtype=torch.long)
        
        return image, label

class RealFakeDetectorTransferLearning(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, freeze_layers=True, dropout_rate=0.5):
        super().__init__()
        # Use EfficientNet-B3 for better feature extraction
        self.base_model = timm.create_model('efficientnet_b3', pretrained=True)
        self.learning_rate = learning_rate
        self.val_predictions = []
        self.val_labels = []
        
        # Freeze EfficientNet layers and set to eval mode
        if freeze_layers:
            self.base_model.eval()  # Set to evaluation mode
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        num_features = self.base_model.get_classifier().in_features
        self.base_model.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        with torch.no_grad():  # No gradients needed for frozen base_model
            x = self.base_model(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_labels.extend(y.cpu().numpy())
        
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def on_validation_epoch_end(self):
        f1 = f1_score(self.val_labels, self.val_predictions, average='weighted')
        self.log('val_f1', f1, prog_bar=True)
        self.val_predictions = []
        self.val_labels = []

        # Calculate F1 score
        f1 = f1_score(
            all_labels.cpu().numpy(),
            all_preds.cpu().numpy(),
            average='weighted'
        )
        
        # Log the F1 score
        self.log('val_f1', f1, prog_bar=True)
        
        # Clear lists for next epoch
        self.val_preds.clear()
        self.val_labels.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "frequency": 1
            }
        }

# Dataset preparation with train/val split
train_dataset_real = ImageDataset(afhq_dataset["train"], is_synthetic=False)
train_dataset_synthetic = ImageDataset(afhq_realvisxl_dataset["train"], is_synthetic=True)
combined_dataset = ConcatDataset([train_dataset_real, train_dataset_synthetic])

# 80-20 train-validation split
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

# DataLoader configuration
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4  # Increase workers for efficiency
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# Instantiate the model
model = RealFakeDetectorTransferLearning()

# Trainer setup as before
trainer = pl.Trainer(
    max_epochs=1,  # Train for more epochs
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    callbacks=[
        pl.callbacks.EarlyStopping(
            monitor='train_loss',
            patience=5,
            mode='min'
        ),
        pl.callbacks.ModelCheckpoint(
            monitor='train_loss',
            dirpath='checkpoints',
            filename='best-checkpoint',
            save_top_k=1,
            mode='min'
        )
    ]
)
trainer.fit(model, train_loader)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")


# Save and load the model
model_name = "hello.pth"
torch.save(model.state_dict(), model_name)
model = RealFakeDetectorTransferLearning()
model.load_state_dict(torch.load(model_name))
model.eval()

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

def predict_and_visualize(model, real_dataset, synthetic_dataset, real_idx=0, synthetic_idx=0):
    """
    Predicts and visualizes results for one real and one synthetic image
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get one sample from each dataset
    real_img, _ = real_dataset[real_idx]
    synthetic_img, _ = synthetic_dataset[synthetic_idx]
    
    # Get raw predictions
    with torch.no_grad():
        real_pred = model(real_img.unsqueeze(0))
        synthetic_pred = model(synthetic_img.unsqueeze(0))
        
        # Apply softmax to get probabilities
        real_prob = F.softmax(real_pred, dim=1)
        synthetic_prob = F.softmax(synthetic_pred, dim=1)
        
        # Get confidence scores
        real_conf = real_prob.max().item() * 100
        synthetic_conf = synthetic_prob.max().item() * 100
        
        # Get predicted labels
        real_label = "Real" if real_prob.argmax().item() == 0 else "Synthetic"
        synthetic_label = "Real" if synthetic_prob.argmax().item() == 0 else "Synthetic"

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Convert tensor to image for visualization
    def tensor_to_img(tensor):
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = tensor * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        return img
    
    # Plot first image
    ax1.imshow(tensor_to_img(real_img))
    ax1.set_title(f'True: Real\nPredicted: {real_label}\nConfidence: {real_conf:.2f}%')
    ax1.axis('off')
    
    # Plot second image
    ax2.imshow(tensor_to_img(synthetic_img))
    ax2.set_title(f'True: Synthetic\nPredicted: {synthetic_label}\nConfidence: {synthetic_conf:.2f}%')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed predictions
    print("\nDetailed Predictions:")
    print(f"\nReal Image:")
    print(f"Raw logits: {real_pred.squeeze().tolist()}")
    print(f"Probabilities: Real: {real_prob[0][0]*100:.2f}%, Synthetic: {real_prob[0][1]*100:.2f}%")
    print(f"\nSynthetic Image:")
    print(f"Raw logits: {synthetic_pred.squeeze().tolist()}")
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=3,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }
    print(f"Probabilities: Real: {synthetic_prob[0][0]*100:.2f}%, Synthetic: {synthetic_prob[0][1]*100:.2f}%")

# Test the visualization with loaded model and datasets
# predict_and_visualize(loaded_model, train_dataset_real, train_dataset_synthetic)

# You can also try different images by changing the indices:
predict_and_visualize(model, train_dataset_real, train_dataset_synthetic, real_idx=32, synthetic_idx=4)