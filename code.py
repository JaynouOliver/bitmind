# bunch of imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import timm
import torch.nn.functional as F
from sklearn.metrics import f1_score
from datasets import load_dataset

@dataclass
class ModelConfig: #model configs
    """Configuration for model hyperparameters and training settings."""
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 10
    dropout_rate: float = 0.5
    train_split: float = 0.8
    image_size: Tuple[int, int] = (224, 224)
    checkpoint_dir: str = "checkpoints"
    model_name: str = "efficientnet_b3"

class BaseDataset(Dataset, ABC):
    """Abstract base class for datasets."""
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

class ImageDataset(BaseDataset):
    """Dataset class for handling real and synthetic images."""
    def __init__(self, dataset: Dataset, is_synthetic: bool = False, config: ModelConfig = ModelConfig()):
        self.dataset = dataset
        self.is_synthetic = is_synthetic
        self.transform = self._create_transform(config.image_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.dataset[idx]['image']
        image = self._preprocess_image(image)
        label = torch.tensor(1 if self.is_synthetic else 0, dtype=torch.long)
        return image, label

    @staticmethod
    def _create_transform(image_size: Tuple[int, int]) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _preprocess_image(self, image: Any) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image)

class RealFakeDetector(pl.LightningModule, ABC):
    """Abstract base class for real/fake image detection models."""
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Dict:
        pass

class EfficientNetDetector(RealFakeDetector):
    """EfficientNet-based model for real/fake image detection."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.val_predictions = []
        self.val_labels = []

    def _build_model(self) -> None:
        self.base_model = timm.create_model(self.config.model_name, pretrained=True)
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

        num_features = self.base_model.get_classifier().in_features
        self.base_model.classifier = nn.Identity()
        
        self.classifier = self._create_classifier(num_features)

    def _create_classifier(self, num_features: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(256, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.base_model(x)
        return self.classifier(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        self.log_dict({
            'train_loss': loss,
            'train_acc': acc,
            'learning_rate': self.optimizers().param_groups[0]['lr']
        }, prog_bar=True)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        preds = torch.argmax(y_hat, dim=1)
        self.val_predictions.extend(preds.cpu().numpy())
        self.val_labels.extend(y.cpu().numpy())
        
        acc = (preds == y).float().mean()
        self.log_dict({
            'val_loss': loss,
            'val_acc': acc
        }, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        f1 = f1_score(self.val_labels, self.val_predictions, average='weighted')
        self.log('val_f1', f1, prog_bar=True)
        self.val_predictions.clear()
        self.val_labels.clear()

    def configure_optimizers(self) -> Dict:
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
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

class DetectorTrainer:
    """Handles the training pipeline for the detector model."""
    def __init__(self, config: ModelConfig):
        self.config = config
        Path(config.checkpoint_dir).mkdir(exist_ok=True)

    def prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        afhq_dataset = load_dataset("bitmind/AFHQ")
        afhq_realvisxl_dataset = load_dataset("bitmind/AFHQ___RealVisXL_V4.0")

        train_dataset_real = ImageDataset(afhq_dataset["train"], is_synthetic=False, config=self.config)
        train_dataset_synthetic = ImageDataset(afhq_realvisxl_dataset["train"], is_synthetic=True, config=self.config)
        combined_dataset = ConcatDataset([train_dataset_real, train_dataset_synthetic])

        train_size = int(self.config.train_split * len(combined_dataset))
        val_size = len(combined_dataset) - train_size
        train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

        return (
            DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers),
            DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)
        )

    def train(self) -> EfficientNetDetector:
        train_loader, val_loader = self.prepare_data()
        model = EfficientNetDetector(self.config)
        
        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=[
                EarlyStopping(monitor='train_loss', patience=5, mode='min'),
                ModelCheckpoint(
                    monitor='train_loss',
                    dirpath=self.config.checkpoint_dir,
                    filename='best-checkpoint',
                    save_top_k=1,
                    mode='min'
                )
            ]
        )
        
        trainer.fit(model, train_loader, val_loader)
        return model

def main():
    config = ModelConfig()
    trainer = DetectorTrainer(config)
    model = trainer.train()
    
    # Save the trained model
    torch.save(model.state_dict(), "final_model.pth")

if __name__ == "__main__":
    main()