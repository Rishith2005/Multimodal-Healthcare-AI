"""
Chest X-ray Dataset Preprocessing
----------------------------------
Builds PyTorch DataLoaders from the ChestXRay2017 dataset folder structure.
Applies augmentation for training, standard transforms for val/test.
Handles grayscale → 3-channel conversion automatically.
"""

import os
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_config():
    """Load project configuration from configs/config.yaml."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_transforms(image_size: int = 224, is_train: bool = True):
    """
    Build image transforms.
    - Training: strong augmentation pipeline for medical imaging robustness
    - Val/Test: deterministic resize + normalization only
    """
    # ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            # Convert grayscale → 3-channel FIRST (before spatial transforms)
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            # Geometric augmentation
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(image_size),
            # Photometric augmentation
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.1
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            # Post-tensor augmentation
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def get_xray_dataloaders(config: dict = None, val_split: float = 0.2):
    """
    Create train, validation, and test DataLoaders for the chest X-ray dataset.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    if config is None:
        config = load_config()

    xray_cfg = config['data']['xray']
    train_cfg = config['training']['cnn']
    general_cfg = config['training']['general']

    # Resolve dataset root: config paths are relative to project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(base_dir, '..', '..'))
    root = os.path.normpath(os.path.join(project_root, xray_cfg['root']))
    image_size = xray_cfg['image_size']
    batch_size = train_cfg['batch_size']
    num_workers = general_cfg['num_workers']

    # Build datasets
    train_transform = get_transforms(image_size, is_train=True)
    test_transform = get_transforms(image_size, is_train=False)

    train_dir = os.path.join(root, xray_cfg['train_dir'])
    test_dir = os.path.join(root, xray_cfg['test_dir'])

    full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    class_names = full_train_dataset.classes

    # Split training into train + validation
    total = len(full_train_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    torch.manual_seed(general_cfg['seed'])
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

    # Custom wrapper to apply val transform without augmentation
    class SubsetWithTransform(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            path, target = self.subset.dataset.samples[self.subset.indices[index]]
            img = self.subset.dataset.loader(path)
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        def __len__(self):
            return len(self.subset)

    train_dataset = train_subset
    val_dataset = SubsetWithTransform(val_subset, test_transform)
    # since augmentation is stochastic and won't harm validation too much.
    # For strict separation, we'd need a wrapper — keeping simple for prototype.

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"[X-ray] Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")
    print(f"[X-ray] Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_xray_dataloaders()
    # Quick sanity check
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels[:8]}")
