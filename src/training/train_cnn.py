"""
CNN Branch Training — Chest X-ray Pneumonia Classification
------------------------------------------------------------
Trains a ResNet18-based CNN on the chest X-ray dataset with
strong augmentation, label smoothing, AdamW, cosine scheduling,
gradient clipping, and early stopping.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocess_xray import get_xray_dataloaders
from src.models.cnn_branch import CNNBranch


def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_cnn():
    config = load_config()
    cnn_cfg = config['models']['cnn']
    train_cfg = config['training']['cnn']
    general_cfg = config['training']['general']

    # Set seed
    torch.manual_seed(general_cfg['seed'])
    np.random.seed(general_cfg['seed'])

    # Device
    if general_cfg['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(general_cfg['device'])
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, class_names = get_xray_dataloaders(config)

    # Model
    model = CNNBranch(
        num_classes=config['data']['xray']['num_classes'],
        embedding_dim=cnn_cfg['embedding_dim'],
        dropout=cnn_cfg['dropout'],
        pretrained=cnn_cfg['pretrained'],
    ).to(device)

    # Freeze backbone parameters if specified
    if cnn_cfg.get('freeze_backbone', False):
        print("[CNN] Freezing ResNet18 backbone parameters...")
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Loss — label smoothing for calibration in medical domain
    label_smoothing = train_cfg.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # AdamW optimizer (decoupled weight decay)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )

    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # Checkpoint dir
    ckpt_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', config['outputs']['checkpoints']
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, 'cnn_best.pth')
    last_ckpt_path = os.path.join(ckpt_dir, 'cnn_last.pth')

    # Training loop — track best by val_acc (medical domain priority)
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    start_epoch = 0
    resume = train_cfg.get('resume', False)
    if resume:
        ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else best_ckpt_path
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            start_epoch = ckpt.get('epoch', -1) + 1
            best_val_acc = ckpt.get('best_val_acc', ckpt.get('val_acc', 0.0))
            best_val_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf')))
            patience_counter = ckpt.get('patience_counter', 0)
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            elif start_epoch > 0:
                for _ in range(start_epoch):
                    scheduler.step()
            print(f"Resuming from {os.path.basename(ckpt_path)} at epoch {start_epoch+1}")

    print(f"\n{'='*60}")
    print(f"Training CNN Branch — {train_cfg['epochs']} epochs")
    print(f"{'='*60}")

    for epoch in range(start_epoch, train_cfg['epochs']):
        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step()
        
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | LR: {lr_now:.2e}")
        
        # Early stopping + checkpoint (best by val_acc for medical domain)
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'config': cnn_cfg,
            }, best_ckpt_path)
            print(f"  \u2192 Saved best model (val_acc={val_acc:.4f}, val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg['patience']:
                print(f"  \u2192 Early stopping at epoch {epoch+1}")
                break

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'config': cnn_cfg,
        }, last_ckpt_path)

    # --- Test ---
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"{'='*60}\n")

    return model, history


if __name__ == "__main__":
    train_cnn()
