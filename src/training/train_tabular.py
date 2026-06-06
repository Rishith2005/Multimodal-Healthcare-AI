"""
Tabular Branch Training — Diabetes Risk Classification
---------------------------------------------------------
Trains the MLP branch on the diabetes/pregnancy tabular dataset with
class-weighted loss, AdamW optimizer, cosine scheduling, and early stopping.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocess_tabular import get_tabular_dataloaders
from src.models.tabular_branch import TabularBranch


def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_tabular():
    config = load_config()
    tab_model_cfg = config['models']['tabular']
    train_cfg = config['training']['tabular']
    general_cfg = config['training']['general']

    torch.manual_seed(general_cfg['seed'])
    np.random.seed(general_cfg['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data — now returns class_weights for imbalanced loss
    train_loader, test_loader, scaler, feature_names, class_weights = \
        get_tabular_dataloaders(config)

    # Model
    model = TabularBranch(
        num_features=config['data']['tabular']['num_features'],
        num_classes=config['data']['tabular']['num_classes'],
        hidden_dims=tab_model_cfg['hidden_dims'],
        embedding_dim=tab_model_cfg['embedding_dim'],
        dropout=tab_model_cfg['dropout'],
    ).to(device)

    # Class-weighted loss for imbalanced dataset
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg['epochs'], eta_min=1e-6
    )

    ckpt_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', config['outputs']['checkpoints']
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    best_test_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    print(f"\n{'='*60}")
    print(f"Training Tabular Branch — {train_cfg['epochs']} epochs")
    print(f"{'='*60}")

    for epoch in range(train_cfg['epochs']):
        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

        scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Test ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                test_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                test_total += y_batch.size(0)
                test_correct += predicted.eq(y_batch).sum().item()

        test_loss /= test_total
        test_acc = test_correct / test_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        lr_now = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | LR: {lr_now:.2e}")

        # Save best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            ckpt_path = os.path.join(ckpt_dir, 'tabular_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'test_acc': test_acc,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
                'feature_names': feature_names,
                'config': tab_model_cfg,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= train_cfg['patience']:
                print(f"  → Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest Test Accuracy: {best_test_acc:.4f}")
    print(f"{'='*60}\n")

    return model, history


if __name__ == "__main__":
    train_tabular()
