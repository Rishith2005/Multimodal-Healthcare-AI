"""
RNN Branch Training — ICU Time-Series Mortality Prediction
------------------------------------------------------------
Trains an LSTM-based model on PhysioNet CinC 2012 ICU vitals data
with class-weighted loss for imbalance, AdamW, cosine scheduling,
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocess_timeseries import get_timeseries_dataloaders
from src.models.rnn_branch import RNNBranch


def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_rnn():
    config = load_config()
    rnn_cfg = config['models']['rnn']
    train_cfg = config['training']['rnn']
    general_cfg = config['training']['general']
    ts_cfg = config['data']['timeseries']

    # Seed
    torch.manual_seed(general_cfg['seed'])
    np.random.seed(general_cfg['seed'])

    # Device
    if general_cfg['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(general_cfg['device'])
    print(f"Using device: {device}")

    # Data — use ALL patients for production-quality training
    max_patients = None  # set to int (e.g. 500) for quick prototyping
    train_loader, test_loader, stats = get_timeseries_dataloaders(
        config, max_patients=max_patients
    )

    # Compute class weights for imbalanced mortality data (~86/14 split)
    # We need class distribution from the training set
    train_dataset = train_loader.dataset
    all_labels = np.array([train_dataset.labels[i] for i in range(len(train_dataset))])
    n_samples = len(all_labels)
    n_classes = ts_cfg['num_classes']
    class_counts = np.bincount(all_labels, minlength=n_classes)
    class_weights = n_samples / (n_classes * class_counts.astype(np.float32))
    print(f"[RNN] Class weights: {class_weights.tolist()}")
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Model
    model = RNNBranch(
        input_size=ts_cfg['num_vitals'],
        hidden_size=rnn_cfg['hidden_size'],
        num_layers=rnn_cfg['num_layers'],
        num_classes=ts_cfg['num_classes'],
        embedding_dim=rnn_cfg['embedding_dim'],
        dropout=rnn_cfg['dropout'],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg['epochs'], eta_min=1e-6
    )

    # Checkpoint dir
    ckpt_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', config['outputs']['checkpoints']
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    best_test_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    print(f"\n{'='*60}")
    print(f"Training RNN Branch — {train_cfg['epochs']} epochs")
    print(f"{'='*60}")

    for epoch in range(train_cfg['epochs']):
        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch+1}/{train_cfg['epochs']} [Train]")
        for sequences, labels in pbar:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for RNN stability (tighter clip)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            running_loss += loss.item() * sequences.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Test ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * sequences.size(0)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_loss /= test_total
        test_acc = test_correct / test_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | LR: {lr_now:.2e}")

        # Early stopping + checkpoint
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            ckpt_path = os.path.join(ckpt_dir, 'rnn_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'stats': {k: v.tolist() if hasattr(v, 'tolist') else v
                          for k, v in stats.items()},
                'config': rnn_cfg,
            }, ckpt_path)
            print(f"  → Saved best model (test_acc={test_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg['patience']:
                print(f"  → Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest Test Accuracy: {best_test_acc:.4f}")
    print(f"{'='*60}\n")

    return model, history


if __name__ == "__main__":
    train_rnn()
