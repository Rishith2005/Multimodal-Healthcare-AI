"""
Fusion Model Training — Multimodal Healthcare Risk Prediction
--------------------------------------------------------------
Loads pretrained CNN, Tabular, and RNN branch checkpoints, freezes
branch weights, and trains the fusion head on simulated aligned batches
(randomly sampled from each modality's dataset per batch).

Multi-task loss: weighted sum of binary risk + severity classification.
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

from src.data.preprocess_xray import get_xray_dataloaders
from src.data.preprocess_tabular import get_tabular_dataloaders
from src.data.preprocess_timeseries import get_timeseries_dataloaders
from src.models.cnn_branch import CNNBranch
from src.models.tabular_branch import TabularBranch
from src.models.rnn_branch import RNNBranch
from src.models.fusion_model import FusionModel


def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_branch_checkpoints(config, device):
    """Load pretrained branch models from checkpoint files."""
    ckpt_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', config['outputs']['checkpoints']
    )

    # --- CNN ---
    cnn_cfg = config['models']['cnn']
    cnn_model = CNNBranch(
        num_classes=config['data']['xray']['num_classes'],
        embedding_dim=cnn_cfg['embedding_dim'],
        dropout=cnn_cfg['dropout'],
        pretrained=cnn_cfg['pretrained'],
    )
    cnn_ckpt_path = os.path.join(ckpt_dir, 'cnn_best.pth')
    if os.path.exists(cnn_ckpt_path):
        cnn_ckpt = torch.load(cnn_ckpt_path, map_location=device, weights_only=False)
        cnn_model.load_state_dict(cnn_ckpt['model_state_dict'])
        print(f"[Fusion] Loaded CNN checkpoint (val_acc={cnn_ckpt.get('val_acc', 'N/A'):.4f})")
    else:
        print(f"[Fusion] WARNING: CNN checkpoint not found at {cnn_ckpt_path}. Using untrained weights.")

    # --- Tabular ---
    tab_cfg = config['models']['tabular']
    tab_model = TabularBranch(
        num_features=config['data']['tabular']['num_features'],
        num_classes=config['data']['tabular']['num_classes'],
        hidden_dims=tab_cfg['hidden_dims'],
        embedding_dim=tab_cfg['embedding_dim'],
        dropout=tab_cfg['dropout'],
    )
    tab_ckpt_path = os.path.join(ckpt_dir, 'tabular_best.pth')
    if os.path.exists(tab_ckpt_path):
        tab_ckpt = torch.load(tab_ckpt_path, map_location=device, weights_only=False)
        tab_model.load_state_dict(tab_ckpt['model_state_dict'])
        print(f"[Fusion] Loaded Tabular checkpoint (test_acc={tab_ckpt.get('test_acc', 'N/A'):.4f})")
    else:
        print(f"[Fusion] WARNING: Tabular checkpoint not found at {tab_ckpt_path}. Using untrained weights.")

    # --- RNN ---
    rnn_cfg = config['models']['rnn']
    ts_cfg = config['data']['timeseries']
    rnn_model = RNNBranch(
        input_size=ts_cfg['num_vitals'],
        hidden_size=rnn_cfg['hidden_size'],
        num_layers=rnn_cfg['num_layers'],
        num_classes=ts_cfg['num_classes'],
        embedding_dim=rnn_cfg['embedding_dim'],
        dropout=rnn_cfg['dropout'],
    )
    rnn_ckpt_path = os.path.join(ckpt_dir, 'rnn_best.pth')
    if os.path.exists(rnn_ckpt_path):
        rnn_ckpt = torch.load(rnn_ckpt_path, map_location=device, weights_only=False)
        rnn_model.load_state_dict(rnn_ckpt['model_state_dict'])
        print(f"[Fusion] Loaded RNN checkpoint (test_acc={rnn_ckpt.get('test_acc', 'N/A'):.4f})")
    else:
        print(f"[Fusion] WARNING: RNN checkpoint not found at {rnn_ckpt_path}. Using untrained weights.")

    return cnn_model, tab_model, rnn_model


def generate_severity_labels(binary_labels: torch.Tensor,
                             cnn_logits: torch.Tensor,
                             tab_logits: torch.Tensor,
                             rnn_logits: torch.Tensor) -> torch.Tensor:
    """
    Generate synthetic severity labels from binary labels and model confidences.

    Academic prototype heuristic:
        - binary=0 (low risk) → severity 0 (Low) or 1 (Moderate)
        - binary=1 (high risk) → severity 2 (High) or 3 (Critical)

    Split is determined by average model confidence (higher confidence → extreme class).
    """
    device = binary_labels.device
    severity = torch.zeros_like(binary_labels)

    # Average max-confidence across branches
    with torch.no_grad():
        cnn_conf = torch.softmax(cnn_logits, dim=1).max(dim=1).values
        tab_conf = torch.softmax(tab_logits, dim=1).max(dim=1).values
        rnn_conf = torch.softmax(rnn_logits, dim=1).max(dim=1).values
        avg_conf = (cnn_conf + tab_conf + rnn_conf) / 3.0
        median_conf = avg_conf.median()

    high_conf_mask = avg_conf >= median_conf

    # binary=0 → Low (0) or Moderate (1); binary=1 → High (2) or Critical (3)
    severity[binary_labels == 0] = torch.where(
        high_conf_mask[binary_labels == 0],
        torch.ones_like(severity[binary_labels == 0]),   # Moderate
        torch.zeros_like(severity[binary_labels == 0])   # Low
    )
    severity[binary_labels == 1] = torch.where(
        high_conf_mask[binary_labels == 1],
        torch.full_like(severity[binary_labels == 1], 3),  # Critical
        torch.full_like(severity[binary_labels == 1], 2)   # High
    )

    return severity


def train_fusion():
    config = load_config()
    fusion_cfg = config['models']['fusion']
    train_cfg = config['training']['fusion']
    general_cfg = config['training']['general']

    torch.manual_seed(general_cfg['seed'])
    np.random.seed(general_cfg['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data from all three modalities
    print("\n[Fusion] Loading datasets...")
    xray_train, xray_val, xray_test, _ = get_xray_dataloaders(config)
    tab_train, tab_test, _, _, _ = get_tabular_dataloaders(config)
    rnn_train, rnn_test, _ = get_timeseries_dataloaders(config, max_patients=None)

    # Load pretrained branches
    print("\n[Fusion] Loading pretrained branch checkpoints...")
    cnn_model, tab_model, rnn_model = load_branch_checkpoints(config, device)

    # Build fusion model
    fusion = FusionModel(
        cnn_branch=cnn_model,
        tabular_branch=tab_model,
        rnn_branch=rnn_model,
        freeze_branches=True,
        hidden_dims=fusion_cfg['hidden_dims'],
        dropout=fusion_cfg['dropout'],
        num_severity_classes=fusion_cfg['num_severity_classes'],
    ).to(device)

    trainable = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    total = sum(p.numel() for p in fusion.parameters())
    print(f"[Fusion] Trainable params: {trainable:,} / {total:,}")

    # Iterators for each modality (cycling through independently)
    batch_size = train_cfg['batch_size']

    # Loss and optimizer (only fusion head parameters are trainable)
    binary_criterion = nn.CrossEntropyLoss()
    severity_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, fusion.parameters()),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg['epochs'], eta_min=1e-6
    )

    ckpt_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', config['outputs']['checkpoints']
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # Estimate training steps from the smallest modality
    n_batches = min(len(xray_train), len(tab_train), len(rnn_train))
    num_epochs = train_cfg['epochs']

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"\n{'='*60}")
    print(f"Training Fusion Model — {num_epochs} epochs, {n_batches} batches/epoch")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        fusion.train()

        # Create fresh iterators each epoch (shuffles data)
        xray_iter = iter(xray_train)
        tab_iter = iter(tab_train)
        rnn_iter = iter(rnn_train)

        running_loss = 0.0
        binary_correct = 0
        total_samples = 0

        pbar = tqdm(range(n_batches),
                    desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for _ in pbar:
            # Sample one batch from each modality (simulated alignment)
            try:
                xray_imgs, xray_labels = next(xray_iter)
            except StopIteration:
                xray_iter = iter(xray_train)
                xray_imgs, xray_labels = next(xray_iter)

            try:
                tab_feats, tab_labels = next(tab_iter)
            except StopIteration:
                tab_iter = iter(tab_train)
                tab_feats, tab_labels = next(tab_iter)

            try:
                rnn_seqs, rnn_labels = next(rnn_iter)
            except StopIteration:
                rnn_iter = iter(rnn_train)
                rnn_seqs, rnn_labels = next(rnn_iter)

            # Use the minimum batch size across modalities to align
            min_bs = min(xray_imgs.size(0), tab_feats.size(0), rnn_seqs.size(0))
            xray_imgs = xray_imgs[:min_bs].to(device)
            tab_feats = tab_feats[:min_bs].to(device)
            rnn_seqs  = rnn_seqs[:min_bs].to(device)
            xray_labels = xray_labels[:min_bs].to(device)

            optimizer.zero_grad()

            # Forward pass
            output = fusion(xray_imgs, tab_feats, rnn_seqs)
            binary_logits = output['binary_logits']
            severity_logits = output['severity_logits']

            # Binary labels: use xray labels as the fused binary target
            binary_labels = xray_labels
            # Generate synthetic severity labels
            with torch.no_grad():
                cnn_logits_for_sev = fusion.cnn_branch(xray_imgs)
                tab_logits_for_sev = fusion.tabular_branch(tab_feats)
                rnn_logits_for_sev = fusion.rnn_branch(rnn_seqs)
            severity_labels = generate_severity_labels(
                binary_labels, cnn_logits_for_sev, tab_logits_for_sev, rnn_logits_for_sev
            )

            # Multi-task loss
            b_loss = binary_criterion(binary_logits, binary_labels)
            s_loss = severity_criterion(severity_logits, severity_labels)
            loss = (train_cfg['binary_loss_weight'] * b_loss +
                    train_cfg['severity_loss_weight'] * s_loss)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * min_bs
            _, pred_binary = binary_logits.max(1)
            total_samples += min_bs
            binary_correct += pred_binary.eq(binary_labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{binary_correct/total_samples:.4f}")

        train_loss = running_loss / total_samples
        train_acc = binary_correct / total_samples

        # --- Validation (use test loaders as proxy) ---
        fusion.eval()
        val_loss = 0.0
        val_samples = 0
        val_batches = len(xray_test)

        # Shuffle X-ray test loader for validation to mix classes
        from torch.utils.data import DataLoader
        xray_test_shuffled = DataLoader(
            xray_test.dataset,
            batch_size=xray_test.batch_size,
            shuffle=True,
            num_workers=xray_test.num_workers,
            pin_memory=xray_test.pin_memory
        )

        xray_val_iter = iter(xray_test_shuffled)
        tab_val_iter  = iter(tab_test)
        rnn_val_iter  = iter(rnn_test)

        with torch.no_grad():
            for _ in range(val_batches):
                try:
                    xi, xl = next(xray_val_iter)
                except StopIteration:
                    xray_val_iter = iter(xray_test_shuffled)
                    xi, xl = next(xray_val_iter)

                try:
                    tf, tl = next(tab_val_iter)
                except StopIteration:
                    tab_val_iter = iter(tab_test)
                    tf, tl = next(tab_val_iter)

                try:
                    rs, rl = next(rnn_val_iter)
                except StopIteration:
                    rnn_val_iter = iter(rnn_test)
                    rs, rl = next(rnn_val_iter)

                min_bs = min(xi.size(0), tf.size(0), rs.size(0))
                xi = xi[:min_bs].to(device)
                tf = tf[:min_bs].to(device)
                rs = rs[:min_bs].to(device)
                xl = xl[:min_bs].to(device)

                out = fusion(xi, tf, rs)
                b_logits = out['binary_logits']
                s_logits = out['severity_logits']

                sev_labels = generate_severity_labels(
                    xl,
                    fusion.cnn_branch(xi),
                    fusion.tabular_branch(tf),
                    fusion.rnn_branch(rs)
                )

                b_loss = binary_criterion(b_logits, xl)
                s_loss = severity_criterion(s_logits, sev_labels)
                loss = (train_cfg['binary_loss_weight'] * b_loss +
                        train_cfg['severity_loss_weight'] * s_loss)

                val_loss += loss.item() * min_bs
                val_samples += min_bs

        val_loss /= max(val_samples, 1)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {lr_now:.2e}")

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(ckpt_dir, 'fusion_best.pth')
            torch.save({
                'fusion_state_dict': fusion.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'config': fusion_cfg,
            }, ckpt_path)
            print(f"  → Saved best fusion model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg['patience']:
                print(f"  → Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest Val Loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")

    return fusion, history


if __name__ == "__main__":
    train_fusion()
