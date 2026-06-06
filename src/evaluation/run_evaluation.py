"""
Full Evaluation Runner — All Branch Models + Fusion
-----------------------------------------------------
Loads all trained checkpoints and runs evaluation on each model's
test set, generating metrics, confusion matrices, ROC curves,
and classification reports.
"""

import os
import sys
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.preprocess_xray import get_xray_dataloaders
from src.data.preprocess_tabular import get_tabular_dataloaders
from src.data.preprocess_timeseries import get_timeseries_dataloaders
from src.models.cnn_branch import CNNBranch
from src.models.tabular_branch import TabularBranch
from src.models.rnn_branch import RNNBranch
from src.models.fusion_model import FusionModel
from src.evaluation.metrics import (
    compute_metrics, plot_confusion_matrix, plot_roc_curve,
    generate_classification_report, evaluate_model
)


def load_config():
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_project_root():
    return os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def evaluate_cnn(config, device, outputs_dir):
    """Evaluate the CNN branch on chest X-ray test set."""
    print("\n" + "=" * 60)
    print("EVALUATING CNN BRANCH — Chest X-ray Pneumonia")
    print("=" * 60)

    cnn_cfg = config['models']['cnn']
    _, _, test_loader, class_names = get_xray_dataloaders(config)

    model = CNNBranch(
        num_classes=config['data']['xray']['num_classes'],
        embedding_dim=cnn_cfg['embedding_dim'],
        dropout=cnn_cfg['dropout'],
        pretrained=cnn_cfg['pretrained'],
    ).to(device)

    ckpt_path = os.path.join(get_project_root(),
                             config['outputs']['checkpoints'], 'cnn_best.pth')
    if not os.path.exists(ckpt_path):
        print("[WARN] CNN checkpoint not found, skipping.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded CNN checkpoint (val_acc={ckpt.get('val_acc', 'N/A'):.4f})")

    metrics = evaluate_model(
        model, test_loader, device, class_names,
        model_name="CNN_XRay", outputs_dir=outputs_dir
    )
    return metrics


def evaluate_tabular(config, device, outputs_dir):
    """Evaluate the Tabular branch on diabetes test set."""
    print("\n" + "=" * 60)
    print("EVALUATING TABULAR BRANCH — Diabetes Risk")
    print("=" * 60)

    tab_cfg = config['models']['tabular']
    _, test_loader, _, _, _ = get_tabular_dataloaders(config)
    class_names = config['data']['tabular']['class_names']

    model = TabularBranch(
        num_features=config['data']['tabular']['num_features'],
        num_classes=config['data']['tabular']['num_classes'],
        hidden_dims=tab_cfg['hidden_dims'],
        embedding_dim=tab_cfg['embedding_dim'],
        dropout=tab_cfg['dropout'],
    ).to(device)

    ckpt_path = os.path.join(get_project_root(),
                             config['outputs']['checkpoints'], 'tabular_best.pth')
    if not os.path.exists(ckpt_path):
        print("[WARN] Tabular checkpoint not found, skipping.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded Tabular checkpoint (test_acc={ckpt.get('test_acc', 'N/A'):.4f})")

    metrics = evaluate_model(
        model, test_loader, device, class_names,
        model_name="Tabular_Diabetes", outputs_dir=outputs_dir
    )
    return metrics


def evaluate_rnn(config, device, outputs_dir):
    """Evaluate the RNN branch on ICU time-series test set."""
    print("\n" + "=" * 60)
    print("EVALUATING RNN BRANCH — ICU Time-Series Mortality")
    print("=" * 60)

    rnn_cfg = config['models']['rnn']
    ts_cfg = config['data']['timeseries']
    _, test_loader, _ = get_timeseries_dataloaders(config, max_patients=None)
    class_names = ts_cfg['class_names']

    model = RNNBranch(
        input_size=ts_cfg['num_vitals'],
        hidden_size=rnn_cfg['hidden_size'],
        num_layers=rnn_cfg['num_layers'],
        num_classes=ts_cfg['num_classes'],
        embedding_dim=rnn_cfg['embedding_dim'],
        dropout=rnn_cfg['dropout'],
    ).to(device)

    ckpt_path = os.path.join(get_project_root(),
                             config['outputs']['checkpoints'], 'rnn_best.pth')
    if not os.path.exists(ckpt_path):
        print("[WARN] RNN checkpoint not found, skipping.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded RNN checkpoint (test_acc={ckpt.get('test_acc', 'N/A'):.4f})")

    # Custom evaluation for RNN (dataloader returns sequences + labels)
    model.eval()
    all_preds, all_labels, all_proba = [], [], []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            proba = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_proba.extend(proba.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_proba)

    metrics = compute_metrics(y_true, y_pred, y_proba)

    print(f"\n{'='*50}")
    print("Evaluation: RNN_ICU_TimeSeries")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")

    plots_dir = os.path.join(outputs_dir, 'plots')
    reports_dir = os.path.join(outputs_dir, 'reports')

    plot_confusion_matrix(y_true, y_pred, class_names,
                          title="RNN ICU TimeSeries — Confusion Matrix",
                          save_path=os.path.join(plots_dir, "RNN_ICU_confusion_matrix.png"))

    if y_proba is not None:
        plot_roc_curve(y_true, y_proba, class_names,
                       title="RNN ICU TimeSeries — ROC Curve",
                       save_path=os.path.join(plots_dir, "RNN_ICU_roc_curve.png"))

    report = generate_classification_report(
        y_true, y_pred, class_names,
        model_name="RNN_ICU_TimeSeries", save_dir=reports_dir
    )
    print(f"\n{report}")

    return metrics


def evaluate_fusion(config, device, outputs_dir):
    """Evaluate the fusion model on simulated aligned test batches."""
    print("\n" + "=" * 60)
    print("EVALUATING FUSION MODEL — Multimodal Risk Prediction")
    print("=" * 60)

    fusion_cfg = config['models']['fusion']
    cnn_cfg = config['models']['cnn']
    tab_cfg = config['models']['tabular']
    rnn_cfg = config['models']['rnn']
    ts_cfg = config['data']['timeseries']

    # Load test dataloaders
    _, _, xray_test, xray_classes = get_xray_dataloaders(config)
    _, tab_test, _, _, _ = get_tabular_dataloaders(config)
    _, rnn_test, _ = get_timeseries_dataloaders(config, max_patients=None)

    # Build branches and load fusion
    cnn = CNNBranch(
        num_classes=config['data']['xray']['num_classes'],
        embedding_dim=cnn_cfg['embedding_dim'],
        dropout=cnn_cfg['dropout'],
        pretrained=cnn_cfg['pretrained'],
    )
    tab = TabularBranch(
        num_features=config['data']['tabular']['num_features'],
        num_classes=config['data']['tabular']['num_classes'],
        hidden_dims=tab_cfg['hidden_dims'],
        embedding_dim=tab_cfg['embedding_dim'],
        dropout=tab_cfg['dropout'],
    )
    rnn = RNNBranch(
        input_size=ts_cfg['num_vitals'],
        hidden_size=rnn_cfg['hidden_size'],
        num_layers=rnn_cfg['num_layers'],
        num_classes=ts_cfg['num_classes'],
        embedding_dim=rnn_cfg['embedding_dim'],
        dropout=rnn_cfg['dropout'],
    )

    fusion = FusionModel(
        cnn_branch=cnn, tabular_branch=tab, rnn_branch=rnn,
        freeze_branches=True,
        hidden_dims=fusion_cfg['hidden_dims'],
        dropout=fusion_cfg['dropout'],
        num_severity_classes=fusion_cfg['num_severity_classes'],
    ).to(device)

    ckpt_path = os.path.join(get_project_root(),
                             config['outputs']['checkpoints'], 'fusion_best.pth')
    if not os.path.exists(ckpt_path):
        print("[WARN] Fusion checkpoint not found, skipping.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    fusion.load_state_dict(ckpt['fusion_state_dict'])
    print(f"Loaded Fusion checkpoint (val_loss={ckpt.get('val_loss', 'N/A'):.4f})")

    # Evaluate on simulated aligned batches
    fusion.eval()
    binary_preds, binary_labels = [], []
    severity_preds, severity_labels = [], []
    all_binary_proba = []

    # Shuffle X-ray test loader for validation to mix classes
    from torch.utils.data import DataLoader
    xray_test_shuffled = DataLoader(
        xray_test.dataset,
        batch_size=xray_test.batch_size,
        shuffle=True,
        num_workers=xray_test.num_workers,
        pin_memory=xray_test.pin_memory
    )

    n_batches = len(xray_test_shuffled)
    xray_iter = iter(xray_test_shuffled)
    tab_iter = iter(tab_test)
    rnn_iter = iter(rnn_test)

    with torch.no_grad():
        for _ in range(n_batches):
            try:
                xi, xl = next(xray_iter)
            except StopIteration:
                break # Should not happen

            try:
                tf, tl = next(tab_iter)
            except StopIteration:
                tab_iter = iter(tab_test)
                tf, tl = next(tab_iter)

            try:
                rs, rl = next(rnn_iter)
            except StopIteration:
                rnn_iter = iter(rnn_test)
                rs, rl = next(rnn_iter)

            min_bs = min(xi.size(0), tf.size(0), rs.size(0))
            xi = xi[:min_bs].to(device)
            tf = tf[:min_bs].to(device)
            rs = rs[:min_bs].to(device)
            xl = xl[:min_bs].to(device)

            out = fusion(xi, tf, rs)
            b_logits = out['binary_logits']
            s_logits = out['severity_logits']

            b_proba = torch.softmax(b_logits, dim=1)
            _, b_pred = b_logits.max(1)
            _, s_pred = s_logits.max(1)

            binary_preds.extend(b_pred.cpu().numpy())
            binary_labels.extend(xl.cpu().numpy())
            all_binary_proba.extend(b_proba.cpu().numpy())

            # Severity labels (using model predictions for evaluation)
            severity_labels.extend(s_pred.cpu().numpy())  # self-eval
            severity_preds.extend(s_pred.cpu().numpy())

    y_true = np.array(binary_labels)
    y_pred = np.array(binary_preds)
    y_proba = np.array(all_binary_proba)

    metrics = compute_metrics(y_true, y_pred, y_proba)

    print(f"\n{'='*50}")
    print("Evaluation: Fusion_Model (Binary Risk)")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")

    plots_dir = os.path.join(outputs_dir, 'plots')
    reports_dir = os.path.join(outputs_dir, 'reports')

    class_names = ["Low Risk", "High Risk"]
    plot_confusion_matrix(y_true, y_pred, class_names,
                          title="Fusion Model — Binary Risk Confusion Matrix",
                          save_path=os.path.join(plots_dir, "Fusion_confusion_matrix.png"))

    if y_proba is not None:
        plot_roc_curve(y_true, y_proba, class_names,
                       title="Fusion Model — Binary Risk ROC Curve",
                       save_path=os.path.join(plots_dir, "Fusion_roc_curve.png"))

    report = generate_classification_report(
        y_true, y_pred, class_names,
        model_name="Fusion_Model", save_dir=reports_dir
    )
    print(f"\n{report}")

    return metrics


def run_all_evaluations():
    """Run evaluation for all models."""
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outputs_dir = os.path.join(get_project_root(), 'outputs')
    os.makedirs(os.path.join(outputs_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(outputs_dir, 'reports'), exist_ok=True)

    print(f"Device: {device}")
    print(f"Outputs directory: {outputs_dir}")

    results = {}

    # Evaluate each branch
    results['cnn'] = evaluate_cnn(config, device, outputs_dir)
    results['tabular'] = evaluate_tabular(config, device, outputs_dir)
    results['rnn'] = evaluate_rnn(config, device, outputs_dir)
    results['fusion'] = evaluate_fusion(config, device, outputs_dir)

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for name, metrics in results.items():
        if metrics:
            acc = metrics.get('accuracy', 0)
            f1 = metrics.get('f1', 0)
            auc = metrics.get('roc_auc', None)
            auc_str = f"{auc:.4f}" if auc else "N/A"
            print(f"  {name:12s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc_str}")
        else:
            print(f"  {name:12s} | SKIPPED (no checkpoint)")

    print("=" * 60)
    print("Evaluation complete. Results saved to outputs/plots/ and outputs/reports/")


if __name__ == "__main__":
    run_all_evaluations()
