"""
Evaluation Metrics & Visualization Utilities
----------------------------------------------
Computes classification metrics (accuracy, precision, recall, F1, ROC-AUC),
generates confusion matrices, ROC curves, and classification reports.
All plots and reports are saved to the outputs/ directory.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray = None) -> dict:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted class labels (N,)
        y_proba: Predicted probabilities for positive class (N,) — optional

    Returns:
        dict with accuracy, precision, recall, f1, and roc_auc (if y_proba provided)
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    if y_proba is not None:
        try:
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            elif y_proba.ndim == 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            else:
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted'
                )
        except ValueError:
            metrics['roc_auc'] = None

    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list, title: str = "Confusion Matrix",
                          save_path: str = None):
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class name strings
        title: Plot title
        save_path: File path to save the figure (None = display only)
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Metrics] Confusion matrix saved to {save_path}")

    plt.close(fig)
    return fig


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   class_names: list = None, title: str = "ROC Curve",
                   save_path: str = None):
    """
    Plot ROC curve for binary or multi-class classification.

    Args:
        y_true: Ground truth labels
        y_proba: Predicted probabilities — shape (N,) or (N, num_classes)
        class_names: List of class names (for multi-class legend)
        title: Plot title
        save_path: File path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if y_proba.ndim == 1 or (y_proba.ndim == 2 and y_proba.shape[1] == 2):
        # Binary classification
        scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='#1f77b4', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        # Multi-class: one-vs-rest
        from sklearn.preprocessing import label_binarize
        n_classes = y_proba.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        colors = plt.cm.get_cmap('tab10', n_classes)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc_val = auc(fpr, tpr)
            label = class_names[i] if class_names else f'Class {i}'
            ax.plot(fpr, tpr, color=colors(i), lw=2,
                    label=f'{label} (AUC = {roc_auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Metrics] ROC curve saved to {save_path}")

    plt.close(fig)
    return fig


def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: list,
                                   model_name: str = "Model",
                                   save_dir: str = None) -> str:
    """
    Generate and save a scikit-learn classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class name strings
        model_name: Name identifier for the report file
        save_dir: Directory to save the report text file

    Returns:
        Report string
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, f"{model_name}_classification_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"Classification Report — {model_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
        print(f"[Metrics] Report saved to {report_path}")

    return report


def evaluate_model(model, dataloader, device, class_names: list,
                   model_name: str = "Model", outputs_dir: str = "outputs"):
    """
    Full evaluation pipeline: compute metrics, plot confusion matrix + ROC curve,
    and save classification report.

    Args:
        model: PyTorch model
        dataloader: Evaluation DataLoader
        device: torch device
        class_names: List of class names
        model_name: Identifier for naming output files
        outputs_dir: Base outputs directory path

    Returns:
        dict of computed metrics
    """
    import torch

    model.eval()
    all_preds = []
    all_labels = []
    all_proba = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs = batch[0]
                labels = batch[-1]

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            proba = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_proba.extend(proba.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_proba)

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)

    # Print metrics
    print(f"\n{'='*50}")
    print(f"Evaluation: {model_name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: N/A")

    # Save outputs
    plots_dir = os.path.join(outputs_dir, 'plots')
    reports_dir = os.path.join(outputs_dir, 'reports')

    plot_confusion_matrix(
        y_true, y_pred, class_names,
        title=f"{model_name} — Confusion Matrix",
        save_path=os.path.join(plots_dir, f"{model_name}_confusion_matrix.png")
    )

    if y_proba is not None:
        plot_roc_curve(
            y_true, y_proba, class_names,
            title=f"{model_name} — ROC Curve",
            save_path=os.path.join(plots_dir, f"{model_name}_roc_curve.png")
        )

    report = generate_classification_report(
        y_true, y_pred, class_names,
        model_name=model_name, save_dir=reports_dir
    )
    print(f"\n{report}")

    return metrics


if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100, 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    print("Dummy metrics:", metrics)

    plot_confusion_matrix(y_true, y_pred, ['Class 0', 'Class 1'],
                          save_path='outputs/plots/test_cm.png')
    plot_roc_curve(y_true, y_proba, ['Class 0', 'Class 1'],
                   save_path='outputs/plots/test_roc.png')
