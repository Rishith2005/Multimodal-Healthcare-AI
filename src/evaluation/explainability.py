"""
Explainability Utilities — Grad-CAM, SHAP, Time-Series Importance
-------------------------------------------------------------------
Provides visualization functions for each modality branch:
  - Grad-CAM heatmaps for chest X-ray CNN predictions
  - SHAP summary/bar plots for tabular feature importance
  - Time-step feature importance for ICU time-series RNN
All plots are saved to outputs/plots/.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------------- #
# Grad-CAM for Chest X-ray CNN
# --------------------------------------------------------------------------- #

def grad_cam_xray(model, image: torch.Tensor, target_class: int = None,
                  save_path: str = None) -> np.ndarray:
    """
    Generate a Grad-CAM heatmap overlay on a chest X-ray image.

    Uses the pytorch-grad-cam library for robust Grad-CAM computation.

    Args:
        model: Trained CNNBranch model (in eval mode)
        image: Single image tensor of shape (1, 3, 224, 224) or (3, 224, 224)
        target_class: Target class index (None = predicted class)
        save_path: Path to save the overlay visualization

    Returns:
        Heatmap numpy array of shape (224, 224)
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("[Explainability] pytorch-grad-cam not installed. "
              "Install with: pip install grad-cam")
        return _grad_cam_fallback(model, image, target_class, save_path)

    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    # Target layer is the last conv layer of ResNet18 (layer4[-1])
    target_layer = model.grad_cam_target_layer

    cam = GradCAM(model=model, target_layers=[target_layer])

    # Run Grad-CAM
    if target_class is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(target_class)]
    else:
        targets = None

    grayscale_cam = cam(input_tensor=image, targets=targets)
    heatmap = grayscale_cam[0]  # (224, 224)

    # Create overlay
    if save_path:
        # Denormalize image for visualization
        img_np = image[0].cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        overlay = show_cam_on_image(img_np, heatmap, use_rgb=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img_np)
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Explainability] Grad-CAM saved to {save_path}")

    return heatmap


def _grad_cam_fallback(model, image, target_class, save_path):
    """Fallback Grad-CAM implementation without pytorch-grad-cam."""
    model.eval()

    if image.dim() == 3:
        image = image.unsqueeze(0)

    # Register hooks on the target layer
    target_layer = model.grad_cam_target_layer
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    image.requires_grad_(True)
    output = model(image)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, target_class].backward(retain_graph=True)

    fh.remove()
    bh.remove()

    if not activations or not gradients:
        print("[Explainability] Grad-CAM fallback: hooks did not capture data.")
        return np.zeros((224, 224))

    acts = activations[0].detach()
    grads = gradients[0].detach()

    # Global average pooling of gradients
    weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    heatmap = (weights * acts).sum(dim=1)  # (1, H, W)
    heatmap = torch.relu(heatmap)
    heatmap = heatmap.squeeze().cpu().numpy()

    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    if save_path:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        img_np = image[0].detach().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(img_np * std + mean, 0, 1)

        axes[0].imshow(img_np)
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')

        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap (Fallback)')
        axes[1].axis('off')

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Explainability] Grad-CAM (fallback) saved to {save_path}")

    return heatmap


# --------------------------------------------------------------------------- #
# SHAP for Tabular Branch
# --------------------------------------------------------------------------- #

def shap_tabular(model, X_test: np.ndarray, feature_names: list = None,
                 max_samples: int = 100, save_dir: str = None):
    """
    Generate SHAP summary and bar plots for the tabular branch model.

    Args:
        model: Trained TabularBranch model
        X_test: Test feature matrix (N, num_features)
        feature_names: List of feature name strings
        max_samples: Max samples to use for SHAP computation
        save_dir: Directory to save SHAP plots
    """
    try:
        import shap
    except ImportError:
        print("[Explainability] SHAP not installed. Install with: pip install shap")
        return

    model.eval()

    # Wrap the model for SHAP (expects a callable returning numpy)
    def model_fn(x):
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32)
            logits = model(tensor)
            proba = torch.softmax(logits, dim=1).numpy()
        return proba

    # Subsample for speed
    if X_test.shape[0] > max_samples:
        idx = np.random.choice(X_test.shape[0], max_samples, replace=False)
        X_sample = X_test[idx]
    else:
        X_sample = X_test

    # Compute SHAP values using KernelExplainer (model-agnostic)
    background = shap.kmeans(X_sample, min(20, len(X_sample)))
    explainer = shap.KernelExplainer(model_fn, background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X_test.shape[1])]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Summary plot (beeswarm)
        fig, ax = plt.subplots(figsize=(10, 6))
        # shap_values is a list for multi-class; take positive class (index 1)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(sv, X_sample, feature_names=feature_names,
                          show=False, plot_size=None)
        plt.title('SHAP Summary Plot — Tabular Branch', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_summary.png'),
                    dpi=150, bbox_inches='tight')
        plt.close('all')

        # Bar plot (mean absolute SHAP)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(sv, X_sample, feature_names=feature_names,
                          plot_type='bar', show=False)
        plt.title('SHAP Feature Importance — Tabular Branch', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'shap_bar.png'),
                    dpi=150, bbox_inches='tight')
        plt.close('all')

        print(f"[Explainability] SHAP plots saved to {save_dir}")

    return shap_values


# --------------------------------------------------------------------------- #
# Time-Series Feature Importance for RNN Branch
# --------------------------------------------------------------------------- #

def timeseries_importance(model, sequence: torch.Tensor,
                          vital_names: list = None,
                          save_path: str = None) -> np.ndarray:
    """
    Compute per-feature, per-timestep importance for the RNN branch using
    input-gradient sensitivity analysis.

    Args:
        model: Trained RNNBranch model
        sequence: Single sequence tensor (1, T, F) or (T, F)
        vital_names: List of vital sign names (length F)
        save_path: Path to save the importance heatmap

    Returns:
        Importance matrix of shape (T, F)
    """
    model.eval()

    if sequence.dim() == 2:
        sequence = sequence.unsqueeze(0)

    sequence = sequence.clone().requires_grad_(True)

    # Forward pass
    output = model(sequence)
    target_class = output.argmax(dim=1).item()

    # Backward pass to get input gradients
    model.zero_grad()
    output[0, target_class].backward()

    # Gradient magnitude as importance
    grad = sequence.grad.detach().squeeze().cpu().numpy()  # (T, F)
    importance = np.abs(grad)

    # Normalize per feature
    max_vals = importance.max(axis=0, keepdims=True)
    max_vals[max_vals == 0] = 1.0
    importance_norm = importance / max_vals

    if vital_names is None:
        vital_names = [f"Vital {i}" for i in range(importance.shape[1])]

    if save_path:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Heatmap
        sns.heatmap(importance_norm.T, cmap='YlOrRd', ax=axes[0],
                    yticklabels=vital_names, xticklabels=False)
        axes[0].set_xlabel('Time Step (hours)', fontsize=12)
        axes[0].set_ylabel('Vital Sign', fontsize=12)
        axes[0].set_title('Feature Importance Over Time — RNN Branch', fontsize=14)

        # Aggregate importance per feature
        agg_importance = importance.sum(axis=0)
        sorted_idx = np.argsort(agg_importance)[::-1]
        axes[1].barh(range(len(vital_names)),
                     agg_importance[sorted_idx], color='#2196F3')
        axes[1].set_yticks(range(len(vital_names)))
        axes[1].set_yticklabels([vital_names[i] for i in sorted_idx])
        axes[1].set_xlabel('Total Importance', fontsize=12)
        axes[1].set_title('Aggregate Feature Importance', fontsize=14)
        axes[1].invert_yaxis()

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[Explainability] Time-series importance saved to {save_path}")

    return importance_norm


# --------------------------------------------------------------------------- #
# Main: Demo usage
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.models.tabular_branch import TabularBranch
    from src.models.rnn_branch import RNNBranch

    print("Testing explainability utilities...")

    # Tabular SHAP test
    tab_model = TabularBranch(num_features=8, num_classes=2)
    X_dummy = np.random.randn(50, 8).astype(np.float32)
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigree', 'Age']
    shap_tabular(tab_model, X_dummy, feature_names, save_dir='outputs/plots')

    # Time-series importance test
    rnn_model = RNNBranch(input_size=12, num_classes=2)
    seq = torch.randn(1, 48, 12)
    vitals = ['HR', 'RespRate', 'Temp', 'NISysABP', 'NIDiasABP', 'NIMAP',
              'GCS', 'Glucose', 'BUN', 'Creatinine', 'HCT', 'Urine']
    timeseries_importance(rnn_model, seq, vitals,
                          save_path='outputs/plots/ts_importance.png')

    print("Done!")
