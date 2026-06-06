"""
Diabetes / Pregnancy Tabular Data Preprocessing
-------------------------------------------------
Loads the Pima Indians Diabetes dataset, handles biological zero imputation,
applies StandardScaler, and returns PyTorch TensorDatasets.
"""

import os
import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_config():
    """Load project configuration from configs/config.yaml."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def preprocess_tabular_data(config: dict = None):
    """
    Load and preprocess the diabetes CSV dataset.

    Steps:
        1. Load CSV
        2. Replace biological zeros with NaN (Glucose, BloodPressure, etc.)
        3. Median-impute NaN values
        4. Stratified 80/20 train/test split
        5. StandardScaler fit on train, transform both

    Returns:
        X_train, X_test, y_train, y_test (numpy arrays), scaler, feature_names
    """
    if config is None:
        config = load_config()

    tab_cfg = config['data']['tabular']
    general_cfg = config['training']['general']

    # Resolve path: config paths are relative to project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(base_dir, '..', '..'))
    csv_path = os.path.normpath(os.path.join(project_root, tab_cfg['csv_path']))

    df = pd.read_csv(csv_path)

    # Columns where 0 is biologically impossible → replace with NaN
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_invalid_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Median imputation
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Separate features and target
    target_col = tab_cfg['target_column']
    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=general_cfg['val_split'],
        random_state=general_cfg['seed'], stratify=y
    )

    # StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    print(f"[Tabular] Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    print(f"[Tabular] Features: {feature_names}")
    print(f"[Tabular] Class distribution (train): "
          f"{dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Compute class weights for imbalanced datasets: w_c = N / (K * N_c)
    n_samples = len(y_train)
    n_classes = len(np.unique(y_train))
    class_counts = np.bincount(y_train.astype(int), minlength=n_classes)
    class_weights = n_samples / (n_classes * class_counts.astype(np.float32))
    print(f"[Tabular] Class weights: {class_weights.tolist()}")

    return X_train, X_test, y_train, y_test, scaler, feature_names, class_weights


def get_tabular_dataloaders(config: dict = None):
    """
    Create PyTorch DataLoaders for the tabular dataset.

    Returns:
        train_loader, test_loader, scaler, feature_names, class_weights
    """
    if config is None:
        config = load_config()

    X_train, X_test, y_train, y_test, scaler, feature_names, class_weights = \
        preprocess_tabular_data(config)

    batch_size = config['training']['tabular']['batch_size']

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, scaler, feature_names, class_weights


if __name__ == "__main__":
    train_loader, test_loader, scaler, feature_names, class_weights = get_tabular_dataloaders()
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch shape: {X_batch.shape}, Labels: {y_batch[:8]}")
    print(f"Class weights: {class_weights}")
