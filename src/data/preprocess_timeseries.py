"""
PhysioNet CinC 2012 ICU Time-Series Preprocessing
----------------------------------------------------
Parses individual patient .txt files (Time, Parameter, Value format),
pivots into fixed time-steps (48h × 12 vitals), handles missing values
with forward-fill + median imputation, and loads Outcomes-a.txt for labels.
"""

import os
import re
import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def load_config():
    """Load project configuration from configs/config.yaml."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_time_to_hours(time_str: str) -> float:
    """Convert HH:MM time string to fractional hours."""
    parts = time_str.strip().split(':')
    return int(parts[0]) + int(parts[1]) / 60.0


def parse_patient_file(filepath: str, selected_vitals: list,
                       sequence_length: int = 48) -> np.ndarray:
    """
    Parse a single PhysioNet patient .txt file into a fixed-size matrix.

    Args:
        filepath: Path to the patient .txt file
        selected_vitals: List of vital parameter names to extract
        sequence_length: Number of hourly time bins (default 48)

    Returns:
        numpy array of shape (sequence_length, num_vitals)
    """
    try:
        df = pd.read_csv(filepath)
    except Exception:
        return None

    if df.empty or 'Time' not in df.columns:
        return None

    # Convert time to hours
    df['Hours'] = df['Time'].apply(parse_time_to_hours)

    # Filter to first 48 hours and selected vitals
    df = df[df['Hours'] < sequence_length]
    df = df[df['Parameter'].isin(selected_vitals)]

    if df.empty:
        return None

    # Assign each measurement to an hourly bin
    df['HourBin'] = df['Hours'].apply(lambda x: min(int(x), sequence_length - 1))

    # Convert Value to numeric, coercing errors
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Pivot: take the mean if multiple readings in same hour
    pivot = df.pivot_table(
        index='HourBin', columns='Parameter', values='Value', aggfunc='mean'
    )

    # Create full matrix with all hours and all vitals
    full_index = np.arange(sequence_length)
    matrix = pd.DataFrame(index=full_index, columns=selected_vitals, dtype=float)

    # Fill in available data
    for hour in pivot.index:
        for vital in pivot.columns:
            if vital in selected_vitals and not pd.isna(pivot.loc[hour, vital]):
                matrix.loc[hour, vital] = pivot.loc[hour, vital]

    # Forward-fill, then backward-fill, then fill remaining with column median
    matrix = matrix.ffill().bfill()

    # Fill any remaining NaNs with global median (0 as ultimate fallback)
    for col in matrix.columns:
        median_val = matrix[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        matrix[col] = matrix[col].fillna(median_val)

    return matrix.values.astype(np.float32)


def load_outcomes(outcomes_path: str) -> dict:
    """
    Load Outcomes-a.txt and return dict mapping RecordID → In-hospital_death.
    """
    df = pd.read_csv(outcomes_path)
    outcomes = {}
    for _, row in df.iterrows():
        record_id = int(row['RecordID'])
        death = int(row['In-hospital_death'])
        outcomes[record_id] = death
    return outcomes


class ICUTimeSeriesDataset(Dataset):
    """PyTorch Dataset for preprocessed ICU time-series data."""

    def __init__(self, sequences: list, labels: list):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def preprocess_timeseries(config: dict = None, max_patients: int = None):
    """
    Process all PhysioNet patient files and create train/test datasets.

    Args:
        config: Configuration dict
        max_patients: Limit number of patients to process (None for all)

    Returns:
        train_loader, test_loader, stats dict
    """
    if config is None:
        config = load_config()

    ts_cfg = config['data']['timeseries']
    general_cfg = config['training']['general']
    train_cfg = config['training']['rnn']

    # Resolve paths: config paths are relative to project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(base_dir, '..', '..'))
    records_dir = os.path.normpath(
        os.path.join(project_root, ts_cfg['records_dir'])
    )
    outcomes_path = os.path.normpath(
        os.path.join(project_root, ts_cfg['outcomes_path'])
    )

    selected_vitals = ts_cfg['selected_vitals']
    seq_length = ts_cfg['sequence_length']

    # Load outcomes
    print(f"[TimeSeries] Loading outcomes from {outcomes_path}")
    outcomes = load_outcomes(outcomes_path)
    print(f"[TimeSeries] Loaded {len(outcomes)} patient outcomes")

    # Parse patient files
    patient_files = sorted([
        f for f in os.listdir(records_dir) if f.endswith('.txt')
    ])

    if max_patients is not None:
        patient_files = patient_files[:max_patients]

    sequences = []
    labels = []
    skipped = 0

    print(f"[TimeSeries] Processing {len(patient_files)} patient files...")
    for i, fname in enumerate(patient_files):
        record_id = int(fname.replace('.txt', ''))

        # Skip if no outcome label available
        if record_id not in outcomes:
            skipped += 1
            continue

        filepath = os.path.join(records_dir, fname)
        matrix = parse_patient_file(filepath, selected_vitals, seq_length)

        if matrix is None:
            skipped += 1
            continue

        sequences.append(matrix)
        labels.append(outcomes[record_id])

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(patient_files)} files...")

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"[TimeSeries] Processed: {len(sequences)}, Skipped: {skipped}")
    print(f"[TimeSeries] Shape: {sequences.shape}")
    print(f"[TimeSeries] Class distribution: "
          f"{dict(zip(*np.unique(labels, return_counts=True)))}")

    # Normalize per-feature across all patients
    # Reshape to (N*T, F), normalize, reshape back
    N, T, F = sequences.shape
    flat = sequences.reshape(-1, F)
    means = np.nanmean(flat, axis=0)
    stds = np.nanstd(flat, axis=0)
    stds[stds == 0] = 1.0  # avoid div by zero
    flat = (flat - means) / stds
    sequences = flat.reshape(N, T, F)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels,
        test_size=general_cfg['val_split'],
        random_state=general_cfg['seed'],
        stratify=labels
    )

    train_dataset = ICUTimeSeriesDataset(X_train, y_train)
    test_dataset = ICUTimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg['batch_size'], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_cfg['batch_size'], shuffle=False
    )

    stats = {
        'means': means, 'stds': stds,
        'total': len(sequences), 'train': len(X_train), 'test': len(X_test)
    }

    print(f"[TimeSeries] Train: {len(X_train)}, Test: {len(X_test)}")

    return train_loader, test_loader, stats


def get_timeseries_dataloaders(config: dict = None, max_patients: int = None):
    """Convenience wrapper matching the interface of other preprocessors."""
    return preprocess_timeseries(config, max_patients)


if __name__ == "__main__":
    train_loader, test_loader, stats = get_timeseries_dataloaders(max_patients=200)
    seq_batch, label_batch = next(iter(train_loader))
    print(f"Batch shape: {seq_batch.shape}, Labels: {label_batch[:8]}")
