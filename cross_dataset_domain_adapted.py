# -*- coding: utf-8 -*-
"""
Cross-Dataset Evaluation with Domain Adaptation
- Train on UNSW-NB15, test on CICIDS-2017 
- Use feature dimension mapping for compatibility
"""

import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_curve
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
LABEL_CANDS = ["Label", "label", "Attack", "attack", "Class", "class"]

def infer_label_binary(s) -> int:
    s = str(s).lower().strip()
    if s == "benign" or "benign" in s or s == "0":
        return 0
    else:
        return 1

def list_csvs(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.csv") if p.is_file()]

def read_one_csv(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc, on_bad_lines="skip")
            break
        except Exception:
            df = None
    if df is None or df.empty:
        raise ValueError(f"Failed to read {path}")
    df.columns = df.columns.str.strip()
    label_col = next((c for c in LABEL_CANDS if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"{path.name}: cannot find label column")
    y = df[label_col].apply(infer_label_binary).astype(int)
    X = df.drop(columns=[label_col])
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype("float32")
    return X, y

def load_dataset(csvdir: str, name: str, max_files: int = None) -> Tuple[pd.DataFrame, pd.Series]:
    root = Path(csvdir)
    if not root.exists():
        raise SystemExit(f"Not found: {csvdir}")
    files = list_csvs(root)
    if not files:
        raise SystemExit(f"No CSVs in {csvdir}")
    
    if max_files is not None:
        files = files[:max_files]
        
    print(f"Loading {name} dataset from {len(files)} CSV files.")
    X_parts, y_parts = [], []
    for f in files:
        try:
            print(f"  [read] {f.relative_to(root)}")
            X, y = read_one_csv(f)
            X_parts.append(X)
            y_parts.append(y)
        except Exception as e:
            print(f"  [warn] skip {f.name}: {e}")
    
    if not X_parts:
        raise SystemExit(f"No valid CSV files loaded from {csvdir}")
    
    X_raw = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    
    print(f"{name} dataset: {X_raw.shape[0]} samples, {X_raw.shape[1]} features")
    print(f"Label distribution - Normal: {(y==0).sum()}, Attack: {(y==1).sum()}")
    
    return X_raw, y

class CatFactorize(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.maps = {}
        self.cols = []
    def fit(self, X):
        if hasattr(X, 'columns'):
            self.cols = list(X.columns)
            X_df = X
        else:
            self.cols = [f"col_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self.cols)
        
        for i, c in enumerate(self.cols):
            if hasattr(X, 'columns'):
                vals = pd.Series(X_df[c]).astype(str).fillna("<NA>")
            else:
                vals = pd.Series(X_df.iloc[:, i]).astype(str).fillna("<NA>")
            cats = pd.Index(vals.unique())
            self.maps[c] = {k: i for i, k in enumerate(cats)}
        return self
    def transform(self, X):
        if hasattr(X, 'columns'):
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.cols)
            
        for i, c in enumerate(self.cols):
            if hasattr(X, 'columns'):
                vals = pd.Series(X_df[c]).astype(str).fillna("<NA>")
                X_df[c] = vals.map(self.maps[c]).fillna(-1).astype('float32')
            else:
                vals = pd.Series(X_df.iloc[:, i]).astype(str).fillna("<NA>")
                X_df.iloc[:, i] = vals.map(self.maps[c]).fillna(-1).astype('float32')
        return X_df.values.astype('float32')

def preprocess_data_with_mapping(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame, target_dim: int = 50):
    """
    Preprocess datasets with different feature sets by mapping to common dimension
    """
    print(f"Training dataset features: {X_train_raw.shape[1]}")
    print(f"Testing dataset features: {X_test_raw.shape[1]}")
    
    # Clean training data
    X_train_clean = X_train_raw.dropna(axis=1, how="all")
    
    # Process training data
    cat_cols_train = X_train_clean.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols_train = [c for c in X_train_clean.columns if c not in cat_cols_train]

    numeric_transformer_train = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer_train = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("factorize", CatFactorize())
    ])

    preprocessor_train = ColumnTransformer([
        ("num", numeric_transformer_train, num_cols_train),
        ("cat", categorical_transformer_train, cat_cols_train)
    ])

    X_train_processed = preprocessor_train.fit_transform(X_train_clean)
    
    # Apply PCA to training data to get target dimension
    pca_train = PCA(n_components=min(target_dim, X_train_processed.shape[1]))
    X_train_final = pca_train.fit_transform(X_train_processed)
    
    print(f"Training data shape after PCA: {X_train_final.shape}")
    
    # Process testing data
    X_test_clean = X_test_raw.dropna(axis=1, how="all")
    
    cat_cols_test = X_test_clean.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols_test = [c for c in X_test_clean.columns if c not in cat_cols_test]

    numeric_transformer_test = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer_test = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("factorize", CatFactorize())
    ])

    preprocessor_test = ColumnTransformer([
        ("num", numeric_transformer_test, num_cols_test),
        ("cat", categorical_transformer_test, cat_cols_test)
    ])

    X_test_processed = preprocessor_test.fit_transform(X_test_clean)
    
    # Apply PCA to testing data to get same dimension
    pca_test = PCA(n_components=min(target_dim, X_test_processed.shape[1]))
    X_test_final = pca_test.fit_transform(X_test_processed)
    
    print(f"Testing data shape after PCA: {X_test_final.shape}")
    
    # Ensure both have same dimensions by padding/truncating
    final_dim = min(X_train_final.shape[1], X_test_final.shape[1])
    X_train_mapped = X_train_final[:, :final_dim]
    X_test_mapped = X_test_final[:, :final_dim]
    
    print(f"Final mapped dimensions: {final_dim}")
    
    return X_train_mapped.astype('float32'), X_test_mapped.astype('float32')

class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleMLP(nn.Module):
    def __init__(self, in_feats: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

def train_torch(model: nn.Module, dl_tr, dl_va, device="cuda", epochs=10, lr=1e-3, pos_weight=None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device)) if pos_weight is not None else nn.BCEWithLogitsLoss()
    best = {"auc": -np.inf, "state": None}
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tqdm(dl_tr, desc=f"train ep{ep}"):
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); y_true, y_prob = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
                y_true.append(yb.numpy()); y_prob.append(prob)
        y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
        auc = roc_auc_score(y_true, y_prob)
        if auc > best["auc"]:
            best["auc"] = auc
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[val] ROC-AUC={auc:.4f}  PR-AUC={average_precision_score(y_true, y_prob):.4f}")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model

def evaluate_arrays(y_true: np.ndarray, y_prob: np.ndarray):
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    eps = 1e-12
    return dict(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        f1=float(f1_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        tn_rate=tn/(total+eps), fp_rate=fp/(total+eps), fn_rate=fn/(total+eps), tp_rate=tp/(total+eps)
    )

def evaluate_torch(model, dl_te, device="cuda"):
    model.eval(); y_true, y_prob = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            y_true.append(yb.numpy()); y_prob.append(prob)
    y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
    return evaluate_arrays(y_true, y_prob)

def main():
    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    
    print("=== Cross-Dataset Evaluation with Domain Adaptation ===")
    print("Training: UNSW-NB15")
    print("Testing: CICIDS-2017")
    
    # Load datasets
    X_train_raw, y_train_all = load_dataset("CSV_NB15/CSV Files/Training and Testing Sets", "UNSW-NB15")
    X_test_raw, y_test = load_dataset("CSV/MachineLearningCSV/MachineLearningCVE", "CICIDS-2017", max_files=2)
    
    # Split training data
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train_all, test_size=0.2, stratify=y_train_all, random_state=42
    )
    
    # Preprocess data with domain adaptation
    print("\nPreprocessing with domain adaptation...")
    X_train, X_test = preprocess_data_with_mapping(X_train_raw, X_test_raw, target_dim=40)
    X_val, _ = preprocess_data_with_mapping(X_val_raw, X_test_raw, target_dim=40)
    
    print(f"Final shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Train and test SimpleMLP
    print("\n=== Training SimpleMLP ===")
    tr = TabDataset(X_train, y_train.values)
    va = TabDataset(X_val, y_val.values) 
    te = TabDataset(X_test, y_test.values)
    
    dl_tr = DataLoader(tr, batch_size=512, shuffle=True)
    dl_va = DataLoader(va, batch_size=512)
    dl_te = DataLoader(te, batch_size=512)
    
    model = SimpleMLP(in_feats=X_train.shape[1])
    pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-6)
    
    model = train_torch(model, dl_tr, dl_va, device=device, epochs=15, lr=1e-3, pos_weight=pos_weight)
    result = evaluate_torch(model, dl_te, device=device)
    
    print("\n=== Cross-Dataset Results (Domain Adaptation) ===")
    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    print(f"PR-AUC: {result['pr_auc']:.4f}")
    print(f"F1: {result['f1']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    
    # Save results
    df = pd.DataFrame([{
        'model': 'SimpleMLP_CrossDataset_DomainAdapted',
        'training': 'UNSW-NB15',
        'testing': 'CICIDS-2017',
        'roc_auc': result['roc_auc'],
        'pr_auc': result['pr_auc'],
        'f1': result['f1'],
        'precision': result['precision'],
        'recall': result['recall'],
        'tn': result['tn'],
        'fp': result['fp'],
        'fn': result['fn'],
        'tp': result['tp']
    }])
    df.to_csv("cross_dataset_domain_adapted.csv", index=False)
    print("\nResults saved to cross_dataset_domain_adapted.csv")

if __name__ == "__main__":
    print("CUDA device:", torch.cuda.get_device_name(0))
    main()