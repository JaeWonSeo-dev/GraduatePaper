# -*- coding: utf-8 -*-
"""
Runner (Mode-B, 3x2 A/B) - v3.6.1 New Experimental Design

NEW EXPERIMENTAL DESIGN (Total: 10 runs)
========================================

Combo1: AFS -> LSTM (Gate L1 OFF, topk=24/28)
  - Gate L1: lambda=0.0 (disabled)
  - AFS ranking: epochs=8 (fixed)
  - AP-lambda: 0.02 (gentle regulation)
  - TopK: [24, 28] (refined search)
  - Experiments: C1_k24_gateL1off, C1_k28_gateL1off
  - Runs: 4 (2 topk x 2 variants)

Combo2: XGB-FS -> LSTM (Precision Maintenance, topk=20)
  - Precision floor: 0.995 * P_base (strengthened)
  - Hard negative mining: topq=0.30, weight=1.8
  - AP-lambda: 0.09
  - Experiment: C2_k20_precMaint
  - Runs: 2 (1 topk x 2 variants)

Combo3: MI+RF -> MLP (Precision Floor OFF, topk=40/50)
  - Precision floor: 0.0 (disabled)
  - AP-lambda: 0.07
  - RF trees: 700
  - TopK: [40, 50]
  - Experiments: C3_k40_varBoost, C3_k50_varBoost
  - Runs: 4 (2 topk x 2 variants)

For each experiment:
  - Variant N-1: Baseline (BCE + precision maintenance)
  - Variant N-2: Proposed (FN-Focal + AP-Head)

Train:    CICIDS-2017
Validate: hold-out from CICIDS (20%)
Test:     UNSW-NB15

Evaluation: Fixed threshold=0.5 (no calibration/optimization)

Outputs:
  - results_run.csv (configurable via --results_path, append mode)
  - runs/cm_{exp_id}_{combo}_{variant}.png (10 confusion matrix images)
"""

# Force matplotlib to use non-GUI backend before any other imports
import os
os.environ['MPLBACKEND'] = 'Agg'

import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import argparse
import numpy as np
import pandas as pd

# Fix matplotlib backend to avoid tkinter threading issues
# MUST be set before any matplotlib/seaborn import
import matplotlib
matplotlib.use('Agg', force=True)  # Force non-GUI backend
import matplotlib.pyplot as plt
# Disable interactive mode
plt.ioff()

# Optional feature mapping (UNSW -> CICIDS)
try:
    from feature_mapping import FEATURE_MAPPING
except Exception:
    FEATURE_MAPPING = {}

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except Exception:
    xgb = None

# >>> PATCH: gate warmup and L1 regularization constants
# >>> PATCH v2: Relaxed gate regularization and extended warmup
# >>> PATCH v3.5 final: Made configurable via CLI
WARMUP_EPOCHS = 5  # Default, can override via --warmup_epochs
GATE_L1_LAMBDA = 1e-6  # Default, can override via --gate_l1_lambda


# ----------------------------
# Args
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cicids_dir", type=str, default="CSV/MachineLearningCSV/MachineLearningCVE")
    ap.add_argument("--unsw_dir",   type=str, default="CSV_NB15/CSV Files/Training and Testing Sets")

    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--sample_size", type=int, default=0, help="Sample size for quick test (0=all)")

    # common train params
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)

    # XGB FS
    # >>> PATCH: Recommended sweep range: 80-120 for better OOD coverage
    ap.add_argument("--xgb_topk", type=int, default=60)

    # MI+RF FS
    # >>> PATCH: Recommended sweep range: 60-120 for better OOD coverage
    # >>> PATCH v2: Increased defaults for MLP recall improvement
    ap.add_argument("--mi_topk", type=int, default=50, help="MI+RF topk (try 100-120)")
    ap.add_argument("--rf_trees", type=int, default=700, help="RF trees (try 500 for MLP)")

    # Proposed method controls
    ap.add_argument("--focal_alpha", type=float, default=0.5)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--fn_lambda", type=float, default=1.0)
    ap.add_argument("--fn_gamma", type=float, default=2.0)
    # >>> PATCH v2: Reduced default ap_lambda for less recall suppression
    # >>> PATCH v3: Lowered ap_min_count to 10 for better rare pattern coverage
    ap.add_argument("--ap_lambda", type=float, default=0.03, help="AP-Head loss weight (0.03 for AFS, 0.12 for XGB)")
    ap.add_argument("--ap_min_count", type=int, default=10, help="Min count for pattern vocab (lower=more coverage)")

    # AP-Head control (DEFAULT: ON)
    # >>> PATCH v2.2: Updated help text to reflect combo3 (MLP) support
    ap.add_argument("--use_ap_head", type=lambda x: x.lower() != 'false', default=True, 
                    help="Use AP-Head for all combos (default: True)")

    # Hyperparameter sweep lists (for grid search)
    ap.add_argument("--ap_lambda_list", type=str, default="", help="Comma-separated ap_lambda values")
    ap.add_argument("--ap_min_count_list", type=str, default="", help="Comma-separated min_count values")
    ap.add_argument("--focal_alpha_list", type=str, default="", help="Comma-separated focal_alpha values")
    ap.add_argument("--focal_gamma_list", type=str, default="", help="Comma-separated focal_gamma values")

    # Reproducibility
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # >>> PATCH v3.5: Precision maintenance training controls (threshold=0.5 fixed)
    ap.add_argument("--prec_floor_train", type=float, default=0.0,
                    help="Training precision floor (0=disabled, recommend baseline_P * 0.95-0.98)")
    ap.add_argument("--prec_penalty_lambda", type=float, default=0.5,
                    help="Soft precision penalty strength")
    ap.add_argument("--neg_guard_lambda", type=float, default=0.1,
                    help="Negative overconfidence guard strength (post-warmup)")
    ap.add_argument("--neg_gamma", type=float, default=2.0,
                    help="Negative guard gamma")
    ap.add_argument("--hard_neg_topq", type=float, default=0.2,
                    help="Hard negative mining top-q fraction")
    ap.add_argument("--hard_neg_weight", type=float, default=1.5,
                    help="Hard negative sample weight multiplier")
    
    # >>> PATCH v3.5 experimental: Gate warmup and L1 lambda controls
    ap.add_argument("--warmup_epochs", type=int, default=5,
                    help="Gate warmup epochs (unfreeze after this)")
    ap.add_argument("--gate_l1_lambda", type=float, default=1e-6,
                    help="Gate L1 regularization lambda")
    
    # >>> PATCH v3.6.1: AFS ranking epochs control (clamped 8-10)
    ap.add_argument("--afs_rank_epochs", type=int, default=8,
                    help="AFS ranking epochs (clamped to 8-10)")
    
    # >>> PATCH v3.5 experimental: Top-k controls per combo
    ap.add_argument("--topk_c1", type=int, default=20, help="Top-k for Combo1 (AFS)")
    ap.add_argument("--topk_c2", type=int, default=20, help="Top-k for Combo2 (XGB)")
    ap.add_argument("--topk_c3", type=int, default=40, help="Top-k for Combo3 (MI+RF)")
    
    # >>> PATCH v3.5 experimental: Experiment ID for tracking
    ap.add_argument("--exp_id", type=str, default="", help="Experiment identifier (e.g., C1_k20_1-1)")
    
    # >>> NEW: Results output path
    ap.add_argument("--results_path", type=str, default="results_run.csv",
                    help="Path to results CSV file (append mode)")

    args = ap.parse_args()

    # >>> PATCH v3.5 experimental: Override global constants from args
    global WARMUP_EPOCHS, GATE_L1_LAMBDA
    WARMUP_EPOCHS = args.warmup_epochs
    GATE_L1_LAMBDA = args.gate_l1_lambda

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # device
    if torch.cuda.is_available():
        args.device = "cuda"
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        args.device = "cpu"
        print("GPU not available. Using CPU.")

    return args


# ----------------------------
# Data IO and preprocessing
# ----------------------------
# >>> PATCH v3: Pattern column priority fix for UNSW unknown issue
LABEL_CANDS = [
    "Label", "label", "Attack", "attack", "Class", "class", "Attack type", "attack_cat"
]
PATTERN_CANDS = ["Attack type", "attack_cat", "Attack", "attack", "Class", "class"]

def list_csvs(root: Path) -> List[Path]:
    files = [p for p in root.rglob("*.csv") if p.is_file()]
    print(f"Found {len(files)} CSV files in {root}")
    return files

def read_one_csv(path: Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Return X, y(binary), ypat(pattern string).
    >>> PATCH v3: Pattern column selection priority fixed to avoid "1" pattern bug.
    >>> PATCH v3.1: Enhanced encoding fallback (utf-8 -> latin-1 -> cp1252 -> iso-8859-1)
    """
    df = None
    for enc in ("utf-8", "latin-1", "cp1252", "iso-8859-1"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc, on_bad_lines="skip")
            if enc != "utf-8":
                print(f"[info] {path.name}: read with encoding={enc}")
            break
        except Exception:
            df = None
    if df is None or df.empty:
        raise ValueError(f"failed to read {path}")

    df.columns = df.columns.str.strip()

    # Find label column
    lab = next((c for c in LABEL_CANDS if c in df.columns), None)
    if lab is None:
        raise ValueError(f"{path.name}: label column not found")

    raw_lab = df[lab].astype(str).str.strip().str.lower()

    def to_bin(s: str) -> int:
        if s in ("0", "normal", "benign", "benign traffic", "", "nan"):
            return 0
        return 1

    # >>> PATCH v3: Find pattern column separately (prioritize attack category columns)
    pat_col = next((c for c in PATTERN_CANDS if c in df.columns), None)
    
    if pat_col is not None:
        raw_pat = df[pat_col].astype(str).str.strip().str.lower()
        def to_pat(s: str) -> str:
            if s in ("", "nan", "none", "normal", "benign", "benign traffic"):
                return "none"
            return s
    else:
        # Fallback: use label column (AP-Head effectiveness reduced)
        print(f"[WARN] {path.name}: No dedicated pattern column found, using label as fallback")
        raw_pat = raw_lab
        def to_pat(s: str) -> str:
            if s in ("0", "normal", "benign", "benign traffic", "", "nan"):
                return "none"
            return s

    y = raw_lab.apply(to_bin).astype(int)
    ypat = raw_pat.apply(to_pat)

    # >>> PATCH v3.2: Remove BOTH label and pattern columns from X (data leakage prevention)
    drop_cols = [lab]
    if pat_col is not None and pat_col != lab:
        drop_cols.append(pat_col)
    
    X = df.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan)
    if len(drop_cols) > 1:
        print(f"[info] {path.name}: dropped {drop_cols} from features")
    return X, y, ypat

def load_all(root_dir: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    root = Path(root_dir)
    if not root.exists():
        raise SystemExit(f"Not found: {root_dir}")
    files = list_csvs(root)
    if not files:
        raise SystemExit(f"No CSV files in {root_dir}")

    X_parts, y_parts, p_parts = [], [], []
    for f in files:
        try:
            X, y, p = read_one_csv(f)
            X_parts.append(X); y_parts.append(y); p_parts.append(p)
        except Exception as e:
            print(f"[warn] skip {f.name}: {e}")

    X_raw = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    ypat = pd.concat(p_parts, ignore_index=True)
    return X_raw, y, ypat


def map_unsw_to_cicids(X_unsw: pd.DataFrame) -> pd.DataFrame:
    """
    Map UNSW-NB15 column names to CICIDS-2017 column names using semantic mapping.
    """
    if not FEATURE_MAPPING:
        print("[warn] Feature mapping not available, returning original dataframe")
        return X_unsw.copy()
    X_mapped = pd.DataFrame(index=X_unsw.index)
    mapped = 0
    for ucol in X_unsw.columns:
        ccol = FEATURE_MAPPING.get(ucol)
        if ccol is not None:
            X_mapped[ccol] = X_unsw[ucol]
            mapped += 1
    print(f"[info] Mapped {mapped}/{len(X_unsw.columns)} UNSW columns to CICIDS feature space")
    return X_mapped


class CatFactorize:
    """Fit on given columns; unseen -> -1."""
    def __init__(self, cols: List[str]):
        self.cols = list(cols)
        self.maps: Dict[str, Dict[str, int]] = {}

    def fit(self, X_df: pd.DataFrame):
        for c in self.cols:
            vals = X_df[c].astype(str).fillna("<NA>")
            cats = pd.Index(vals.unique())
            self.maps[c] = {k: i for i, k in enumerate(cats)}
        return self

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        out = np.zeros((len(X_df), len(self.cols)), dtype=np.float32)
        for j, c in enumerate(self.cols):
            vals = X_df[c].astype(str).fillna("<NA>")
            mp = self.maps.get(c, {})
            ids = vals.map(mp).fillna(-1).astype("float32").values
            out[:, j] = ids
        return out


class Preprocessor:
    """
    Fit on CICIDS: numeric median+scale; categorical factorize to ints.
    Transform for both datasets with same mappings/scaler.
    SAFE: if a required column is missing at transform-time, create it.
    Uses RobustScaler for cross-dataset robustness.
    """
    def __init__(self, use_robust=True):
        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.num_imputer = SimpleImputer(strategy="median")
        self.num_scaler = RobustScaler() if use_robust else StandardScaler()
        self.cat_fac: Optional[CatFactorize] = None
        self.use_robust = use_robust

    def fit(self, X_df: pd.DataFrame):
        self.cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.num_cols = [c for c in X_df.columns if c not in self.cat_cols]
        if self.num_cols:
            num_df = X_df[self.num_cols].apply(pd.to_numeric, errors="coerce")
            Xn = self.num_imputer.fit_transform(num_df)
            self.num_scaler.fit(Xn)
        Xc_fit = pd.DataFrame(index=X_df.index)
        for c in self.cat_cols:
            Xc_fit[c] = X_df[c].astype(str).fillna("<NA>")
        self.cat_fac = CatFactorize(self.cat_cols).fit(Xc_fit)
        return self

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        parts = []
        if self.num_cols:
            num_df = pd.DataFrame(index=X_df.index)
            for c in self.num_cols:
                if c in X_df.columns:
                    num_df[c] = pd.to_numeric(X_df[c], errors="coerce")
                else:
                    num_df[c] = np.nan
            Xn = self.num_imputer.transform(num_df[self.num_cols])
            # Step 1: Pre-scaling clip to prevent extreme outliers
            Xn = np.clip(Xn, -1e5, 1e5)
            # Step 2: Apply RobustScaler
            Xn = self.num_scaler.transform(Xn).astype("float32")
            # Step 3: Post-scaling clip to prevent LSTM saturation (CRITICAL for stability)
            Xn = np.clip(Xn, -5.0, 5.0)
            parts.append(Xn)
        if self.cat_cols:
            cat_df = pd.DataFrame(index=X_df.index)
            for c in self.cat_cols:
                if c in X_df.columns:
                    cat_df[c] = X_df[c].astype(str).fillna("<NA>")
                else:
                    cat_df[c] = "<NA>"
            Xc = self.cat_fac.transform(cat_df)
            parts.append(Xc)
        Xt = np.concatenate(parts, axis=1).astype("float32") if parts else np.zeros((len(X_df), 0), dtype=np.float32)
        return np.nan_to_num(Xt, copy=False)


# ----------------------------
# Feature selection helpers
# ----------------------------
# >>> PATCH v3.3: AFS gate-based ranking for Combo1
# >>> PATCH v4: Ensemble-based AFS ranking for stability (5 runs averaged)
def fs_afs_ranking(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_va: np.ndarray, y_va: np.ndarray,
                   hidden: int, device: str,
                   epochs: int = None, lr: float = 1e-3,
                   batch_size: int = 256) -> np.ndarray:
    """
    Train AFSLSTM on full features briefly with ENSEMBLE (5 runs), 
    then rank features by averaged gate activation values (descending order).
    >>> PATCH v3.5: Default epochs ensures gate unfreezes (WARMUP_EPOCHS+1 minimum)
    >>> PATCH v4: Ensemble averaging (5 runs) to reduce variance in feature selection
    """
    if epochs is None:
        epochs = max(6, WARMUP_EPOCHS + 1)
    
    tr = TabDataset(X_tr, y_tr); va = TabDataset(X_va, y_va)
    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch_size)

    # Ensemble: accumulate gate values across 5 runs
    n_ensemble = 5
    accumulated_gate = np.zeros(X_tr.shape[1], dtype=np.float32)
    
    print(f"[afs-ranking] Running ensemble feature selection ({n_ensemble} runs)...")
    
    for run_idx in range(n_ensemble):
        # Re-initialize model for each run
        model = AFSLSTM(in_feats=X_tr.shape[1], hidden=hidden).to(device)

        pos_weight = (len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-6)
        crit = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        # Gate warmup freeze (same as existing training)
        _freeze_gate_lin(model)
        for ep in range(1, epochs + 1):
            if ep == WARMUP_EPOCHS + 1:
                _unfreeze_gate_lin(model)
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                logit = model(xb)
                loss = crit(logit, yb)

                # Gate L1 regularization (same as existing)
                gate_l1 = _get_gate_l1(model)
                if gate_l1 > 0:
                    loss = loss + gate_l1

                opt.zero_grad(); loss.backward(); opt.step()

        # Compute mean gate activations over training set for this run
        model.eval()
        sum_g = torch.zeros(X_tr.shape[1], device=device)
        cnt = 0
        with torch.no_grad():
            for xb, _ in dl_tr:
                xb = xb.to(device)
                _ = model(xb)  # Forward to populate last_gate
                if model.last_gate is not None:
                    sum_g += model.last_gate.sum(dim=0)
                    cnt += model.last_gate.size(0)
        
        mean_g = (sum_g / max(cnt, 1)).detach().cpu().numpy()  # (D,)
        accumulated_gate += mean_g
        print(f"  Run {run_idx+1}/{n_ensemble}: mean gate={mean_g.mean():.3f}, std={mean_g.std():.3f}")
    
    # Average across ensemble runs
    averaged_gate = accumulated_gate / n_ensemble
    
    # Higher gate = more important feature -> descending sort
    print(f"[afs-ranking] Final ensemble: mean gate={averaged_gate.mean():.3f}, std={averaged_gate.std():.3f}")
    return np.argsort(-averaged_gate)

def fs_mi_rf_train(X_tr: np.ndarray, y_tr: np.ndarray, topk: int = 40, rf_trees: int = 300) -> np.ndarray:
    mi = mutual_info_classif(X_tr, y_tr, discrete_features=False)
    order = np.argsort(mi)[::-1]
    keep = order[: max(topk * 3, topk)]
    X_mi = X_tr[:, keep]
    rf = RandomForestClassifier(n_estimators=rf_trees, class_weight="balanced",
                                n_jobs=-1, random_state=42).fit(X_mi, y_tr)
    imp = rf.feature_importances_
    loc = np.argsort(imp)[::-1][:topk]
    idx = np.sort(keep[loc])
    # >>> PATCH: log actual selected topk
    print(f"[mi-rf-fs] selected topk={len(idx)} (requested={topk})")
    return idx

def fs_xgb_importance_try_gpu(X_tr: np.ndarray, y_tr: np.ndarray, topk: int = 30) -> Tuple[np.ndarray, str]:
    if xgb is None:
        raise ImportError("xgboost not installed. pip install xgboost")
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    used = "gpu_hist"
    params = dict(max_depth=5, eta=0.1, subsample=0.8, colsample_bytree=0.8,
                  objective="binary:logistic", eval_metric="auc",
                  tree_method="gpu_hist", predictor="gpu_predictor")
    try:
        model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
    except Exception:
        used = "hist"
        params = dict(max_depth=5, eta=0.1, subsample=0.8, colsample_bytree=0.8,
                      objective="binary:logistic", eval_metric="auc",
                      tree_method="hist")
        model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
        print("[info] XGBoost: GPU not available, using CPU hist.")
    score = model.get_score(importance_type="gain")
    ranked = sorted([(int(k[1:]), v) for k, v in score.items()], key=lambda t: t[1], reverse=True)
    idx = [i for i, _ in ranked[:min(topk, len(ranked))]] if ranked else list(range(X_tr.shape[1]))
    # >>> PATCH: log actual selected topk
    print(f"[xgb-fs] selected topk={len(idx)} (requested={topk})")
    return np.array(sorted(idx), dtype=int), used


# ----------------------------
# Pattern vocab (AP-Head)
# ----------------------------
def build_pattern_vocab(ypat_train: pd.Series, min_count: int = 10) -> Dict[str, int]:
    s = ypat_train[ypat_train != "none"].astype(str)
    vc = s.value_counts()
    keep = sorted(vc[vc >= min_count].index.tolist())
    return {p: i for i, p in enumerate(keep)}

def to_pattern_ids(ypat: pd.Series, vocab: Dict[str, int]) -> np.ndarray:
    ids = []
    for v in ypat.astype(str):
        ids.append(-1 if v == "none" else vocab.get(v, -1))
    return np.array(ids, dtype=np.int64)

# >>> PATCH: helper to map -1 (unknown) to explicit class index
def map_unknown_numpy(ids: np.ndarray, n_patterns: int) -> np.ndarray:
    """Map -1 (unknown pattern) to n_patterns index."""
    out = ids.copy()
    out[out == -1] = n_patterns
    return out


# ----------------------------
# Torch datasets/models
# ----------------------------
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, ypat: Optional[np.ndarray] = None):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.ypat = ypat
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i]; y = self.y[i]
        return (x, y) if self.ypat is None else (x, y, self.ypat[i])

# --- Feature Gate with mode selection ---
class FeatureGate(nn.Module):
    """
    Adaptive feature gate with multiple modes:
    - sigmoid: gate = sigmoid(LN -> Linear)
    - softplus_norm: gate = normalized softplus, clipped to gate_max
    """
    def __init__(self, in_feats: int, mode: str = "sigmoid", gate_max: float = 2.0):
        super().__init__()
        self.mode = mode
        self.gate_max = gate_max
        self.ln = nn.LayerNorm(in_feats)
        self.lin = nn.Linear(in_feats, in_feats, bias=True)

    def forward(self, x):
        # x: (B, D)
        h = self.lin(self.ln(x))
        
        if self.mode == "sigmoid":
            g = torch.sigmoid(h)
        elif self.mode == "softplus_norm":
            g_raw = torch.nn.functional.softplus(h, beta=1.0)
            g = g_raw / (g_raw.mean(dim=1, keepdim=True) + 1e-6)
            g = torch.clamp(g, 0, self.gate_max)
        else:
            g = torch.sigmoid(h)  # fallback
        
        return x * g, g

# Simple LSTM block used in several places
class SimpleLSTM(nn.Module):
    def __init__(self, in_feats: int, hidden: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(in_feats, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(2*hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)

# --- NEW: AFSLSTM (baseline for Chapter 1) ---
class AFSLSTM(nn.Module):
    def __init__(self, in_feats: int, hidden: int = 128, gate_mode: str = "sigmoid", gate_max: float = 2.0):
        super().__init__()
        self.gate = FeatureGate(in_feats, mode=gate_mode, gate_max=gate_max)
        self.lstm = nn.LSTM(in_feats, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(2*hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.last_gate = None  # cache for logging/regularization
    
    def forward(self, x):
        xg, g = self.gate(x)
        self.last_gate = g  # cache gate values
        out, _ = self.lstm(xg.unsqueeze(1))
        feat = out[:, -1, :]
        return self.fc(feat).squeeze(1)

# --- NEW: AFSLSTM with AP head (proposed for Chapter 1) ---
class AFSLSTMWithAPHead(nn.Module):
    def __init__(self, in_feats: int, hidden: int, n_patterns: int, gate_mode: str = "sigmoid", gate_max: float = 2.0):
        super().__init__()
        self.gate = FeatureGate(in_feats, mode=gate_mode, gate_max=gate_max)
        self.lstm = nn.LSTM(in_feats, hidden, batch_first=True, bidirectional=True)
        shared_dim = 2*hidden
        self.main_head = nn.Linear(shared_dim, 1)
        self.ap_head = nn.Linear(shared_dim, n_patterns)
        self.last_gate = None  # cache for logging/regularization
    
    def forward(self, x):
        xg, g = self.gate(x)
        self.last_gate = g  # cache gate values
        out, _ = self.lstm(xg.unsqueeze(1))
        feat = out[:, -1, :]
        return self.main_head(feat).squeeze(1), self.ap_head(feat)

# (for combo2 proposed head reuse)
class LSTMWithAPHead(nn.Module):
    def __init__(self, in_feats: int, hidden: int, n_patterns: int):
        super().__init__()
        self.trunk_lstm = nn.LSTM(in_feats, hidden, batch_first=True, bidirectional=True)
        shared_dim = 2*hidden
        self.main_head = nn.Linear(shared_dim, 1)
        self.ap_head = nn.Linear(shared_dim, n_patterns)
    def forward(self, x):
        out, _ = self.trunk_lstm(x.unsqueeze(1))
        feat = out[:, -1, :]
        return self.main_head(feat).squeeze(1), self.ap_head(feat)

# Simple MLP for combo3 baseline
class TorchMLP(nn.Module):
    def __init__(self, in_feats: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# >>> PATCH v2.2: MLP with AP-Head for combo3
class TorchMLPWithAPHead(nn.Module):
    """MLP with Attack Pattern auxiliary head for combo3."""
    def __init__(self, in_feats: int, n_patterns: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_feats, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU()
        )
        self.main_head = nn.Linear(128, 1)
        self.ap_head = nn.Linear(128, n_patterns)
    
    def forward(self, x):
        feat = self.trunk(x)
        return self.main_head(feat).squeeze(1), self.ap_head(feat)

# ----------------------------
# Losses
# ----------------------------
class FNFocalLoss(nn.Module):
    """
    BCEWithLogits + focal term + extra FN term on positives + optional neg guard.
    >>> PATCH v3.5: Added neg_guard_lambda for FP suppression
    >>> PATCH v3.5 final: Support reduction='none' for hard negative mining
    """
    def __init__(self, alpha=0.5, gamma=2.0, fn_lambda=1.0, fn_gamma=2.0, 
                 neg_guard_lambda=0.0, neg_gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.fn_lambda = fn_lambda
        self.fn_gamma = fn_gamma
        self.neg_guard_lambda = neg_guard_lambda
        self.neg_gamma = neg_gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    
    def forward(self, logits, targets):
        targets = targets.float()
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits).clamp(1e-7, 1-1e-7)
        pt = torch.where(targets > 0.5, p, 1 - p)
        focal = (1 - pt) ** self.gamma
        alpha_w = torch.where(targets > 0.5, self.alpha, 1.0 - self.alpha)
        base = (alpha_w * focal * bce)
        fn_term = ((1 - p) ** self.fn_gamma) * (targets > 0.5)
        
        # Element-wise loss (before reduction)
        loss_vec = base + self.fn_lambda * fn_term
        
        # >>> PATCH v3.5: Negative overconfidence guard (applied to mean)
        if self.neg_guard_lambda > 0:
            neg_mask = (targets < 0.5)
            if neg_mask.sum() > 0:
                neg_overconf = (p[neg_mask] ** self.neg_gamma).mean()
                if self.reduction == 'none':
                    # For 'none', we add guard as a per-sample constant
                    loss_vec = loss_vec + (self.neg_guard_lambda * neg_overconf) / len(loss_vec)
                else:
                    # For 'mean', add to final scalar
                    return loss_vec.mean() + self.neg_guard_lambda * neg_overconf
        
        # Apply reduction
        if self.reduction == 'none':
            return loss_vec
        elif self.reduction == 'mean':
            return loss_vec.mean()
        else:
            return loss_vec.sum()

# >>> PATCH: CE for AP-Head now includes unknown class (no ignore)
ce_ap = nn.CrossEntropyLoss()


# ----------------------------
# >>> PATCH v3.5: Precision maintenance helpers
# ----------------------------
def soft_precision_penalty(logits, y_true, floor, lam=0.5, eps=1e-7):
    """
    Compute soft precision penalty to maintain precision >= floor.
    Uses differentiable soft TP/FP approximation.
    """
    if floor <= 0.0:
        return 0.0
    
    probs = torch.sigmoid(logits).squeeze(-1)
    soft_tp = (probs * y_true).sum()
    soft_fp = (probs * (1 - y_true)).sum()
    soft_prec = soft_tp / (soft_tp + soft_fp + eps)
    
    # Penalty when precision drops below floor
    shortfall = torch.clamp(floor - soft_prec, min=0.0)
    return lam * (shortfall ** 2)


def hard_negative_weights(logits, y_true, topq=0.2, weight=1.5):
    """
    Compute sample weights for hard negative mining.
    Returns weights with topq hardest negatives weighted by 'weight'.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits).squeeze(-1)
        neg_mask = (y_true == 0)
        
        if neg_mask.sum() == 0:
            return torch.ones_like(y_true, dtype=torch.float32)
        
        # Get negative probabilities
        neg_probs = probs[neg_mask]
        k = max(1, int(len(neg_probs) * topq))
        topk_threshold = torch.topk(neg_probs, k).values[-1]
        
        # Create weights
        weights = torch.ones_like(y_true, dtype=torch.float32)
        hard_neg_mask = neg_mask & (probs >= topk_threshold)
        weights[hard_neg_mask] = weight
        
        return weights


# ----------------------------
# Train/Eval helpers
# ----------------------------
def to_device(batch, device, has_pattern=False):
    if has_pattern:
        xb, yb, pb = batch
        return xb.to(device), yb.to(device), pb.to(device)
    xb, yb = batch
    return xb.to(device), yb.to(device), None

# >>> PATCH: gate control helpers for warmup
def _freeze_gate_lin(model):
    """Freeze FeatureGate linear layer parameters."""
    if hasattr(model, "gate") and hasattr(model.gate, "lin"):
        for p in model.gate.lin.parameters():
            p.requires_grad = False

def _unfreeze_gate_lin(model):
    """Unfreeze FeatureGate linear layer parameters."""
    if hasattr(model, "gate") and hasattr(model.gate, "lin"):
        for p in model.gate.lin.parameters():
            p.requires_grad = True

def _get_gate_l1(model, y_batch=None, neg_boost=2.0):
    """
    Compute L1 regularization term for FeatureGate.
    >>> PATCH v3.5: Optional negative-only boosted L1 for FP suppression
    """
    if not hasattr(model, "gate") or not hasattr(model.gate, "lin"):
        return 0.0
    
    g = torch.sigmoid(model.gate.lin.weight)  # (1, D)
    base_l1 = GATE_L1_LAMBDA * torch.mean(torch.abs(g - 1.0))
    
    # If y_batch provided, add boosted L1 for negative samples
    if y_batch is not None and neg_boost > 1.0:
        neg_mask = (y_batch < 0.5)
        if neg_mask.sum() > 0:
            # Boost L1 penalty for negatives only
            neg_l1 = GATE_L1_LAMBDA * neg_boost * torch.mean(torch.abs(g - 1.0))
            return base_l1 + neg_l1 * (neg_mask.float().mean())
    
    return base_l1

def train_baseline(model, dl_tr, dl_va, device="cuda", epochs=10, lr=1e-3, pos_weight=None, tag="baseline",
                   prec_floor=0.0, prec_penalty_lambda=0.5, hard_neg_topq=0.2, hard_neg_weight=1.5):
    """
    >>> PATCH v3.5: Added precision maintenance controls
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction='none') if pos_weight is not None else nn.BCEWithLogitsLoss(reduction='none')

    # >>> PATCH: freeze gate during warmup
    _freeze_gate_lin(model)

    best = {"auc": -1.0, "state": None}
    for ep in range(1, epochs+1):
        # >>> PATCH: unfreeze gate after warmup
        if ep == WARMUP_EPOCHS + 1:
            _unfreeze_gate_lin(model)
        
        model.train()
        ep_loss, ep_prec_proxy = [], []
        for batch in dl_tr:
            xb, yb, _ = to_device(batch, device, has_pattern=False)
            logits = model(xb)
            
            # Hard negative mining weights
            sample_weights = hard_negative_weights(logits, yb, topq=hard_neg_topq, weight=hard_neg_weight)
            
            # Weighted BCE loss
            bce_loss = crit(logits, yb)
            loss = (bce_loss * sample_weights).mean()
            
            # Soft precision penalty
            prec_penalty = soft_precision_penalty(logits, yb, floor=prec_floor, lam=prec_penalty_lambda)
            loss = loss + prec_penalty
            
            # Gate L1 regularization (neg-boosted)
            gate_l1 = _get_gate_l1(model, y_batch=yb, neg_boost=2.0)
            loss = loss + gate_l1
            
            opt.zero_grad(); loss.backward(); opt.step()
            
            # Track metrics
            ep_loss.append(loss.item())
            with torch.no_grad():
                probs = torch.sigmoid(logits).squeeze(-1)
                soft_tp = (probs * yb).sum()
                soft_fp = (probs * (1 - yb)).sum()
                soft_prec = (soft_tp / (soft_tp + soft_fp + 1e-7)).item()
                ep_prec_proxy.append(soft_prec)

        model.eval(); y_true, y_prob = [], []
        with torch.no_grad():
            for batch in dl_va:
                xb, yb, _ = to_device(batch, device, has_pattern=False)
                prob = torch.sigmoid(model(xb)).cpu().numpy()
                y_true.append(yb.cpu().numpy()); y_prob.append(prob)
        y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
        auc = roc_auc_score(y_true, y_prob)
        if auc > best["auc"]:
            best["auc"] = auc
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        avg_prec = np.mean(ep_prec_proxy) if ep_prec_proxy else 0.0
        print(f"[{tag}] ep{ep:02d} AUC={auc:.4f} loss={np.mean(ep_loss):.4f} soft_P={avg_prec:.4f}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model

def train_proposed(model, dl_tr, dl_va, device="cuda", epochs=10, lr=1e-3,
                   focal_cfg=None, ap_lambda=0.2, has_ap=True, tag="proposed", n_known=0,
                   prec_floor=0.0, prec_penalty_lambda=0.5, neg_guard_lambda=0.1, neg_gamma=2.0,
                   hard_neg_topq=0.2, hard_neg_weight=1.5):
    """
    >>> PATCH v3.5: Added precision maintenance controls + neg_guard schedule
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if focal_cfg is None:
        focal_cfg = dict(alpha=0.5, gamma=2.0, fn_lambda=1.0, fn_gamma=2.0)
    
    # Start with neg_guard=0, will activate after warmup
    focal_cfg_dynamic = {**focal_cfg, 'neg_guard_lambda': 0.0, 'neg_gamma': neg_gamma}
    crit = FNFocalLoss(**focal_cfg_dynamic)

    # >>> PATCH: freeze gate during warmup
    _freeze_gate_lin(model)

    best = {"auc": -1.0, "state": None}
    _first_batch_prec_penalty_logged = False
    for ep in range(1, epochs+1):
        # >>> PATCH: unfreeze gate after warmup
        if ep == WARMUP_EPOCHS + 1:
            _unfreeze_gate_lin(model)
            # >>> PATCH v3.5: Activate neg_guard after warmup
            crit.neg_guard_lambda = neg_guard_lambda
        
        model.train()
        ep_loss, ep_prec_proxy, ep_neg_guard = [], [], []
        for batch in dl_tr:
            xb, yb, pb = to_device(batch, device, has_pattern=has_ap)
            
            if has_ap:
                main_logit, ap_logits = model(xb)
                
                # Hard negative mining weights
                sample_weights = hard_negative_weights(main_logit, yb, topq=hard_neg_topq, weight=hard_neg_weight)
                
                # Main loss with FN-Focal (reduction='none' for weighting)
                crit.reduction = 'none'
                loss_vec = crit(main_logit, yb)
                loss_main = (loss_vec * sample_weights).mean()
                crit.reduction = 'mean'  # restore
                
                # Soft precision penalty
                prec_penalty = soft_precision_penalty(main_logit, yb, floor=prec_floor, lam=prec_penalty_lambda)
                loss = loss_main + prec_penalty
                
                # Log first batch precision penalty (once)
                if not _first_batch_prec_penalty_logged:
                    _first_batch_prec_penalty_logged = True
                    penalty_val = prec_penalty.item() if hasattr(prec_penalty, 'item') else float(prec_penalty)
                    print(f"  [C2] precision penalty active: {penalty_val > 0} (value={penalty_val:.6f})")
                
                # AP-Head loss only on positives
                mask_pos = (yb > 0.5)
                if mask_pos.any().item():
                    pb_cpu = pb.detach().cpu().numpy()
                    target_pos = torch.from_numpy(map_unknown_numpy(pb_cpu, n_known)).to(device).long()
                    loss_ap = ce_ap(ap_logits[mask_pos], target_pos[mask_pos])
                    loss = loss + ap_lambda * loss_ap
            else:
                main_logit = model(xb)
                
                # Hard negative mining weights
                sample_weights = hard_negative_weights(main_logit, yb, topq=hard_neg_topq, weight=hard_neg_weight)
                
                # Main loss with FN-Focal (reduction='none' for weighting)
                crit.reduction = 'none'
                loss_vec = crit(main_logit, yb)
                loss = (loss_vec * sample_weights).mean()
                crit.reduction = 'mean'  # restore
                
                # Soft precision penalty
                prec_penalty = soft_precision_penalty(main_logit, yb, floor=prec_floor, lam=prec_penalty_lambda)
                loss = loss + prec_penalty
            
            # Gate L1 regularization (neg-boosted)
            gate_l1 = _get_gate_l1(model, y_batch=yb, neg_boost=2.0)
            loss = loss + gate_l1
            
            opt.zero_grad(); loss.backward(); opt.step()
            
            # Track metrics
            ep_loss.append(loss.item())
            with torch.no_grad():
                probs = torch.sigmoid(main_logit).squeeze(-1)
                soft_tp = (probs * yb).sum()
                soft_fp = (probs * (1 - yb)).sum()
                soft_prec = (soft_tp / (soft_tp + soft_fp + 1e-7)).item()
                ep_prec_proxy.append(soft_prec)
                
                # Track neg_guard contribution
                neg_mask = (yb < 0.5)
                if neg_mask.sum() > 0:
                    neg_conf = (probs[neg_mask] ** neg_gamma).mean().item()
                    ep_neg_guard.append(neg_conf)

        model.eval(); y_true, y_prob = [], []
        with torch.no_grad():
            for batch in dl_va:
                xb, yb, pb = to_device(batch, device, has_pattern=has_ap)
                mlogit = model(xb)[0] if has_ap else model(xb)
                prob = torch.sigmoid(mlogit).cpu().numpy()
                y_true.append(yb.cpu().numpy()); y_prob.append(prob)
        y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
        auc = roc_auc_score(y_true, y_prob)
        if auc > best["auc"]:
            best["auc"] = auc
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        avg_prec = np.mean(ep_prec_proxy) if ep_prec_proxy else 0.0
        avg_neg = np.mean(ep_neg_guard) if ep_neg_guard else 0.0
        print(f"[{tag}] ep{ep:02d} AUC={auc:.4f} loss={np.mean(ep_loss):.4f} soft_P={avg_prec:.4f} neg_guard={avg_neg:.4f}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model

def compute_recall_at_precision(y_true: np.ndarray, y_prob: np.ndarray, precision_floor: float = 0.90) -> float:
    """Find maximum recall where precision >= precision_floor."""
    thresholds = np.linspace(0, 1, 101)
    best_recall = 0.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        if prec >= precision_floor:
            rec = recall_score(y_true, y_pred, zero_division=0)
            best_recall = max(best_recall, rec)
    return float(best_recall)

def compute_precision_at_recall(y_true: np.ndarray, y_prob: np.ndarray, recall_floor: float = 0.95) -> float:
    """Find maximum precision where recall >= recall_floor."""
    thresholds = np.linspace(0, 1, 101)
    best_precision = 0.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= recall_floor:
            prec = precision_score(y_true, y_pred, zero_division=0)
            best_precision = max(best_precision, prec)
    return float(best_precision)

def compute_fpr_at_recall(y_true: np.ndarray, y_prob: np.ndarray, recall_floor: float = 0.95) -> float:
    """Find minimum FPR where recall >= recall_floor."""
    thresholds = np.linspace(0, 1, 101)
    best_fpr = 1.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if rec >= recall_floor:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            fpr = fp / (fp + tn + 1e-12)
            best_fpr = min(best_fpr, fpr)
    return float(best_fpr)

def evaluate_arrays(y_true: np.ndarray, y_prob: np.ndarray, threshold: Optional[float] = None):
    threshold = 0.5 if threshold is None else float(threshold)
    y_pred = (y_prob >= threshold).astype(int)
    out = dict(
        threshold=threshold,
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        f1=float(f1_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred)),
        # New metrics
        recall_at_p90=compute_recall_at_precision(y_true, y_prob, 0.90),
        precision_at_r95=compute_precision_at_recall(y_true, y_prob, 0.95),
        fpr_at_r95=compute_fpr_at_recall(y_true, y_prob, 0.95),
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp + 1e-12
    out.update(dict(
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        tn_rate=tn/total, fp_rate=fp/total, fn_rate=fn/total, tp_rate=tp/total,
        _y_true=y_true, _y_pred=y_pred, _y_prob=y_prob
    ))
    return out

def eval_fixed_threshold(model, dl_test, device="cuda", has_ap=False):
    """
    >>> PATCH v3.5: Evaluate model on test set with FIXED threshold=0.5.
    No calibration, no threshold optimization.
    """
    model.eval()
    
    # Collect test probabilities
    yt, pt = [], []
    with torch.no_grad():
        for batch in dl_test:
            xb, yb, pb = to_device(batch, device, has_pattern=has_ap)
            logit = model(xb)[0] if has_ap else model(xb)
            prob = torch.sigmoid(logit).cpu().numpy()
            yt.append(yb.cpu().numpy())
            pt.append(prob)
    yt = np.concatenate(yt)
    pt = np.concatenate(pt)
    
    # Log gate statistics if available
    if getattr(model, "last_gate", None) is not None:
        g = model.last_gate.detach().cpu().numpy()
        print(f"[Gate] mean={g.mean():.3f}, std={g.std():.3f}, active(>0.5)={(g>0.5).mean():.3f}")
    
    # Evaluate with fixed threshold=0.5
    result = evaluate_arrays(yt, pt, threshold=0.5)
    # Store raw data for threshold search
    result["_y_true"] = yt
    result["_y_prob"] = pt
    return result


def find_threshold_for_precision(y_true: np.ndarray, y_prob: np.ndarray, precision_floor: float = 0.95):
    """Find threshold that achieves Precision >= precision_floor with maximum Recall on validation set."""
    thresholds = np.linspace(0, 1, 1001)
    best_thr, best_rec = 0.5, 0.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if y_pred.sum() == 0:
            continue
        prec = precision_score(y_true, y_pred, zero_division=0)
        if prec >= precision_floor:
            rec = recall_score(y_true, y_pred, zero_division=0)
            if rec > best_rec:
                best_rec, best_thr = rec, thr
    return best_thr, best_rec


def evaluate_with_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    """Evaluate with given threshold (standard metrics + confusion matrix)."""
    return evaluate_arrays(y_true, y_prob, threshold=float(thr))


# ----------------------------
# Debug helpers
# ----------------------------
def save_confusion_matrix(exp_id, combo, variant, metrics_dict, output_dir="runs"):
    """Save confusion matrix as PNG image with heatmap visualization."""
    import os
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"cm_{exp_id}_{combo}_{variant}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Extract confusion matrix values
    tn = metrics_dict.get('tn', 0)
    fp = metrics_dict.get('fp', 0)
    fn = metrics_dict.get('fn', 0)
    tp = metrics_dict.get('tp', 0)
    
    # Create confusion matrix array [[TN, FP], [FN, TP]]
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Try seaborn first, fallback to matplotlib
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    cbar_kws={'label': 'Count'}, ax=ax)
    except ImportError:
        # Matplotlib-only fallback
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
        # Annotate cells
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i, j]:.0f}', ha='center', va='center', fontsize=14)
        plt.colorbar(im, ax=ax, label='Count')
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    # Two-line title to prevent overlap
    title_line1 = f'Confusion Matrix: {exp_id} - {combo} - {variant}'
    title_line2 = f'P={metrics_dict.get("precision", 0.0):.4f} R={metrics_dict.get("recall", 0.0):.4f} F1={metrics_dict.get("f1", 0.0):.4f}'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    
    # Clean up to prevent memory leaks and threading issues
    plt.close(fig)
    plt.close('all')
    del fig, ax
    
    print(f"[CM] Saved to {filepath}")


def _debug_label_stats(y, name):
    y = np.asarray(y)
    pos = int((y == 1).sum()); neg = int((y == 0).sum()); tot = len(y)
    print(f"[debug] {name} label counts: pos={pos} neg={neg} total={tot} (pos_rate={pos/(tot+1e-12):.3f})")

def _debug_feature_variance(X, name):
    stds = X.std(axis=0)
    print(f"[debug] {name} feature std: mean={float(stds.mean()):.6f}, min={float(stds.min()):.6f}, max={float(stds.max()):.6f}")
    if float(stds.max()) < 1e-6:
        print(f"[debug][WARNING] {name} appears almost constant across all features. Check column alignment!")

# >>> PATCH: AP pattern coverage logging
def _log_ap_coverage(name, ypat_test, n_known):
    """Log known/unknown pattern distribution in test set."""
    if ypat_test is None:
        return
    ypat_arr = np.asarray(ypat_test)
    unk_ratio = float((ypat_arr == -1).mean())
    print(f"[AP][{name}] test pattern coverage: known={1-unk_ratio:.3f}, unknown={unk_ratio:.3f}")

# >>> PATCH: Known/unknown split recall logging
def _log_split_recall(name, y_true, y_pred, ypat_test):
    """Log recall separately for known vs unknown patterns."""
    if ypat_test is None:
        return
    ypat_arr = np.asarray(ypat_test)
    ymask_known = (ypat_arr != -1)
    ymask_unknown = (ypat_arr == -1)
    
    if ymask_known.any() and ymask_unknown.any():
        rec_known = recall_score(y_true[ymask_known], y_pred[ymask_known], zero_division=0)
        rec_unknown = recall_score(y_true[ymask_unknown], y_pred[ymask_unknown], zero_division=0)
        print(f"[AP][{name}] recall known={rec_known:.3f} unknown={rec_unknown:.3f}")

# >>> PATCH v2.1: AP-lambda resolver for combo-specific defaults
# >>> PATCH v2.2: Added mi_rf_mlp case
def _resolve_ap_lambda(ap_lambda_cli: float, combo_name: str) -> float:
    """
    Resolve AP-lambda with combo-specific defaults if not explicitly set.
    - afs_lstm: 0.03 (default)
    - xgbfs_lstm: 0.12 (recommended for XGB-FS)
    - mi_rf_mlp: 0.06 (recommended for MLP)
    """
    if ap_lambda_cli != 0.03:  # User explicitly changed from new default
        return ap_lambda_cli
    # Apply combo-specific override
    if combo_name == "xgbfs_lstm":
        print(f"[AP-Lambda] Using combo-specific default: 0.12 for {combo_name}")
        return 0.12
    if combo_name == "mi_rf_mlp":
        print(f"[AP-Lambda] Using combo-specific default: 0.07 for {combo_name}")
        return 0.07
    return ap_lambda_cli


# ----------------------------
# Combos
# ----------------------------
# 1) AFS -> LSTM (baseline / proposed)
def run_combo1_baseline(Xtr, ytr, Xva, yva, Xte, yte, device, epochs, lr, batch, hidden,
                        prec_floor_train=0.0, prec_penalty_lambda=0.5, hard_neg_topq=0.2, hard_neg_weight=1.5):
    tr = TabDataset(Xtr, ytr); va = TabDataset(Xva, yva); te = TabDataset(Xte, yte)
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch); dl_te = DataLoader(te, batch_size=batch)
    model = AFSLSTM(in_feats=Xtr.shape[1], hidden=hidden)
    pos_weight = (len(ytr) - ytr.sum()) / (ytr.sum() + 1e-6)
    model = train_baseline(model, dl_tr, dl_va, device=device, epochs=epochs, lr=lr, pos_weight=pos_weight, tag="baseline",
                           prec_floor=prec_floor_train, prec_penalty_lambda=prec_penalty_lambda,
                           hard_neg_topq=hard_neg_topq, hard_neg_weight=hard_neg_weight)
    # Collect predictions without fixed threshold
    test_result = eval_fixed_threshold(model, dl_te, device=device, has_ap=False)
    val_result = eval_fixed_threshold(model, dl_va, device=device, has_ap=False)
    test_result["_val_precision"] = val_result["precision"]  # Store for Combo2 precision maintenance
    
    # P>=0.75 threshold evaluation
    thr_p75, _ = find_threshold_for_precision(val_result["_y_true"], val_result["_y_prob"], precision_floor=0.75)
    result = evaluate_with_threshold(test_result["_y_true"], test_result["_y_prob"], thr_p75)
    result["_val_precision"] = test_result["_val_precision"]
    result["_y_true"] = test_result["_y_true"]
    result["_y_prob"] = test_result["_y_prob"]
    
    return result


def run_combo1_proposed(Xtr, ytr, Xva, yva, Xte, yte, ypat_tr, ypat_va, ypat_te,
                        device, epochs, lr, batch, hidden, vocab, focal_cfg, ap_lambda, use_ap_head=True,
                        prec_floor_train=0.0, prec_penalty_lambda=0.5, neg_guard_lambda=0.1, neg_gamma=2.0,
                        hard_neg_topq=0.2, hard_neg_weight=1.5):
    # >>> PATCH: n_known from vocab, n_total includes unknown class
    n_known = len(vocab)
    n_total = n_known + 1  # +1 for unknown class
    has_ap = use_ap_head and n_known > 0
    
    if has_ap:
        # >>> PATCH: log AP coverage
        _log_ap_coverage("1_AFS_LSTM+AP", ypat_te, n_known)
        
        tr = TabDataset(Xtr, ytr, ypat_tr); va = TabDataset(Xva, yva, ypat_va); te = TabDataset(Xte, yte, ypat_te)
        dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
        dl_va = DataLoader(va, batch_size=batch); dl_te = DataLoader(te, batch_size=batch)
        # >>> PATCH: use n_total (includes unknown)
        model = AFSLSTMWithAPHead(in_feats=Xtr.shape[1], hidden=hidden, n_patterns=n_total)
        model = train_proposed(model, dl_tr, dl_va, device=device, epochs=epochs, lr=lr,
                               focal_cfg=focal_cfg, ap_lambda=ap_lambda, has_ap=True, tag="proposed", n_known=n_known,
                               prec_floor=prec_floor_train, prec_penalty_lambda=prec_penalty_lambda,
                               neg_guard_lambda=neg_guard_lambda, neg_gamma=neg_gamma,
                               hard_neg_topq=hard_neg_topq, hard_neg_weight=hard_neg_weight)
        # Collect predictions without fixed threshold
        test_result = eval_fixed_threshold(model, dl_te, device=device, has_ap=True)
        val_result = eval_fixed_threshold(model, dl_va, device=device, has_ap=True)
        
        # P>=0.75 threshold evaluation
        thr_p75, _ = find_threshold_for_precision(val_result["_y_true"], val_result["_y_prob"], precision_floor=0.75)
        result = evaluate_with_threshold(test_result["_y_true"], test_result["_y_prob"], thr_p75)
        
        # >>> PATCH: log split recall
        y_pred_p75 = (test_result["_y_prob"] >= thr_p75).astype(int)
        _log_split_recall("1_AFS_LSTM+AP", test_result["_y_true"], y_pred_p75, ypat_te)
        
        return result
    else:
        # AP-Head OFF: use baseline AFSLSTM with FN Focal Loss only
        tr = TabDataset(Xtr, ytr); va = TabDataset(Xva, yva); te = TabDataset(Xte, yte)
        dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
        dl_va = DataLoader(va, batch_size=batch); dl_te = DataLoader(te, batch_size=batch)
        model = AFSLSTM(in_feats=Xtr.shape[1], hidden=hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = FNFocalLoss(**focal_cfg)
        
        # >>> PATCH: freeze gate during warmup
        _freeze_gate_lin(model)
        
        best = {"auc": -1.0, "state": None}
        for ep in range(1, epochs+1):
            # >>> PATCH: unfreeze gate after warmup
            if ep == WARMUP_EPOCHS + 1:
                _unfreeze_gate_lin(model)
            
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                loss = crit(model(xb), yb)
                
                # >>> PATCH: add gate L1 regularization
                gate_l1 = 0.0
                if getattr(model, "last_gate", None) is not None:
                    g = model.last_gate
                    gate_l1 = GATE_L1_LAMBDA * torch.mean(torch.abs(g - 1.0))
                loss = loss + gate_l1
                
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
            print(f"[proposed-noAP] val ep{ep:02d} AUC={auc:.4f}")
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        
        # Collect predictions without fixed threshold
        test_result = eval_fixed_threshold(model, dl_te, device=device, has_ap=False)
        val_result = eval_fixed_threshold(model, dl_va, device=device, has_ap=False)
        
        # P>=0.75 threshold evaluation
        thr_p75, _ = find_threshold_for_precision(val_result["_y_true"], val_result["_y_prob"], precision_floor=0.75)
        result = evaluate_with_threshold(test_result["_y_true"], test_result["_y_prob"], thr_p75)
        
        return result

# 2) XGB FS -> LSTM
def run_combo2_baseline(Xtr, ytr, Xva, yva, Xte, yte, device, epochs, lr, batch, hidden, topk,
                        prec_floor_train=0.0, prec_penalty_lambda=0.5, hard_neg_topq=0.2, hard_neg_weight=1.5):
    # >>> PATCH v3.2: XGBoost fallback - return NaN metrics if unavailable
    if xgb is None:
        print("[WARN] xgboost not installed; skipping Combo2 baseline")
        return {
            "threshold": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "f1": np.nan,
            "precision": np.nan, "recall": np.nan, "recall_at_p90": np.nan,
            "precision_at_r95": np.nan, "fpr_at_r95": np.nan,
            "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
            "tn_rate": np.nan, "fp_rate": np.nan, "fn_rate": np.nan, "tp_rate": np.nan,
            "_y_true": np.array([]), "_y_pred": np.array([]), "_y_prob": np.array([])
        }
    
    try:
        idx, used_tree = fs_xgb_importance_try_gpu(Xtr, ytr, topk=topk)
        print(f"[xgb-fs] tree_method used: {used_tree}")
    except Exception as e:
        print(f"[WARN] XGBoost FS failed: {e}; skipping Combo2 baseline")
        return {
            "threshold": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "f1": np.nan,
            "precision": np.nan, "recall": np.nan, "recall_at_p90": np.nan,
            "precision_at_r95": np.nan, "fpr_at_r95": np.nan,
            "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
            "tn_rate": np.nan, "fp_rate": np.nan, "fn_rate": np.nan, "tp_rate": np.nan,
            "_y_true": np.array([]), "_y_pred": np.array([]), "_y_prob": np.array([])
        }
    
    Xtr2 = Xtr[:, idx]; Xva2 = Xva[:, idx]; Xte2 = Xte[:, idx]
    tr = TabDataset(Xtr2, ytr); va = TabDataset(Xva2, yva); te = TabDataset(Xte2, yte)
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch); dl_te = DataLoader(te, batch_size=batch)
    model = SimpleLSTM(in_feats=Xtr2.shape[1], hidden=hidden)
    pos_weight = (len(ytr) - ytr.sum()) / (ytr.sum() + 1e-6)
    model = train_baseline(model, dl_tr, dl_va, device=device, epochs=epochs, lr=lr, pos_weight=pos_weight, tag="baseline",
                           prec_floor=prec_floor_train, prec_penalty_lambda=prec_penalty_lambda,
                           hard_neg_topq=hard_neg_topq, hard_neg_weight=hard_neg_weight)
    # Collect predictions without fixed threshold
    test_result = eval_fixed_threshold(model, dl_te, device=device, has_ap=False)
    # >>> CRITICAL FIX: Collect validation precision for C2 P_base calculation
    val_result = eval_fixed_threshold(model, dl_va, device=device, has_ap=False)
    
    # P>=0.75 threshold evaluation
    thr_p75, _ = find_threshold_for_precision(val_result["_y_true"], val_result["_y_prob"], precision_floor=0.75)
    result = evaluate_with_threshold(test_result["_y_true"], test_result["_y_prob"], thr_p75)
    result["_val_precision"] = val_result["precision"]
    
    return result

def run_combo2_proposed(Xtr, ytr, Xva, yva, Xte, yte, ypat_tr, ypat_va, ypat_te,
                        device, epochs, lr, batch, hidden, topk, vocab, focal_cfg, ap_lambda, use_ap_head=True,
                        neg_guard_lambda=0.1, neg_gamma=2.0, prec_floor_train=0.0, prec_penalty_lambda=0.5,
                        hard_neg_topq=0.2, hard_neg_weight=1.5, baseline_val_precision=None):
    # >>> PATCH v3.2: XGBoost fallback - return NaN metrics if unavailable
    if xgb is None:
        print("[WARN] xgboost not installed; skipping Combo2 proposed")
        return {
            "threshold": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "f1": np.nan,
            "precision": np.nan, "recall": np.nan, "recall_at_p90": np.nan,
            "precision_at_r95": np.nan, "fpr_at_r95": np.nan,
            "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
            "tn_rate": np.nan, "fp_rate": np.nan, "fn_rate": np.nan, "tp_rate": np.nan,
            "_y_true": np.array([]), "_y_pred": np.array([]), "_y_prob": np.array([])
        }
    
    try:
        idx, used_tree = fs_xgb_importance_try_gpu(Xtr, ytr, topk=topk)
        print(f"[xgb-fs] tree_method used: {used_tree}")
    except Exception as e:
        print(f"[WARN] XGBoost FS failed: {e}; skipping Combo2 proposed")
        return {
            "threshold": np.nan, "roc_auc": np.nan, "pr_auc": np.nan, "f1": np.nan,
            "precision": np.nan, "recall": np.nan, "recall_at_p90": np.nan,
            "precision_at_r95": np.nan, "fpr_at_r95": np.nan,
            "tn": np.nan, "fp": np.nan, "fn": np.nan, "tp": np.nan,
            "tn_rate": np.nan, "fp_rate": np.nan, "fn_rate": np.nan, "tp_rate": np.nan,
            "_y_true": np.array([]), "_y_pred": np.array([]), "_y_prob": np.array([])
        }
    
    Xtr2 = Xtr[:, idx]; Xva2 = Xva[:, idx]; Xte2 = Xte[:, idx]
    # >>> PATCH: n_known from vocab, n_total includes unknown class
    n_known = len(vocab)
    n_total = n_known + 1  # +1 for unknown class
    has_ap = use_ap_head and n_known > 0
    
    if has_ap:
        # >>> PATCH: log AP coverage
        _log_ap_coverage("2_XGBFS_LSTM+AP", ypat_te, n_known)
        
        tr = TabDataset(Xtr2, ytr, ypat_tr); va = TabDataset(Xva2, yva, ypat_va); te = TabDataset(Xte2, yte, ypat_te)
        dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
        dl_va = DataLoader(va, batch_size=batch); dl_te = DataLoader(te, batch_size=batch)
        # >>> PATCH: use n_total (includes unknown)
        model = LSTMWithAPHead(in_feats=Xtr2.shape[1], hidden=hidden, n_patterns=n_total)
        model = train_proposed(model, dl_tr, dl_va, device=device, epochs=epochs, lr=lr,
                               focal_cfg=focal_cfg, ap_lambda=ap_lambda, has_ap=True, tag="proposed", n_known=n_known,
                               prec_floor=prec_floor_train, prec_penalty_lambda=prec_penalty_lambda,
                               hard_neg_topq=hard_neg_topq, hard_neg_weight=hard_neg_weight,
                               neg_guard_lambda=neg_guard_lambda, neg_gamma=neg_gamma)
        # Collect predictions without fixed threshold
        test_result = eval_fixed_threshold(model, dl_te, device=device, has_ap=True)
        val_result = eval_fixed_threshold(model, dl_va, device=device, has_ap=True)
        
        # >>> NEW: Combo2 precision maintenance check
        _precision_maintained = True
        if baseline_val_precision is not None and not np.isnan(baseline_val_precision):
            if val_result["precision"] < baseline_val_precision:
                print(f"[WARN] C2 precision not maintained: P_proposed={val_result['precision']:.4f} < P_base={baseline_val_precision:.4f}")
                _precision_maintained = False
        
        # P>=0.75 threshold evaluation
        thr_p75, _ = find_threshold_for_precision(val_result["_y_true"], val_result["_y_prob"], precision_floor=0.75)
        result = evaluate_with_threshold(test_result["_y_true"], test_result["_y_prob"], thr_p75)
        result["_precision_maintained"] = _precision_maintained
        
        # >>> PATCH: log split recall
        y_pred_p75 = (test_result["_y_prob"] >= thr_p75).astype(int)
        _log_split_recall("2_XGBFS_LSTM+AP", test_result["_y_true"], y_pred_p75, ypat_te)
        
        return result
    else:
        # AP-Head OFF: use SimpleLSTM with FN Focal Loss only
        tr = TabDataset(Xtr2, ytr); va = TabDataset(Xva2, yva); te = TabDataset(Xte2, yte)
        dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
        dl_va = DataLoader(va, batch_size=batch); dl_te = DataLoader(te, batch_size=batch)
        model = SimpleLSTM(in_feats=Xtr2.shape[1], hidden=hidden).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = FNFocalLoss(**focal_cfg)
        
        best = {"auc": -1.0, "state": None}
        for ep in range(1, epochs+1):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                loss = crit(model(xb), yb)
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
            print(f"[proposed-noAP] val ep{ep:02d} AUC={auc:.4f}")
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        
        # Collect predictions without fixed threshold
        test_result = eval_fixed_threshold(model, dl_te, device=device, has_ap=False)
        val_result = eval_fixed_threshold(model, dl_va, device=device, has_ap=False)
        
        # P>=0.75 threshold evaluation
        thr_p75, _ = find_threshold_for_precision(val_result["_y_true"], val_result["_y_prob"], precision_floor=0.75)
        result = evaluate_with_threshold(test_result["_y_true"], test_result["_y_prob"], thr_p75)
        
        return result

# 3) MI+RF FS -> MLP
def run_combo3_baseline(Xtr, ytr, Xva, yva, Xte, yte, device, epochs, lr, batch, topk, rf_trees):
    idx = fs_mi_rf_train(Xtr, ytr, topk=topk, rf_trees=rf_trees)
    Xtr2 = Xtr[:, idx]; Xva2 = Xva[:, idx]; Xte2 = Xte[:, idx]
    tr = TabDataset(Xtr2, ytr); va = TabDataset(Xva2, yva); te = TabDataset(Xte2, yte)
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch); dl_te = DataLoader(te, batch_size=batch)
    model = TorchMLP(in_feats=Xtr2.shape[1])
    pos_weight = (len(ytr) - ytr.sum()) / (ytr.sum() + 1e-6)
    model = train_baseline(model, dl_tr, dl_va, device=device, epochs=epochs, lr=lr, pos_weight=pos_weight, 
                          tag="baseline-MLP")
    
    # Collect predictions without fixed threshold
    test_result = eval_fixed_threshold(model, dl_te, device=device, has_ap=False)
    val_result = eval_fixed_threshold(model, dl_va, device=device, has_ap=False)
    
    # P>=0.75 threshold evaluation
    thr_p75, _ = find_threshold_for_precision(val_result["_y_true"], val_result["_y_prob"], precision_floor=0.75)
    result = evaluate_with_threshold(test_result["_y_true"], test_result["_y_prob"], thr_p75)
    
    return result

# >>> PATCH v2.2: Extended to support AP-Head for combo3
def run_combo3_proposed(Xtr, ytr, Xva, yva, Xte, yte,
                        ypat_tr, ypat_va, ypat_te,  # Added for AP-Head support
                        device, epochs, lr, batch, topk, rf_trees, 
                        vocab, focal_cfg, ap_lambda, use_ap_head=True):
    """
    Combo3: MI+RF -> MLP with optional AP-Head support.
    """
    idx = fs_mi_rf_train(Xtr, ytr, topk=topk, rf_trees=rf_trees)
    Xtr2 = Xtr[:, idx]; Xva2 = Xva[:, idx]; Xte2 = Xte[:, idx]

    n_known = len(vocab)
    n_total = n_known + 1
    has_ap = use_ap_head and n_known > 0

    # AP-Head path
    if has_ap:
        _log_ap_coverage("3_MI_RF_MLP+AP", ypat_te, n_known)
        
        tr = TabDataset(Xtr2, ytr, ypat_tr)
        va = TabDataset(Xva2, yva, ypat_va)
        te = TabDataset(Xte2, yte, ypat_te)
        dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
        dl_va = DataLoader(va, batch_size=batch)
        dl_te = DataLoader(te, batch_size=batch)

        # Resolve combo-specific ap_lambda
        ap_lambda_eff = _resolve_ap_lambda(ap_lambda, "mi_rf_mlp")

        model = TorchMLPWithAPHead(in_feats=Xtr2.shape[1], n_patterns=n_total)
        model = train_proposed(model, dl_tr, dl_va, device=device, epochs=epochs, lr=lr,
                               focal_cfg=focal_cfg, ap_lambda=ap_lambda_eff, has_ap=True,
                               tag="proposed-MLP+AP", n_known=n_known)
        
        # Collect predictions without fixed threshold
        test_result = eval_fixed_threshold(model, dl_te, device=device, has_ap=True)
        val_result = eval_fixed_threshold(model, dl_va, device=device, has_ap=True)
        
        # P>=0.75 threshold evaluation
        thr_p75, _ = find_threshold_for_precision(val_result["_y_true"], val_result["_y_prob"], precision_floor=0.75)
        result = evaluate_with_threshold(test_result["_y_true"], test_result["_y_prob"], thr_p75)
        
        # Log split recall with optimized threshold
        y_pred_p75 = (test_result["_y_prob"] >= thr_p75).astype(int)
        _log_split_recall("3_MI_RF_MLP+AP", test_result["_y_true"], y_pred_p75, ypat_te)
        
        return result

    # Focal-only path (AP-Head OFF)
    tr = TabDataset(Xtr2, ytr)
    va = TabDataset(Xva2, yva)
    te = TabDataset(Xte2, yte)
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch)
    dl_te = DataLoader(te, batch_size=batch)

    model = TorchMLP(in_feats=Xtr2.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = FNFocalLoss(**focal_cfg)

    best = {"auc": -1.0, "state": None}
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        
        model.eval()
        y_true, y_prob = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
                y_true.append(yb.numpy())
                y_prob.append(prob)
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        if auc > best["auc"]:
            best["auc"] = auc
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        print(f"[proposed-MLP] val ep{ep:02d} AUC={auc:.4f}")
    
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    
    # Collect predictions without fixed threshold
    test_result = eval_fixed_threshold(model, dl_te, device=device, has_ap=False)
    val_result = eval_fixed_threshold(model, dl_va, device=device, has_ap=False)
    
    # P>=0.75 threshold evaluation
    thr_p75, _ = find_threshold_for_precision(val_result["_y_true"], val_result["_y_prob"], precision_floor=0.75)
    result = evaluate_with_threshold(test_result["_y_true"], test_result["_y_prob"], thr_p75)
    
    return result


# ----------------------------
# Summaries
# ----------------------------
def chapter_summary_print(title: str, params: Dict, test_metrics: Dict):
    # >>> PATCH v3: Show fixed threshold=0.5
    thr = test_metrics.get("threshold", 0.5)
    print("\n===== SUMMARY:", title, "=====")
    for k, v in params.items():
        print(f"{k}: {v}")
    print(f"eval_threshold_fixed: {thr:.3f}")
    print("================================\n")


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    os.makedirs("runs", exist_ok=True)
    
    # Environment summary
    import torch.backends.cudnn as cudnn
    device_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[ENV] seed={args.seed}, random_state={args.random_state}, device={device_info}, cudnn.deterministic={cudnn.deterministic}")

    # Load CICIDS (train/val)
    print("Loading CICIDS-2017 (train/val) ...")
    X_cicids_raw, y_cicids, ypat_cicids = load_all(args.cicids_dir)

    # Optional sampling
    if args.sample_size > 0 and len(X_cicids_raw) > args.sample_size:
        print(f"[info] Sampling {args.sample_size} from CICIDS ({len(X_cicids_raw)} total)")
        rng = np.random.RandomState(args.random_state)
        idx = rng.choice(len(X_cicids_raw), args.sample_size, replace=False)
        X_cicids_raw = X_cicids_raw.iloc[idx].reset_index(drop=True)
        y_cicids = y_cicids.iloc[idx].reset_index(drop=True)
        ypat_cicids = ypat_cicids.iloc[idx].reset_index(drop=True)

    # Load UNSW (test)
    print("Loading UNSW-NB15 (test) ...")
    X_unsw_raw, y_unsw, ypat_unsw = load_all(args.unsw_dir)

    if args.sample_size > 0 and len(X_unsw_raw) > args.sample_size:
        print(f"[info] Sampling {args.sample_size} from UNSW ({len(X_unsw_raw)} total)")
        rng = np.random.RandomState(args.random_state)
        idx = rng.choice(len(X_unsw_raw), args.sample_size, replace=False)
        X_unsw_raw = X_unsw_raw.iloc[idx].reset_index(drop=True)
        y_unsw = y_unsw.iloc[idx].reset_index(drop=True)
        ypat_unsw = ypat_unsw.iloc[idx].reset_index(drop=True)

    # Map UNSW columns to CICIDS space (if mapping available)
    print("Mapping UNSW features to CICIDS feature space ...")
    X_unsw_mapped = map_unsw_to_cicids(X_unsw_raw)

    # Fit preprocessor on CICIDS; transform both
    pre = Preprocessor(use_robust=True).fit(X_cicids_raw)
    X_cicids = pre.transform(X_cicids_raw)
    X_te     = pre.transform(X_unsw_mapped)

    # Train/val split
    X_tr, X_va, y_tr, y_va, p_tr, p_va = train_test_split(
        X_cicids, y_cicids.values, ypat_cicids.values,
        test_size=args.val_size, stratify=y_cicids.values, random_state=args.random_state
    )

    # Pattern vocab from CICIDS positives
    # >>> PATCH v2.1: Use ap_min_count from args instead of hardcoded value
    vocab = build_pattern_vocab(pd.Series(p_tr), min_count=args.ap_min_count)
    ypat_tr = to_pattern_ids(pd.Series(p_tr), vocab)
    ypat_va = to_pattern_ids(pd.Series(p_va), vocab)
    y_te    = y_unsw.values
    ypat_te = to_pattern_ids(ypat_unsw, vocab)

    # Quick debug stats
    _debug_label_stats(y_tr,  "CICIDS-train")
    _debug_label_stats(y_va,  "CICIDS-val")
    _debug_label_stats(y_te,  "UNSW-test")
    _debug_feature_variance(X_tr, "CICIDS-train (X)")
    _debug_feature_variance(X_va, "CICIDS-val   (X)")
    _debug_feature_variance(X_te, "UNSW-test    (X)")

    # Common params
    device = args.device
    epochs = args.epochs
    batch  = args.batch_size
    lr     = args.lr
    hidden = args.hidden
    rf_trees = args.rf_trees

    focal_cfg = dict(alpha=args.focal_alpha, gamma=args.focal_gamma,
                     fn_lambda=args.fn_lambda, fn_gamma=args.fn_gamma)
    ap_lambda = args.ap_lambda
    use_ap_head = args.use_ap_head

    # >>> PATCH v3.5 experimental: Configurable top-k per combo
    TOPK_C1 = args.topk_c1  # AFS 입력 슬라이싱 (AFS gate-based ranking)
    TOPK_C2 = args.topk_c2  # XGB FS
    TOPK_C3 = args.topk_c3  # MI+RF FS
    exp_id = args.exp_id
    
    fname = Path(__file__).name
    print("\n" + "="*80)
    print(f"ALL COMBOS ACTIVE - P>=0.75 THRESHOLD (VERY HIGH-K EXPLORATION)")
    print(f"     - C1: AFS+LSTM, topk=[55,60] (2 values), ENSEMBLE(5x)")
    print(f"     - C2: XGB+LSTM, topk=[55,60,65,70] (4 values)")
    print(f"     - C3: MI+RF+MLP, topk=[55,60,65,70] (4 values)")
    print(f"     - Total experiments: 20 (C1: 2x2, C2: 4x2, C3: 4x2)")
    if exp_id:
        print(f"     - Experiment ID: {exp_id}")
    print("     - UNIFIED METRICS:")
    print("       * P>=0.75 threshold (ALL COMBOS) - RELAXED for maximum recall")
    print("       * Loss parameters: fn_lambda=1.0, hard_neg_weight=1.5 (DEFAULT)")
    print("       * TOP-K range: C1=[55,60], C2=[55,60,65,70], C3=[55,60,65,70] (VERY HIGH-K)")
    print("     - STABILITY FIXES (PRESERVED):")
    print("       * Preprocessor: Clipping [-5, +5] post-scaling (LSTM stability)")
    print("       * AFS Ensemble: 5-run averaging (feature selection stability)")
    print(f"     - Epochs: {epochs}")
    print(f"     - Results append to: {args.results_path}")
    print(f"     - Confusion matrices saved to: runs/ (.png)")
    print(f"     - Usage: python {fname}")
    print("="*80 + "\n")

    results = []
    D = X_cicids.shape[1]
    print(f"[info] Total features: D={D}")
    
    # ============================================================================
    # Combo1: AFS-LSTM (ACTIVE with 5x ENSEMBLE)
    # ============================================================================
    # Clamp AFS ranking epochs to [8, 10]
    afs_rank_epochs = max(8, min(10, args.afs_rank_epochs))
    print(f"\n[Combo1-PreFS] Computing AFS ranking with ENSEMBLE (epochs={afs_rank_epochs}, warmup={WARMUP_EPOCHS}, gate_l1=0.0)...")
    
    # Force gate_l1_lambda to 0.0 for Combo1
    global GATE_L1_LAMBDA
    original_gate_l1 = GATE_L1_LAMBDA
    GATE_L1_LAMBDA = 0.0
    
    afs_rank = fs_afs_ranking(
        X_tr, y_tr, X_va, y_va,
        hidden=hidden, device=device,
        epochs=afs_rank_epochs,
        lr=lr, batch_size=batch
    )
    
    # AP-lambda for Combo1: Use default ap_lambda from args
    ap_lambda_c1 = ap_lambda
    
    # Loop over topk = [55, 60] (VERY HIGH-K RANGE for Combo1)
    for tk in [55, 60]:
        # Reset seed for each experiment to ensure independence and reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        exp_id_c1 = f"C1_k{tk}_gateL1off"
        print(f"\n{'='*80}")
        print(f"[Combo1] AFS-LSTM @ topk={tk} (gate_l1=0.0, eval P>=0.75, DEFAULT LOSS, ENSEMBLE 5x)")
        print(f"{'='*80}")
        
        idx1 = afs_rank[:tk]
        Xtr1 = X_tr[:, idx1]; Xva1 = X_va[:, idx1]; Xte1 = X_te[:, idx1]
        
        # Baseline N-1 (NO training precision floor)
        print(f"\n[RUN] {exp_id_c1} | var=N-1 (baseline, no training prec_floor)")
        m1a = run_combo1_baseline(Xtr1, y_tr, Xva1, y_va, Xte1, y_te, device, epochs, lr, batch, hidden,
                                  hard_neg_topq=args.hard_neg_topq,
                                  hard_neg_weight=args.hard_neg_weight)
        
        notes_c1_baseline = f"no_train_floor, gate_l1=0.0, afs_rank_epochs={afs_rank_epochs}, ap_lambda={ap_lambda_c1}"
        
        results.append({
            "exp_id": exp_id_c1,
            "combo": "1_AFS_LSTM",
            "variant": "N-1",
            "topk_used": tk,
            "notes": notes_c1_baseline,
            **{k: v for k, v in m1a.items() if not str(k).startswith('_')}
        })
        print(f"  [{exp_id_c1}] N-1: P={m1a['precision']:.4f}, R={m1a['recall']:.4f}, F1={m1a['f1']:.4f}, AUC={m1a['roc_auc']:.4f}")
        save_confusion_matrix(exp_id_c1, "1_AFS_LSTM", "N-1", m1a)
        
        # Proposed N-2 (NO training precision floor)
        print(f"\n[RUN] {exp_id_c1} | var=N-2 (proposed, no training prec_floor)")
        
        m1b = run_combo1_proposed(Xtr1, y_tr, Xva1, y_va, Xte1, y_te,
                                  ypat_tr, ypat_va, ypat_te,
                                  device, epochs, lr, batch, hidden, vocab, focal_cfg, ap_lambda_c1, use_ap_head,
                                  neg_guard_lambda=args.neg_guard_lambda,
                                  neg_gamma=args.neg_gamma,
                                  hard_neg_topq=args.hard_neg_topq,
                                  hard_neg_weight=args.hard_neg_weight)
        combo_name = "1_AFS_LSTM+AP_FN" if use_ap_head else "1_AFS_LSTM+FN"
        notes_c1_proposed = f"no_train_floor, gate_l1=0.0, afs_rank_epochs={afs_rank_epochs}, ap_lambda={ap_lambda_c1}"
        
        results.append({
            "exp_id": exp_id_c1,
            "combo": combo_name,
            "variant": "N-2",
            "topk_used": tk,
            "notes": notes_c1_proposed,
            **{k: v for k, v in m1b.items() if not str(k).startswith('_')}
        })
        print(f"  [{exp_id_c1}] N-2: P={m1b['precision']:.4f}, R={m1b['recall']:.4f}, F1={m1b['f1']:.4f}, AUC={m1b['roc_auc']:.4f}")
        save_confusion_matrix(exp_id_c1, combo_name, "N-2", m1b)
    
    # Restore original gate_l1_lambda
    GATE_L1_LAMBDA = original_gate_l1
    
    # ============================================================================
    # Combo2: XGB-FS -> LSTM (VERY HIGH-K sweep)
    # ============================================================================
    ap_lambda_c2 = ap_lambda  # Use default ap_lambda from args
    
    # Loop over topk = [55, 60, 65, 70] (VERY HIGH-K RANGE for Combo2)
    for tk in [55, 60, 65, 70]:
        # Reset seed for each experiment to ensure independence and reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        exp_id_c2 = f"C2_k{tk}_noTrainFloor"
        print(f"\n{'='*80}")
        print(f"[Combo2] XGB-FS-LSTM @ topk={tk} (no training prec_floor, eval P>=0.75, DEFAULT LOSS)")
        print(f"{'='*80}")
        
        # Baseline N-1 (NO training precision floor)
        print(f"\n[RUN] {exp_id_c2} | var=N-1 (baseline, no training prec_floor)")
        m2a = run_combo2_baseline(X_tr, y_tr, X_va, y_va, X_te, y_te, device, epochs, lr, batch, hidden,
                                  topk=tk,
                                  hard_neg_topq=0.2,
                                  hard_neg_weight=1.5)
        
        notes_c2_baseline = f"topk={tk}, no_train_floor"
        
        results.append({
            "exp_id": exp_id_c2,
            "combo": "2_XGBFS_LSTM",
            "variant": "N-1",
            "topk_used": tk,
            "notes": notes_c2_baseline,
            **{k: v for k, v in m2a.items() if not str(k).startswith('_')}
        })
        if not np.isnan(m2a['precision']):
            print(f"  [{exp_id_c2}] N-1: P={m2a['precision']:.4f}, R={m2a['recall']:.4f}, F1={m2a['f1']:.4f}, AUC={m2a['roc_auc']:.4f}")
            save_confusion_matrix(exp_id_c2, "2_XGBFS_LSTM", "N-1", m2a)
        else:
            print("  [N-1] SKIPPED (XGBoost unavailable)")
        
        # Proposed N-2 (NO training precision floor)
        print(f"\n[RUN] {exp_id_c2} | var=N-2 (proposed, no training prec_floor)")
        
        m2b = run_combo2_proposed(X_tr, y_tr, X_va, y_va, X_te, y_te,
                                  ypat_tr, ypat_va, ypat_te,
                                  device, epochs, lr, batch, hidden, topk=tk, vocab=vocab, focal_cfg=focal_cfg,
                                  ap_lambda=ap_lambda_c2, use_ap_head=use_ap_head,
                                  neg_guard_lambda=args.neg_guard_lambda,
                                  neg_gamma=args.neg_gamma,
                                  hard_neg_topq=0.25,  # Relaxed from 0.30 for better recall
                                  hard_neg_weight=1.5)  # Relaxed from 1.8 for better recall
        
        combo_name = "2_XGBFS_LSTM+AP_FN" if use_ap_head else "2_XGBFS_LSTM+FN"
        notes_c2_proposed = f"no_train_floor, hard_neg=0.30/1.8, ap_lambda={ap_lambda_c2}"
        
        results.append({
            "exp_id": exp_id_c2,
            "combo": combo_name,
            "variant": "N-2",
            "topk_used": tk,
            "notes": notes_c2_proposed,
            **{k: v for k, v in m2b.items() if not str(k).startswith('_')}
        })
        if not np.isnan(m2b['precision']):
            print(f"  [{exp_id_c2}] N-2: P={m2b['precision']:.4f}, R={m2b['recall']:.4f}, F1={m2b['f1']:.4f}, AUC={m2b['roc_auc']:.4f}")
            save_confusion_matrix(exp_id_c2, combo_name, "N-2", m2b)
        else:
            print("  [N-2] SKIPPED (XGBoost unavailable)")
    
    # ============================================================================
    # Combo3: MI+RF -> MLP (VERY HIGH-K sweep)
    # ============================================================================
    ap_lambda_c3 = ap_lambda  # Use default ap_lambda from args
    
    # Loop over topk = [55, 60, 65, 70] (VERY HIGH-K RANGE for Combo3)
    for tk in [55, 60, 65, 70]:
        # Reset seed for each experiment to ensure independence and reproducibility
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        exp_id_c3 = f"C3_k{tk}_noTrainFloor"
        print(f"\n{'='*80}")
        print(f"[Combo3] MI+RF-MLP @ topk={tk} (no training prec_floor, eval P>=0.75, DEFAULT LOSS)")
        print(f"{'='*80}")
        
        # Baseline N-1 (NO training precision floor)
        print(f"\n[RUN] {exp_id_c3} | var=N-1 (baseline, no training prec_floor)")
        m3a = run_combo3_baseline(X_tr, y_tr, X_va, y_va, X_te, y_te, device, epochs, lr, batch,
                                  topk=tk, rf_trees=rf_trees)
        
        notes_c3_baseline = f"topk={tk}, no_train_floor, rf_trees={rf_trees}"
        
        results.append({
            "exp_id": exp_id_c3,
            "combo": "3_MI_RF_MLP",
            "variant": "N-1",
            "topk_used": tk,
            "notes": notes_c3_baseline,
            **{k: v for k, v in m3a.items() if not str(k).startswith('_')}
        })
        print(f"  [{exp_id_c3}] N-1: P={m3a['precision']:.4f}, R={m3a['recall']:.4f}, F1={m3a['f1']:.4f}, AUC={m3a['roc_auc']:.4f}")
        save_confusion_matrix(exp_id_c3, "3_MI_RF_MLP", "N-1", m3a)
        
        # Proposed N-2 (NO training precision floor)
        print(f"\n[RUN] {exp_id_c3} | var=N-2 (proposed, no training prec_floor)")
        
        m3b = run_combo3_proposed(X_tr, y_tr, X_va, y_va, X_te, y_te,
                                  ypat_tr, ypat_va, ypat_te,
                                  device, epochs, lr, batch, topk=tk, rf_trees=rf_trees,
                                  vocab=vocab, focal_cfg=focal_cfg, ap_lambda=ap_lambda_c3, use_ap_head=use_ap_head)
        combo_name = "3_MI_RF_MLP+AP_FN" if use_ap_head else "3_MI_RF_MLP+FN"
        notes_c3_proposed = f"no_train_floor, ap_lambda={ap_lambda_c3}, rf_trees={rf_trees}"
        
        results.append({
            "exp_id": exp_id_c3,
            "combo": combo_name,
            "variant": "N-2",
            "topk_used": tk,
            "notes": notes_c3_proposed,
            **{k: v for k, v in m3b.items() if not str(k).startswith('_')}
        })
        print(f"  [{exp_id_c3}] N-2: P={m3b['precision']:.4f}, R={m3b['recall']:.4f}, F1={m3b['f1']:.4f}, AUC={m3b['roc_auc']:.4f}")
        save_confusion_matrix(exp_id_c3, combo_name, "N-2", m3b)    # >>> NEW: Save results to specified path with TRUE append mode
    df = pd.DataFrame(results)
    out_csv = Path(args.results_path)
    
    # Create parent directory if needed
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    write_header = not out_csv.exists()
    df.to_csv(out_csv, mode="a" if out_csv.exists() else "w", 
              header=write_header, index=False)
    
    if write_header:
        print(f"\n[info] Created new {out_csv} with {len(results)} rows")
    else:
        print(f"\n[info] Appended {len(results)} rows to existing {out_csv}")
    
    # Save summary to console
    print(f"\n{'='*80}")
    print(f"EXPERIMENTAL RUN SUMMARY")
    print(f"{'='*80}")
    for r in results:
        print(f"[{r['exp_id']}] {r['combo']} {r['variant']} topk={r['topk_used']}: "
              f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} "
              f"ROC-AUC={r['roc_auc']:.4f} PR-AUC={r['pr_auc']:.4f} "
              f"TN={r['tn']:.0f} FP={r['fp']:.0f} FN={r['fn']:.0f} TP={r['tp']:.0f}")
    print(f"{'='*80}")
    print(f"Results saved to: {out_csv.resolve()}")
    print(f"Confusion matrices saved to: runs/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
