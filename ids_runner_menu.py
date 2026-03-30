# ids_runner_menu.py
# Minimal-encoding (ASCII-only strings) to avoid Windows codepage issues.

import os
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, roc_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ----------------------------
# Paths (EDIT THESE IF NEEDED)
# ----------------------------
CICIDS_DIR = "CSV/MachineLearningCSV/MachineLearningCVE"
UNSW_DIR   = "CSV_NB15/CSV Files/Training and Testing Sets"

# ----------------------------
# Optional deps
# ----------------------------
try:
    import optuna
except Exception:
    optuna = None

# ----------------------------
# Utils
# ----------------------------
def device_auto():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"GPU: {name}  |  CUDA: {torch.version.cuda}")
        return "cuda"
    print("GPU not found. Using CPU.")
    return "cpu"

def ensure_dirs():
    Path("runs").mkdir(parents=True, exist_ok=True)

# ----------------------------
# Data loading
# ----------------------------
LABEL_CANDS = ["Label", "label", "Attack", "attack", "Class", "class", "attack_cat", "Attack_cat", "attack_cat ", "category"]

def infer_label_binary(val) -> int:
    s = str(val).strip().lower()
    if s in ("", "nan"):
        return 0
    if s in ("0", "benign", "normal"):
        return 0
    if s in ("1",):
        return 1
    if "benign" in s or "normal" in s:
        return 0
    return 1

def list_csvs(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.csv") if p.is_file()]

def read_one_csv(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = None
    for enc in ("utf-8", "cp949", "euc-kr", "latin-1"):
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
    X = X.dropna(axis=1, how="all")
    return X, y

def load_all(csvdir: str) -> Tuple[pd.DataFrame, pd.Series]:
    root = Path(csvdir)
    if not root.exists():
        raise SystemExit(f"Not found: {csvdir}")
    files = list_csvs(root)
    if not files:
        raise SystemExit(f"No CSVs in {csvdir}")
    X_parts, y_parts = [], []
    for f in files:
        try:
            X, y = read_one_csv(f)
            X_parts.append(X)
            y_parts.append(y)
        except Exception as e:
            print(f"[warn] skip {f.name}: {e}")
    X_raw = pd.concat(X_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    return X_raw, y

def stratified_split(X: pd.DataFrame, y: pd.Series, test_size=0.2, val_size=0.2, random_state=42):
    X_tr_all, X_te, y_tr_all, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_all, y_tr_all, test_size=val_size, stratify=y_tr_all, random_state=random_state
    )
    return X_tr, X_va, X_te, y_tr, y_va, y_te

# ----------------------------
# Fast Hashed Featurizer
# ----------------------------
from sklearn.feature_extraction import FeatureHasher

class HashedFeaturizer:
    def __init__(self, n_features: int = 2048, n_bins: int = 16, batch_rows: int = 50000, show_progress: bool = True):
        self.n_features = int(n_features)
        self.n_bins = int(n_bins)
        self.batch_rows = int(batch_rows)
        self.show_progress = show_progress
        self.hasher = FeatureHasher(n_features=self.n_features, input_type="pair", alternate_sign=False)
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.bin_edges_: dict[str, np.ndarray] = {}

    @staticmethod
    def _is_numeric_series(s: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(s)

    def fit(self, X: pd.DataFrame):
        X = X.copy()
        for c in X.columns:
            if not self._is_numeric_series(X[c]):
                conv = pd.to_numeric(X[c], errors="coerce")
                if conv.notna().mean() > 0.7:
                    X[c] = conv
        self.num_cols_ = [c for c in X.columns if self._is_numeric_series(X[c])]
        self.cat_cols_ = [c for c in X.columns if c not in self.num_cols_]
        for c in self.num_cols_:
            col = pd.to_numeric(X[c], errors="coerce")
            finite = col[np.isfinite(col)]
            if finite.empty:
                self.bin_edges_[c] = np.array([0.0, 1.0], dtype=np.float64)
                continue
            qs = np.nanpercentile(finite, np.linspace(0, 100, self.n_bins + 1))
            qs = np.unique(qs)
            if len(qs) < 2:
                qs = np.array([finite.min(), finite.max() + 1e-12], dtype=np.float64)
            self.bin_edges_[c] = qs
        return self

    def _materialize_pairs_chunk(self, X_chunk: pd.DataFrame):
        num_bins_per_col: dict[str, np.ndarray] = {}
        for c in self.num_cols_:
            col = pd.to_numeric(X_chunk[c], errors="coerce").to_numpy(copy=False)
            col = np.where(np.isfinite(col), col, 0.0)
            edges = self.bin_edges_[c]
            b = np.digitize(col, edges, right=True).astype(np.int32, copy=False)
            num_bins_per_col[c] = b

        cat_vals_per_col: dict[str, np.ndarray] = {}
        for c in self.cat_cols_:
            cat_vals_per_col[c] = X_chunk[c].astype(str).to_numpy(copy=False)

        n = len(X_chunk)
        for i in range(n):
            pairs = []
            for c in self.num_cols_:
                b = int(num_bins_per_col[c][i])
                pairs.append((f"N:{c}:{b}", 1.0))
            for c in self.cat_cols_:
                sval = cat_vals_per_col[c][i]
                pairs.append((f"C:{c}:{sval}", 1.0))
            if not pairs:
                pairs = [("BIAS", 1.0)]
            yield pairs

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        for c in self.num_cols_:
            if c not in X.columns:
                X[c] = 0.0
        for c in self.cat_cols_:
            if c not in X.columns:
                X[c] = ""
        for c in self.num_cols_:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        for c in self.cat_cols_:
            X[c] = X[c].astype(str)

        n = len(X)
        chunks = range(0, n, self.batch_rows)
        mats = []
        iterator = chunks if not self.show_progress else tqdm(chunks, desc="hashing", unit="rows")
        for start in iterator:
            stop = min(start + self.batch_rows, n)
            X_chunk = X.iloc[start:stop]
            gen = self._materialize_pairs_chunk(X_chunk)
            X_sp = self.hasher.transform(gen)
            X_dn = X_sp.astype(np.float32).toarray()
            mats.append(X_dn)

        X_all = np.vstack(mats) if len(mats) > 1 else mats[0]
        norms = np.linalg.norm(X_all, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        X_all = X_all / norms
        return X_all.astype(np.float32)

# ----------------------------
# Torch dataset & models
# ----------------------------
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class TorchMLP(nn.Module):
    def __init__(self, in_feats: int, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden), nn.Dropout(0.3),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.BatchNorm1d(hidden//2), nn.Dropout(0.2),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

class SimpleLSTM(nn.Module):
    def __init__(self, in_feats: int, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(in_feats, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(2*hidden_size, 1))
    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,D)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out).squeeze(1)

class CNNLSTM(nn.Module):
    def __init__(self, in_feats: int, hidden_size=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(in_feats, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(32*in_feats + 2*hidden_size, 1))
    def forward(self, x):
        x_seq = x.unsqueeze(1)                # (B,1,D)
        conv_out = self.conv(x_seq).view(x.size(0), -1)
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        lstm_flat = lstm_out.view(x.size(0), -1)
        h = torch.cat([conv_out, lstm_flat], dim=1)
        return self.fc(h).squeeze(1)

# ----------------------------
# Train / Eval helpers
# ----------------------------
def train_torch(model: nn.Module, dl_tr, dl_va, y_tr_arr, device="cpu", epochs=12, lr=1e-3):
    """
    y_tr_arr: numpy array or pandas Series of train labels (0/1)
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # pos_weight from y_tr_arr (robust casting)
    y_tr_np = np.asarray(y_tr_arr, dtype=np.float32).reshape(-1)
    pos_w = float((len(y_tr_np) - y_tr_np.sum()) / (y_tr_np.sum() + 1e-6))
    pos_w = max(pos_w, 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device))

    best = {"auc": -np.inf, "state": None}
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device).float()
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        # val
        model.eval(); y_true, y_prob = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
                # yb might be CPU tensor already; ensure CPU then numpy
                if hasattr(yb, "detach"):
                    y_true.append(yb.detach().cpu().numpy())
                else:
                    y_true.append(np.asarray(yb))
                y_prob.append(prob)
        y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
        auc = roc_auc_score(y_true, y_prob)
        if auc > best["auc"]:
            best["auc"] = auc
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[val] ep{ep:02d} ROC-AUC={auc:.4f}  PR-AUC={average_precision_score(y_true, y_prob):.4f}")
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model

def confusion_counts_rates(y_true: np.ndarray, y_pred: np.ndarray):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    total = tn + fp + fn + tp
    eps = 1e-12
    return dict(
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        tn_rate=tn/(total+eps), fp_rate=fp/(total+eps), fn_rate=fn/(total+eps), tp_rate=tp/(total+eps)
    )

def evaluate_arrays(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    out = dict(
        threshold=float(threshold),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        f1=float(f1_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred)),
        recall=float(recall_score(y_true, y_pred)),
    )
    out.update(confusion_counts_rates(y_true, y_pred))
    out.update({"_y_true": y_true, "_y_prob": y_prob, "_y_pred": y_pred})
    return out

def best_threshold_on_val(y_val: np.ndarray, p_val: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y_val, (p_val >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t

def infer_probs(model, dl, device):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for xb, yb in dl:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy()
            if hasattr(yb, "detach"):
                ys.append(yb.detach().cpu().numpy())
            else:
                ys.append(np.asarray(yb))
            ps.append(prob)
    return np.concatenate(ys), np.concatenate(ps)

# ----------------------------
# Plot saving
# ----------------------------
def save_confusion_bar(name: str, rates: dict, outdir: str = "runs"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    labels = ["TN","FP","FN","TP"]
    vals = [rates.get("tn_rate",0), rates.get("fp_rate",0), rates.get("fn_rate",0), rates.get("tp_rate",0)]
    plt.figure(); plt.bar(labels, vals); plt.ylim(0,1)
    plt.title(f"{name} - confusion rates"); plt.ylabel("ratio")
    plt.savefig(Path(outdir)/f"{name}_confusion_rates.png", bbox_inches="tight"); plt.close()

def save_confusion_heatmap(name: str, y_true: np.ndarray, y_pred: np.ndarray, outdir: str = "runs"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title(f"{name} - Confusion Matrix"); plt.colorbar()
    ticks = np.arange(2); plt.xticks(ticks, ["Pred 0","Pred 1"]); plt.yticks(ticks, ["True 0","True 1"])
    thr = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i,j]), ha="center", color="white" if cm[i,j]>thr else "black")
    plt.ylabel("True"); plt.xlabel("Pred")
    plt.savefig(Path(outdir)/f"{name}_confusion_matrix.png", bbox_inches="tight"); plt.close()

def save_roc_curve(name: str, y_true: np.ndarray, y_prob: np.ndarray, outdir: str = "runs"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob); auc = roc_auc_score(y_true, y_prob)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc:.4f}"); plt.plot([0,1],[0,1],"--")
    plt.xlim([0,1]); plt.ylim([0,1]); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name} - ROC"); plt.legend(loc="lower right")
    plt.savefig(Path(outdir)/f"{name}_roc.png", bbox_inches="tight"); plt.close()

def save_plots(name: str, metrics: dict):
    rates = {k: metrics[k] for k in ("tn_rate","fp_rate","fn_rate","tp_rate")}
    save_confusion_bar(name, rates)
    save_confusion_heatmap(name, metrics["_y_true"], metrics["_y_pred"])
    save_roc_curve(name, metrics["_y_true"], metrics["_y_prob"])

# ----------------------------
# Combos (val->threshold, test apply)
# ----------------------------
def combo_cnn_lstm(X_tr, y_tr, X_va, y_va, X_te, y_te, device, epochs=10, lr=1e-3, batch=512):
    tr = TabDataset(X_tr, y_tr.values); va = TabDataset(X_va, y_va.values); te = TabDataset(X_te, y_te.values)
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch); dl_te = DataLoader(te, batch_size=batch)
    model = CNNLSTM(in_feats=X_tr.shape[1], hidden_size=64)
    model = train_torch(model, dl_tr, dl_va, y_tr.values, device=device, epochs=epochs, lr=lr)
    yv, pv = infer_probs(model, dl_va, device); t = best_threshold_on_val(yv, pv)
    yt, pt = infer_probs(model, dl_te, device)
    return evaluate_arrays(yt, pt, threshold=t)

def combo_lstm(X_tr, y_tr, X_va, y_va, X_te, y_te, device, epochs=10, lr=1e-3, batch=512):
    tr = TabDataset(X_tr, y_tr.values); va = TabDataset(X_va, y_va.values); te = TabDataset(X_te, y_te.values)
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch); dl_te = DataLoader(te, batch_size=batch)
    model = SimpleLSTM(in_feats=X_tr.shape[1], hidden_size=128)
    model = train_torch(model, dl_tr, dl_va, y_tr.values, device=device, epochs=epochs, lr=lr)
    yv, pv = infer_probs(model, dl_va, device); t = best_threshold_on_val(yv, pv)
    yt, pt = infer_probs(model, dl_te, device)
    return evaluate_arrays(yt, pt, threshold=t)

def combo_mlp(X_tr, y_tr, X_va, y_va, X_te, y_te, device, epochs=10, lr=1e-3, batch=512):
    tr = TabDataset(X_tr, y_tr.values); va = TabDataset(X_va, y_va.values); te = TabDataset(X_te, y_te.values)
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch); dl_te = DataLoader(te, batch_size=batch)
    model = TorchMLP(in_feats=X_tr.shape[1], hidden=256)
    model = train_torch(model, dl_tr, dl_va, y_tr.values, device=device, epochs=epochs, lr=lr)
    yv, pv = infer_probs(model, dl_va, device); t = best_threshold_on_val(yv, pv)
    yt, pt = infer_probs(model, dl_te, device)
    return evaluate_arrays(yt, pt, threshold=t)

def combo_optuna_lstm(X_tr, y_tr, X_va, y_va, X_te, y_te, device, epochs=10, lr=1e-3, batch=512):
    if optuna is None:
        return combo_lstm(X_tr, y_tr, X_va, y_va, X_te, y_te, device, epochs=epochs, lr=lr, batch=batch)

    def objective(trial):
        hidden = trial.suggest_int("hidden", 64, 384, step=64)
        lr_ = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        ep_ = trial.suggest_int("epochs", 6, 16)
        batch_ = trial.suggest_categorical("batch", [256, 512, 1024])

        model = SimpleLSTM(in_feats=X_tr.shape[1], hidden_size=hidden)
        dl_tr = DataLoader(TabDataset(X_tr, y_tr.values), batch_size=batch_, shuffle=True)
        dl_va = DataLoader(TabDataset(X_va, y_va.values), batch_size=batch_)
        model = train_torch(model, dl_tr, dl_va, y_tr.values, device=device, epochs=ep_, lr=lr_)
        yv, pv = infer_probs(model, dl_va, device)
        return roc_auc_score(yv, pv)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    best = study.best_params
    hidden = best.get("hidden", 128); lr_ = best.get("lr", lr); ep_ = best.get("epochs", epochs); batch_ = best.get("batch", batch)

    model = SimpleLSTM(in_feats=X_tr.shape[1], hidden_size=hidden)
    dl_tr = DataLoader(TabDataset(X_tr, y_tr.values), batch_size=batch_, shuffle=True)
    dl_va = DataLoader(TabDataset(X_va, y_va.values), batch_size=batch_)
    dl_te = DataLoader(TabDataset(X_te, y_te.values), batch_size=batch_)
    model = train_torch(model, dl_tr, dl_va, y_tr.values, device=device, epochs=ep_, lr=lr_)
    yv, pv = infer_probs(model, dl_va, device); t = best_threshold_on_val(yv, pv)
    yt, pt = infer_probs(model, dl_te, device)
    return evaluate_arrays(yt, pt, threshold=t)

# ----------------------------
# Runner
# ----------------------------
def run_all_combos(X_tr, y_tr, X_va, y_va, X_te, y_te, device):
    ensure_dirs()
    results = []
    combos = [
        ("1_CNN_LSTM",    combo_cnn_lstm,   {}),
        ("2_LSTM",        combo_lstm,       {}),
        ("3_MLP",         combo_mlp,        {}),
        ("4_OPTUNA_LSTM", combo_optuna_lstm,{}),
    ]
    for name, fn, kw in combos:
        print(f"\n>>> Running {name} ...")
        try:
            m = fn(X_tr, y_tr, X_va, y_va, X_te, y_te, device, **kw)
            row = {"combo": name, **{k: v for k, v in m.items() if not str(k).startswith('_')}}
            results.append(row)
            save_plots(name, m)
            print(f"=== {name} ===")
            for k, v in row.items():
                if k == "combo": continue
                print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
        except Exception as e:
            print(f"  [error] {name}: {e}")

    # Save results
    df = pd.DataFrame(results)
    out_csv = Path("results_run.csv"); df.to_csv(out_csv, index=False)
    out_xlsx = Path("results_run.xlsx")
    engine_ok = None
    for eng in ("xlsxwriter", "openpyxl"):
        try:
            with pd.ExcelWriter(out_xlsx, engine=eng) as writer:
                for row in results:
                    nm = row["combo"]
                    pd.DataFrame([{k: v for k, v in row.items() if k != "combo"}]).to_excel(writer, sheet_name=nm[:31], index=False)
            engine_ok = eng; break
        except Exception:
            engine_ok = None
    if engine_ok is None:
        print("Warning: could not write XLSX (xlsxwriter/openpyxl missing).")
    print(f"\nAll done. CSV: {out_csv.resolve()}  XLSX: {out_xlsx.resolve() if engine_ok else 'not written'}")

# ----------------------------
# Console UI
# ----------------------------
def console_ui():
    dev = device_auto()
    print("")
    print("Select dataset option:")
    print("1) CICIDS2017 (train/val/test)")
    print("2) UNSW-NB15 (train/val/test)")
    print("3) CICIDS2017 train/val  ->  UNSW-NB15 test (cross)")
    sel = input("Enter 1/2/3: ").strip()

    fe = HashedFeaturizer(n_features=2048, n_bins=16, batch_rows=40000, show_progress=True)

    if sel == "1":
        print("\n[CICIDS2017: train/val/test]")
        X_raw, y = load_all(CICIDS_DIR)
        X_tr_raw, X_va_raw, X_te_raw, y_tr, y_va, y_te = stratified_split(X_raw, y, test_size=0.2, val_size=0.2, random_state=42)
        fe.fit(X_tr_raw)
        X_tr = fe.transform(X_tr_raw); X_va = fe.transform(X_va_raw); X_te = fe.transform(X_te_raw)
        run_all_combos(X_tr, y_tr, X_va, y_va, X_te, y_te, dev)

    elif sel == "2":
        print("\n[UNSW-NB15: train/val/test]")
        X_raw, y = load_all(UNSW_DIR)
        X_tr_raw, X_va_raw, X_te_raw, y_tr, y_va, y_te = stratified_split(X_raw, y, test_size=0.2, val_size=0.2, random_state=42)
        fe.fit(X_tr_raw)
        X_tr = fe.transform(X_tr_raw); X_va = fe.transform(X_va_raw); X_te = fe.transform(X_te_raw)
        run_all_combos(X_tr, y_tr, X_va, y_va, X_te, y_te, dev)

    elif sel == "3":
        print("\n[CICIDS2017 train/val  ->  UNSW-NB15 test]")
        X_src_raw, y_src = load_all(CICIDS_DIR)
        X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
            X_src_raw, y_src, test_size=0.2, stratify=y_src, random_state=42
        )
        X_tgt_raw, y_te = load_all(UNSW_DIR)
        fe.fit(X_tr_raw)
        X_tr = fe.transform(X_tr_raw); X_va = fe.transform(X_va_raw); X_te = fe.transform(X_tgt_raw)
        run_all_combos(X_tr, y_tr, X_va, y_va, X_te, y_te, dev)

    else:
        print("Invalid input. Exit.")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ensure_dirs()
    console_ui()
