# -*- coding: utf-8 -*-
"""
Test feature mapping
"""
import pandas as pd
from pathlib import Path
from feature_mapping import FEATURE_MAPPING

def list_csvs(root: Path):
    return [p for p in root.rglob("*.csv") if p.is_file()]

def read_one_csv(path: Path):
    df = None
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc, on_bad_lines="skip", nrows=1000)
            break
        except Exception:
            df = None
    return df

def map_unsw_to_cicids(X_unsw: pd.DataFrame) -> pd.DataFrame:
    X_mapped = pd.DataFrame(index=X_unsw.index)
    mapped_count = 0
    
    for unsw_col in X_unsw.columns:
        cicids_col = FEATURE_MAPPING.get(unsw_col)
        if cicids_col is not None:
            X_mapped[cicids_col] = X_unsw[unsw_col]
            mapped_count += 1
    
    print(f"Mapped {mapped_count}/{len(X_unsw.columns)} UNSW columns")
    return X_mapped

# Load sample data
cicids_dir = Path("CSV/MachineLearningCSV/MachineLearningCVE")
unsw_dir = Path("CSV_NB15/CSV Files/Training and Testing Sets")

cicids_files = list_csvs(cicids_dir)
unsw_files = list_csvs(unsw_dir)

if cicids_files and unsw_files:
    cicids_df = read_one_csv(cicids_files[0])
    unsw_df = read_one_csv(unsw_files[0])
    
    # Remove label column
    unsw_df = unsw_df.drop(columns=['label', 'attack_cat', 'id'], errors='ignore')
    cicids_df = cicids_df.drop(columns=['Label'], errors='ignore')
    
    print(f"\nBefore mapping:")
    print(f"  CICIDS: {cicids_df.shape}")
    print(f"  UNSW: {unsw_df.shape}")
    
    unsw_mapped = map_unsw_to_cicids(unsw_df)
    
    print(f"\nAfter mapping:")
    print(f"  UNSW mapped: {unsw_mapped.shape}")
    
    cicids_cols = set(cicids_df.columns)
    unsw_mapped_cols = set(unsw_mapped.columns)
    common = cicids_cols & unsw_mapped_cols
    
    print(f"\nCommon columns: {len(common)}")
    print(f"Sample common: {sorted(list(common))[:10]}")
    
    # Check for NaN values
    print(f"\nNaN ratio in UNSW mapped:")
    print(f"  {unsw_mapped.isna().mean().mean():.3f}")
    
    print(f"\nSample values (first row):")
    for col in sorted(list(common))[:5]:
        print(f"  {col}: UNSW={unsw_mapped[col].iloc[0]:.4f if pd.notna(unsw_mapped[col].iloc[0]) else 'NaN'}, CICIDS={cicids_df[col].iloc[0]:.4f if pd.notna(cicids_df[col].iloc[0]) else 'NaN'}")
