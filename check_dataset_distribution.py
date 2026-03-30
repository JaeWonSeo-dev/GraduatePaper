import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 80)
print("DATASET CLASS DISTRIBUTION CHECK")
print("=" * 80)

# Check CICIDS-2017
print("\n[1] CICIDS-2017 (Training Dataset)")
print("-" * 80)
cicids_dir = Path("CSV/MachineLearningCSV/MachineLearningCVE")
if cicids_dir.exists():
    cicids_files = sorted(list(cicids_dir.glob("*.csv")))
    print(f"Total files found: {len(cicids_files)}")
    
    total_normal = 0
    total_attack = 0
    
    for f in cicids_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            # Find label column
            label_col = None
            for c in df.columns:
                if 'label' in c.lower():
                    label_col = c
                    break
            
            if label_col:
                labels = df[label_col].astype(str).str.strip().str.lower()
                normal = labels.isin(['benign', '0', 'normal']).sum()
                attack = len(df) - normal
                total_normal += normal
                total_attack += attack
                print(f"  {f.name:50s} | Normal: {normal:>8,} | Attack: {attack:>8,}")
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
    
    total = total_normal + total_attack
    attack_ratio = (total_attack / total * 100) if total > 0 else 0
    print("-" * 80)
    print(f"  {'TOTAL':50s} | Normal: {total_normal:>8,} | Attack: {total_attack:>8,}")
    print(f"  Attack Ratio: {attack_ratio:.2f}%")
else:
    print(f"  Directory not found: {cicids_dir}")

# Check UNSW-NB15
print("\n[2] UNSW-NB15 (Test Dataset)")
print("-" * 80)
unsw_dir = Path("CSV_NB15/CSV Files/Training and Testing Sets")
if unsw_dir.exists():
    unsw_files = sorted(list(unsw_dir.glob("*.csv")))
    print(f"Total files found: {len(unsw_files)}")
    
    total_normal = 0
    total_attack = 0
    
    for f in unsw_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            # Find label column
            label_col = None
            for c in df.columns:
                if 'label' in c.lower():
                    label_col = c
                    break
            
            if label_col:
                labels = df[label_col].astype(str).str.strip().str.lower()
                normal = labels.isin(['benign', '0', 'normal', '0.0']).sum()
                attack = len(df) - normal
                total_normal += normal
                total_attack += attack
                print(f"  {f.name:50s} | Normal: {normal:>8,} | Attack: {attack:>8,}")
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
    
    total = total_normal + total_attack
    attack_ratio = (total_attack / total * 100) if total > 0 else 0
    print("-" * 80)
    print(f"  {'TOTAL':50s} | Normal: {total_normal:>8,} | Attack: {total_attack:>8,}")
    print(f"  Attack Ratio: {attack_ratio:.2f}%")
else:
    print(f"  Directory not found: {unsw_dir}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("CICIDS-2017: Used for training/validation (cross-domain source)")
print("UNSW-NB15:   Used for testing (cross-domain target)")
print("=" * 80)
