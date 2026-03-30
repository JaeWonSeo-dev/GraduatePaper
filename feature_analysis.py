# -*- coding: utf-8 -*-
"""
Dataset Feature Analysis
Compare features between CICIDS-2017 and UNSW-NB15
"""

import pandas as pd
from pathlib import Path

def check_features():
    # CICIDS-2017 features
    cicids_file = Path("CSV/MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    df1 = pd.read_csv(cicids_file, nrows=5)
    cicids_features = set(col.strip() for col in df1.columns if col.strip().lower() not in ["label", "attack"])
    
    # UNSW-NB15 features  
    nb15_file = Path("CSV_NB15/CSV Files/Training and Testing Sets/UNSW_NB15_training-set.csv")
    df2 = pd.read_csv(nb15_file, nrows=5)
    nb15_features = set(col.strip() for col in df2.columns if col.strip().lower() not in ["label", "attack"])
    
    print(f"CICIDS-2017 features ({len(cicids_features)}):")
    for i, feat in enumerate(sorted(cicids_features)):
        print(f"  {i+1:2d}. {feat}")
    
    print(f"\nUNSW-NB15 features ({len(nb15_features)}):")
    for i, feat in enumerate(sorted(nb15_features)):
        print(f"  {i+1:2d}. {feat}")
    
    common = cicids_features & nb15_features
    print(f"\nCommon features ({len(common)}):")
    for feat in sorted(common):
        print(f"  - {feat}")
    
    cicids_only = cicids_features - nb15_features
    print(f"\nCICIDS-2017 only ({len(cicids_only)}):")
    for feat in sorted(cicids_only):
        print(f"  - {feat}")
    
    nb15_only = nb15_features - cicids_features  
    print(f"\nUNSW-NB15 only ({len(nb15_only)}):")
    for feat in sorted(nb15_only):
        print(f"  - {feat}")

if __name__ == "__main__":
    check_features()