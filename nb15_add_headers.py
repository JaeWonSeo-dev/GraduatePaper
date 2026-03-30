# -*- coding: utf-8 -*-
import os
from pathlib import Path
import pandas as pd

BASE = Path("CSV_NB15/CSV Files")
FEATURES = BASE / "NUSW-NB15_features.csv"
OUTDIR = BASE / "labeled_orig"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Read feature names (try utf-8 then latin-1)
try:
    fdf = pd.read_csv(FEATURES, encoding='utf-8')
except Exception:
    fdf = pd.read_csv(FEATURES, encoding='latin-1')

if 'Name' in fdf.columns:
    cols = fdf['Name'].astype(str).str.strip().tolist()
else:
    cols = [c.strip() for c in fdf.iloc[:,1].astype(str).tolist()]

print(f"Found {len(cols)} feature names.\nSample: {cols[:5]}")

# target files: UNSW-NB15_1.csv ... _4.csv
for i in range(1,5):
    src = BASE / f"UNSW-NB15_{i}.csv"
    if not src.exists():
        print(f"Skip (not found): {src}")
        continue
    dst = OUTDIR / src.name
    print(f"Processing {src} -> {dst}")
    
    # write header and then copy content, handling BOM properly
    with open(dst, 'w', encoding='utf-8', newline='') as outf:
        outf.write(','.join(cols) + '\n')
        
        # Try different encodings to handle BOM and other issues
        success = False
        for encoding in ['utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(src, 'r', encoding=encoding, newline='') as inf:
                    for line in inf:
                        line = line.strip()
                        if line:
                            # Remove any remaining BOM characters
                            line = line.replace('\ufeff', '')  # Remove BOM
                            if line and not line.startswith('srcip'):  # Skip any existing headers
                                outf.write(line + '\n')
                success = True
                break
            except Exception as e:
                continue
        
        if not success:
            print(f"    Warning: Could not process {src} with any encoding")
    
    print(f"Wrote: {dst}")

print('Done.')