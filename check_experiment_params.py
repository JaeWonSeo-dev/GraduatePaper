# -*- coding: utf-8 -*-
"""
Quick experiment parameter checker
Run this to verify settings before full experiment run
"""

import argparse

def main():
    print("="*80)
    print("NEW EXPERIMENTAL DESIGN - PARAMETER SUMMARY")
    print("="*80)
    
    experiments = [
        # Combo1: Gate L1 OFF
        {
            "exp_id": "C1_k20_gateL1off",
            "combo": "1_AFS_LSTM",
            "topk": 20,
            "gate_l1": 0.0,
            "ap_lambda": 0.02,
            "runs": ["N-1", "N-2"],
            "notes": "gate_l1=0.0, neg_guard=0.1, hard_neg=0.2/1.5, ap_lambda=0.02"
        },
        {
            "exp_id": "C1_k30_gateL1off",
            "combo": "1_AFS_LSTM",
            "topk": 30,
            "gate_l1": 0.0,
            "ap_lambda": 0.02,
            "runs": ["N-1", "N-2"],
            "notes": "gate_l1=0.0, neg_guard=0.1, hard_neg=0.2/1.5, ap_lambda=0.02"
        },
        # Combo2: Precision maintenance
        {
            "exp_id": "C2_k20_precMaint",
            "combo": "2_XGBFS_LSTM",
            "topk": 20,
            "prec_floor": "0.97*P_base (dynamic)",
            "hard_neg": "0.30/1.8",
            "ap_lambda": 0.09,
            "runs": ["N-1", "N-2"],
            "notes": "prec_floor=0.97*P_base, hard_neg=0.30/1.8, ap_lambda=0.09"
        },
        # Combo3: Precision floor OFF
        {
            "exp_id": "C3_k40_varBoost",
            "combo": "3_MI_RF_MLP",
            "topk": 40,
            "prec_floor": 0.0,
            "ap_lambda": 0.07,
            "rf_trees": 700,
            "runs": ["N-1", "N-2"],
            "notes": "prec_floor=0, ap_lambda=0.07, rf_trees=700"
        },
        {
            "exp_id": "C3_k50_varBoost",
            "combo": "3_MI_RF_MLP",
            "topk": 50,
            "prec_floor": 0.0,
            "ap_lambda": 0.07,
            "rf_trees": 700,
            "runs": ["N-1", "N-2"],
            "notes": "prec_floor=0, ap_lambda=0.07, rf_trees=700"
        },
    ]
    
    print("\nTotal experiments: 10 (5 configs x 2 variants)")
    print("\nBreakdown:")
    print("  - Combo1 (AFS-LSTM): 4 runs (2 topk x 2 variants)")
    print("  - Combo2 (XGB-LSTM): 2 runs (1 topk x 2 variants)")
    print("  - Combo3 (MI+RF-MLP): 4 runs (2 topk x 2 variants)")
    
    print("\n" + "="*80)
    print("DETAILED CONFIGURATION")
    print("="*80)
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{i}. {exp['exp_id']}")
        print(f"   Combo: {exp['combo']}")
        print(f"   TopK: {exp['topk']}")
        print(f"   Variants: {', '.join(exp['runs'])}")
        
        if 'gate_l1' in exp:
            print(f"   Gate L1: {exp['gate_l1']}")
        if 'prec_floor' in exp:
            print(f"   Prec Floor: {exp['prec_floor']}")
        if 'hard_neg' in exp:
            print(f"   Hard Neg: {exp['hard_neg']}")
        if 'ap_lambda' in exp:
            print(f"   AP Lambda: {exp['ap_lambda']}")
        if 'rf_trees' in exp:
            print(f"   RF Trees: {exp['rf_trees']}")
        
        print(f"   Notes: {exp['notes']}")
    
    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print("\n1. Results CSV: /mnt/data/results_run.csv")
    print("   - Mode: Append")
    print("   - Columns: exp_id, combo, variant, topk_used, notes, metrics...")
    
    print("\n2. Confusion Matrices: runs/cm_*.txt")
    print("   - Format: cm_{exp_id}_{combo}_{variant}.txt")
    print("   - Total files: 10")
    
    print("\n" + "="*80)
    print("READY TO RUN")
    print("="*80)
    print("\nCommand:")
    print("  python runner_combine.py --results_path /mnt/data/results_run.csv")
    print("\nEstimated time: 90-115 minutes")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
