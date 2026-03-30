"""
NEW EXPERIMENTAL DESIGN
======================

Combo1: AFS → LSTM (Gate L1 제거, topk=20/30)
    - gate_l1_lambda = 0.0
    - ap_lambda = 0.02
    - topk = [20, 30]
    - exp_id: C1_k20_gateL1off, C1_k30_gateL1off

Combo2: XGB-FS → LSTM (정밀도 유지 강제, topk=20)
    - Baseline 먼저 실행하여 P_base 기록
    - Proposed는 prec_floor = 0.97 * P_base 적용
    - hard_neg_topq = 0.30, hard_neg_weight = 1.8
    - ap_lambda = 0.09
    - exp_id: C2_k20_precMaint

Combo3: MI+RF → MLP (정밀도 유지 비활성, topk=40/50)
    - prec_floor_train = 0.0 (OFF)
    - ap_lambda = 0.07
    - topk = [40, 50]
    - exp_id: C3_k40_varBoost, C3_k50_varBoost

Total: 10 runs (C1: 4, C2: 2, C3: 4)
"""

import subprocess
import sys

def run_experiment(exp_id, combo_num, topk, extra_args=""):
    """Run single experiment configuration."""
    
    # Base command
    cmd = [
        sys.executable, "runner_combine.py",
        "--exp_id", exp_id,
        "--results_path", "/mnt/data/results_run.csv"
    ]
    
    # Combo-specific settings
    if combo_num == 1:  # AFS-LSTM
        cmd.extend([
            f"--topk_c1", str(topk),
            "--gate_l1_lambda", "0.0",  # Gate L1 OFF
            "--ap_lambda", "0.02",  # Gentler AP regulation
        ])
    elif combo_num == 2:  # XGB-FS LSTM
        cmd.extend([
            f"--topk_c2", str(topk),
            "--ap_lambda", "0.09",
            "--hard_neg_topq", "0.30",
            "--hard_neg_weight", "1.8",
        ])
        # Precision maintenance will be handled programmatically in runner
    elif combo_num == 3:  # MI+RF MLP
        cmd.extend([
            f"--topk_c3", str(topk),
            "--prec_floor_train", "0.0",  # Precision maintenance OFF
            "--ap_lambda", "0.07",
            "--rf_trees", "700",
        ])
    
    # Add extra args
    if extra_args:
        cmd.extend(extra_args.split())
    
    print(f"\n{'='*80}")
    print(f"Running: {exp_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"[ERROR] Experiment {exp_id} failed with code {result.returncode}")
        return False
    return True


def main():
    print(__doc__)
    print("\n" + "="*80)
    print("STARTING NEW EXPERIMENTAL RUNS")
    print("="*80)
    
    experiments = [
        # Combo1: topk=20, 30
        ("C1_k20_gateL1off", 1, 20),
        ("C1_k30_gateL1off", 1, 30),
        
        # Combo2: topk=20 (precision maintenance)
        ("C2_k20_precMaint", 2, 20),
        
        # Combo3: topk=40, 50
        ("C3_k40_varBoost", 3, 40),
        ("C3_k50_varBoost", 3, 50),
    ]
    
    failed = []
    for exp_id, combo, topk in experiments:
        success = run_experiment(exp_id, combo, topk)
        if not success:
            failed.append(exp_id)
    
    print("\n" + "="*80)
    print("EXPERIMENTAL RUN SUMMARY")
    print("="*80)
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {len(experiments) - len(failed)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed experiments: {', '.join(failed)}")
    print(f"\nResults saved to: /mnt/data/results_run.csv")
    print(f"Confusion matrices saved to: runs/")
    print("="*80)


if __name__ == "__main__":
    main()
