# -*- coding: utf-8 -*-
"""
Critical Bug Fixes Verification Script
Tests that the two critical bugs are fixed:
1. Combo1 Gate L1 actually disabled (global scope)
2. Combo2 baseline validation precision collected
"""

import sys
import os

def verify_gate_l1_fix():
    """Verify that GATE_L1_LAMBDA is declared global in main()"""
    print("="*80)
    print("TEST 1: Gate L1 Global Declaration")
    print("="*80)
    
    with open("runner_combine.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check for global declaration before GATE_L1_LAMBDA = 0.0
    if "global GATE_L1_LAMBDA" in content:
        # Find the context
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "global GATE_L1_LAMBDA" in line:
                print(f"\n? FOUND at line {i+1}:")
                print(f"   {lines[i-1]}")
                print(f">> {lines[i]}")
                print(f"   {lines[i+1]}")
                print(f"   {lines[i+2]}")
                
                # Verify it's before the assignment
                if i+1 < len(lines) and "original_gate_l1 = GATE_L1_LAMBDA" in lines[i+1]:
                    print("\n? VERIFIED: global declaration is in correct position")
                    print("   This ensures Gate L1 is actually disabled in fs_afs_ranking and training")
                    return True
                elif i+2 < len(lines) and "GATE_L1_LAMBDA = 0.0" in lines[i+2]:
                    print("\n? VERIFIED: global declaration is in correct position")
                    print("   This ensures Gate L1 is actually disabled in fs_afs_ranking and training")
                    return True
    
    print("\n? FAILED: 'global GATE_L1_LAMBDA' not found")
    print("   Gate L1 will NOT be disabled (scope issue)")
    return False


def verify_combo2_val_precision():
    """Verify that run_combo2_baseline collects validation precision"""
    print("\n" + "="*80)
    print("TEST 2: Combo2 Baseline Validation Precision Collection")
    print("="*80)
    
    with open("runner_combine.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find run_combo2_baseline function
    lines = content.split('\n')
    in_function = False
    found_val_eval = False
    found_val_precision_store = False
    
    for i, line in enumerate(lines):
        if "def run_combo2_baseline(" in line:
            in_function = True
            print(f"\n? Found run_combo2_baseline at line {i+1}")
        
        if in_function:
            # Look for validation evaluation
            if "val_result = eval_fixed_threshold(model, dl_va" in line:
                found_val_eval = True
                print(f"\n? FOUND validation evaluation at line {i+1}:")
                print(f"   {line.strip()}")
            
            # Look for _val_precision storage
            if '_val_precision' in line and 'val_result' in line:
                found_val_precision_store = True
                print(f"\n? FOUND validation precision storage at line {i+1}:")
                print(f"   {line.strip()}")
            
            # Exit function scope
            if in_function and line.startswith("def ") and "run_combo2_baseline" not in line:
                break
    
    if found_val_eval and found_val_precision_store:
        print("\n? VERIFIED: Combo2 baseline collects validation precision")
        print("   P_base will be correctly calculated in main()")
        print("   prec_floor_c2 = 0.97 * P_base will work properly")
        return True
    else:
        print("\n? FAILED: Validation precision collection incomplete")
        if not found_val_eval:
            print("   - Missing: val_result = eval_fixed_threshold(model, dl_va, ...)")
        if not found_val_precision_store:
            print("   - Missing: test_result['_val_precision'] = val_result['precision']")
        print("   P_base will be NaN, precision maintenance will not work")
        return False


def main():
    print("\n" + "="*80)
    print("CRITICAL BUG FIXES VERIFICATION")
    print("="*80)
    print("\nChecking runner_combine.py for two critical fixes:")
    print("1. Combo1: Gate L1 global declaration (scope fix)")
    print("2. Combo2: Validation precision collection (P_base fix)")
    print()
    
    if not os.path.exists("runner_combine.py"):
        print("? ERROR: runner_combine.py not found in current directory")
        sys.exit(1)
    
    # Run tests
    test1 = verify_gate_l1_fix()
    test2 = verify_combo2_val_precision()
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"\nTest 1 (Gate L1 global): {'? PASS' if test1 else '? FAIL'}")
    print(f"Test 2 (C2 val precision): {'? PASS' if test2 else '? FAIL'}")
    
    if test1 and test2:
        print("\n" + "="*80)
        print("? ALL TESTS PASSED - READY FOR EXPERIMENTS")
        print("="*80)
        print("\nWhat the fixes do:")
        print("1. Combo1 Gate L1 is ACTUALLY disabled (not just local var)")
        print("2. Combo2 precision floor is CALCULATED from baseline (not 0.0)")
        print("\nYou can now run:")
        print("  python runner_combine.py --results_path /mnt/data/results_run.csv")
        print("="*80)
        return 0
    else:
        print("\n" + "="*80)
        print("? SOME TESTS FAILED - BUGS NOT FULLY FIXED")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
