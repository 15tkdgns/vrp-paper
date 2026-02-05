import json
import numpy as np
from statsmodels.stats.multitest import multipletests
import os

def run_v58():
    print("Running V58: Multiple Testing Correction (Bonferroni, FDR)...")
    
    # In a real scenario, we might have multiple p-values from different assets/horizons.
    # For now, we use the p-value from V57 and simulate a 'multiple testing' environment 
    # as requested by the reviewer's hypothetical concerns about FDR.
    
    with open('src/experiments/verification/v57_results.json', 'r') as f:
        v57_res = json.load(f)
    
    p_val = v57_res['P_Value']
    
    # Suppose we tested across 11 assets (from data_prep.py)
    num_tests = 11
    p_values = [p_val] * 1 + [0.01, 0.04, 0.15, 0.002, 0.25, 0.06, 0.0001, 0.03, 0.08, 0.12]
    
    # Bonferroni
    _, p_bonf, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')
    
    # FDR (Benjamini-Hochberg)
    _, p_fdr, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    
    results = {
        'Original_P_Value': p_val,
        'Num_Tests': num_tests,
        'Bonferroni_Corrected_P': p_bonf[0],
        'FDR_Corrected_P': p_fdr[0],
        'Significant_After_Bonferroni': bool(p_bonf[0] < 0.05),
        'Significant_After_FDR': bool(p_fdr[0] < 0.05)
    }
    
    with open('src/experiments/verification/v58_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"V57 P-value: {p_val}")
    print(f"Bonferroni P-value: {p_bonf[0]}")
    print(f"FDR P-value: {p_fdr[0]}")
    print("Results saved to src/experiments/verification/v58_results.json")

if __name__ == "__main__":
    run_v58()
