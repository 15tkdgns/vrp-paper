"""
Multiple testing correction for DM test results.
Applies Benjamini-Hochberg (BH) FDR correction to dm_test_results.csv.
Focuses on WEns vs {HAR-3, NaiveRV, GARCH, IV-only} comparisons.
Output: paper/csv/dm_corrected.csv
"""
import pandas as pd
import numpy as np
import os

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels not found — using manual Bonferroni correction")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_CSV  = os.path.join(_SCRIPT_DIR, '..', 'paper', 'csv', 'dm_test_results.csv')
OUT_CSV = os.path.join(_SCRIPT_DIR, '..', 'paper', 'csv', 'dm_corrected.csv')

HORIZONS = ['1d', '5d', '22d', '60d', '90d', '120d', '180d', '252d']

# Comparisons of interest: WEns as challenger vs key references
TARGET_PAIRS = [
    ('HAR-3',   'WEns'),
    ('NaiveRV', 'WEns'),
    ('GARCH',   'WEns'),
    ('IV-only', 'WEns'),
]


def bonferroni_manual(p_values):
    n = len(p_values)
    corrected = np.minimum(np.array(p_values) * n, 1.0)
    return corrected


def bh_manual(p_values):
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    p = np.array(p_values)
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    corrected = np.minimum(p * n / ranks, 1.0)
    # Enforce monotonicity (cumulative min from right)
    corrected = np.minimum.accumulate(corrected[order][::-1])[::-1][np.argsort(order)]
    return corrected


def main():
    print("=" * 60)
    print("DM Test Multiple Testing Correction")
    print("=" * 60)

    if not os.path.exists(IN_CSV):
        print(f"ERROR: {IN_CSV} not found")
        return

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} rows from {IN_CSV}")
    print(f"Columns: {list(df.columns)}")

    # Filter for WEns comparisons
    rows = []
    for ref, chal in TARGET_PAIRS:
        sub = df[(df['Reference'] == ref) & (df['Challenger'] == chal)].copy()
        if sub.empty:
            # Try reverse direction
            sub = df[(df['Reference'] == chal) & (df['Challenger'] == ref)].copy()
            if not sub.empty:
                # Flip sign of DM stat
                sub['DM_stat'] = -sub['DM_stat']
                sub['Reference'], sub['Challenger'] = ref, chal
        rows.append(sub)

    filtered = pd.concat(rows, ignore_index=True)
    print(f"\nFiltered to {len(filtered)} WEns comparisons")

    if filtered.empty:
        print("No matching rows found. Check column names in dm_test_results.csv")
        return

    # Apply corrections
    p_vals = filtered['p_value'].values.copy()

    if HAS_STATSMODELS:
        _, p_bh, _, _  = multipletests(p_vals, method='fdr_bh')
        _, p_bon, _, _ = multipletests(p_vals, method='bonferroni')
    else:
        p_bh  = bh_manual(p_vals)
        p_bon = bonferroni_manual(p_vals)

    filtered = filtered.copy()
    filtered['p_raw']         = p_vals.round(4)
    filtered['p_BH']          = p_bh.round(4)
    filtered['p_Bonferroni']  = p_bon.round(4)
    filtered['Sig_raw']       = p_vals < 0.05
    filtered['Sig_BH']        = p_bh < 0.05
    filtered['Sig_Bonferroni']= p_bon < 0.05

    # Select and sort output columns
    out_cols = ['Horizon', 'Reference', 'Challenger', 'DM_stat',
                'p_raw', 'p_BH', 'p_Bonferroni',
                'Sig_raw', 'Sig_BH', 'Sig_Bonferroni',
                'Ref_Pooled_R2', 'Chal_Pooled_R2']
    out_cols = [c for c in out_cols if c in filtered.columns]
    filtered = filtered[out_cols]

    # Sort by comparison pair then horizon
    hz_order = {h: i for i, h in enumerate(HORIZONS)}
    filtered['_hz_order'] = filtered['Horizon'].map(hz_order)
    filtered = filtered.sort_values(['Reference', 'Challenger', '_hz_order']).drop(columns='_hz_order')

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    filtered.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: WEns superiority — raw vs BH-corrected significance")
    print("=" * 80)
    print(f"{'Comparison':<25} {'Horizon':<8} {'DM':>7} {'p_raw':>8} {'p_BH':>8} {'Sig_BH':>8}")
    print("-" * 80)
    for _, row in filtered.iterrows():
        comp = f"{row['Reference']} vs {row['Challenger']}"
        sig  = '✓' if row['Sig_BH'] else '✗'
        print(f"{comp:<25} {row['Horizon']:<8} {row['DM_stat']:>7.3f} {row['p_raw']:>8.4f} {row['p_BH']:>8.4f} {sig:>8}")

    # Count how many remain significant after correction
    n_raw = filtered['Sig_raw'].sum()
    n_bh  = filtered['Sig_BH'].sum()
    n_bon = filtered['Sig_Bonferroni'].sum()
    print(f"\nSignificant at 5%: raw={n_raw}/{len(filtered)}, BH={n_bh}/{len(filtered)}, Bonferroni={n_bon}/{len(filtered)}")


if __name__ == '__main__':
    main()
