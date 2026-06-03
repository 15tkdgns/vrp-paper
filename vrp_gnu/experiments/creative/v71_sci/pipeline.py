"""
V71 SCI Review Response Pipeline Runner
=========================================
모든 Step을 순차적으로 실행합니다.

Usage:
    python -m src.experiments.creative.v71_sci.pipeline
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def main():
    t0 = time.time()
    print("="*70, flush=True)
    print("V71 SCI REVIEW RESPONSE PIPELINE", flush=True)
    print("="*70, flush=True)
    
    # Step 0: Build dataset
    print("\n[Pipeline] Step 0: Building dataset...", flush=True)
    from src.experiments.creative.v71_sci.data_builder import build_dataset
    ds = build_dataset()
    print(f"[Pipeline] Dataset ready: {len(ds['data'])} samples, {len(ds['feats'])} features\n", flush=True)
    
    # Step 1: Fair comparison
    print("\n[Pipeline] Step 1: Fair Model Comparison", flush=True)
    from src.experiments.creative.v71_sci.step01_fair_comparison import run as run_step1
    run_step1()
    
    # Step 2: Metrics + DM test
    print("\n[Pipeline] Step 2: Metrics & DM Test", flush=True)
    from src.experiments.creative.v71_sci.step02_metrics_dm import run as run_step2
    run_step2()
    
    # Step 3: Per-asset performance
    print("\n[Pipeline] Step 3: Per-Asset & Per-Class Performance", flush=True)
    from src.experiments.creative.v71_sci.step03_asset_class import run as run_step3
    run_step3()
    
    # Step 4: Regime analysis
    print("\n[Pipeline] Step 4: Regime & Subsample Analysis", flush=True)
    from src.experiments.creative.v71_sci.step04_regime import run as run_step4
    run_step4()
    
    # Step 5: Correlation & coefficients
    print("\n[Pipeline] Step 5: Feature Correlation & Coefficients", flush=True)
    from src.experiments.creative.v71_sci.step05_correlation_coef import run as run_step5
    run_step5()
    
    elapsed = time.time() - t0
    print("\n" + "="*70, flush=True)
    print(f"PIPELINE COMPLETE ({elapsed:.1f}s)", flush=True)
    print("="*70, flush=True)
    print("\nGenerated files:", flush=True)
    result_dir = 'src/experiments/creative/v71_sci'
    for f in sorted(os.listdir(result_dir)):
        if f.startswith('results_'):
            fpath = os.path.join(result_dir, f)
            size = os.path.getsize(fpath)
            print(f"  {f} ({size:,} bytes)", flush=True)

if __name__ == '__main__':
    main()
