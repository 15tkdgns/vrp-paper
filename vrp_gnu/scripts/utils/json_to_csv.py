import json
import pandas as pd
import os

def convert_v60():
    try:
        path = 'src/experiments/verification/v60_results.json'
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Check if Format matches expectation
        # Expected: {'Full Model': 0.51, 'No Attention': 0.49, ...}
        if not data:
             print("v60 data is empty")
             return

        df = pd.DataFrame(list(data.items()), columns=['Model', 'R2_Score'])
        
        # Calculate Drop
        full_model_score = data.get('Full Model', 0)
        df['Drop_Point'] = df['R2_Score'].apply(lambda x: (full_model_score - x) * 100 if full_model_score != 0 else 0)
        
        output_path = 'paper/csv/v60_ablation_study.csv'
        df.to_csv(output_path, index=False)
        print(f"Successfully created {output_path}")
        
    except Exception as e:
        print(f"Error converting v60: {e}")

def convert_v61():
    try:
        path = 'src/experiments/verification/v61_results.json'
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return

        with open(path, 'r') as f:
            data = json.load(f)
            
        # Expected: {'baseline_r2': 0.51, 'importances': {'Feature': {'mean': ..., 'std': ...}}}
        importances = data.get('importances', {})
        if not importances:
            print("v61 importances not found or empty")
            return

        rows = []
        for feat, stats in importances.items():
            rows.append({
                'Feature': feat, 
                'Importance_Mean': stats.get('mean', 0), 
                'Importance_Std': stats.get('std', 0)
            })
            
        df = pd.DataFrame(rows)
        # Sort by importance
        df = df.sort_values('Importance_Mean', ascending=False)
        
        output_path = 'paper/csv/v61_feature_importance.csv'
        df.to_csv(output_path, index=False)
        print(f"Successfully created {output_path}")

    except Exception as e:
        print(f"Error converting v61: {e}")


def convert_v71_main():
    try:
        path = 'src/experiments/creative/v71_results.json'
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # 1. Ablation Results
        ablation = data.get('ablation', {})
        if ablation:
            df_abl = pd.DataFrame(list(ablation.items()), columns=['Model', 'R2_Score'])
            df_abl = df_abl.sort_values('R2_Score', ascending=False)
            df_abl.to_csv('paper/csv/v71_ablation.csv', index=False)
            print("Successfully created paper/csv/v71_ablation.csv")
            
        # 2. Feature Importance
        fi = data.get('feature_importance', [])
        if fi:
            df_fi = pd.DataFrame(fi)
            df_fi = df_fi[['feature', 'importance', 'std']]
            df_fi.to_csv('paper/csv/v71_feature_importance.csv', index=False)
            print("Successfully created paper/csv/v71_feature_importance.csv")

    except Exception as e:
        print(f"Error converting v71 main: {e}")

def convert_v71_robustness():
    try:
        path = 'src/experiments/creative/v71_robustness_results.json'
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        # 1. Alpha Sensitivity
        alpha_sens = data.get('alpha_sensitivity', [])
        if alpha_sens:
            df_alpha = pd.DataFrame(alpha_sens)
            df_alpha.to_csv('paper/csv/v71_alpha_sensitivity.csv', index=False)
            print("Successfully created paper/csv/v71_alpha_sensitivity.csv")
            
        # 2. Cross Validation
        cv = data.get('cross_validation', {}).get('folds', [])
        if cv:
            df_cv = pd.DataFrame(cv)
            df_cv.to_csv('paper/csv/v71_cross_validation.csv', index=False)
            print("Successfully created paper/csv/v71_cross_validation.csv")

    except Exception as e:
        print(f"Error converting v71 robustness: {e}")


def convert_v71_sci():
    base_dir = 'src/experiments/creative/v71_sci'
    
    # 1. Fair Comparison (Existing)
    try:
        path = os.path.join(base_dir, 'results_01_fair_comparison.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            rows = []
            for feature_set, models in data.items():
                for model_name, metrics in models.items():
                    row = {'Feature_Set': feature_set, 'Model': model_name}
                    row.update(metrics)
                    rows.append(row)
            if rows:
                pd.DataFrame(rows).to_csv('paper/csv/v71_sci_01_fair_comparison.csv', index=False)
                print("Created v71_sci_01_fair_comparison.csv")
    except Exception as e:
        print(f"Error 01: {e}")

    # 2. Metrics & DM Test
    try:
        path = os.path.join(base_dir, 'results_02_metrics_dm.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Metrics
            metrics = data.get('metrics', {})
            if metrics:
                rows = [{'Model': k, **v} for k, v in metrics.items()]
                pd.DataFrame(rows).to_csv('paper/csv/v71_sci_02_metrics.csv', index=False)
                print("Created v71_sci_02_metrics.csv")
                
            # DM Tests
            dm = data.get('dm_tests', {})
            if dm:
                rows = [{'Comparison': k, **v} for k, v in dm.items()]
                pd.DataFrame(rows).to_csv('paper/csv/v71_sci_02_dm_tests.csv', index=False)
                print("Created v71_sci_02_dm_tests.csv")
    except Exception as e:
        print(f"Error 02: {e}")

    # 3. Asset Class
    try:
        path = os.path.join(base_dir, 'results_03_asset_class.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Per Asset
            per_asset = data.get('per_asset', {})
            if per_asset:
                rows = [{'Asset': k, **v} for k, v in per_asset.items()]
                pd.DataFrame(rows).to_csv('paper/csv/v71_sci_03_per_asset.csv', index=False)
                print("Created v71_sci_03_per_asset.csv")
                
            # Per Class
            per_class = data.get('per_class', {})
            if per_class:
                rows = [{'Class': k, **v} for k, v in per_class.items()]
                pd.DataFrame(rows).to_csv('paper/csv/v71_sci_03_per_class.csv', index=False)
                print("Created v71_sci_03_per_class.csv")
    except Exception as e:
        print(f"Error 03: {e}")

    # 4. Regime
    try:
        path = os.path.join(base_dir, 'results_04_regime.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Period
            period = data.get('period_subsample', {})
            if period:
                rows = [{'Period': k, **v} for k, v in period.items()]
                pd.DataFrame(rows).to_csv('paper/csv/v71_sci_04_period.csv', index=False)
                print("Created v71_sci_04_period.csv")
                
            # VIX Regime
            vix = data.get('vix_regime', {})
            if vix:
                rows = [{'Regime': k, **v} for k, v in vix.items()]
                pd.DataFrame(rows).to_csv('paper/csv/v71_sci_04_vix_regime.csv', index=False)
                print("Created v71_sci_04_vix_regime.csv")
    except Exception as e:
        print(f"Error 04: {e}")
        
    # 5. Correlation (Coefficients only, matrix is too big/complex for simple CSV view usually, but we can try)
    try:
        path = os.path.join(base_dir, 'results_05_correlation_coef.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Coef per Class
            coefs = data.get('coefficients_per_class', {})
            rows = []
            for cls, feats in coefs.items():
                for feat, val in feats.items():
                    rows.append({'Class': cls, 'Feature': feat, 'Coefficient': val})
            if rows:
                pd.DataFrame(rows).to_csv('paper/csv/v71_sci_05_coefficients.csv', index=False)
                print("Created v71_sci_05_coefficients.csv")
    except Exception as e:
        print(f"Error 05: {e}")

if __name__ == "__main__":
    os.makedirs('paper/csv', exist_ok=True)
    # convert_v60() # Already done
    # convert_v61() # Already done
    # convert_v71_main() # Already done
    # convert_v71_robustness() # Already done
    convert_v71_sci()
