import json
import pandas as pd
import os

# Paths
json_path = '/root/vrp/final_championship_results.json'
csv_output_path = '/root/vrp/paper/csv/complete_championship_performance_v4.csv'

def main():
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    results_list = []
    horizons_data = data.get('Data', {})

    # Sort horizons to ensure logical order (1d, 5d, 22d, 60d, 90d, 120d, 180d, 252d)
    # The keys are like "1d", "22d", so we extract digits for sorting
    sorted_horizons = sorted(horizons_data.keys(), key=lambda x: int(x.replace('d', '')))

    for hz in sorted_horizons:
        models_data = horizons_data[hz]
        # Sort models alphabetically or in a preferred order
        for model, metrics in models_data.items():
            row = {
                'Model': model,
                'Horizon': hz,
                'Pooled_R2': metrics.get('Pooled_R2'),
                'Median_R2': metrics.get('Median_R2'),
                'Mean_R2': metrics.get('Mean_R2'),
                'RMSE': metrics.get('RMSE')
            }
            results_list.append(row)

    df = pd.DataFrame(results_list)
    
    # Optional: Fill Parameters for specific known contexts
    df['Parameters'] = ""
    
    # Fill known parameters from v3 for 22d context if needed
    # But usually V4 should be a pure evaluation table.
    
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df.to_csv(csv_output_path, index=False)
    print(f"Successfully generated {csv_output_path}")
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    main()
