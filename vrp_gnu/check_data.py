import pandas as pd
d = '/mnt/c/Users/user/Desktop/vrp/vrp_gnu/data'
for f in ['VIX.parquet', 'SPY.parquet']:
    df = pd.read_parquet(f'{d}/{f}')
    print(f'=== {f} ===')
    print('columns:', df.columns.tolist())
    print('shape:', df.shape)
    print(df.head(2))
    print()
