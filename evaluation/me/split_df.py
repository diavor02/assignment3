import pandas as pd

df = pd.read_csv('demand_raw.csv', index_col=0)

df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
filtered_df = df[df['timestamp_utc'].dt.year <= 2019]

filtered_df.to_csv('filtered_demand_raw.csv')
print(f"Original rows: {len(df)}, Filtered rows: {len(filtered_df)}")