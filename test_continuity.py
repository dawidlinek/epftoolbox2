import pandas as pd

df = pd.read_csv("output.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index, utc=True)

# Check frequency by month
deltas = df.index.to_series().diff().dropna()

print("Data frequency breakdown:")
print(f"  15min intervals: {(deltas == pd.Timedelta('15min')).sum()}")
print(f"  1h intervals: {(deltas == pd.Timedelta('1h')).sum()}")
print(f"  30min intervals: {(deltas == pd.Timedelta('30min')).sum()}")
print(f"  Other: {((deltas != pd.Timedelta('15min')) & (deltas != pd.Timedelta('1h')) & (deltas != pd.Timedelta('30min'))).sum()}")

# Find where frequency changes
hourly_mask = deltas == pd.Timedelta("1h")
first_15min = df.index[~hourly_mask.values].min() if any(~hourly_mask.values) else None
last_1h = df.index[hourly_mask.values].max() if any(hourly_mask.values) else None

print(f"\nFrequency change point:")
print(f"  Last 1h timestamp: {last_1h}")
print(f"  First 15min timestamp: {first_15min}")
