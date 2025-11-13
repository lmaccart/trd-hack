import pandas as pd
import numpy as np

print("Loading telemetry data...")
df_long = pd.read_csv('../../data/raw/datasets/indianapolis/R1_indianapolis_motor_speedway_telemetry.csv')

print("Transforming to wide format...")
df_wide = df_long.pivot_table(
    index=['timestamp', 'lap', 'vehicle_id', 'vehicle_number', 'outing'],
    columns='telemetry_name',
    values='telemetry_value',
    aggfunc='first'
).reset_index()
df_wide.columns.name = None

print(f"Wide telemetry shape: {df_wide.shape}")

print("\nLoading lap timing data...")
lap_time = pd.read_csv('../../data/raw/datasets/indianapolis/R1_indianapolis_motor_speedway_lap_time.csv')

# Convert timestamp to datetime
lap_time['timestamp'] = pd.to_datetime(lap_time['timestamp'])
lap_time = lap_time.sort_values(['vehicle_number', 'timestamp'])

# Calculate lap times (difference between consecutive lap end timestamps)
print("\nCalculating lap times...")
lap_time['lap_time_seconds'] = lap_time.groupby('vehicle_number')['timestamp'].diff().dt.total_seconds()

# Remove rows without valid lap times and filter out invalid laps (lap 32768 is likely warmup)
lap_time_clean = lap_time[lap_time['lap_time_seconds'].notna() & (lap_time['lap'] < 32768)].copy()

print(f"Valid laps with timing: {len(lap_time_clean)}")
print(f"Lap time range: {lap_time_clean['lap_time_seconds'].min():.2f}s to {lap_time_clean['lap_time_seconds'].max():.2f}s")

# Aggregate telemetry data per lap
print("\nAggregating telemetry per lap...")
df_wide['timestamp'] = pd.to_datetime(df_wide['timestamp'])

# Filter out invalid laps from telemetry
df_wide_clean = df_wide[df_wide['lap'] < 32768].copy()

# Aggregate telemetry by lap and vehicle
telemetry_agg = df_wide_clean.groupby(['vehicle_number', 'lap']).agg({
    'accx_can': 'mean',
    'accy_can': 'mean',
    'aps': 'mean',
    'gear': 'mean',
    'nmot': 'mean',
    'pbrake_f': 'mean',
    'pbrake_r': 'mean',
    'speed': 'mean',
    'Steering_Angle': 'mean',
    'VBOX_Lat_Min': 'first',
    'VBOX_Long_Minutes': 'first'
}).reset_index()

print(f"Aggregated telemetry shape: {telemetry_agg.shape}")

# Merge lap times with aggregated telemetry
print("\nMerging lap times with telemetry...")
merged_df = lap_time_clean.merge(
    telemetry_agg,
    on=['vehicle_number', 'lap'],
    how='inner'
)

print(f"Merged dataframe shape: {merged_df.shape}")
print(f"\nColumns: {merged_df.columns.tolist()}")

# Save merged dataframe
merged_df.to_csv('../../data/processed/merged_lap_telemetry.csv', index=False)
print("\nSaved merged data to: data/processed/merged_lap_telemetry.csv")

# Calculate correlations
print("\n" + "="*80)
print("CALCULATING CORRELATIONS WITH LAP TIME")
print("="*80)

telemetry_columns = ['accx_can', 'accy_can', 'aps', 'gear', 'nmot',
                     'pbrake_f', 'pbrake_r', 'speed', 'Steering_Angle']

correlations = {}
for col in telemetry_columns:
    if col in merged_df.columns and merged_df[col].notna().sum() > 0:
        corr = merged_df[['lap_time_seconds', col]].corr().iloc[0, 1]
        correlations[col] = corr
        print(f"{col:20s}: {corr:+.4f}")

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n" + "="*80)
print("TOP 3 CORRELATIONS (by absolute value):")
print("="*80)
for i, (col, corr) in enumerate(sorted_corr[:3], 1):
    print(f"{i}. {col:20s}: {corr:+.4f}")

# Save correlation results
with open('../../docs/correlation_results.txt', 'w') as f:
    f.write("CORRELATION ANALYSIS: Telemetry vs Lap Time\n")
    f.write("="*80 + "\n\n")
    for col, corr in sorted_corr:
        f.write(f"{col:20s}: {corr:+.4f}\n")
    f.write("\n" + "="*80 + "\n")
    f.write("TOP 3 CORRELATIONS:\n")
    f.write("="*80 + "\n")
    for i, (col, corr) in enumerate(sorted_corr[:3], 1):
        f.write(f"{i}. {col:20s}: {corr:+.4f}\n")

print("\nSaved correlation results to: docs/correlation_results.txt")

# Save top 3 variables for streamlit
import json
top_3_vars = [col for col, _ in sorted_corr[:3]]
with open('../../data/results/top_correlations.json', 'w') as f:
    json.dump({
        'top_3_variables': top_3_vars,
        'correlations': dict(sorted_corr[:3])
    }, f, indent=2)

print(f"\nTop 3 variables saved to: data/results/top_correlations.json")
print(f"Top 3: {top_3_vars}")
