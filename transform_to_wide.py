import pandas as pd

# Load the telemetry data in long format
df_long = pd.read_csv('datasets/indianapolis/R1_indianapolis_motor_speedway_telemetry.csv')

print("Original Long Format:")
print(f"Shape: {df_long.shape}")
print(f"\nFirst few rows:")
print(df_long.head(20))
print(f"\nUnique telemetry types: {df_long['telemetry_name'].nunique()}")
print(f"Telemetry types: {sorted(df_long['telemetry_name'].unique())}")

# Transform to wide format
# Pivot on telemetry_name to create columns for each telemetry type
# Use timestamp as the index to handle multiple measurements
df_wide = df_long.pivot_table(
    index=['timestamp', 'lap', 'vehicle_id', 'vehicle_number', 'outing',
           'meta_event', 'meta_session', 'meta_source', 'meta_time'],
    columns='telemetry_name',
    values='telemetry_value',
    aggfunc='first'  # Use first value if there are duplicates
).reset_index()

# Flatten column names
df_wide.columns.name = None

print("\n" + "="*80)
print("\nTransformed Wide Format:")
print(f"Shape: {df_wide.shape}")
print(f"\nFirst few rows:")
print(df_wide.head())
print(f"\nColumn names:")
print(df_wide.columns.tolist())

# Show sample of a few telemetry columns
print("\n" + "="*80)
print("\nSample data (selected columns):")
sample_cols = ['timestamp', 'vehicle_number', 'lap', 'speed', 'nmot', 'gear', 'aps']
available_cols = [col for col in sample_cols if col in df_wide.columns]
print(df_wide[available_cols].head(10))

print("\n" + "="*80)
print(f"\nSummary:")
print(f"Long format: {df_long.shape[0]} rows x {df_long.shape[1]} columns")
print(f"Wide format: {df_wide.shape[0]} rows x {df_wide.shape[1]} columns")
print(f"Each row now represents a unique timestamp with all telemetry values as separate columns")
