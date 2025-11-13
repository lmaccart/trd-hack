"""
Clean lap data to focus on actual racing laps (excluding warmup, cooldown, pit stops)
"""
import pandas as pd
import numpy as np

print("="*80)
print("CLEANING LAP DATA FOR RACING LAPS ONLY")
print("="*80)

# Load data
print("\nLoading data...")
df = pd.read_csv('merged_lap_telemetry.csv')
print(f"Original dataset: {len(df)} laps")

# Display current distribution
print(f"\nLap time distribution:")
print(f"  Min: {df['lap_time_seconds'].min():.2f}s")
print(f"  25th percentile: {df['lap_time_seconds'].quantile(0.25):.2f}s")
print(f"  Median: {df['lap_time_seconds'].median():.2f}s")
print(f"  75th percentile: {df['lap_time_seconds'].quantile(0.75):.2f}s")
print(f"  Max: {df['lap_time_seconds'].max():.2f}s")

# Calculate reasonable lap time bounds using IQR method
Q1 = df['lap_time_seconds'].quantile(0.25)
Q3 = df['lap_time_seconds'].quantile(0.75)
IQR = Q3 - Q1

# Use 1.5 * IQR for outlier detection (standard method)
lower_bound = max(80, Q1 - 1.5 * IQR)  # Don't go below 80s (too fast to be real lap)
upper_bound = min(150, Q3 + 1.5 * IQR)  # Don't go above 150s (likely pit stop)

print(f"\nCalculated bounds using IQR method:")
print(f"  Lower bound: {lower_bound:.2f}s")
print(f"  Upper bound: {upper_bound:.2f}s")

# Apply filters
df_clean = df.copy()

# Filter 1: Remove extreme lap times
df_clean = df_clean[
    (df_clean['lap_time_seconds'] >= lower_bound) &
    (df_clean['lap_time_seconds'] <= upper_bound)
]
print(f"\nAfter lap time filter: {len(df_clean)} laps ({len(df) - len(df_clean)} removed)")

# Filter 2: Remove warmup laps (first 2 laps of each outing)
df_clean = df_clean[df_clean['lap'] > 2]
print(f"After warmup lap filter: {len(df_clean)} laps ({len(df) - len(df_clean)} removed from original)")

# Filter 3: Remove laps with suspiciously low speed (likely not racing)
speed_threshold = df_clean['speed'].quantile(0.20)  # Bottom 20% probably not racing
df_clean = df_clean[df_clean['speed'] >= speed_threshold]
print(f"After speed filter (>{speed_threshold:.1f}): {len(df_clean)} laps ({len(df) - len(df_clean)} removed from original)")

# Filter 4: Remove laps with very low RPM (coasting/mechanical issue)
rpm_threshold = df_clean['nmot'].quantile(0.20)
df_clean = df_clean[df_clean['nmot'] >= rpm_threshold]
print(f"After RPM filter (>{rpm_threshold:.0f}): {len(df_clean)} laps ({len(df) - len(df_clean)} removed from original)")

# Display cleaned distribution
print(f"\n{'='*80}")
print(f"CLEANED DATASET SUMMARY")
print(f"{'='*80}")
print(f"Total laps retained: {len(df_clean)} ({len(df_clean)/len(df)*100:.1f}% of original)")
print(f"Laps removed: {len(df) - len(df_clean)}")
print(f"\nCleaned lap time distribution:")
print(f"  Min: {df_clean['lap_time_seconds'].min():.2f}s")
print(f"  Mean: {df_clean['lap_time_seconds'].mean():.2f}s")
print(f"  Median: {df_clean['lap_time_seconds'].median():.2f}s")
print(f"  Max: {df_clean['lap_time_seconds'].max():.2f}s")
print(f"  Std Dev: {df_clean['lap_time_seconds'].std():.2f}s")

# Show per-vehicle summary
print(f"\nLaps per vehicle:")
vehicle_counts = df_clean.groupby('vehicle_number').size().sort_values(ascending=False)
for vehicle, count in vehicle_counts.head(10).items():
    best_lap = df_clean[df_clean['vehicle_number'] == vehicle]['lap_time_seconds'].min()
    avg_lap = df_clean[df_clean['vehicle_number'] == vehicle]['lap_time_seconds'].mean()
    print(f"  Vehicle {vehicle:3d}: {count:3d} laps (best: {best_lap:6.2f}s, avg: {avg_lap:6.2f}s)")

if len(vehicle_counts) > 10:
    print(f"  ... and {len(vehicle_counts) - 10} more vehicles")

# Save cleaned data
output_file = 'merged_lap_telemetry_CLEAN.csv'
df_clean.to_csv(output_file, index=False)
print(f"\n{'='*80}")
print(f"Saved cleaned data to: {output_file}")
print(f"{'='*80}")

# Create a summary of what was removed
removed_df = df[~df.index.isin(df_clean.index)]
removed_summary = pd.DataFrame({
    'Reason': ['Extreme lap times', 'Warmup laps', 'Low speed', 'Low RPM'],
    'Count': [
        len(df[(df['lap_time_seconds'] < lower_bound) | (df['lap_time_seconds'] > upper_bound)]),
        len(df[df['lap'] <= 2]),
        len(df[df['speed'] < speed_threshold]),
        len(df[df['nmot'] < rpm_threshold])
    ]
})

print(f"\nBreakdown of removed laps:")
print(removed_summary.to_string(index=False))
print(f"\nNote: Some laps may have been removed for multiple reasons")

# Quality check
print(f"\n{'='*80}")
print(f"QUALITY CHECK")
print(f"{'='*80}")
consistency = df_clean.groupby('vehicle_number')['lap_time_seconds'].std().mean()
print(f"Average lap time consistency (std dev): {consistency:.2f}s")
print(f"Lap time range: {df_clean['lap_time_seconds'].max() - df_clean['lap_time_seconds'].min():.2f}s")

# Suggest next steps
print(f"\n{'='*80}")
print(f"NEXT STEPS")
print(f"{'='*80}")
print("1. Re-run analysis with cleaned data:")
print("   Update 'merged_lap_telemetry.csv' reference to 'merged_lap_telemetry_CLEAN.csv'")
print("   in advanced_lap_optimization.py and optimization_dashboard.py")
print("")
print("2. Or use this cleaned data for new analysis:")
print("   df = pd.read_csv('merged_lap_telemetry_CLEAN.csv')")
print("")
print("3. Consider additional filtering:")
print(f"   - Further narrow lap time range if needed")
print("   - Remove specific vehicles with mechanical issues")
print("   - Focus on specific lap ranges (e.g., laps 5-15 for consistency)")
