"""
Investigate data quality issues in Sonoma Raceway dataset
"""
import pandas as pd
import numpy as np

print("="*80)
print("INVESTIGATING SONOMA RACEWAY DATA")
print("="*80)

# Load Sonoma data from combined dataset
df = pd.read_csv('../../data/processed/all_tracks_combined.csv')
sonoma = df[df['track'] == 'Sonoma'].copy()

print(f"\nTotal Sonoma laps in combined dataset: {len(sonoma)}")
print(f"Races: {sonoma['race'].unique()}")

for race in sorted(sonoma['race'].unique()):
    race_data = sonoma[sonoma['race'] == race]
    print(f"\n{'='*80}")
    print(f"RACE {race}")
    print(f"{'='*80}")

    print(f"Total laps: {len(race_data)}")
    print(f"Vehicles: {race_data['vehicle_id'].nunique()}")

    # Lap time statistics
    print(f"\nLap Time Statistics:")
    print(f"  Min: {race_data['lap_time_seconds'].min():.2f}s")
    print(f"  Max: {race_data['lap_time_seconds'].max():.2f}s")
    print(f"  Mean: {race_data['lap_time_seconds'].mean():.2f}s")
    print(f"  Median: {race_data['lap_time_seconds'].median():.2f}s")
    print(f"  Std: {race_data['lap_time_seconds'].std():.2f}s")

    # Check for problematic values
    zero_laps = len(race_data[race_data['lap_time_seconds'] == 0])
    extreme_laps = len(race_data[race_data['lap_time_seconds'] > 600])

    print(f"\nProblematic Laps:")
    print(f"  Zero lap times (0s): {zero_laps}")
    print(f"  Extreme lap times (>10 min): {extreme_laps}")

    # Show distribution
    print(f"\nLap Time Distribution:")
    bins = [0, 30, 60, 90, 120, 180, 300, 600, float('inf')]
    labels = ['0-30s', '30-60s', '60-90s', '90-120s', '120-180s', '180-300s', '300-600s', '>600s']
    race_data['bin'] = pd.cut(race_data['lap_time_seconds'], bins=bins, labels=labels)
    distribution = race_data['bin'].value_counts().sort_index()

    for bin_label, count in distribution.items():
        pct = (count / len(race_data)) * 100
        print(f"  {bin_label:12s}: {count:4d} laps ({pct:5.1f}%)")

    # Sample some extreme cases
    if extreme_laps > 0:
        print(f"\nSample of extreme lap times:")
        extreme = race_data[race_data['lap_time_seconds'] > 600].sort_values('lap_time_seconds', ascending=False).head(10)
        for idx, row in extreme.iterrows():
            minutes = row['lap_time_seconds'] / 60
            print(f"  Vehicle {row['vehicle_id']}, Lap {row['lap']}: {row['lap_time_seconds']:.1f}s ({minutes:.1f} minutes)")

# Compare with typical lap times at Sonoma
print(f"\n{'='*80}")
print("COMPARISON WITH EXPECTED LAP TIMES")
print(f"{'='*80}")
print(f"\nTypical Sonoma lap times should be around 90-120 seconds")
print(f"Anything below 30s is likely a data error")
print(f"Anything above 300s is likely a pit stop or data error")

# Suggest filtering
reasonable_sonoma = sonoma[(sonoma['lap_time_seconds'] >= 80) & (sonoma['lap_time_seconds'] <= 180)]
print(f"\nWith reasonable filtering (80-180s):")
print(f"  Would keep: {len(reasonable_sonoma)} laps ({len(reasonable_sonoma)/len(sonoma)*100:.1f}%)")
print(f"  Would remove: {len(sonoma) - len(reasonable_sonoma)} laps")
print(f"  New range: {reasonable_sonoma['lap_time_seconds'].min():.1f}s - {reasonable_sonoma['lap_time_seconds'].max():.1f}s")
print(f"  New average: {reasonable_sonoma['lap_time_seconds'].mean():.1f}s")

print(f"\n{'='*80}")
print("LIKELY CAUSES")
print(f"{'='*80}")
print("1. Timestamp issues - some lap end times may be incorrect")
print("2. Missing lap markers - causing calculation of time between non-consecutive laps")
print("3. Session includes pit stops, crashes, or red flags")
print("4. Data logging errors during race")
print("\nRecommendation: Apply track-specific outlier filtering")
