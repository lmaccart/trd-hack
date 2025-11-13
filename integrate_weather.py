"""
Integrate weather data with lap telemetry and analyze correlations
"""
import pandas as pd
import numpy as np
import json

print("="*80)
print("INTEGRATING WEATHER DATA WITH LAP ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: LOAD WEATHER DATA
# ============================================================================
print("\n[1/5] Loading weather data...")
weather_df = pd.read_csv(
    'datasets/indianapolis/26_Weather_Race 1.CSV',
    sep=';',
    encoding='utf-8-sig'  # Handle BOM character
)

# Clean column names (remove trailing semicolons)
weather_df.columns = weather_df.columns.str.strip().str.rstrip(';')

# Convert timestamp and localize to UTC
weather_df['timestamp'] = pd.to_datetime(weather_df['TIME_UTC_STR'], format='%m/%d/%Y %I:%M:%S %p')
weather_df['timestamp'] = weather_df['timestamp'].dt.tz_localize('UTC')

print(f"   + Loaded {len(weather_df)} weather observations")
print(f"   + Time range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")
print(f"   + Weather variables: {[col for col in weather_df.columns if col not in ['TIME_UTC_SECONDS', 'TIME_UTC_STR', 'timestamp']]}")

# Display weather summary
print(f"\n   Weather conditions during race:")
print(f"   - Air temp: {weather_df['AIR_TEMP'].min():.1f}°C to {weather_df['AIR_TEMP'].max():.1f}°C")
print(f"   - Track temp: {weather_df['TRACK_TEMP'].min():.1f}°C to {weather_df['TRACK_TEMP'].max():.1f}°C")
print(f"   - Humidity: {weather_df['HUMIDITY'].min():.1f}% to {weather_df['HUMIDITY'].max():.1f}%")
print(f"   - Wind speed: {weather_df['WIND_SPEED'].min():.1f} to {weather_df['WIND_SPEED'].max():.1f} m/s")
print(f"   - Rain: {'YES' if weather_df['RAIN'].max() > 0 else 'NO'}")

# ============================================================================
# STEP 2: LOAD LAP DATA
# ============================================================================
print("\n[2/5] Loading lap data...")
lap_df = pd.read_csv('merged_lap_telemetry.csv')
lap_df['timestamp'] = pd.to_datetime(lap_df['timestamp'])

print(f"   + Loaded {len(lap_df)} laps")
print(f"   + Time range: {lap_df['timestamp'].min()} to {lap_df['timestamp'].max()}")

# ============================================================================
# STEP 3: MERGE WEATHER WITH LAPS (NEAREST TIMESTAMP)
# ============================================================================
print("\n[3/5] Merging weather data with laps...")

# Use merge_asof to match each lap with the nearest weather observation
lap_df_sorted = lap_df.sort_values('timestamp')
weather_df_sorted = weather_df.sort_values('timestamp')

merged_df = pd.merge_asof(
    lap_df_sorted,
    weather_df_sorted[['timestamp', 'AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY',
                       'PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION', 'RAIN']],
    on='timestamp',
    direction='nearest',
    tolerance=pd.Timedelta('5 minutes')  # Only match within 5 minutes
)

# Count successful merges
weather_matches = merged_df['AIR_TEMP'].notna().sum()
print(f"   + Matched {weather_matches} laps with weather data ({weather_matches/len(merged_df)*100:.1f}%)")

if weather_matches == 0:
    print("\n   WARNING: No weather data matched! Checking timestamp alignment...")
    print(f"   Lap timestamps range: {lap_df['timestamp'].min()} to {lap_df['timestamp'].max()}")
    print(f"   Weather timestamps range: {weather_df['timestamp'].min()} to {weather_df['timestamp'].max()}")
    print("\n   This might be due to:")
    print("   1. Date/timezone mismatch between datasets")
    print("   2. Weather data from different session")
    print("   3. Timestamp format issues")
    print("\n   Attempting to match by time-of-day only...")

    # Extract time of day and try matching
    lap_df['time_minutes'] = lap_df['timestamp'].dt.hour * 60 + lap_df['timestamp'].dt.minute
    weather_df['time_minutes'] = weather_df['timestamp'].dt.hour * 60 + weather_df['timestamp'].dt.minute

    # For each lap, find closest weather reading by time of day
    merged_weather = []
    for _, lap in lap_df.iterrows():
        time_diff = abs(weather_df['time_minutes'] - lap['time_minutes'])
        closest_idx = time_diff.idxmin()

        if time_diff[closest_idx] < 10:  # Within 10 minutes
            weather_row = weather_df.loc[closest_idx]
            merged_weather.append({
                'AIR_TEMP': weather_row['AIR_TEMP'],
                'TRACK_TEMP': weather_row['TRACK_TEMP'],
                'HUMIDITY': weather_row['HUMIDITY'],
                'PRESSURE': weather_row['PRESSURE'],
                'WIND_SPEED': weather_row['WIND_SPEED'],
                'WIND_DIRECTION': weather_row['WIND_DIRECTION'],
                'RAIN': weather_row['RAIN']
            })
        else:
            merged_weather.append({
                'AIR_TEMP': np.nan,
                'TRACK_TEMP': np.nan,
                'HUMIDITY': np.nan,
                'PRESSURE': np.nan,
                'WIND_SPEED': np.nan,
                'WIND_DIRECTION': np.nan,
                'RAIN': np.nan
            })

    weather_features_df = pd.DataFrame(merged_weather, index=lap_df.index)
    merged_df = pd.concat([lap_df, weather_features_df], axis=1)

    weather_matches = merged_df['AIR_TEMP'].notna().sum()
    print(f"   + After time-of-day matching: {weather_matches} laps matched ({weather_matches/len(merged_df)*100:.1f}%)")

# ============================================================================
# STEP 4: CORRELATION ANALYSIS
# ============================================================================
print("\n[4/5] Analyzing weather correlations with lap time...")

weather_features = ['AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'PRESSURE', 'WIND_SPEED']

correlations = {}
for feat in weather_features:
    if feat in merged_df.columns and merged_df[feat].notna().sum() > 10:
        valid_mask = merged_df[feat].notna() & merged_df['lap_time_seconds'].notna()
        if valid_mask.sum() > 10:
            corr = merged_df.loc[valid_mask, ['lap_time_seconds', feat]].corr().iloc[0, 1]
            correlations[feat] = corr

if len(correlations) > 0:
    print("\n   WEATHER CORRELATIONS WITH LAP TIME:")
    print("   " + "-"*60)
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, corr in sorted_corr:
        interpretation = ""
        if abs(corr) > 0.3:
            interpretation = " [STRONG]"
        elif abs(corr) > 0.1:
            interpretation = " [MODERATE]"
        else:
            interpretation = " [WEAK]"

        print(f"   {feat:20s}: {corr:+.4f}{interpretation}")

    # Interpretations
    print("\n   INTERPRETATIONS:")
    print("   " + "-"*60)
    for feat, corr in sorted_corr:
        if feat == 'AIR_TEMP':
            if corr < 0:
                print("   - Warmer air temperature -> FASTER laps (less air resistance)")
            else:
                print("   - Warmer air temperature -> SLOWER laps (less engine power)")
        elif feat == 'TRACK_TEMP':
            if corr < 0:
                print("   - Warmer track -> FASTER laps (better tire grip)")
            else:
                print("   - Warmer track -> SLOWER laps (tire degradation)")
        elif feat == 'HUMIDITY':
            if corr < 0:
                print("   - Higher humidity -> FASTER laps (denser air, more downforce)")
            else:
                print("   - Higher humidity -> SLOWER laps (less engine power)")
        elif feat == 'WIND_SPEED':
            if corr < 0:
                print("   - Higher wind -> FASTER laps (unusual, check data)")
            else:
                print("   - Higher wind -> SLOWER laps (increased drag/instability)")
        elif feat == 'PRESSURE':
            if corr < 0:
                print("   - Higher pressure -> FASTER laps (denser air, more power)")
            else:
                print("   - Higher pressure -> SLOWER laps (increased drag)")
else:
    print("\n   WARNING: Not enough weather data matched to calculate correlations")
    print("   Proceeding with weather data merge only...")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================
print("\n[5/5] Saving results...")

# Save merged data
output_file = 'merged_lap_telemetry_with_weather.csv'
merged_df.to_csv(output_file, index=False)
print(f"   + Saved: {output_file}")

# Save correlation results
if len(correlations) > 0:
    weather_results = {
        'weather_correlations': dict(sorted_corr),
        'weather_summary': {
            'air_temp_range': [float(merged_df['AIR_TEMP'].min()), float(merged_df['AIR_TEMP'].max())],
            'track_temp_range': [float(merged_df['TRACK_TEMP'].min()), float(merged_df['TRACK_TEMP'].max())],
            'humidity_range': [float(merged_df['HUMIDITY'].min()), float(merged_df['HUMIDITY'].max())],
            'wind_speed_range': [float(merged_df['WIND_SPEED'].min()), float(merged_df['WIND_SPEED'].max())],
            'rain_occurred': bool(merged_df['RAIN'].max() > 0) if 'RAIN' in merged_df.columns else False
        },
        'laps_with_weather': int(weather_matches),
        'total_laps': int(len(merged_df))
    }

    with open('weather_analysis.json', 'w') as f:
        json.dump(weather_results, f, indent=2)
    print(f"   + Saved: weather_analysis.json")

# Create weather impact summary
if len(correlations) > 0:
    # Analyze fastest vs slowest laps with weather
    fastest_threshold = merged_df['lap_time_seconds'].quantile(0.10)
    slowest_threshold = merged_df['lap_time_seconds'].quantile(0.90)

    fastest_laps = merged_df[merged_df['lap_time_seconds'] <= fastest_threshold]
    slowest_laps = merged_df[merged_df['lap_time_seconds'] >= slowest_threshold]

    print("\n" + "="*80)
    print("WEATHER CONDITIONS: FAST vs SLOW LAPS")
    print("="*80)
    print(f"\n   {'Metric':<20} {'Fast Laps':>15} {'Slow Laps':>15} {'Difference':>15}")
    print("   " + "-"*70)

    weather_comparison = []
    for feat in weather_features:
        if feat in merged_df.columns and merged_df[feat].notna().sum() > 10:
            fast_mean = fastest_laps[feat].mean()
            slow_mean = slowest_laps[feat].mean()
            diff = fast_mean - slow_mean

            if not np.isnan(fast_mean) and not np.isnan(slow_mean):
                print(f"   {feat:<20} {fast_mean:>15.2f} {slow_mean:>15.2f} {diff:>15.2f}")
                weather_comparison.append({
                    'metric': feat,
                    'fast_laps': fast_mean,
                    'slow_laps': slow_mean,
                    'difference': diff
                })

    # Save comparison
    weather_comparison_df = pd.DataFrame(weather_comparison)
    weather_comparison_df.to_csv('weather_fast_vs_slow.csv', index=False)
    print(f"\n   + Saved: weather_fast_vs_slow.csv")

print("\n" + "="*80)
print("WEATHER INTEGRATION COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Use 'merged_lap_telemetry_with_weather.csv' for analysis")
print("2. Check 'weather_analysis.json' for correlation results")
print("3. Run updated dashboard with weather data")
print("\nTo update existing analysis scripts:")
print("   Change: pd.read_csv('merged_lap_telemetry.csv')")
print("   To:     pd.read_csv('merged_lap_telemetry_with_weather.csv')")
