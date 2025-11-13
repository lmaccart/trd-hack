"""
Universal lap time analysis across all tracks
Runs optimization analysis on all combined track data
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json
from pathlib import Path

print("="*80)
print("UNIVERSAL LAP TIME OPTIMIZATION - ALL TRACKS")
print("="*80)

# Load combined data
print("\n[1/6] Loading combined track data...")
df = pd.read_csv('../../data/processed/all_tracks_combined.csv')
print(f"   + Loaded {len(df)} laps from {df['track'].nunique()} tracks")
print(f"   + Vehicles: {df['vehicle_id'].nunique()}")
print(f"   + Tracks: {', '.join(df['track'].unique())}")

# Remove laps without valid lap times
df_clean = df[df['lap_time_seconds'].notna()].copy()
print(f"   + Valid laps with timing: {len(df_clean)}")

# Feature engineering
print("\n[2/6] Engineering features...")

# Identify numeric telemetry columns
exclude_cols = ['timestamp', 'lap', 'vehicle_id', 'vehicle_number', 'outing',
                'track', 'track_name', 'race', 'lap_time_seconds',
                'meta_event', 'meta_session', 'meta_source', 'meta_time',
                'expire_at', 'original_vehicle_id', 'value', 'expire_at_x', 'expire_at_y']

numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col not in exclude_cols]

print(f"   + Feature columns: {len(feature_cols)}")
print(f"   + Features: {', '.join(feature_cols[:15])}{'...' if len(feature_cols) > 15 else ''}")

# Clean data: first check column completeness
df_model = df_clean[['lap_time_seconds', 'track', 'track_name', 'race', 'vehicle_id'] + feature_cols].copy()

# Remove outlier lap times (keep laps between 30s and 600s - 10 minutes)
print(f"\n   Filtering outlier lap times...")
print(f"   Before: {len(df_model)} laps (range: {df_model['lap_time_seconds'].min():.1f}s - {df_model['lap_time_seconds'].max():.1f}s)")
df_model = df_model[(df_model['lap_time_seconds'] >= 30) & (df_model['lap_time_seconds'] <= 600)]
print(f"   After: {len(df_model)} laps (range: {df_model['lap_time_seconds'].min():.1f}s - {df_model['lap_time_seconds'].max():.1f}s)")

# Check how complete each column is
print("\n   Checking data completeness...")
completeness = {}
for col in feature_cols:
    pct_valid = (df_model[col].notna().sum() / len(df_model)) * 100
    completeness[col] = pct_valid
    if pct_valid < 50:
        print(f"   ⚠️  {col}: only {pct_valid:.1f}% complete")

# Keep only columns that are at least 50% complete
good_cols = [col for col in feature_cols if completeness.get(col, 0) >= 50]
print(f"   + Keeping {len(good_cols)} features with >= 50% data")

# Update feature list and dataframe
feature_cols = good_cols
df_model = df_model[['lap_time_seconds', 'track', 'track_name', 'race', 'vehicle_id'] + feature_cols].copy()

# Now clean: replace inf and drop remaining NaNs
df_model = df_model.replace([np.inf, -np.inf], np.nan)
df_model = df_model.dropna()

print(f"   + Clean dataset: {len(df_model)} laps")

# Train model
print("\n[3/6] Training Random Forest model...")

X = df_model[feature_cols]
y = df_model['lap_time_seconds']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf_model.predict(X_test_scaled)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"   + R² Score: {r2:.4f}")
print(f"   + RMSE: {rmse:.3f} seconds")
print(f"   + MAE: {mae:.3f} seconds")

# Feature importance
print("\n[4/6] Analyzing feature importance...")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:20s}: {row['importance']:.4f}")

# Per-track analysis
print("\n[5/6] Per-track analysis...")
track_stats = []

for track in df_model['track'].unique():
    track_data = df_model[df_model['track'] == track]
    track_name = track_data['track_name'].iloc[0]

    stats = {
        'track': track,
        'track_name': track_name,
        'total_laps': len(track_data),
        'avg_lap_time': float(track_data['lap_time_seconds'].mean()),
        'min_lap_time': float(track_data['lap_time_seconds'].min()),
        'max_lap_time': float(track_data['lap_time_seconds'].max()),
        'std_lap_time': float(track_data['lap_time_seconds'].std()),
        'vehicles': int(track_data['vehicle_id'].nunique())
    }

    track_stats.append(stats)

    print(f"\n   {track_name}:")
    print(f"     Laps: {stats['total_laps']}")
    print(f"     Vehicles: {stats['vehicles']}")
    print(f"     Lap time: {stats['min_lap_time']:.2f}s - {stats['max_lap_time']:.2f}s (avg: {stats['avg_lap_time']:.2f}s ± {stats['std_lap_time']:.2f}s)")

# Fast vs Slow comparison across all tracks
print("\n[6/6] Fast vs Slow lap analysis...")

fastest_threshold = df_model['lap_time_seconds'].quantile(0.10)
slowest_threshold = df_model['lap_time_seconds'].quantile(0.90)

fastest_laps = df_model[df_model['lap_time_seconds'] <= fastest_threshold]
slowest_laps = df_model[df_model['lap_time_seconds'] >= slowest_threshold]

comparison = []
for feature in feature_cols:
    if feature in fastest_laps.columns and fastest_laps[feature].notna().sum() > 10:
        fast_mean = fastest_laps[feature].mean()
        slow_mean = slowest_laps[feature].mean()
        diff = fast_mean - slow_mean
        pct_diff = (diff / slow_mean * 100) if slow_mean != 0 else 0

        comparison.append({
            'feature': feature,
            'fast_laps_mean': fast_mean,
            'slow_laps_mean': slow_mean,
            'difference': diff,
            'pct_difference': pct_diff
        })

comparison_df = pd.DataFrame(comparison).sort_values('pct_difference', key=abs, ascending=False)

print(f"\n   Top 10 Differentiating Features (Fast vs Slow):")
for idx, row in comparison_df.head(10).iterrows():
    print(f"   {row['feature']:20s}: {row['pct_difference']:+7.2f}%")

# Save results
print("\n[7/7] Saving results...")

results = {
    'model_performance': {
        'r2_score': float(r2),
        'rmse': float(rmse),
        'mae': float(mae),
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test))
    },
    'track_statistics': track_stats,
    'feature_importance': feature_importance.head(20).to_dict('records'),
    'fast_vs_slow': comparison_df.head(20).to_dict('records')
}

# Save JSON results
results_file = '../../data/results/universal_analysis_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"   ✓ {results_file}")

# Save CSVs
feature_importance.to_csv('../../data/results/universal_feature_importance.csv', index=False)
print(f"   ✓ universal_feature_importance.csv")

comparison_df.to_csv('../../data/results/universal_fast_vs_slow.csv', index=False)
print(f"   ✓ universal_fast_vs_slow.csv")

pd.DataFrame(track_stats).to_csv('../../data/results/track_statistics.csv', index=False)
print(f"   ✓ track_statistics.csv")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"\nProcessed {len(df_model)} laps across {df_model['track'].nunique()} tracks")
print(f"Model R²: {r2:.4f} | RMSE: {rmse:.3f}s")
print(f"\nResults saved to data/results/")
print(f"{'='*80}")
