import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADVANCED LAP TIME OPTIMIZATION ANALYSIS")
print("Indianapolis Motor Speedway - Race 1")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n[1/6] Loading data...")
df = pd.read_csv('merged_lap_telemetry.csv')
print(f"   + Loaded {len(df)} laps from {df['vehicle_number'].nunique()} vehicles")

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n[2/6] Engineering advanced features...")

# Basic features (already have these)
basic_features = ['nmot', 'speed', 'aps', 'gear', 'accx_can', 'accy_can',
                  'pbrake_f', 'pbrake_r', 'Steering_Angle']

# ADVANCED FEATURES - These can reveal optimization opportunities
df_features = df.copy()

# 1. Driving efficiency metrics
df_features['throttle_efficiency'] = df['aps'] / (df['nmot'] + 1)  # How much throttle per RPM
df_features['brake_balance'] = df['pbrake_f'] / (df['pbrake_r'] + 0.1)  # Front/rear brake balance
df_features['total_brake_pressure'] = df['pbrake_f'] + df['pbrake_r']

# 2. G-force and handling metrics
df_features['total_g_force'] = np.sqrt(df['accx_can']**2 + df['accy_can']**2)  # Total g-force
df_features['lateral_to_longitudinal_ratio'] = df['accy_can'] / (df['accx_can'] + 0.01)

# 3. Speed-related metrics
df_features['speed_per_gear'] = df['speed'] / (df['gear'] + 1)
df_features['speed_rpm_ratio'] = df['speed'] / (df['nmot'] + 1) * 1000  # Transmission efficiency

# 4. Steering metrics
df_features['abs_steering'] = np.abs(df['Steering_Angle'])  # Absolute steering input
df_features['steering_to_lateral_g'] = df['Steering_Angle'] / (df['accy_can'] + 0.01)  # Steering efficiency

# 5. Combined power metrics
df_features['power_index'] = df['nmot'] * df['aps'] / 1000  # Rough power estimation
df_features['braking_to_speed'] = df_features['total_brake_pressure'] / (df['speed'] + 1)

# List all features
engineered_features = ['throttle_efficiency', 'brake_balance', 'total_brake_pressure',
                       'total_g_force', 'lateral_to_longitudinal_ratio',
                       'speed_per_gear', 'speed_rpm_ratio', 'abs_steering',
                       'steering_to_lateral_g', 'power_index', 'braking_to_speed']

all_features = basic_features + engineered_features

print(f"   + Created {len(engineered_features)} engineered features")
print(f"   + Total features: {len(all_features)}")

# ============================================================================
# STEP 3: CORRELATION ANALYSIS WITH NEW FEATURES
# ============================================================================
print("\n[3/6] Analyzing correlations with lap time...")

# Calculate correlations
correlations = {}
for feat in all_features:
    if feat in df_features.columns and df_features[feat].notna().sum() > 0:
        # Remove infinite values
        valid_mask = np.isfinite(df_features[feat]) & np.isfinite(df_features['lap_time_seconds'])
        if valid_mask.sum() > 10:
            corr = df_features.loc[valid_mask, ['lap_time_seconds', feat]].corr().iloc[0, 1]
            correlations[feat] = corr

# Sort by absolute correlation
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print("\n   TOP 10 FEATURES MOST CORRELATED WITH LAP TIME:")
print("   " + "-"*60)
for i, (feat, corr) in enumerate(sorted_corr[:10], 1):
    feat_type = "ENGINEERED" if feat in engineered_features else "BASIC"
    print(f"   {i:2d}. {feat:30s} {corr:+.4f}  [{feat_type}]")

# ============================================================================
# STEP 4: PREDICTIVE MODELING
# ============================================================================
print("\n[4/6] Building predictive models...")

# Prepare data
feature_cols = [f for f in all_features if f in df_features.columns]
X = df_features[feature_cols].copy()
y = df_features['lap_time_seconds'].copy()

# Remove rows with NaN or infinite values
valid_mask = X.notna().all(axis=1) & np.isfinite(X).all(axis=1) & y.notna() & np.isfinite(y)
X = X[valid_mask]
y = y[valid_mask]

print(f"   + Training on {len(X)} valid laps")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print("\n   Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

rf_pred_train = rf_model.predict(X_train_scaled)
rf_pred_test = rf_model.predict(X_test_scaled)

rf_train_r2 = r2_score(y_train, rf_pred_train)
rf_test_r2 = r2_score(y_test, rf_pred_test)
rf_test_mae = mean_absolute_error(y_test, rf_pred_test)
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_pred_test))

print(f"   + Train R2: {rf_train_r2:.4f}")
print(f"   + Test R2:  {rf_test_r2:.4f}")
print(f"   + Test MAE: {rf_test_mae:.3f} seconds")
print(f"   + Test RMSE: {rf_test_rmse:.3f} seconds")

# Train Gradient Boosting
print("\n   Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_train_scaled, y_train)

gb_pred_test = gb_model.predict(X_test_scaled)
gb_test_r2 = r2_score(y_test, gb_pred_test)
gb_test_mae = mean_absolute_error(y_test, gb_pred_test)

print(f"   + Test R2:  {gb_test_r2:.4f}")
print(f"   + Test MAE: {gb_test_mae:.3f} seconds")

# ============================================================================
# STEP 5: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n[5/6] Analyzing feature importance...")

# Get feature importance from Random Forest
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   TOP 10 MOST IMPORTANT FEATURES FOR PREDICTION:")
print("   " + "-"*60)
for i, row in feature_importance.head(10).iterrows():
    feat = row['feature']
    imp = row['importance']
    feat_type = "ENGINEERED" if feat in engineered_features else "BASIC"
    print(f"   {row.name+1:2d}. {feat:30s} {imp:.4f}  [{feat_type}]")

# ============================================================================
# STEP 6: OPTIMIZATION RECOMMENDATIONS
# ============================================================================
print("\n[6/6] Generating optimization recommendations...")

# Find the fastest 10% of laps
fastest_threshold = df_features['lap_time_seconds'].quantile(0.10)
slowest_threshold = df_features['lap_time_seconds'].quantile(0.90)

fastest_laps = df_features[df_features['lap_time_seconds'] <= fastest_threshold]
slowest_laps = df_features[df_features['lap_time_seconds'] >= slowest_threshold]

print(f"\n   Comparing FASTEST 10% vs SLOWEST 10% of laps:")
print(f"   Fastest lap: {df_features['lap_time_seconds'].min():.2f}s")
print(f"   Slowest lap: {df_features['lap_time_seconds'].max():.2f}s")
print(f"   Fast threshold: {fastest_threshold:.2f}s ({len(fastest_laps)} laps)")
print(f"   Slow threshold: {slowest_threshold:.2f}s ({len(slowest_laps)} laps)")

# Compare key metrics
print("\n   KEY DIFFERENCES (Fast vs Slow laps):")
print("   " + "-"*75)
print(f"   {'Metric':<30} {'Fast Laps':>15} {'Slow Laps':>15} {'Difference':>10}")
print("   " + "-"*75)

comparison_features = ['nmot', 'speed', 'aps', 'gear', 'total_g_force',
                       'abs_steering', 'total_brake_pressure', 'throttle_efficiency',
                       'speed_per_gear', 'power_index']

recommendations = []
for feat in comparison_features:
    if feat in fastest_laps.columns:
        fast_mean = fastest_laps[feat].mean()
        slow_mean = slowest_laps[feat].mean()
        diff = fast_mean - slow_mean
        pct_diff = (diff / slow_mean * 100) if slow_mean != 0 else 0

        print(f"   {feat:<30} {fast_mean:>15.2f} {slow_mean:>15.2f} {diff:>9.2f} ({pct_diff:+.1f}%)")

        # Generate recommendation
        if abs(pct_diff) > 5:  # Only significant differences
            if feat == 'nmot':
                direction = "higher" if diff > 0 else "lower"
                recommendations.append(f"Maintain {direction} RPM (avg {fast_mean:.0f} vs {slow_mean:.0f})")
            elif feat == 'speed':
                direction = "higher" if diff > 0 else "lower"
                recommendations.append(f"Carry {direction} average speed ({abs(pct_diff):.1f}% difference)")
            elif feat == 'aps':
                direction = "more aggressive" if diff > 0 else "smoother"
                recommendations.append(f"Use {direction} throttle application")
            elif feat == 'total_g_force':
                direction = "higher" if diff > 0 else "lower"
                recommendations.append(f"Push {direction} g-forces in corners")
            elif feat == 'abs_steering':
                direction = "more" if diff > 0 else "less"
                recommendations.append(f"Use {direction} steering input (smoother is often faster)")
            elif feat == 'total_brake_pressure':
                direction = "harder" if diff > 0 else "earlier/lighter"
                recommendations.append(f"Brake {direction} (total pressure)")
            elif feat == 'throttle_efficiency':
                recommendations.append(f"Optimize throttle/RPM relationship (efficiency: {fast_mean:.4f} vs {slow_mean:.4f})")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("OPTIMIZATION RECOMMENDATIONS")
print("="*80)

for i, rec in enumerate(recommendations[:8], 1):  # Top 8 recommendations
    print(f"{i}. {rec}")

# Save detailed results
results = {
    'model_performance': {
        'random_forest_r2': rf_test_r2,
        'random_forest_mae': rf_test_mae,
        'random_forest_rmse': rf_test_rmse,
        'gradient_boosting_r2': gb_test_r2,
        'gradient_boosting_mae': gb_test_mae
    },
    'top_correlations': dict(sorted_corr[:10]),
    'top_feature_importance': feature_importance.head(10).to_dict('records'),
    'optimization_recommendations': recommendations
}

import json
with open('optimization_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)

# Save comparison
comparison_df = pd.DataFrame({
    'feature': comparison_features,
    'fast_laps_mean': [fastest_laps[f].mean() if f in fastest_laps.columns else np.nan for f in comparison_features],
    'slow_laps_mean': [slowest_laps[f].mean() if f in slowest_laps.columns else np.nan for f in comparison_features],
})
comparison_df['difference'] = comparison_df['fast_laps_mean'] - comparison_df['slow_laps_mean']
comparison_df['pct_difference'] = (comparison_df['difference'] / comparison_df['slow_laps_mean'] * 100)
comparison_df.to_csv('fast_vs_slow_comparison.csv', index=False)

print("\n" + "="*80)
print("SAVED FILES:")
print("="*80)
print("+ optimization_results.json       - Complete analysis results")
print("+ feature_importance.csv          - Feature importance rankings")
print("+ fast_vs_slow_comparison.csv     - Fast vs slow lap comparison")
print("="*80)

# ============================================================================
# WHAT-IF ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("WHAT-IF ANALYSIS: Lap Time Impact")
print("="*80)
print("\nIf you improve each metric by 10%, your lap time would change by:")
print("-"*75)

# Calculate impact of 10% improvement in each feature
baseline_lap = X_test.mean()
for feat in feature_importance.head(8)['feature'].values:
    # Create modified version
    modified_lap = baseline_lap.copy()

    # Check correlation direction
    if feat in correlations:
        corr = correlations[feat]
        # If negative correlation, increase; if positive, decrease
        multiplier = 1.1 if corr < 0 else 0.9
    else:
        multiplier = 1.1

    modified_lap[feat] = baseline_lap[feat] * multiplier

    # Predict
    baseline_pred = rf_model.predict(scaler.transform([baseline_lap]))[0]
    modified_pred = rf_model.predict(scaler.transform([modified_lap]))[0]

    impact = modified_pred - baseline_pred
    impact_pct = (impact / baseline_pred) * 100

    direction = "increase" if multiplier > 1 else "decrease"
    print(f"10% {direction} in {feat:30s}: {impact:+.3f}s ({impact_pct:+.2f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
