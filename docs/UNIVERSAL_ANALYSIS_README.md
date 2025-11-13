# Universal Multi-Track Racing Analysis

This branch provides comprehensive analysis across **all 7 race tracks** with **14 total races** (2 races per track).

## 🏁 Tracks Analyzed

1. **Barber Motorsports Park**
2. **Circuit of the Americas (COTA)**
3. **Indianapolis Motor Speedway**
4. **Road America**
5. **Sebring International Raceway**
6. **Sonoma Raceway**
7. **Virginia International Raceway (VIR)**

## 📊 What's New

### Universal Data Processing
- Automatically processes **all tracks** in a unified format
- Handles both LONG and WIDE data formats
- Merges telemetry and lap timing data consistently
- Outputs single combined dataset: `data/processed/all_tracks_combined.csv`

### Cross-Track Analysis
- Machine learning model trained on **all tracks combined**
- Feature importance analysis across different circuits
- Fast vs Slow lap comparison (top 10% vs bottom 10%)
- Per-track statistics and comparisons

### Centralized Dashboard
- **One dashboard** to view all tracks
- Switch between individual tracks or view all combined
- Compare performance metrics across circuits
- Interactive visualizations and insights

## 🚀 Quick Start

### Option 1: Run Everything
```bash
cd scripts
./run_universal_analysis.sh
```

This will:
1. Process all track data
2. Run lap time optimization analysis
3. Launch the universal dashboard

### Option 2: Step by Step
```bash
# 1. Process all track data
python src/data_processing/universal_data_processor.py

# 2. Run analysis
python src/analysis/universal_lap_analysis.py

# 3. Launch dashboard
streamlit run src/dashboards/universal_dashboard.py --server.port 8503
```

## 📁 Key Files

### Data Processing
- `src/data_processing/explore_all_datasets.py` - Explore and verify all datasets
- `src/data_processing/universal_data_processor.py` - Process all tracks into unified format

### Analysis
- `src/analysis/universal_lap_analysis.py` - ML analysis across all tracks

### Dashboard
- `src/dashboards/universal_dashboard.py` - Centralized multi-track dashboard

### Output Files
- `data/processed/all_tracks_combined.csv` - Combined dataset (all laps, all tracks)
- `data/results/universal_analysis_results.json` - Analysis results
- `data/results/universal_feature_importance.csv` - Feature importance rankings
- `data/results/universal_fast_vs_slow.csv` - Fast vs slow lap comparison
- `data/results/track_statistics.csv` - Per-track statistics

## 📈 Dashboard Features

The universal dashboard includes 5 main tabs:

### 1. Overview
- Total laps, average lap times, best lap times
- Lap time distribution across all tracks
- Box plots comparing lap time ranges

### 2. Track Comparison
- Average lap times by track
- Lap count by track
- Lap time trends over time
- Single-track deep dives

### 3. Feature Analysis
- Top features influencing lap time (using Random Forest)
- Model performance metrics (R², RMSE, MAE)
- Interactive feature importance visualization

### 4. Fast vs Slow
- Comparison of fastest 10% vs slowest 10% of laps
- Percentage differences in telemetry features
- Identify key performance differentiators

### 5. Model Performance
- ML model validation metrics
- Performance interpretation
- Training/test split information

## 🔍 Key Insights

### Dataset Format Consistency
- **13 of 14 races** use LONG format (telemetry_name/telemetry_value structure)
- **1 race** (Sebring R2) uses WIDE format (pre-pivoted)
- Universal processor handles both automatically

### Telemetry Features
- Most tracks have **9-12 telemetry types**
- Common features: speed, gear, RPM (nmot), throttle (aps), steering angle, accelerations
- Some tracks have additional sensors

### Data Quality
- All tracks have consistent column structure
- Lap timing data follows same format
- Vehicle IDs are standardized across tracks

## 💡 Usage Tips

1. **Compare Tracks**: Use "All Tracks" view to see which circuits are faster/slower
2. **Optimize Performance**: Check "Fast vs Slow" tab to see what separates fast laps from slow ones
3. **Feature Focus**: Use "Feature Analysis" to understand which telemetry signals matter most
4. **Track-Specific**: Select individual tracks for detailed race-by-race analysis

## 🎯 Next Steps

- Clean data to remove outliers (warmup laps, pit stops, etc.)
- Add weather data integration for all tracks
- Build predictive models for optimal race strategy
- Create vehicle-specific performance profiles

## 📝 Technical Details

### Data Processing Pipeline
1. **Discovery**: Scan all track directories for telemetry and lap time files
2. **Transformation**: Convert LONG format to WIDE format (pivot on telemetry_name)
3. **Merging**: Aggregate telemetry by lap, merge with lap timing data
4. **Enrichment**: Add track metadata (name, race number)
5. **Combination**: Concatenate all tracks into single dataset

### Machine Learning Model
- **Algorithm**: Random Forest Regressor
- **Features**: All numeric telemetry columns
- **Target**: Lap time (seconds)
- **Validation**: 80/20 train/test split
- **Metrics**: R², RMSE, MAE

### Performance Considerations
- Processing all tracks takes 2-5 minutes depending on hardware
- Combined dataset typically contains 5,000-10,000+ laps
- Dashboard caching improves responsiveness

## 🐛 Troubleshooting

**Missing files error**: Run `universal_data_processor.py` first to generate combined dataset

**Import errors**: Make sure you're running from project root and venv is activated

**Memory issues**: The combined dataset is large; ensure you have adequate RAM

**Dashboard slow**: First load will cache data, subsequent views are faster

---

Generated on the `universal-analysis` branch | Multi-Track Racing Analytics
