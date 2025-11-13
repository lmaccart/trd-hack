# Dashboard Quick Start Guide

## Quick Launch (Easiest Method)

Just run this single command:
```bash
./start_dashboards.sh
```

This will start both dashboards automatically:
- **Weather Dashboard**: http://localhost:8501
- **Optimization Dashboard**: http://localhost:8502

Press `Ctrl+C` to stop both dashboards.

---

## Manual Launch (Individual Dashboards)

### Start Weather Dashboard Only:
```bash
./venv/bin/streamlit run weather_dashboard.py
```
Opens at: http://localhost:8501

### Start Optimization Dashboard Only:
```bash
./venv/bin/streamlit run optimization_dashboard.py
```
Opens at: http://localhost:8501

### Start Both (on different ports):
```bash
# Terminal 1:
./venv/bin/streamlit run weather_dashboard.py

# Terminal 2:
./venv/bin/streamlit run optimization_dashboard.py --server.port 8502
```

---

## What Each Dashboard Shows

### Weather Dashboard (port 8501)
- Weather conditions overview
- Weather correlations with lap time
- Fast vs slow lap weather comparison
- Weather time series evolution
- Combined telemetry + weather analysis

**Key Finding**: Wind speed is strongest weather factor (r=-0.14)

### Optimization Dashboard (port 8502)
- Performance metrics and improvement potential
- Top optimization recommendations
- Feature importance rankings
- Fast vs slow lap telemetry comparison
- Vehicle-by-vehicle analysis
- Detailed scatter plots

**Key Finding**: RPM, speed, and throttle are top predictors (r=-0.42, -0.35, -0.30)

---

## Troubleshooting

**Port already in use?**
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Or use different ports
./venv/bin/streamlit run weather_dashboard.py --server.port 8503
```

**Module not found?**
```bash
# Install required packages
./venv/bin/pip install streamlit pandas plotly scikit-learn seaborn matplotlib
```

**Dashboard not loading?**
- Check that the CSV files exist: `merged_lap_telemetry_with_weather.csv`
- Check that JSON files exist: `optimization_results.json`, `weather_analysis.json`
- If missing, run the analysis scripts first:
  ```bash
  ./venv/bin/python integrate_weather.py
  ./venv/bin/python advanced_lap_optimization.py
  ```
