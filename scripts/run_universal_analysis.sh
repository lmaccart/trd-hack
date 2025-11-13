#!/bin/bash
# Run universal analysis across all tracks

echo "="
echo "UNIVERSAL RACING ANALYSIS PIPELINE"
echo "========================================="
echo ""

# Step 1: Process all track data
echo "[1/3] Processing all track data..."
../venv/bin/python ../src/data_processing/universal_data_processor.py
if [ $? -ne 0 ]; then
    echo "❌ Data processing failed"
    exit 1
fi
echo "✓ Data processing complete"
echo ""

# Step 2: Run analysis
echo "[2/3] Running universal lap time analysis..."
../venv/bin/python ../src/analysis/universal_lap_analysis.py
if [ $? -ne 0 ]; then
    echo "❌ Analysis failed"
    exit 1
fi
echo "✓ Analysis complete"
echo ""

# Step 3: Launch dashboard
echo "[3/3] Launching universal dashboard..."
echo ""
echo "Dashboard will be available at: http://localhost:8503"
echo "Press Ctrl+C to stop the dashboard"
echo ""

../venv/bin/streamlit run ../src/dashboards/universal_dashboard.py --server.port 8503
