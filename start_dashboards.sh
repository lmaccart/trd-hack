#!/bin/bash
# Start both optimization dashboards

echo "Starting dashboards..."
echo ""
echo "Weather Dashboard will be available at: http://localhost:8501"
echo "Optimization Dashboard will be available at: http://localhost:8502"
echo ""
echo "Press Ctrl+C to stop both dashboards"
echo ""

# Start both dashboards in background
./venv/bin/streamlit run weather_dashboard.py --server.port 8501 &
WEATHER_PID=$!

./venv/bin/streamlit run optimization_dashboard.py --server.port 8502 &
OPT_PID=$!

# Wait for user to press Ctrl+C
trap "echo ''; echo 'Stopping dashboards...'; kill $WEATHER_PID $OPT_PID; exit" INT

# Keep script running
wait
