import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np

# Page configuration
st.set_page_config(page_title="Lap Time Optimization Dashboard", layout="wide")

# Title
st.title("Lap Time Optimization Dashboard")
st.markdown("### Indianapolis Motor Speedway - Race 1")
st.markdown("**Predictive Analysis for Performance Improvement**")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('merged_lap_telemetry.csv')
    with open('optimization_results.json', 'r') as f:
        opt_results = json.load(f)
    feature_importance = pd.read_csv('feature_importance.csv')
    comparison = pd.read_csv('fast_vs_slow_comparison.csv')
    return df, opt_results, feature_importance, comparison

try:
    df, opt_results, feature_importance, comparison = load_data()

    # Filter out extreme outliers for better visualization
    df_clean = df[(df['lap_time_seconds'] > 80) & (df['lap_time_seconds'] < 200)].copy()

    # Sidebar for filtering
    st.sidebar.header("Filters")

    # Vehicle selection
    vehicles = sorted(df_clean['vehicle_number'].unique())
    selected_vehicles = st.sidebar.multiselect(
        "Select Vehicles",
        vehicles,
        default=vehicles[:5] if len(vehicles) > 5 else vehicles
    )

    if selected_vehicles:
        df_filtered = df_clean[df_clean['vehicle_number'].isin(selected_vehicles)]
    else:
        df_filtered = df_clean

    # Performance Overview
    st.header("Performance Overview")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Laps Analyzed", len(df_filtered))
    with col2:
        st.metric("Vehicles", df_filtered['vehicle_number'].nunique())
    with col3:
        st.metric("Best Lap Time", f"{df_filtered['lap_time_seconds'].min():.2f}s")
    with col4:
        st.metric("Avg Lap Time", f"{df_filtered['lap_time_seconds'].mean():.2f}s")
    with col5:
        st.metric("Time Improvement Potential",
                  f"{df_filtered['lap_time_seconds'].mean() - df_filtered['lap_time_seconds'].min():.2f}s")

    st.markdown("---")

    # Optimization Recommendations
    st.header("Key Optimization Recommendations")

    recommendations = opt_results['optimization_recommendations']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Actions to Improve Lap Time")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"**{i}.** {rec}")

    with col2:
        st.subheader("Model Performance")
        model_perf = opt_results['model_performance']
        st.metric("Random Forest R-squared", f"{model_perf['random_forest_r2']:.4f}")
        st.metric("Mean Absolute Error", f"{model_perf['random_forest_mae']:.2f}s")
        st.info("Note: Low R-squared suggests high variance in lap times. Consider filtering to racing laps only (excluding warmup, cooldown, pit stops).")

    st.markdown("---")

    # Feature Importance
    st.header("Feature Importance Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_importance = px.bar(
            feature_importance.head(15),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 15 Features for Predicting Lap Time',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig_importance.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_importance, use_container_width=True)

    with col2:
        st.subheader("Feature Correlations")
        top_corr = opt_results['top_correlations']
        corr_df = pd.DataFrame(list(top_corr.items()), columns=['Feature', 'Correlation'])
        corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False).head(10)

        fig_corr = px.bar(
            corr_df,
            x='Correlation',
            y='Feature',
            orientation='h',
            title='Top 10 Correlations with Lap Time',
            color='Correlation',
            color_continuous_scale='RdBu_r',
            range_color=[-0.5, 0.5]
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")

    # Fast vs Slow Lap Comparison
    st.header("Fast vs Slow Lap Analysis")

    comparison_clean = comparison.dropna()
    comparison_clean['feature_display'] = comparison_clean['feature'].str.replace('_', ' ').str.title()

    fig_comparison = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Values: Fast vs Slow Laps', 'Percentage Difference'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Fast vs Slow values
    fig_comparison.add_trace(
        go.Bar(name='Fast Laps', x=comparison_clean['feature_display'],
               y=comparison_clean['fast_laps_mean'], marker_color='green'),
        row=1, col=1
    )
    fig_comparison.add_trace(
        go.Bar(name='Slow Laps', x=comparison_clean['feature_display'],
               y=comparison_clean['slow_laps_mean'], marker_color='red'),
        row=1, col=1
    )

    # Percentage difference
    fig_comparison.add_trace(
        go.Bar(name='% Difference', x=comparison_clean['feature_display'],
               y=comparison_clean['pct_difference'],
               marker_color=comparison_clean['pct_difference'].apply(
                   lambda x: 'green' if x > 0 else 'red')),
        row=1, col=2
    )

    fig_comparison.update_xaxes(title_text="Feature", row=1, col=1, tickangle=-45)
    fig_comparison.update_xaxes(title_text="Feature", row=1, col=2, tickangle=-45)
    fig_comparison.update_yaxes(title_text="Average Value", row=1, col=1)
    fig_comparison.update_yaxes(title_text="% Difference", row=1, col=2)

    fig_comparison.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig_comparison, use_container_width=True)

    st.markdown("---")

    # Lap Time Distribution
    st.header("Lap Time Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(
            df_filtered,
            x='lap_time_seconds',
            nbins=50,
            title='Lap Time Distribution (Filtered)',
            labels={'lap_time_seconds': 'Lap Time (seconds)'},
            color_discrete_sequence=['#636EFA'],
            marginal='box'
        )
        fig_hist.add_vline(x=df_filtered['lap_time_seconds'].median(),
                           line_dash="dash", line_color="red",
                           annotation_text="Median")
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Lap time by vehicle
        vehicle_stats = df_filtered.groupby('vehicle_number')['lap_time_seconds'].agg([
            ('min', 'min'),
            ('mean', 'mean'),
            ('max', 'max')
        ]).reset_index()

        fig_vehicle = px.scatter(
            vehicle_stats,
            x='vehicle_number',
            y='mean',
            error_y=vehicle_stats['max'] - vehicle_stats['mean'],
            error_y_minus=vehicle_stats['mean'] - vehicle_stats['min'],
            title='Lap Time by Vehicle (Mean ± Range)',
            labels={'mean': 'Mean Lap Time (s)', 'vehicle_number': 'Vehicle #'}
        )
        fig_vehicle.update_traces(marker_size=10)
        st.plotly_chart(fig_vehicle, use_container_width=True)

    st.markdown("---")

    # Key Telemetry Correlations
    st.header("Key Telemetry vs Lap Time")

    # Select top correlated features to plot
    top_features = ['speed', 'nmot', 'aps', 'gear']

    fig_scatter = make_subplots(
        rows=2, cols=2,
        subplot_titles=tuple([f.upper() + ' vs Lap Time' for f in top_features])
    )

    for i, feat in enumerate(top_features):
        row = (i // 2) + 1
        col = (i % 2) + 1

        fig_scatter.add_trace(
            go.Scatter(
                x=df_filtered[feat],
                y=df_filtered['lap_time_seconds'],
                mode='markers',
                marker=dict(size=5, opacity=0.6, color=df_filtered['vehicle_number']),
                showlegend=False,
                name=feat
            ),
            row=row, col=col
        )

        fig_scatter.update_xaxes(title_text=feat.upper(), row=row, col=col)
        fig_scatter.update_yaxes(title_text='Lap Time (s)', row=row, col=col)

    fig_scatter.update_layout(height=800, title_text="Key Telemetry Parameters vs Lap Time")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Detailed Data Table
    with st.expander("View Detailed Data"):
        st.subheader("Filtered Lap Data")
        display_cols = ['vehicle_number', 'lap', 'lap_time_seconds', 'speed', 'nmot',
                       'aps', 'gear', 'total_brake_pressure', 'Steering_Angle']
        display_cols = [c for c in display_cols if c in df_filtered.columns]
        st.dataframe(df_filtered[display_cols].sort_values('lap_time_seconds'))

    # Download Section
    st.markdown("---")
    st.header("Download Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name='filtered_lap_data.csv',
            mime='text/csv'
        )

    with col2:
        importance_csv = feature_importance.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Feature Importance",
            data=importance_csv,
            file_name='feature_importance.csv',
            mime='text/csv'
        )

    with col3:
        comparison_csv = comparison.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Fast vs Slow Analysis",
            data=comparison_csv,
            file_name='fast_vs_slow_comparison.csv',
            mime='text/csv'
        )

except FileNotFoundError as e:
    st.error("Required data files not found. Please run 'advanced_lap_optimization.py' first.")
    st.info("Run: `./venv/bin/python advanced_lap_optimization.py`")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
