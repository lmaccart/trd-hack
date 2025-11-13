"""
Universal Racing Dashboard - All Tracks Analysis
Compare and analyze performance across all race tracks
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np

# Page configuration
st.set_page_config(page_title="Universal Racing Dashboard", layout="wide", page_icon="🏁")

# Title
st.title("🏁 Universal Racing Analytics Dashboard")
st.markdown("### Multi-Track Performance Analysis")
st.markdown("**Comprehensive lap time optimization across all circuits**")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../../data/processed/all_tracks_combined.csv')
        with open('../../data/results/universal_analysis_results.json', 'r') as f:
            analysis_results = json.load(f)
        feature_importance = pd.read_csv('../../data/results/universal_feature_importance.csv')
        comparison = pd.read_csv('../../data/results/universal_fast_vs_slow.csv')
        track_stats = pd.read_csv('../../data/results/track_statistics.csv')
        return df, analysis_results, feature_importance, comparison, track_stats
    except FileNotFoundError as e:
        st.error(f"Data files not found. Please run the universal_data_processor.py and universal_lap_analysis.py first.")
        st.stop()

try:
    df, results, feature_importance, comparison, track_stats = load_data()

    # Sidebar - Track Selection
    st.sidebar.header("🏎️ Track Selection")

    all_tracks = ['All Tracks'] + sorted(df['track_name'].unique().tolist())
    selected_track = st.sidebar.selectbox("Select Track", all_tracks)

    # Filter data based on selection
    if selected_track == 'All Tracks':
        df_filtered = df
        st.sidebar.metric("Total Laps", f"{len(df):,}")
        st.sidebar.metric("Tracks", df['track'].nunique())
        st.sidebar.metric("Vehicles", df['vehicle_id'].nunique())
    else:
        df_filtered = df[df['track_name'] == selected_track]
        st.sidebar.metric("Total Laps", f"{len(df_filtered):,}")
        st.sidebar.metric("Races", df_filtered['race'].nunique())
        st.sidebar.metric("Vehicles", df_filtered['vehicle_id'].nunique())

    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🏁 Track Comparison",
        "🎯 Feature Analysis",
        "⚡ Fast vs Slow",
        "📈 Model Performance"
    ])

    # TAB 1: OVERVIEW
    with tab1:
        st.header("Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Laps Analyzed",
                f"{len(df_filtered):,}",
                help="Total number of racing laps across all selected tracks"
            )

        with col2:
            avg_lap_time = df_filtered['lap_time_seconds'].mean()
            st.metric(
                "Avg Lap Time",
                f"{avg_lap_time:.2f}s",
                help="Average lap time across all selected laps"
            )

        with col3:
            best_lap_time = df_filtered['lap_time_seconds'].min()
            st.metric(
                "Best Lap Time",
                f"{best_lap_time:.2f}s",
                help="Fastest lap time recorded"
            )

        with col4:
            if selected_track == 'All Tracks':
                st.metric(
                    "Tracks Analyzed",
                    df_filtered['track'].nunique(),
                    help="Number of different race tracks"
                )
            else:
                st.metric(
                    "Races",
                    df_filtered['race'].nunique(),
                    help="Number of races at this track"
                )

        st.markdown("---")

        # Lap time distribution
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Lap Time Distribution")
            fig = px.histogram(
                df_filtered,
                x='lap_time_seconds',
                nbins=50,
                title='Distribution of Lap Times',
                labels={'lap_time_seconds': 'Lap Time (seconds)', 'count': 'Frequency'},
                color_discrete_sequence=['#636EFA']
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Lap Times by Track")
            fig = px.box(
                df if selected_track == 'All Tracks' else df_filtered,
                x='track_name',
                y='lap_time_seconds',
                title='Lap Time Ranges by Track',
                labels={'track_name': 'Track', 'lap_time_seconds': 'Lap Time (seconds)'},
                color='track_name'
            )
            fig.update_layout(showlegend=False, height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        # Track statistics table
        if selected_track == 'All Tracks':
            st.subheader("Track Statistics Summary")
            display_stats = track_stats.copy()
            display_stats['avg_lap_time'] = display_stats['avg_lap_time'].round(2)
            display_stats['min_lap_time'] = display_stats['min_lap_time'].round(2)
            display_stats['std_lap_time'] = display_stats['std_lap_time'].round(2)
            st.dataframe(
                display_stats,
                use_container_width=True,
                hide_index=True
            )

    # TAB 2: TRACK COMPARISON
    with tab2:
        st.header("Track-by-Track Comparison")

        if selected_track == 'All Tracks':
            # Compare tracks
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Average Lap Times by Track")
                track_avg = track_stats.sort_values('avg_lap_time')
                fig = px.bar(
                    track_avg,
                    x='track_name',
                    y='avg_lap_time',
                    title='Average Lap Times',
                    labels={'track_name': 'Track', 'avg_lap_time': 'Avg Lap Time (s)'},
                    color='avg_lap_time',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Lap Count by Track")
                fig = px.bar(
                    track_stats.sort_values('total_laps', ascending=False),
                    x='track_name',
                    y='total_laps',
                    title='Total Laps Analyzed',
                    labels={'track_name': 'Track', 'total_laps': 'Total Laps'},
                    color='total_laps',
                    color_continuous_scale='blues'
                )
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Lap time trends over time
            st.subheader("Lap Time Trends Across All Tracks")
            fig = px.scatter(
                df.sample(min(5000, len(df))),  # Sample for performance
                x='lap',
                y='lap_time_seconds',
                color='track_name',
                title='Lap Times Throughout Races',
                labels={'lap': 'Lap Number', 'lap_time_seconds': 'Lap Time (s)', 'track_name': 'Track'},
                opacity=0.6
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Single track analysis
            st.subheader(f"{selected_track} - Detailed Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Lap Time Evolution")
                fig = px.scatter(
                    df_filtered,
                    x='lap',
                    y='lap_time_seconds',
                    color='race',
                    title=f'Lap Times by Lap Number',
                    labels={'lap': 'Lap Number', 'lap_time_seconds': 'Lap Time (s)', 'race': 'Race'},
                    opacity=0.6
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Vehicle Performance")
                vehicle_avg = df_filtered.groupby('vehicle_id')['lap_time_seconds'].agg(['mean', 'min', 'count']).reset_index()
                vehicle_avg = vehicle_avg.sort_values('mean').head(20)
                fig = px.bar(
                    vehicle_avg,
                    x='vehicle_id',
                    y='mean',
                    error_y=None,
                    title='Top 20 Fastest Vehicles (Avg Lap Time)',
                    labels={'vehicle_id': 'Vehicle ID', 'mean': 'Avg Lap Time (s)'},
                    color='mean',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

    # TAB 3: FEATURE ANALYSIS
    with tab3:
        st.header("Feature Importance Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Top Features Influencing Lap Time")
            top_n = st.slider("Number of features to display", 5, 30, 15)

            fig = px.bar(
                feature_importance.head(top_n),
                x='importance',
                y='feature',
                orientation='h',
                title=f'Top {top_n} Most Important Features',
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='reds'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Key Insights")
            st.markdown(f"""
            **Model Performance:**
            - R² Score: `{results['model_performance']['r2_score']:.4f}`
            - RMSE: `{results['model_performance']['rmse']:.3f}s`
            - MAE: `{results['model_performance']['mae']:.3f}s`

            **Top 5 Features:**
            """)

            for idx, row in feature_importance.head(5).iterrows():
                st.markdown(f"**{idx+1}. {row['feature']}**")
                st.progress(float(row['importance']))

    # TAB 4: FAST VS SLOW
    with tab4:
        st.header("Fast vs Slow Lap Analysis")

        st.markdown("""
        Comparing the **fastest 10%** of laps vs the **slowest 10%** to identify key performance differentiators.
        """)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Biggest Differences Between Fast and Slow Laps")
            top_diff = st.slider("Number of features to compare", 5, 20, 10, key='diff_slider')

            fig = px.bar(
                comparison.head(top_diff),
                x='pct_difference',
                y='feature',
                orientation='h',
                title=f'Top {top_diff} Differentiating Features (% Difference)',
                labels={'pct_difference': 'Percent Difference (%)', 'feature': 'Feature'},
                color='pct_difference',
                color_continuous_scale='rdbu',
                color_continuous_midpoint=0
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Top Differentiators")
            for idx, row in comparison.head(10).iterrows():
                direction = "🔴" if row['pct_difference'] > 0 else "🟢"
                st.metric(
                    f"{direction} {row['feature']}",
                    f"{abs(row['pct_difference']):.1f}%",
                    delta=None
                )

        # Detailed comparison table
        st.subheader("Detailed Comparison Table")
        comparison_display = comparison.head(20).copy()
        comparison_display['pct_difference'] = comparison_display['pct_difference'].round(2)
        comparison_display['fast_laps_mean'] = comparison_display['fast_laps_mean'].round(2)
        comparison_display['slow_laps_mean'] = comparison_display['slow_laps_mean'].round(2)
        st.dataframe(comparison_display, use_container_width=True, hide_index=True)

    # TAB 5: MODEL PERFORMANCE
    with tab5:
        st.header("Model Performance & Validation")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Metrics")
            metrics = results['model_performance']

            st.metric("R² Score", f"{metrics['r2_score']:.4f}", help="Coefficient of determination - how well the model explains variance")
            st.metric("RMSE", f"{metrics['rmse']:.3f}s", help="Root Mean Squared Error - average prediction error")
            st.metric("MAE", f"{metrics['mae']:.3f}s", help="Mean Absolute Error - average absolute prediction error")

            st.markdown("---")
            st.markdown(f"""
            **Training Data:**
            - Training samples: `{metrics['training_samples']:,}`
            - Test samples: `{metrics['test_samples']:,}`
            - Split: 80/20
            """)

        with col2:
            st.subheader("Performance Interpretation")

            r2 = metrics['r2_score']
            if r2 > 0.8:
                performance = "Excellent"
                color = "green"
            elif r2 > 0.6:
                performance = "Good"
                color = "blue"
            elif r2 > 0.4:
                performance = "Moderate"
                color = "orange"
            else:
                performance = "Needs Improvement"
                color = "red"

            st.markdown(f"""
            <div style='padding: 20px; background-color: {color}; color: white; border-radius: 10px;'>
                <h3>Model Performance: {performance}</h3>
                <p>The model explains {r2*100:.1f}% of the variance in lap times.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            st.markdown("""
            **Key Findings:**
            - The model can predict lap times with reasonable accuracy
            - Telemetry features have significant impact on performance
            - Feature importance reveals optimization opportunities
            """)

    # Footer
    st.markdown("---")
    st.markdown(f"""
    **Data Summary:** {len(df):,} laps | {df['track'].nunique()} tracks | {df['vehicle_id'].nunique()} vehicles |
    Generated with Universal Racing Analytics
    """)

except Exception as e:
    st.error(f"Error loading dashboard: {str(e)}")
    import traceback
    st.code(traceback.format_exc())
