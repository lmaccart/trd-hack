import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np

# Page configuration
st.set_page_config(page_title="Lap Time + Weather Analysis", layout="wide")

# Title
st.title("Lap Time & Weather Optimization Dashboard")
st.markdown("### Indianapolis Motor Speedway - Race 1")
st.markdown("**How Weather Impacts Performance**")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('../../data/processed/merged_lap_telemetry_with_weather.csv')
    with open('../../data/results/optimization_results.json', 'r') as f:
        opt_results = json.load(f)
    with open('../../data/results/weather_analysis.json', 'r') as f:
        weather_results = json.load(f)
    feature_importance = pd.read_csv('../../data/results/feature_importance.csv')
    return df, opt_results, weather_results, feature_importance

try:
    df, opt_results, weather_results, feature_importance = load_data()

    # Filter out extreme outliers for better visualization
    df_clean = df[(df['lap_time_seconds'] > 80) & (df['lap_time_seconds'] < 200)].copy()

    # ========================================================================
    # WEATHER OVERVIEW
    # ========================================================================
    st.header("Weather Conditions Overview")

    weather_summary = weather_results['weather_summary']

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        air_temp_range = weather_summary['air_temp_range']
        st.metric("Air Temp Range",
                  f"{air_temp_range[0]:.1f}°C - {air_temp_range[1]:.1f}°C",
                  f"{air_temp_range[1] - air_temp_range[0]:.1f}°C change")

    with col2:
        track_temp_range = weather_summary['track_temp_range']
        st.metric("Track Temp Range",
                  f"{track_temp_range[0]:.1f}°C - {track_temp_range[1]:.1f}°C",
                  f"{track_temp_range[1] - track_temp_range[0]:.1f}°C change")

    with col3:
        humidity_range = weather_summary['humidity_range']
        st.metric("Humidity Range",
                  f"{humidity_range[0]:.1f}% - {humidity_range[1]:.1f}%",
                  f"{humidity_range[1] - humidity_range[0]:.1f}% change")

    with col4:
        wind_range = weather_summary['wind_speed_range']
        st.metric("Wind Speed Range",
                  f"{wind_range[0]:.1f} - {wind_range[1]:.1f} m/s",
                  f"{wind_range[1] - wind_range[0]:.1f} m/s variation")

    with col5:
        rain_status = "YES" if weather_summary['rain_occurred'] else "NO"
        st.metric("Rain During Race", rain_status,
                  "Dry conditions" if rain_status == "NO" else "Wet conditions")

    st.info(f"Weather data matched to {weather_results['laps_with_weather']} out of {weather_results['total_laps']} laps ({weather_results['laps_with_weather']/weather_results['total_laps']*100:.1f}%)")

    st.markdown("---")

    # ========================================================================
    # WEATHER CORRELATIONS
    # ========================================================================
    st.header("Weather Impact on Lap Time")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Weather Correlations with Lap Time")

        weather_corr = weather_results['weather_correlations']
        weather_corr_df = pd.DataFrame(list(weather_corr.items()), columns=['Weather Variable', 'Correlation'])
        weather_corr_df = weather_corr_df.sort_values('Correlation', key=abs, ascending=False)

        fig_weather_corr = px.bar(
            weather_corr_df,
            x='Correlation',
            y='Weather Variable',
            orientation='h',
            title='Weather Variable Correlations',
            color='Correlation',
            color_continuous_scale='RdBu_r',
            range_color=[-0.3, 0.3]
        )
        fig_weather_corr.update_layout(height=400)
        st.plotly_chart(fig_weather_corr, use_container_width=True)

        # Interpretation
        st.markdown("**Interpretation:**")
        for var, corr in weather_corr.items():
            if abs(corr) > 0.1:
                direction = "slower" if corr > 0 else "faster"
                st.markdown(f"- **{var}**: Higher values → {direction} laps (corr: {corr:+.3f})")

    with col2:
        st.subheader("Fast vs Slow Lap Weather Conditions")

        # Calculate fast vs slow
        fastest_threshold = df_clean['lap_time_seconds'].quantile(0.10)
        slowest_threshold = df_clean['lap_time_seconds'].quantile(0.90)

        fastest_laps = df_clean[df_clean['lap_time_seconds'] <= fastest_threshold]
        slowest_laps = df_clean[df_clean['lap_time_seconds'] >= slowest_threshold]

        weather_vars = ['AIR_TEMP', 'TRACK_TEMP', 'HUMIDITY', 'WIND_SPEED']
        comparison_data = []

        for var in weather_vars:
            if var in fastest_laps.columns and fastest_laps[var].notna().sum() > 0:
                fast_mean = fastest_laps[var].mean()
                slow_mean = slowest_laps[var].mean()
                comparison_data.append({
                    'Variable': var,
                    'Fast Laps': fast_mean,
                    'Slow Laps': slow_mean
                })

        comparison_df = pd.DataFrame(comparison_data)

        fig_comparison = go.Figure()
        fig_comparison.add_trace(go.Bar(
            name='Fast Laps',
            x=comparison_df['Variable'],
            y=comparison_df['Fast Laps'],
            marker_color='green'
        ))
        fig_comparison.add_trace(go.Bar(
            name='Slow Laps',
            x=comparison_df['Variable'],
            y=comparison_df['Slow Laps'],
            marker_color='red'
        ))

        fig_comparison.update_layout(
            title='Average Weather: Fast vs Slow Laps',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

    st.markdown("---")

    # ========================================================================
    # WEATHER TIME SERIES
    # ========================================================================
    st.header("Weather & Performance Evolution")

    # Convert timestamp for plotting
    df_clean['timestamp_plot'] = pd.to_datetime(df_clean['timestamp'])

    col1, col2 = st.columns(2)

    with col1:
        # Track temperature over time with lap times
        fig_temp_time = make_subplots(specs=[[{"secondary_y": True}]])

        fig_temp_time.add_trace(
            go.Scatter(
                x=df_clean['timestamp_plot'],
                y=df_clean['TRACK_TEMP'],
                name='Track Temp (°C)',
                line=dict(color='red', width=2),
                mode='lines'
            ),
            secondary_y=False
        )

        fig_temp_time.add_trace(
            go.Scatter(
                x=df_clean['timestamp_plot'],
                y=df_clean['lap_time_seconds'],
                name='Lap Time (s)',
                mode='markers',
                marker=dict(size=4, opacity=0.5, color='blue')
            ),
            secondary_y=True
        )

        fig_temp_time.update_xaxes(title_text="Time")
        fig_temp_time.update_yaxes(title_text="Track Temp (°C)", secondary_y=False)
        fig_temp_time.update_yaxes(title_text="Lap Time (s)", secondary_y=True)
        fig_temp_time.update_layout(title="Track Temperature & Lap Times Over Session", height=400)

        st.plotly_chart(fig_temp_time, use_container_width=True)

    with col2:
        # Wind speed over time with lap times
        fig_wind_time = make_subplots(specs=[[{"secondary_y": True}]])

        fig_wind_time.add_trace(
            go.Scatter(
                x=df_clean['timestamp_plot'],
                y=df_clean['WIND_SPEED'],
                name='Wind Speed (m/s)',
                line=dict(color='green', width=2),
                mode='lines'
            ),
            secondary_y=False
        )

        fig_wind_time.add_trace(
            go.Scatter(
                x=df_clean['timestamp_plot'],
                y=df_clean['lap_time_seconds'],
                name='Lap Time (s)',
                mode='markers',
                marker=dict(size=4, opacity=0.5, color='blue')
            ),
            secondary_y=True
        )

        fig_wind_time.update_xaxes(title_text="Time")
        fig_wind_time.update_yaxes(title_text="Wind Speed (m/s)", secondary_y=False)
        fig_wind_time.update_yaxes(title_text="Lap Time (s)", secondary_y=True)
        fig_wind_time.update_layout(title="Wind Speed & Lap Times Over Session", height=400)

        st.plotly_chart(fig_wind_time, use_container_width=True)

    st.markdown("---")

    # ========================================================================
    # WEATHER SCATTER PLOTS
    # ========================================================================
    st.header("Lap Time vs Weather Conditions")

    # Create 2x2 grid of scatter plots
    weather_features = ['TRACK_TEMP', 'AIR_TEMP', 'HUMIDITY', 'WIND_SPEED']
    weather_labels = {
        'TRACK_TEMP': 'Track Temperature (°C)',
        'AIR_TEMP': 'Air Temperature (°C)',
        'HUMIDITY': 'Humidity (%)',
        'WIND_SPEED': 'Wind Speed (m/s)'
    }

    fig_scatter = make_subplots(
        rows=2, cols=2,
        subplot_titles=[weather_labels[f] for f in weather_features]
    )

    for i, feat in enumerate(weather_features):
        row = (i // 2) + 1
        col = (i % 2) + 1

        if feat in df_clean.columns:
            # Filter valid data
            valid_data = df_clean[df_clean[feat].notna()]

            fig_scatter.add_trace(
                go.Scatter(
                    x=valid_data[feat],
                    y=valid_data['lap_time_seconds'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        opacity=0.6,
                        color=valid_data['lap_time_seconds'],
                        colorscale='RdYlGn_r',
                        showscale=(i == 0)  # Only show colorbar once
                    ),
                    showlegend=False,
                    name=feat
                ),
                row=row, col=col
            )

        fig_scatter.update_xaxes(title_text=weather_labels[feat], row=row, col=col)
        fig_scatter.update_yaxes(title_text='Lap Time (s)', row=row, col=col)

    fig_scatter.update_layout(height=800, title_text="Weather Variables vs Lap Time")
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # ========================================================================
    # COMBINED CORRELATIONS
    # ========================================================================
    st.header("All Features: Telemetry + Weather")

    # Combine telemetry and weather correlations
    all_correlations = opt_results['top_correlations'].copy()
    all_correlations.update(weather_results['weather_correlations'])

    corr_df = pd.DataFrame(list(all_correlations.items()), columns=['Feature', 'Correlation'])
    corr_df['Type'] = corr_df['Feature'].apply(
        lambda x: 'Weather' if x in weather_results['weather_correlations'] else 'Telemetry'
    )
    corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False).head(20)

    fig_all_corr = px.bar(
        corr_df,
        x='Correlation',
        y='Feature',
        orientation='h',
        title='Top 20 Features Correlated with Lap Time (Telemetry + Weather)',
        color='Type',
        color_discrete_map={'Telemetry': '#636EFA', 'Weather': '#EF553B'}
    )
    fig_all_corr.update_layout(height=600)
    st.plotly_chart(fig_all_corr, use_container_width=True)

    st.markdown("---")

    # ========================================================================
    # KEY INSIGHTS
    # ========================================================================
    st.header("Key Weather Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Optimal Weather Conditions")
        st.markdown("Based on fast lap analysis:")

        fast_weather = fastest_laps[weather_vars].mean()
        for var in weather_vars:
            if var in fast_weather.index:
                unit = {'AIR_TEMP': '°C', 'TRACK_TEMP': '°C', 'HUMIDITY': '%', 'WIND_SPEED': 'm/s'}
                st.markdown(f"- **{var}**: {fast_weather[var]:.1f} {unit.get(var, '')}")

    with col2:
        st.subheader("Weather Impact Ranking")

        weather_impact = [(var, abs(corr)) for var, corr in weather_results['weather_correlations'].items()]
        weather_impact.sort(key=lambda x: x[1], reverse=True)

        for i, (var, impact) in enumerate(weather_impact, 1):
            strength = "Strong" if impact > 0.3 else "Moderate" if impact > 0.1 else "Weak"
            st.markdown(f"{i}. **{var}**: {strength} impact ({impact:.3f})")

    st.markdown("---")

    # ========================================================================
    # DOWNLOAD SECTION
    # ========================================================================
    st.header("Download Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Full Data with Weather",
            data=csv,
            file_name='lap_telemetry_weather.csv',
            mime='text/csv'
        )

    with col2:
        weather_csv = pd.DataFrame(weather_results['weather_correlations'].items(),
                                   columns=['Variable', 'Correlation']).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Weather Correlations",
            data=weather_csv,
            file_name='weather_correlations.csv',
            mime='text/csv'
        )

    with col3:
        # Load weather comparison if exists
        try:
            weather_comp = pd.read_csv('../../data/results/weather_fast_vs_slow.csv')
            weather_comp_csv = weather_comp.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Weather Fast vs Slow",
                data=weather_comp_csv,
                file_name='weather_fast_vs_slow.csv',
                mime='text/csv'
            )
        except:
            pass

except FileNotFoundError as e:
    st.error("Required data files not found. Please run 'integrate_weather.py' first.")
    st.info("Run: `./venv/bin/python integrate_weather.py`")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)

# Sidebar
st.sidebar.header("About Weather Analysis")
st.sidebar.markdown("""
This dashboard shows how weather conditions affect lap times at Indianapolis Motor Speedway.

**Weather Variables:**
- **Air Temperature**: Affects air density and engine power
- **Track Temperature**: Affects tire grip and degradation
- **Humidity**: Affects air density and engine performance
- **Wind Speed**: Affects drag and downforce
- **Pressure**: Affects air density

**Key Findings:**
Weather had relatively small impact on lap times compared to driving technique and vehicle setup, but track temperature and wind showed measurable correlations.
""")
