import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import statsmodels.api as sm

# Page configuration
st.set_page_config(page_title="Lap Time Telemetry Analysis", layout="wide")

# Title
st.title("🏁 Lap Time vs Telemetry Correlation Analysis")
st.markdown("### Indianapolis Motor Speedway - R1 Race Data")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('merged_lap_telemetry.csv')
    with open('top_correlations.json', 'r') as f:
        top_corr = json.load(f)
    return df, top_corr

try:
    df, top_corr = load_data()

    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Laps", len(df))
    with col2:
        st.metric("Vehicles", df['vehicle_number'].nunique())
    with col3:
        st.metric("Avg Lap Time", f"{df['lap_time_seconds'].mean():.2f}s")
    with col4:
        st.metric("Best Lap Time", f"{df['lap_time_seconds'].min():.2f}s")

    st.markdown("---")

    # Display top correlations
    st.header("🔍 Top 3 Telemetry Variables Correlated with Lap Time")

    top_vars = top_corr['top_3_variables']
    corr_values = top_corr['correlations']

    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        var1 = top_vars[0]
        st.metric(
            label=f"1. {var1.upper()}",
            value=f"Correlation: {corr_values[var1]:.4f}",
            delta="Strongest"
        )

    with col2:
        var2 = top_vars[1]
        st.metric(
            label=f"2. {var2.upper()}",
            value=f"Correlation: {corr_values[var2]:.4f}",
            delta="2nd Strongest"
        )

    with col3:
        var3 = top_vars[2]
        st.metric(
            label=f"3. {var3.upper()}",
            value=f"Correlation: {corr_values[var3]:.4f}",
            delta="3rd Strongest"
        )

    st.markdown("---")

    # Variable descriptions
    var_descriptions = {
        'nmot': 'Engine RPM (Revolutions Per Minute)',
        'speed': 'Vehicle Speed',
        'aps': 'Accelerator Pedal Sensor (Throttle Position)',
        'gear': 'Transmission Gear',
        'accx_can': 'Longitudinal Acceleration',
        'accy_can': 'Lateral Acceleration',
        'pbrake_f': 'Front Brake Pressure',
        'pbrake_r': 'Rear Brake Pressure',
        'Steering_Angle': 'Steering Wheel Angle'
    }

    # Create scatter plots for top 3 correlations
    st.header("📊 Scatter Plots: Lap Time vs Top 3 Variables")

    for i, var in enumerate(top_vars, 1):
        st.subheader(f"{i}. {var.upper()} - {var_descriptions.get(var, var)}")
        st.markdown(f"**Correlation coefficient: {corr_values[var]:.4f}**")

        # Create scatter plot with trendline
        fig = px.scatter(
            df,
            x=var,
            y='lap_time_seconds',
            color='vehicle_number',
            title=f'Lap Time vs {var.upper()}',
            labels={
                var: f'{var_descriptions.get(var, var)} ({var})',
                'lap_time_seconds': 'Lap Time (seconds)',
                'vehicle_number': 'Vehicle #'
            },
            hover_data=['vehicle_number', 'lap'],
            trendline='ols',
            height=500
        )

        fig.update_layout(
            xaxis_title=f'{var_descriptions.get(var, var)}',
            yaxis_title='Lap Time (seconds)',
            hovermode='closest'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add interpretation
        if corr_values[var] < 0:
            st.info(f"💡 **Interpretation**: Higher {var} values are associated with faster (lower) lap times. This suggests that maintaining higher {var} during a lap leads to better performance.")
        else:
            st.info(f"💡 **Interpretation**: Higher {var} values are associated with slower (higher) lap times. This suggests that maintaining lower {var} during a lap leads to better performance.")

        st.markdown("---")

    # Additional analysis: Distribution of lap times
    st.header("📈 Lap Time Distribution")
    fig_hist = px.histogram(
        df,
        x='lap_time_seconds',
        nbins=30,
        title='Distribution of Lap Times',
        labels={'lap_time_seconds': 'Lap Time (seconds)'},
        color_discrete_sequence=['#636EFA']
    )
    fig_hist.update_layout(showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

    # Show raw data
    with st.expander("📋 View Raw Merged Data"):
        st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Merged Data as CSV",
        data=csv,
        file_name='merged_lap_telemetry.csv',
        mime='text/csv'
    )

except FileNotFoundError as e:
    st.error("⚠️ Data files not found. Please run 'merge_and_analyze.py' first to generate the necessary data files.")
    st.info("Run: `python merge_and_analyze.py`")
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)
