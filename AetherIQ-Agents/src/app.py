from agent_engine import run_aetheriq_query
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import base64
import os

#--- Page Configuration
st.set_page_config(page_title="AetherIQ Command Center", layout="wide", initial_sidebar_state="expanded")

#--- Background Image Encoding Function ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

bg_path = "background.png" if os.path.exists("background.png") else "../background.png"
bg_base64 = get_base64_of_bin_file(bg_path)

#--- Dynamic CSS Injection for Background, Fonts, & Cards ---
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
.stApp {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
.stApp::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(10, 15, 25, 0.4);
    z-index: -1;
}}
section[data-testid="stSidebar"] {{
    background-color: rgba(15, 20, 30, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}}
section[data-testid="stSidebar"] label, .stTextInput label {{
    color: #ffffff !important;
    font-weight: 600 !important;
}}
.telemetry-card {{
    background: rgba(15, 22, 35, 0.8) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(0, 255, 102, 0.15) !important;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
}}
[data-testid="stPlotlyChart"] {{
    background: rgba(15, 22, 35, 0.8) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(0, 255, 102, 0.15) !important;
    border-radius: 12px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    padding-top: 10px;
}}
.card-title {{
    color: #a0aec0;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    font-weight: 600;
}}
.card-value {{
    font-family: monospace;
    font-size: 2.5rem;
    font-weight: bold;
    color: #ffffff;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}}
.status-stable {{ color: #00ff66 !important; font-size: 0.9rem; font-weight: 600; background: rgba(0,255,102,0.1); padding: 4px 10px; border-radius: 4px;}}
.status-hazard {{ color: #ff3333 !important; font-size: 0.9rem; font-weight: 600; background: rgba(255,51,51,0.1); padding: 4px 10px; border-radius: 4px;}}
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div style='background: rgba(15, 22, 35, 0.85); backdrop-filter: blur(10px); padding: 25px; border-radius: 12px; margin-bottom: 25px; border: 1px solid rgba(255,255,255,0.1);'>
<h1 style='color: #00ff66; margin: 0; font-weight: 800; letter-spacing: -0.5px;'>🌍 AetherIQ Geo-Spatial Command Center</h1>
<p style='color: #a0aec0; font-size: 1.1rem; margin-top: 5px; margin-bottom: 0;'>Predictive Urban Telemetry Operations & Multi-Agent Grid Diagnostics</p>
</div>
""", unsafe_allow_html=True)

# API Authentication Key Section
api_key = st.text_input("🔑 Authenticate Google Gemini API Key (Required for AI Agents):", type="password")

#--- Model Loading Infrastructure
@st.cache_resource
def load_models():
    regressor = joblib.load('models/rf_regressor.pkl')
    classifier = joblib.load('models/rf_classifier.pkl')
    features = joblib.load('models/model_features.pkl')
    return regressor, classifier, features

try:
    regressor, classifier, feature_cols = load_models()
except Exception as e:
    st.error("Fatal Error: Machine Learning analytical layers could not be resolved from local storage.")
    st.stop()

#--- Sidebar Controls ---
st.sidebar.markdown("<h3 style='color:#00ff66; font-weight:800;'>⚙️ SENSOR CONFIGURATION</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#a0aec0;'>Adjust live telemetry to simulate environmental conditions.</p>", unsafe_allow_html=True)

zone_name = st.sidebar.selectbox("📍 Geospatial Target Sector", ["Downtown Core", "Industrial District", "Green Belt Park", "Residential Suburbs", "Commercial Hub"])
traffic_density = st.sidebar.slider("🚗 Congestion Vector (Traffic Density Index)", 0.0, 1.0, 0.85)
ambient_temp = st.sidebar.slider("🌡️ Thermal Vector (Ambient Temperature °C)", 15.0, 48.0, 38.5)
energy_consumption = st.sidebar.slider("⚡ Baseload Consumption Profile (MWh)", 10.0, 150.0, 112.0)
green_canopy = st.sidebar.slider("🌳 Mitigation Factor (Green Canopy %)", 0.0, 100.0, 12.0)
aqi = st.sidebar.slider("🏭 Particulate Density (Air Quality Index)", 10, 500, 185)
hour = st.sidebar.slider("⏱️ Temporal Phase Coordinate (Hour)", 0, 23, 15)

# Dynamic Feature Array Preprocessing
input_df = pd.DataFrame(columns=feature_cols)
input_df.loc[0] = 0.0
input_df['Traffic_Density_Index'] = traffic_density
input_df['Energy_Consumption_MWh'] = energy_consumption
input_df['Ambient_Temperature_C'] = ambient_temp
input_df['Air_Quality_Index_AQI'] = aqi
input_df['Green_Canopy_Coverage_Pct'] = green_canopy
input_df['Hour_Sin'] = np.sin(2 * np.pi * hour / 24.0)
input_df['Hour_Cos'] = np.cos(2 * np.pi * hour / 24.0)
input_df['Thermal_Energy_Stress_Index'] = energy_consumption * max(0, ambient_temp - 25.0)

if "Zone_ZONE_A" in feature_cols and zone_name == "Downtown Core": input_df['Zone_ZONE_A'] = 1.0
elif "Zone_ZONE_B" in feature_cols and zone_name == "Industrial District": input_df['Zone_ZONE_B'] = 1.0
elif "Zone_ZONE_C" in feature_cols and zone_name == "Green Belt Park": input_df['Zone_ZONE_C'] = 1.0
elif "Zone_ZONE_D" in feature_cols and zone_name == "Residential Suburbs": input_df['Zone_ZONE_D'] = 1.0
elif "Zone_ZONE_E" in feature_cols and zone_name == "Commercial Hub": input_df['Zone_ZONE_E'] = 1.0

#--- Compute Predictive Models
predicted_carbon = regressor.predict(input_df)[0]
grid_risk_class = classifier.predict(input_df)[0]

#--- Tactical Layout Matrix --
col1, col2, col3 = st.columns([1, 1, 1.3])

with col1:
    carbon_status_html = "<span class='status-hazard'>▲ INTENSE METRIC IMPACT</span>" if predicted_carbon > 40 else "<span class='status-stable'> NOMINAL SENSOR STATE</span>"
    st.markdown(f"""
    <div class='telemetry-card'>
    <div class='card-title'>☁️ Carbon Emission Output</div>
    <div class='card-value'>{predicted_carbon:.2f} <span style='font-size:1.2rem; color:#a0aec0;'>t/hr</span></div>
    <div style='margin-top:15px;'>{carbon_status_html}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    aqi_status_html = "<span class='status-hazard'>▲ CRITICAL POLLUTION LEVEL</span>" if aqi > 150 else "<span class='status-stable'> SAFE SATURATION SCALE</span>"
    st.markdown(f"""
    <div class='telemetry-card'>
    <div class='card-title'>🏭 Atmospheric Quality</div>
    <div class='card-value'>{aqi}</div>
    <div style='margin-top:15px;'>{aqi_status_html}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=grid_risk_class,
        domain={'x': [0, 1], 'y': [0, 0.75]},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#ffffff", 'tickvals': [0, 1], 'ticktext': ['SAFE', 'CRITICAL']},
            'bar': {'color': "#ff3333" if grid_risk_class == 1 else "#00ff66"},
            'bgcolor': "rgba(0, 0, 0, 0)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.2)",
            'steps': [
                {'range': [0, 0.5], 'color': "rgba(0, 255, 102, 0.1)"},
                {'range': [0.5, 1], 'color': "rgba(255, 51, 51, 0.1)"}
            ],
        }
    ))
    fig.update_layout(
        title={'text':"⚡ UTILITY GRID STABILITY", 'y': 0.9, 'x': 0.05, 'xanchor': 'left', 'yanchor': 'top', 'font': {'color': '#a0aec0', 'size': 14, 'family': 'Inter, sans-serif'}},
        height=210,
        margin=dict(l=75, r=75, t=10, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#ffffff", 'family': 'Inter, sans-serif'}
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# Decision Intelligence Autonomous AI Matrix (ADK INTEGRATION)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#ffffff; font-weight:800;'>🧠 MULTI-AGENT DIAGNOSTIC ENGINE</h3>", unsafe_allow_html=True)

if st.button("RUN GENERATIVE DIAGNOSTICS LOG", use_container_width=True):
    if not api_key:
        st.error("Access Prohibited: Local machine terminal lacks authorized Gemini credentials. Please insert your API key at the top.")
    else:
        with st.spinner("Routing query to ADK Orchestrator & Analyst Agents..."):
            
            # Package live telemetry for the multi-agent system
            current_telemetry = {
                "zone_name": zone_name,
                "traffic_density": traffic_density,
                "ambient_temp": ambient_temp,
                "grid_load_mwh": energy_consumption,
                "green_canopy": green_canopy,
                "aqi": aqi,
                "predicted_carbon_tons": round(predicted_carbon, 2)
            }
            
            # Formulate the instruction for the Orchestrator
            user_query = f"Assess this live telemetry for {zone_name}. Provide a 3-step tactical risk management action plan for city engineers."
            
            try:
                # Call the custom Agent Workflow we built!
                agent_response = run_aetheriq_query(user_query, current_telemetry, api_key)
                
                st.markdown("<div class='telemetry-card'>", unsafe_allow_html=True)
                st.markdown(f"**TACTICAL ACTION PLAN: {zone_name.upper()}**\n\n{agent_response}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Multi-Agent Execution Failed: Ensure your API key is valid. Error details: {str(e)}")