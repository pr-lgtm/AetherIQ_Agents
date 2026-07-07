from agent_engine import run_aetheriq_query
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import base64
import os
import subprocess
import sys
import time

# --- PATH-AGNOSTIC BACKGROUND MCP RUNNER ---
@st.cache_resource
def start_background_mcp():
    print("Booting FastMCP Server in the background...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mcp_path = os.path.join(current_dir, "mcp_server.py")
    process = subprocess.Popen([sys.executable, mcp_path])
    time.sleep(2) 
    return process

start_background_mcp()

st.set_page_config(page_title="AetherIQ Command Center", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
bg_path = os.path.join(BASE_DIR, "background.png")
bg_base64 = get_base64_of_bin_file(bg_path)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
.stApp::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(10, 15, 25, 0.4); z-index: -1;
}}
section[data-testid="stSidebar"] {{
    background-color: rgba(15, 20, 30, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}}
section[data-testid="stSidebar"] label, .stTextInput label {{ color: #ffffff !important; font-weight: 600 !important; }}
.telemetry-card {{
    background: rgba(15, 22, 35, 0.8) !important; backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(0, 255, 102, 0.15) !important; border-radius: 12px; padding: 24px; margin-bottom: 20px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
}}
[data-testid="stPlotlyChart"] {{
    background: rgba(15, 22, 35, 0.8) !important; backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(0, 255, 102, 0.15) !important; border-radius: 12px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4); padding-top: 10px;
}}
.card-title {{ color: #a0aec0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; font-weight: 600; }}
.card-value {{ font-family: monospace; font-size: 2.5rem; font-weight: bold; color: #ffffff; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }}
.status-stable {{ color: #00ff66 !important; font-size: 0.9rem; font-weight: 600; background: rgba(0,255,102,0.1); padding: 4px 10px; border-radius: 4px;}}
.status-hazard {{ color: #ff3333 !important; font-size: 0.9rem; font-weight: 600; background: rgba(255,51,51,0.1); padding: 4px 10px; border-radius: 4px;}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background: rgba(15, 22, 35, 0.85); backdrop-filter: blur(10px); padding: 25px; border-radius: 12px; margin-bottom: 25px; border: 1px solid rgba(255,255,255,0.1);'>
<h1 style='color: #00ff66; margin: 0; font-weight: 800; letter-spacing: -0.5px;'>🌍 AetherIQ Geo-Spatial Command Center</h1>
<p style='color: #a0aec0; font-size: 1.1rem; margin-top: 5px; margin-bottom: 0;'>Predictive Urban Telemetry Operations & Multi-Agent Grid Diagnostics</p>
</div>
""", unsafe_allow_html=True)

api_key = st.text_input("🔑 Authenticate Google Gemini API Key (Required for AI Agents):", type="password")

@st.cache_resource
def load_models():
    reg = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_regressor.pkl'))
    clf = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_classifier.pkl'))
    feats = joblib.load(os.path.join(BASE_DIR, 'models', 'model_features.pkl'))
    return reg, clf, feats

try:
    regressor, classifier, feature_cols = load_models()
except Exception as e:
    st.error("Fatal Error: Models not found. Please run `python src/ml_pipeline.py` in your terminal first to generate the models.")
    st.stop()

st.sidebar.markdown("<h3 style='color:#00ff66; font-weight:800;'>⚙️ SENSOR CONFIGURATION</h3>", unsafe_allow_html=True)
zone_name = st.sidebar.selectbox("📍 Geospatial Target Sector", ["Downtown Core", "Industrial District", "Green Belt Park", "Residential Suburbs", "Commercial Hub"])
traffic_density = st.sidebar.slider("🚗 Congestion Vector (Traffic Density Index)", 0.0, 1.0, 0.85)
ambient_temp = st.sidebar.slider("🌡️ Thermal Vector (Ambient Temperature °C)", 15.0, 48.0, 38.5)
energy_consumption = st.sidebar.slider("⚡ Baseload Consumption Profile (MWh)", 10.0, 150.0, 112.0)
green_canopy = st.sidebar.slider("🌳 Mitigation Factor (Green Canopy %)", 0.0, 100.0, 12.0)
aqi = st.sidebar.slider("🏭 Particulate Density (Air Quality Index)", 10, 500, 185)
hour = st.sidebar.slider("⏱️ Temporal Phase Coordinate (Hour)", 0, 23, 15)

# Dynamically map UI inputs to whatever features ml_pipeline.py exported to prevent crashes
input_dict = {col: 0.0 for col in feature_cols}

if 'Traffic_Density_Index' in input_dict: input_dict['Traffic_Density_Index'] = traffic_density
if 'Energy_Consumption_MWh' in input_dict: input_dict['Energy_Consumption_MWh'] = energy_consumption
if 'Ambient_Temperature_C' in input_dict: input_dict['Ambient_Temperature_C'] = ambient_temp
if 'Air_Quality_Index_AQI' in input_dict: input_dict['Air_Quality_Index_AQI'] = aqi
if 'Green_Canopy_Coverage_Pct' in input_dict: input_dict['Green_Canopy_Coverage_Pct'] = green_canopy
if 'Hour_Sin' in input_dict: input_dict['Hour_Sin'] = np.sin(2 * np.pi * hour / 24.0)
if 'Hour_Cos' in input_dict: input_dict['Hour_Cos'] = np.cos(2 * np.pi * hour / 24.0)
if 'Thermal_Energy_Stress_Index' in input_dict: input_dict['Thermal_Energy_Stress_Index'] = energy_consumption * max(0, ambient_temp - 25.0)

if "Zone_ZONE_A" in input_dict and zone_name == "Downtown Core": input_dict['Zone_ZONE_A'] = 1.0
elif "Zone_ZONE_B" in input_dict and zone_name == "Industrial District": input_dict['Zone_ZONE_B'] = 1.0
elif "Zone_ZONE_C" in input_dict and zone_name == "Green Belt Park": input_dict['Zone_ZONE_C'] = 1.0
elif "Zone_ZONE_D" in input_dict and zone_name == "Residential Suburbs": input_dict['Zone_ZONE_D'] = 1.0
elif "Zone_ZONE_E" in input_dict and zone_name == "Commercial Hub": input_dict['Zone_ZONE_E'] = 1.0

input_df = pd.DataFrame([input_dict])

# Run Inference
predicted_carbon = regressor.predict(input_df)[0]
grid_risk_class = int(classifier.predict(input_df)[0])

col1, col2, col3 = st.columns([1, 1, 1.3])
with col1:
    carbon_status_html = "<span class='status-hazard'>▲ INTENSE METRIC IMPACT</span>" if predicted_carbon > 40 else "<span class='status-stable'> NOMINAL SENSOR STATE</span>"
    st.markdown(f"<div class='telemetry-card'><div class='card-title'>☁️ Carbon Emission Output</div><div class='card-value'>{predicted_carbon:.2f} <span style='font-size:1.2rem; color:#a0aec0;'>t/hr</span></div><div style='margin-top:15px;'>{carbon_status_html}</div></div>", unsafe_allow_html=True)
with col2:
    aqi_status_html = "<span class='status-hazard'>▲ CRITICAL POLLUTION LEVEL</span>" if aqi > 150 else "<span class='status-stable'> SAFE SATURATION SCALE</span>"
    st.markdown(f"<div class='telemetry-card'><div class='card-title'>🏭 Atmospheric Quality</div><div class='card-value'>{aqi}</div><div style='margin-top:15px;'>{aqi_status_html}</div></div>", unsafe_allow_html=True)
with col3:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=grid_risk_class, domain={'x': [0, 1], 'y': [0, 0.75]},
        gauge={'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#ffffff", 'tickvals': [0, 1], 'ticktext': ['SAFE', 'CRITICAL']},
               'bar': {'color': "#ff3333" if grid_risk_class == 1 else "#00ff66"}, 'bgcolor': "rgba(0, 0, 0, 0)",
               'borderwidth': 1, 'bordercolor': "rgba(255, 255, 255, 0.2)",
               'steps': [{'range': [0, 0.5], 'color': "rgba(0, 255, 102, 0.1)"}, {'range': [0.5, 1], 'color': "rgba(255, 51, 51, 0.1)"}]}))
    fig.update_layout(title={'text':"⚡ UTILITY GRID STABILITY", 'y': 0.9, 'x': 0.05, 'xanchor': 'left', 'yanchor': 'top', 'font': {'color': '#a0aec0', 'size': 14, 'family': 'Inter, sans-serif'}}, height=210, margin=dict(l=75, r=75, t=10, b=15), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "#ffffff", 'family': 'Inter, sans-serif'})
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- MULTI-AGENT DIAGNOSTIC ENGINE ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h3 style='color:#ffffff; font-weight:800;'>🧠 MULTI-AGENT DIAGNOSTIC ENGINE</h3>", unsafe_allow_html=True)

if "hitl_auth_required" not in st.session_state:
    st.session_state.hitl_auth_required = False
if "override_granted" not in st.session_state:
    st.session_state.override_granted = False

if st.button("RUN GENERATIVE DIAGNOSTICS LOG", use_container_width=True):
    if not api_key:
        st.error("Access Prohibited: Terminal requires valid API credentials.")
    else:
        with st.spinner("Synchronizing ML Pipeline outputs with ADK Orchestrator Graph..."):
            current_telemetry = {
                "zone_name": zone_name,
                "traffic_density": traffic_density,
                "ambient_temp": ambient_temp,
                "grid_load_mwh": energy_consumption,
                "green_canopy": green_canopy,
                "aqi": aqi,
                "hitl_authorized": st.session_state.override_granted
            }
            
            ml_results = {
                "predicted_carbon_tons": predicted_carbon,
                "grid_overload_predicted": grid_risk_class
            }
            
            # The query dynamically adapts based on the ML Risk Prediction
            if grid_risk_class == 1:
                user_query = f"CRITICAL CONGESTION WARNING for {zone_name}. Deploy emergency physical circuit breakers and downscale grid parameters."
            else:
                user_query = f"Provide standard spatial operational health assessment for {zone_name}."
                
            try:
                result = run_aetheriq_query(user_query, current_telemetry, ml_results, api_key)
                
                if result["status"] == "SECURITY_HALT":
                    st.session_state.hitl_auth_required = True
                    st.rerun()
                else:
                    st.session_state.hitl_auth_required = False
                    st.session_state.override_granted = False
                    st.markdown("<div class='telemetry-card'>", unsafe_allow_html=True)
                    st.markdown(f"**TACTICAL ACTION PLAN: {zone_name.upper()}**\n\n{result['message']}")
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as graph_err:
                st.error(f"ADK Runtime Exception: {str(graph_err)}")

# --- HUMAN-IN-THE-LOOP INTERACTION UI ---
if st.session_state.hitl_auth_required:
    st.markdown("""
    <div style='background: rgba(255, 51, 51, 0.12); border: 2px solid #ff3333; padding: 20px; border-radius: 8px; margin-top: 15px;'>
    <h4 style='color: #ff3333; margin-top: 0; font-weight:800;'>🚨 HUMAN-IN-THE-LOOP GUARDRAIL ACTIVATED</h4>
    <p style='color: #ffffff; font-size:14px;'>The AetherIQ Security Agent intercepted an infrastructure downscaling directive driven by high Random Forest grid overload scores. Manual oversight authorization required to override safety thresholds.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_auth, col_abort = st.columns(2)
    with col_auth:
        if st.button("🔓 APPROVE EXECUTIVE INTERVENTION", use_container_width=True):
            st.session_state.override_granted = True
            st.session_state.hitl_auth_required = False
            st.rerun()
    with col_abort:
        if st.button("🔒 ENFORCE SYSTEM LOCKDOWN", use_container_width=True):
            st.session_state.override_granted = False
            st.session_state.hitl_auth_required = False
            st.info("Emergency mitigation cancelled. Infrastructure locked down safely.")