import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_smart_city_telemetry(num_samples=2000, output_path='data/urban_environmental_telemetry.csv'):
    """Generates a mathematically dense, highly correlated urban sensor telemetry dataset."""
    print("Phase 1: Launching Synthesizer Engine...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    
    zones = {
        'ZONE_A': ('Downtown Core', 12.5),
        'ZONE_B': ('Industrial District', 4.2),
        'ZONE_C': ('Green Belt Park', 78.3),
        'ZONE_D': ('Residential Suburbs', 35.0),
        'ZONE_E': ('Commercial Hub', 15.8)
    }
    zone_keys = list(zones.keys())
    data_records = []
    start_time = datetime(2026, 6, 1, 0, 0, 0)

    for i in range(num_samples):
        timestamp = start_time + timedelta(minutes=15 * i)
        zone_id = np.random.choice(zone_keys)
        zone_name, canopy = zones[zone_id]
        hour = timestamp.hour
        is_weekend = 1 if timestamp.weekday() >= 5 else 0
        
        # Traffic Density Distribution Models (Peak vs Off-Peak Cycles)
        if 8 <= hour <= 10 or 17 <= hour <= 19:
            base_traffic = np.random.uniform(0.72, 0.96) if not is_weekend else np.random.uniform(0.35, 0.62)
        else:
            base_traffic = np.random.uniform(0.12, 0.48)
            
        if zone_id in ['ZONE_A', 'ZONE_E']:
            traffic_density = np.clip(base_traffic + np.random.uniform(0.02, 0.08), 0.0, 1.0)
        elif zone_id == 'ZONE_C':
            traffic_density = np.clip(base_traffic - np.random.uniform(0.25, 0.45), 0.0, 1.0)
        else:
            traffic_density = np.clip(base_traffic, 0.0, 1.0)
            
        # Local Microclimates (Urban Heat Island effects logic linked to lack of Green Canopy)
        diurnal_cycle = 27.5 + 6.5 * np.sin((hour - 6) * np.pi / 12.0)
        heat_island_delta = (80.0 - canopy) * 0.062
        ambient_temp = np.round(diurnal_cycle + heat_island_delta + np.random.normal(0, 0.65), 2)
        
        # Energy Baseload Calculations (Cooling Loads + Industrial Stabilities)
        cooling_load = max(0, (ambient_temp - 23.5) * 1.75)
        if zone_id in ['ZONE_A', 'ZONE_E']:
            energy_consumption = np.round(42.0 + (traffic_density * 22.5) + cooling_load + np.random.normal(0, 2.5), 2)
        elif zone_id == 'ZONE_B':
            energy_consumption = np.round(115.0 + np.random.normal(0, 7.0), 2)
        else:
            energy_consumption = np.round(14.0 + cooling_load + np.random.normal(0, 1.2), 2)
            
        # Dynamic Air Quality Index (AQI Factors)
        traffic_pollution = traffic_density * 135.0
        industrial_pollution = 195.0 if zone_id == 'ZONE_B' else 12.0
        canopy_cleansing = canopy * 0.68
        aqi = int(np.clip(25 + traffic_pollution + industrial_pollution - canopy_cleansing + np.random.normal(8, 4), 10, 500))
        
        # Regression Target Variable (Carbon Footprint Output Engine)
        carbon_emission = np.round((energy_consumption * 0.44) + (traffic_density * 14.8) + (aqi * 0.022), 3)
        
        # Probabilistic Classification Target (Utility Grid Failure Constraints)
        log_odds_grid = (0.085 * energy_consumption) + (0.16 * (ambient_temp - 31.0)) - 5.25
        grid_risk = np.round(1.0 / (1.0 + np.exp(-log_odds_grid)), 3)
        grid_risk = np.clip(grid_risk, 0.001, 0.999)

        data_records.append({
            'Timestamp': timestamp,
            'Zone_ID': zone_id,
            'Zone_Name': zone_name,
            'Traffic_Density_Index': np.round(traffic_density, 2),
            'Energy_Consumption_MWh': energy_consumption,
            'Ambient_Temperature_C': ambient_temp,
            'Air_Quality_Index_AQI': aqi,
            'Green_Canopy_Coverage_Pct': canopy,
            'Carbon_Emission_Tons': carbon_emission,
            'Grid_Overload_Risk': grid_risk
        })

    df = pd.DataFrame(data_records)
    df.to_csv(output_path, index=False)
    print(f"Success! Base telemetry core saved to: {output_path}")
    return df

def execute_automated_eda(df):
    """Performs programmatic deep summary analytics on the ingested telemetry fields."""
    print("\nPhase 2: Launching Automated Exploratory Data Analysis Pipeline...")
    print(f"• Dataset Dimensionality: {df.shape[0]} Rows x {df.shape[1]} Columns")
    print("\n• Missing Record Assessment:")
    print(df.isnull().sum())
    
    print("\n• Statistical Descriptive Analysis:")
    print(df.describe().T[['mean', 'std', 'min', 'max']])
    
    print("\n• Target Correlations with Carbon_Emission_Tons:")
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['Carbon_Emission_Tons'].sort_values(ascending=False)
    print(correlations)

def compute_advanced_feature_engineering(df, output_path='data/engineered_telemetry.csv'):
    """Applies math and transformation logic to generate predictive feature spaces."""
    print("\nPhase 3: Commencing Feature Engineering Matrix...")
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day_Of_Week'] = df['Timestamp'].dt.dayofweek
    
    # Cyclic time transformations using sine and cosine functions
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24.0)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24.0)
    
    # Interaction features for thermal load
    df['Thermal_Energy_Stress_Index'] = df['Energy_Consumption_MWh'] * np.maximum(0, df['Ambient_Temperature_C'] - 25.0)
    
    # Ratios tracking emission density relative to urban canopy mitigation
    df['Emission_To_Canopy_Ratio'] = df['Carbon_Emission_Tons'] / (df['Green_Canopy_Coverage_Pct'] + 1.0)
    
    # One-hot encode the categorical zone structures
    df = pd.get_dummies(df, columns=['Zone_ID'], prefix='Zone', drop_first=False)
    
    df.to_csv(output_path, index=False)
    print(f"Feature Engineering Completed! Dataset exported to: {output_path}\n")
    return df

if __name__ == "__main__":
    raw_df = generate_smart_city_telemetry()
    execute_automated_eda(raw_df)
    engineered_df = compute_advanced_feature_engineering(raw_df)
    print("Data Engine Execution Completed Successfully.")