import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_models():
    print("Loading data...")
    df = pd.read_csv('data/synthetic_aws_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    features = ['temperature_c', 'humidity_percent', 'pressure_hpa']

    # We train models ONLY on normal data to learn the valid distribution
    normal_data = df[df['is_anomaly'] == 0].copy()

    print("Training Scaler...")
    scaler = StandardScaler()
    scaled_normal_data = scaler.fit_transform(normal_data[features])

    print("Training Isolation Forest (Anomaly Detection)...")
    # Contamination set to 0.02 (assume 2% expected anomalies in real world to set threshold)
    iso_forest = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    iso_forest.fit(scaled_normal_data)

    print("Training Self-Healing Model (Imputation)...")
    # For self-healing, we train a model to predict current state based on recent history
    # Simple approach for MVP: Train an RF to predict each variable based on the other two + time of day
    normal_data['hour'] = normal_data['timestamp'].dt.hour

    # Train Temp Imputer
    rf_temp = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
    rf_temp.fit(normal_data[['hour', 'humidity_percent', 'pressure_hpa']], normal_data['temperature_c'])

    # Train Hum Imputer
    rf_hum = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
    rf_hum.fit(normal_data[['hour', 'temperature_c', 'pressure_hpa']], normal_data['humidity_percent'])

    # Train Pres Imputer
    rf_pres = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
    rf_pres.fit(normal_data[['hour', 'temperature_c', 'humidity_percent']], normal_data['pressure_hpa'])

    # Save models
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(iso_forest, 'models/iso_forest.pkl')

    # Save Self-healing models
    joblib.dump(rf_temp, 'models/imputer_temp.pkl')
    joblib.dump(rf_hum, 'models/imputer_hum.pkl')
    joblib.dump(rf_pres, 'models/imputer_pres.pkl')

    print("All models trained and saved to 'models/' directory.")

if __name__ == "__main__":
    train_models()
