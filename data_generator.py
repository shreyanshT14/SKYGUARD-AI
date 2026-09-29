import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_synthetic_data(days=30, interval_mins=5):
    """Generates synthetic automatic weather station data with injected anomalies."""
    print("Generating baseline weather data...")
    num_records = int((days * 24 * 60) / interval_mins)

    # Time index
    start_time = datetime(2023, 1, 1)
    timestamps = [start_time + timedelta(minutes=i*interval_mins) for i in range(num_records)]

    # Baseline signals (smooth daily variations)
    t = np.linspace(0, days * 2 * np.pi, num_records)

    # Temperature: 15 to 35 C, daily cycle
    temp_base = 25 + 10 * np.sin(t - (np.pi/2)) # Peak in afternoon
    # Humidity: 40% to 90%, inversely proportional to temp
    hum_base = 65 - 25 * np.sin(t - (np.pi/2))
    # Pressure: 1000 to 1020 hPa, slow variations
    pres_base = 1010 + 5 * np.cos(t / 5)

    # Add natural noise
    temp = temp_base + np.random.normal(0, 0.5, num_records)
    hum = hum_base + np.random.normal(0, 2.0, num_records)
    pres = pres_base + np.random.normal(0, 0.5, num_records)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature_c': temp,
        'humidity_percent': hum,
        'pressure_hpa': pres,
        'is_anomaly': 0,
        'anomaly_type': 'None'
    })

    print("Injecting anomalies (Spikes, Frozen values, Contextual errors)...")

    # 1. Inject Spikes (e.g., Temp leaps to 55C)
    spike_indices = np.random.choice(num_records, size=int(num_records*0.01), replace=False)
    for idx in spike_indices:
        df.loc[idx, 'temperature_c'] += np.random.choice([15, 20, 25, 30])
        df.loc[idx, 'is_anomaly'] = 1
        df.loc[idx, 'anomaly_type'] = 'Spike'

    # 2. Inject Frozen Data (Sensor stuck)
    freeze_starts = np.random.choice(num_records - 20, size=5, replace=False)
    for start in freeze_starts:
        frozen_val_t = df.loc[start, 'temperature_c']
        frozen_val_h = df.loc[start, 'humidity_percent']
        df.loc[start:start+15, 'temperature_c'] = frozen_val_t
        df.loc[start:start+15, 'humidity_percent'] = frozen_val_h
        df.loc[start+1:start+15, 'is_anomaly'] = 1
        df.loc[start+1:start+15, 'anomaly_type'] = 'Frozen'

    # 3. Inject Contextual Anomalies (Temp high, Hum high - unusual physics)
    context_indices = np.random.choice(num_records, size=int(num_records*0.005), replace=False)
    for idx in context_indices:
        # Avoid overriding spikes
        if df.loc[idx, 'is_anomaly'] == 0:
            df.loc[idx, 'temperature_c'] = 35  # High temp
            df.loc[idx, 'humidity_percent'] = 98 # High hum (rare combination without rain/pressure drop)
            df.loc[idx, 'is_anomaly'] = 1
            df.loc[idx, 'anomaly_type'] = 'Contextual (Physics)'

    # Save to CSV
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/synthetic_aws_data.csv', index=False)
    print(f"Dataset generated successfully! Shape: {df.shape}")
    print(df['anomaly_type'].value_counts())

if __name__ == "__main__":
    generate_synthetic_data(days=30, interval_mins=5)
