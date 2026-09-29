import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import time
import shap

# --- Configuration ---
st.set_page_config(page_title="SkyGuard AI Dashboard", layout="wide")
st.title("☁️ SkyGuard AI: Intelligent AWS Monitoring")

# --- Load Models & Data ---
@st.cache_resource
def load_resources():
    scaler = joblib.load('models/scaler.pkl')
    model = joblib.load('models/iso_forest.pkl')
    # Load imputers
    imp_t = joblib.load('models/imputer_temp.pkl')
    imp_h = joblib.load('models/imputer_hum.pkl')
    imp_p = joblib.load('models/imputer_pres.pkl')
    df = pd.read_csv('data/synthetic_aws_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return scaler, model, (imp_t, imp_h, imp_p), df

scaler, model, imputers, df = load_resources()
imp_t, imp_h, imp_p = imputers

# --- Sidebar ---
st.sidebar.header("Configuration")
simulate_speed = st.sidebar.slider("Simulation Speed (ms)", 100, 1000, 300)
start_button = st.sidebar.button("Run Real-time Stream")

# --- Dashboad Layout ---
col1, col2, col3 = st.columns(3)
temp_metric = col1.empty()
hum_metric = col2.empty()
pres_metric = col3.empty()

chart_placeholder = st.empty()
alert_placeholder = st.empty()
xai_placeholder = st.empty()

# --- Simulation Logic ---
if start_button:
    # Prepare XAI
    feature_names = ['Temperature', 'Humidity', 'Pressure']

    # Simple XAI: TreeExplainer for IF doesn't work directly, using random data for background
    # Background for XAI
    X_train_scaled = scaler.transform(df[df['is_anomaly']==0][['temperature_c', 'humidity_percent', 'pressure_hpa']])
    explainer = shap.KernelExplainer(model.decision_function, X_train_scaled[:100])

    for i in range(len(df)):
        row = df.iloc[i]
        features = row[['temperature_c', 'humidity_percent', 'pressure_hpa']].values.reshape(1, -1)
        scaled_features = scaler.transform(features)

        # Inference
        is_anomaly = model.predict(scaled_features)[0] == -1

        # UI Updates
        temp_metric.metric("Temperature", f"{row['temperature_c']:.1f} °C")
        hum_metric.metric("Humidity", f"{row['humidity_percent']:.1f} %")
        pres_metric.metric("Pressure", f"{row['pressure_hpa']:.1f} hPa")

        # Plotting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'].iloc[:i+1], y=df['temperature_c'].iloc[:i+1], name="Temp", line=dict(color='orange')))

        if is_anomaly:
            # Highlight anomaly
            fig.add_trace(go.Scatter(x=[row['timestamp']], y=[row['temperature_c']], mode='markers',
                                    marker=dict(color='red', size=10), name="Anomaly"))

            # XAI & Correction
            alert_placeholder.error(f"🚨 ANOMALY DETECTED (Type: {row['anomaly_type']}) at {row['timestamp']}")

            # Calculate correction
            hour = row['timestamp'].hour
            pred_temp = imp_t.predict(np.array([[hour, row['humidity_percent'], row['pressure_hpa']]]))[0]
            st.write(f"**Self-Healing Suggestion:** Replaced anomaly {row['temperature_c']:.1f}°C with predicted {pred_temp:.1f}°C")

        chart_placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(simulate_speed / 1000)
else:
    st.info("Press 'Run Real-time Stream' to start simulation.")
