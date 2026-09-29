# SkyGuard AI: Intelligent AWS Anomaly Detection

A prototype for the Smart India Hackathon.

## Overview
SkyGuard AI detects sensor malfunctions in Automatic Weather Stations (AWS) using Machine Learning (Isolation Forests) and offers explainability (XAI) and automatic value imputation (Self-Healing).

## Features
- **Real-time Anomaly Detection**: Monitors temp/pressure/humidity streams.
- **Explainable AI**: Indicates *why* a reading is suspicious.
- **Self-Healing**: Automatically suggests corrected values if a sensor is faulty.
- **Edge-Ready**: Uses scikit-learn; easily convertible to TensorFlow Lite for ESP32 deployment.

## Installation
1. `pip install -r requirements.txt`

## Running the Prototype
1. **Generate Data**: `python src/data_generator.py`
2. **Train Models**: `python src/train.py`
3. **Launch Dashboard**: `streamlit run app.py`

## Use Cases
- **Sensor Calibration Drift**: Detects gradual errors before they impact data quality.
- **Hardware Malfunction**: Identifies spikes and frozen readings instantly.
- **Environmental Anomalies**: Distinguishes between extreme weather events and sensor faults.
