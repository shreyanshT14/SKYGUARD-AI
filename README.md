# SkyGuard AI

SkyGuard AI is an intelligent anomaly detection and self-healing monitoring prototype for Automatic Weather Stations (AWS). It is designed to identify faulty sensor behavior, distinguish true meteorological events from data-quality issues, and suggest corrected values for downstream weather forecasting and climate monitoring workflows.

This project was developed as a Smart India Hackathon-style prototype to address real-world problems in meteorological data quality, sensor reliability, and near-real-time operational monitoring.

## Overview
Automatic Weather Stations continuously collect temperature, pressure, and humidity observations. In practice, sensor drift, stuck readings, communication failures, and transient electrical faults can produce bad measurements that reduce forecast accuracy and compromise station health monitoring.

SkyGuard AI tackles this by combining:

- Isolation Forest-based anomaly detection
- Physical consistency checks across temperature, pressure, and humidity
- Explainable AI (XAI) reasoning for why a value was flagged
- Self-healing imputation to estimate corrected readings
- A desktop and dashboard interface for demonstration and monitoring

## Key Features

- Real-time AWS monitoring for temperature, pressure, and humidity
- Detection of common sensor faults such as:
  - temperature spikes
  - frozen sensor values
  - humidity faults
  - pressure failures
  - communication loss
  - calibration drift
- Genuine weather event detection to avoid false alarms during storm-like conditions
- Explainable AI output describing which variable contributed most to the anomaly
- Self-healing correction suggestions for suspicious readings
- Offline-ready prototype suitable for demonstration and local deployment

## Technology Stack

- Python 3.11+
- Streamlit for dashboard UI
- Tkinter desktop demo interface
- scikit-learn for Isolation Forest and regression-based imputation
- pandas and numpy for data processing
- plotly for charts and diagnostics
- joblib for model serialization

## Project Structure

```text
.
├── README.md                     # Project documentation
├── main.py                       # Desktop GUI demo for AWS monitoring
├── app.py                        # Streamlit live monitoring dashboard
├── anomaly_detector.py           # ML anomaly engine and classification logic
├── aws_simulator.py              # AWS sensor simulation and fault injection
├── requirements.txt              # Python dependencies
├── build_exe.bat                 # Windows packaging script for PyInstaller
├── DEMO_SCRIPT.md               # Presentation/demo script for judges
├── data/
│   └── synthetic_aws_data.csv   # Generated synthetic AWS dataset
├── models/
│   ├── scaler.pkl               # StandardScaler artifact
│   ├── iso_forest.pkl           # Isolation Forest model
│   ├── imputer_temp.pkl         # Temperature imputation model
│   ├── imputer_hum.pkl          # Humidity imputation model
│   └── imputer_pres.pkl         # Pressure imputation model
├── src/
│   ├── data_generator.py        # Synthetic AWS data generation
│   └── train.py                 # Model training pipeline
├── build/                       # Build artifacts
├── dist/                        # Packaged executables
├── SkyGuardAI.spec              # PyInstaller spec file
├── .gitignore
└── ...
```

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Generation and Model Training

Before running the app, generate the synthetic meteorological dataset and train the models.

```bash
python src/data_generator.py
python src/train.py
```

This creates the dataset in `data/synthetic_aws_data.csv` and saves trained model files in `models/`.

## Running the Project

### 1. Desktop GUI Demo

```bash
python main.py
```

This launches the Tkinter-based monitoring interface, which includes:

- live sensor cards
- anomaly status
- explainable AI reasoning
- self-healing correction values
- manual fault injection controls
- demo mode

### 2. Streamlit Dashboard

```bash
streamlit run app.py
```

This opens the browser-based dashboard that simulates a live AWS stream and visualizes anomalies in real time.

### 3. Build Windows Executable

```powershell
build_exe.bat
```

This uses PyInstaller to build a packaged Windows executable for offline presentation/demo deployment.

## Demonstration Flow

The project includes a structured demo experience for judging or presentation use:

- normal station behavior
- temperature spike detection
- frozen sensor detection
- communication loss handling
- humidity and pressure faults
- genuine weather event recognition
- self-healing recommendations

For detailed judge/demo guidance, see [DEMO_SCRIPT.md](DEMO_SCRIPT.md).

## How the Detection Logic Works

SkyGuard AI follows a practical multi-stage pipeline:

1. Learn the normal weather baseline from historical data
2. Detect outliers using Isolation Forest
3. Evaluate whether the pattern is physically coherent
4. Classify the root cause as sensor fault or genuine weather event
5. Explain the anomaly using feature contribution reasoning
6. Estimate a corrected replacement value using imputation models

This makes the system more robust than threshold-only alerting, especially when the weather event and sensor fault patterns overlap.

## Example Use Cases

- Station health monitoring for meteorological field networks
- Automated sensor quality control for AWS maintenance teams
- Reducing false alerts in forecasting pipelines
- Early detection of calibration drift and hardware degradation
- Supporting explainable AI workflows for operational meteorology

## Limitations

This is a prototype and demonstration project, not a production-grade national weather monitoring system. It is intended to showcase the concept, architecture, and experimental validation of AI-driven sensor anomaly detection for AWS data quality control.

## License

No explicit license has been provided in this repository.

## Acknowledgements

This project is designed around a Smart India Hackathon-style problem statement focused on intelligent meteorological monitoring, explainability, and self-healing sensor systems.
