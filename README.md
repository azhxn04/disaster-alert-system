# 🛡️ Resilience AI

**A Unified 4-Discipline Engineering Architecture for Statewide Disaster Prediction & Early Warning**

Final Year Academic Capstone Project — Deep Learning • Data Visualization • Data Security & Privacy • Data Engineering

![Coverage](https://img.shields.io/badge/coverage-36%20Maharashtra%20districts-blue)
![Records](https://img.shields.io/badge/historical%20records-75%2C240-informational)
![Data](https://img.shields.io/badge/data%20span-5%2B%20years-success)
![Status](https://img.shields.io/badge/status-active%20development-yellow)

---

## 📖 Overview

**Resilience AI** is a multi-hazard early-warning system covering all 36 districts of Maharashtra. It ingests five years of historical and live meteorological, hydrological, and seismic data, engineers it into a predictive feature store, and serves risk scores through a neural-network inference engine — all built on a security-first data pipeline.

The project is organized around four engineering disciplines:

| Discipline | Focus |
|---|---|
| 🧠 **DL** — Deep Learning | ANN-based multi-hazard risk prediction |
| 📊 **DV** — Data Visualization | Interactive maps, trends & KPI dashboards |
| 🔒 **DSP** — Data Security & Privacy | Credential hashing, encryption, access control |
| ⚙️ **DE** — Data Engineering | ETL pipeline, feature engineering, storage |

---

## 🌍 Problem Statement

Maharashtra faces distinct, region-specific disaster risks:

- **🌊 Coastal Konkan & Ghats** (Mumbai, Thane, Palghar, Raigad, Ratnagiri, Sindhudurg) — monsoon deluges, flash floods, landslides, storm surges
- **☀️ Vidarbha & Marathwada** (Nagpur, Chandrapur, Akola, Wardha, Latur, Beed, Jalna) — extreme heatwaves, agrarian drought, forest fires
- **⛰️ Desh & Seismic Faults** (Pune, Satara, Kolhapur, Sangli, Solapur, Dharashiv) — river flooding, dam spillway discharge, seismic activity (BIS Zone IV)

Resilience AI unifies these hazard types into a single statewide prediction platform.

---

## 🗂️ Data Sources

| Source | Data |
|---|---|
| **Open-Meteo API** | ECMWF/GFS forecasts — precipitation, temperature, humidity, pressure, wind speed (2021–present) |
| **IMD Ground Normals** | Climatological baselines, heatwave thresholds, rainfall classification standards |

All 75,240+ records span 5+ years across all 36 districts.

---

## ⚙️ Discipline 1: Data Engineering

**5-Stage ETL Pipeline**

1. **Extract** — Open-Meteo, USGS & NDMA live feeds
2. **Transform** — Timezone correction, missing-value cleanup
3. **Feature Engineering** — Antecedent Precipitation Index (API), Heat Index, Landslide Susceptibility Index (LSI)
4. **Validate** — Pydantic schema enforcement, 100% compliance
5. **Load** — SQLite + Parquet columnar storage

**Key Metrics:** 36 districts · 75,240 records · 5+ years of data · <50ms query latency

---

## 🧠 Discipline 2: Deep Learning

**Model:** Artificial Neural Network (ANN) — a simple feedforward network chosen for learning non-linear relationships between weather features without the overhead of sequence memory.

- **Input Layer** — Normalized weather and geographic features
- **Hidden Layers** — Dense layers with ReLU activation
- **Output Layer** — Calibrated risk probabilities via sigmoid/softmax

**5 Computed Multi-Hazard Risk Indicators**

1. Flood Risk Indicator (0–100%)
2. Heavy Rainfall Forecaster (1d / 3d / 7d lead time)
3. Heatwave Risk Classifier
4. Western Ghats Landslide Susceptibility
5. Seismic Anomaly & Statistical Risk

---

## 🔒 Discipline 3: Data Security & Privacy

- **Password Hashing** — User passwords are never stored in plain text; they are hashed with **SHA-256** and a unique per-user salt before being written to the database
- **Credential Protection** — API keys and access tokens are encrypted at rest and rotated periodically
- **Access Control** — Role-based access control (RBAC) restricts read/write permissions on the data warehouse and audit ledger
- **Data-in-Transit** — All API and dashboard traffic secured over TLS 1.2+

---

## 📊 Discipline 4: Data Visualization

- Historical data volume trend charts
- Interactive district map for location-based hazard lookup
- Real-time KPI dashboard (districts covered, total records, years of data, query latency)

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/resilience-ai.git
cd resilience-ai

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python run_pipeline.py
```

> Update the commands above to match your actual project scripts and entry points.

---

## 📁 Project Structure

```
resilience-ai/
├── data/               # Raw and processed datasets
├── etl/                # Extraction, transformation, validation, loading
├── models/             # ANN training & inference code
├── security/           # Hashing, encryption, access control
├── dashboard/          # Visualization & map interface
└── README.md
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or file an issue.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
