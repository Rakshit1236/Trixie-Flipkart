# Trixie — Parking Intelligence Platform

AI-powered dashboard for parking congestion intelligence. Detects illegal parking hotspots, quantifies their impact on traffic flow, and enables targeted enforcement.

## Features

- **Executive Overview** — KPIs, heatmaps, forecast with confidence gauges
- **What-If Simulator** — One-click preset scenarios + counterfactual AI mode
- **Congestion Propagation** — Step-by-step cascade timeline visualization
- **Insights** — SHAP-based Explainable AI + Root Cause Attribution cards
- **Actions** — Early Warning Timeline + Enhanced Dispatch cards with ETA
- **Validation** — Backtesting + drift detection

## Innovation Features

### 1. Scenario Simulator with Presets
One-click buttons for common scenarios: Remove Illegal Parking, Festival Day, Heavy Rain, Metro Delay, Increase Capacity.

### 2. Counterfactual AI
"What if parking occupancy were 15% lower?" — See predicted impact on speed, violations, and queue length.

### 3. Congestion Propagation Timeline
Step-by-step cascade visualization: "8:10 Metro Station → 8:18 Ring Road → 8:26 Main Road → 8:40 Cross Road"

### 4. Root Cause Attribution
Human-readable cards showing: "Illegal Parking 48%, Road Width 21%, Weather 13%, Event 18%"

### 5. Early Warning Timeline
Visual progression: Normal → Risk Rising → Congestion Expected → Critical

### 6. Enhanced Dispatch
Cards with ETA, resource type (Officers/Tow Truck), and expected delay reduction.

## Architecture

```
┌─────────────────────┐     REST API     ┌──────────────────────┐
│   Streamlit Cloud   │ ◄──────────────► │  HuggingFace Spaces  │
│   (Dashboard UI)    │                  │  (ML Pipeline)       │
│                     │  GET /hotspots   │  FastAPI + LightGBM  │
│  - Executive KPIs   │  POST /scenario  │  XGBoost + Optuna    │
│  - What-If Simulator│  GET /predictions│  HDBSCAN + SHAP      │
│  - Propagation      │  GET /warnings   │                      │
│  - Insights (XAI)   │  POST /propagate │                      │
│  - Actions          │  POST /counterf. │                      │
└─────────────────────┘                  └──────────────────────┘
```

## Setup

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python -c "from src.data_pipeline import run_pipeline; from src.clustering import cluster_hotspots; from src.traffic_impact import run_impact_analysis; from src.scoring import run_scoring; from src.predictive_model import run_ml_pipeline; from src.analytics import run_analytics; print('Pipeline modules loaded')"

# Run Streamlit app
streamlit run app.py
```

### Deploy to HuggingFace Spaces

1. Create a new HuggingFace Space with Docker SDK (port 7860)
2. Upload `hf_backend/` contents
3. Set `HF_REPO_ID` environment variable if using dataset repo

### Deploy Dashboard to Streamlit Cloud

1. Push to GitHub
2. Connect at https://share.streamlit.io
3. Set main file to `app.py`
4. Add `BACKEND_URL` secret pointing to your HuggingFace Space

## Tech Stack

- **ML**: LightGBM + XGBoost ensemble with Optuna tuning
- **XAI**: True SHAP explanations (not heuristic weights)
- **Traffic**: Greenshields model + M/M/1 queue approximation
- **Clustering**: HDBSCAN with KD-tree adjacency
- **Prediction**: Conformal prediction with coverage guarantees
- **Frontend**: Streamlit + Plotly + Folium
- **Backend**: FastAPI + HuggingFace Spaces
