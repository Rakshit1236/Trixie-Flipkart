<div align="center">

# Trixie — Parking Intelligence Platform

**AI-powered platform that detects parking hotspots, predicts violations, and recommends enforcement actions using real traffic data.**

[![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?logo=streamlit)](https://trixie-flipkart.streamlit.app/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow?logo=huggingface)](https://rakshit1236-trixie-backend.hf.space)

</div>

---

## What It Does

Trixie analyzes 115,000+ real Bengaluru traffic police violations to build an end-to-end parking intelligence system. It clusters 1,263 hotspots across the city, quantifies their impact on traffic speed using physics-based models, and predicts tomorrow's violations with a tuned ML ensemble — all served through a full-stack dashboard.

## Features

| Tab | Feature | What It Does |
|-----|---------|-------------|
| **Overview** | Executive KPIs + Heatmaps | Hotspot distribution, confidence gauges, tomorrow's forecast with error bars |
| **What-If** | Scenario Simulator + Counterfactual AI | One-click presets (Festival, Rain, etc.) + custom occupancy reduction |
| **Propagation** | Congestion Cascade Timeline | BFS simulation of how congestion ripples through road networks |
| **Insights** | SHAP Root Cause Analysis + PRI | Why each hotspot exists — with ranked risk scores and factor breakdowns |
| **Actions** | Dynamic Warnings + 30-Day Forecast | Hour-specific threat levels + ML-powered monthly forecast with CI bands |
| **Validation** | Model Health Dashboard | Live R², MAE, pipeline status, and drift detection |

## Architecture

```
┌─────────────────────┐         ┌──────────────────────────┐
│   Streamlit Cloud   │  REST   │    HuggingFace Spaces    │
│   (Dashboard UI)    │ ◄─────► │    (FastAPI Backend)     │
│                     │  API    │                          │
│  6 tabs + sidebar   │         │  JSON cache (23 MB)      │
│  Plotly + Folium    │         │  16 endpoints             │
└─────────────────────┘         └──────────────────────────┘
```

**Pipeline:** Data (115K records) → HDBSCAN Clustering (1,263 hotspots) → Greenshields Traffic Impact → Priority Scoring → LGB+XGB+CatBoost Ensemble → SHAP XAI → Dispatch Recommendations

## Tech Stack

| Layer | |
|-------|--|
| **ML** | LightGBM + XGBoost + CatBoost (stacking with RidgeCV meta-learner) |
| **Tuning** | Optuna (30 trials, TPE sampler, 5-fold time-series CV) |
| **Explainability** | SHAP TreeExplainer — real per-feature attributions |
| **Traffic Model** | Greenshields speed-density + M/M/1 queue approximation |
| **Clustering** | HDBSCAN + KD-tree adjacency (2 km radius) |
| **Prediction** | Conformal prediction intervals (90% coverage) |
| **Frontend** | Streamlit + Plotly + Folium |
| **Backend** | FastAPI + GZip compression on HuggingFace Spaces |
| **Data** | Bengaluru Police Violations (Jan–May), 150+ pincode areas |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

## Links

- **Dashboard:** [trixie-flipkart.streamlit.app](https://trixie-flipkart.streamlit.app/)
- **Backend API:** [rakshit1236-trixie-backend.hf.space](https://rakshit1236-trixie-backend.hf.space)
- **Health Check:** [/health](https://rakshit1236-trixie-backend.hf.space/health)
