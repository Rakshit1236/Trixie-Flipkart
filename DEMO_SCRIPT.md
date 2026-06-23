# Trixie — Hackathon Demo Script
# Parking Intelligence Platform | Flipkart Grid

## Pre-Demo Setup (Do Before Judges Arrive)
1. Open https://trixie-flipkart.streamlit.app/ in Chrome (full screen)
2. Verify sidebar shows: Status=healthy, Hotspots=1263, Critical=1263, R²=0.177, MAE=2.09
3. Keep a second tab open with Google Maps satellite view of Bengaluru for context

---

## OPENING (30 seconds)
"Traffic congestion from illegal parking costs Bengaluru ₹38,000 crore annually.
We analyzed 115,000+ real police violations to build Trixie — a system that
detects, predicts, and recommends actions for parking hotspots across the city."

Click: Sidebar metrics — "Our ML pipeline clusters 1,263 hotspots, identifies
1,263 critical zones, and predicts tomorrow's violations with a stacking
ensemble of LightGBM, XGBoost, and CatBoost."

---

## TAB 1: OVERVIEW & MAPS (1 minute)

### 1a. Executive KPIs
Action: Point to the 4 top-level metric cards
Say: "At a glance — total hotspots, critical count, average speed drop %,
and total vehicle-hours lost to congestion."

### 1b. Road Type Distribution
Action: Scroll to the bar chart
Say: "We classify hotspots by road type — Ring roads and main arteries
carry the heaviest congestion load."

### 1c. Confidence Gauges (click sub-tab)
Action: Click "Confidence Gauges" sub-tab
Say: "Each prediction comes with a conformal prediction interval at 90%
coverage. These gauges show model confidence per hotspot."

Action: Point to the horizontal bar chart of top 15 hotspots
Say: "Higher confidence means we're more certain about the prediction,
enabling reliable resource allocation."

### 1d. Tomorrow's Forecast (click sub-tab)
Action: Click "Tomorrow's Forecast" sub-tab
Say: "Tomorrow's citywide prediction with confidence intervals.
The error bars represent our conformal bounds — not just a point
estimate, but a calibrated range."

### 1e. Dispatch Report (click sub-tab)
Action: Click "Dispatch Report" sub-tab
Say: "Auto-generated dispatch instructions — which officers go where,
how many units, and expected impact."

---

## TAB 2: WHAT-IF SIMULATOR (1.5 minutes)

### 2a. Quick Scenario — Festival Day
Action: Click "What-If Simulator" tab
Action: Select preset "Festival Day" from the dropdown
Action: Click "Run Scenario"
Say: "One click — what happens during a festival? The simulator applies
weather and day-type multipliers to our physics model."

Action: Point to the citywide metrics
Say: "Violations increase, speed drops further, vehicle-hours lost spike.
Now every hotspot is re-scoreed in real-time."

### 2b. Custom Scenario — Remove Illegal Parking
Action: Change preset to "Remove Illegal Parking" (or manually set 50 vehicles)
Action: Click "Run Scenario"
Say: "Now the flip side — what if we remove 50 illegally parked vehicles?
Watch the improvement metrics."

Action: Point to violations_reduced and speed_gained
Say: "Citywide violation reduction and speed recovery, computed using the
Greenshields speed-density model and M/M/1 queue theory."

### 2c. Counterfactual AI
Action: Switch to "Counterfactual AI" sub-tab
Action: Set occupancy reduction to 20%
Action: Click "Run Counterfactual"
Say: "A question every urban planner asks — 'What if parking occupancy
were 20% lower?' Our counterfactual engine answers with physics."

### 2d. Per-Hotspot Focus
Action: Select a specific hotspot (e.g., Koramangala)
Say: "We can zoom into any single hotspot — Koramangala, one of Bengaluru's
busiest areas — and simulate the exact local impact."

---

## TAB 3: CONGESTION PROPAGATION (1 minute)

Action: Click "Congestion Propagation" tab
Action: Select a source hotspot (pick one with many neighbors)
Action: Set start hour to 8 (morning rush)
Action: Set horizon to 60 minutes
Action: Click "Simulate Cascade"
Say: "This is the cascade view — pick a source hotspot, and watch how
congestion ripples outward through the road network."

Action: Point to the HTML cascade timeline
Say: "Each node shows the time, speed drop, and queue length.
The cascade follows real road adjacency within 2km, using BFS
traversal with distance-based decay."

Action: Scroll to the scatter plot
Say: "Speed drop vs time — larger bubbles mean longer queues.
Red = critical severity, orange = high."

Action: Scroll to the Folium map
Say: "Interactive map showing the propagation spatially.
Circle size = speed drop magnitude."

---

## TAB 4: INSIGHTS — XAI + PRI (1.5 minutes)

### 4a. Explainable AI
Action: Click "Insights" tab
Action: Select a hotspot from the XAI dropdown
Say: "Every prediction is explainable. This is SHAP-based root cause
analysis from our LightGBM model — not heuristics."

Action: Point to the dominant factor card
Say: "For this hotspot, the dominant factor is Illegal Parking at 45%.
The progress bars show each factor's contribution."

Action: Scroll to the pie chart
Say: "The donut chart gives a proportional view — illegal parking,
road width, density, and junction effects."

Action: Scroll to the |SHAP| bar chart
Say: "Top features by mean absolute SHAP value — these are the
features the model actually relies on."

### 4b. Parking Risk Index (PRI)
Action: Click "Parking Risk Index (PRI)" sub-tab
Say: "Our proprietary composite risk score: 40% illegal parking,
30% density, 20% road importance, 10% event score."

Action: Scroll to the stacked bar chart
Say: "The stacked bars break down each hotspot's risk into its
four components — you can see exactly what drives risk."

Action: Point to the top-15 ranked table
Say: "Ranked table with all component scores and the dominant
risk factor for each hotspot."

---

## TAB 5: ACTIONS (2 minutes)

### 5a. Dynamic Early Warnings
Action: Click "Actions" tab
Action: Change the hour selector to 9:00 (morning rush)
Say: "This is live — not cached, not static. Every page load
recomputes warnings for the current hour or any hour you select."

Action: Point to the warning timeline cards (+15, +30, +60 min)
Say: "Three time horizons: 15, 30, and 60 minutes ahead.
At 9 AM rush hour — most are CRITICAL. Each card shows the
count of HIGH, MEDIUM, and LOW threats."

Action: Scroll to the warning map
Say: "The map shows exactly where threats are concentrated.
Red = high threat, orange = medium, green = low."

Action: Scroll to the 24-hour threat progression chart
Say: "This dual-axis chart reveals the rush-hour pattern —
average threat score peaks during 8-11 AM and 5-9 PM.
The red bars show how many hotspots hit HIGH severity each hour."

### 5b. 30-Day Forecast
Action: Click "30-Day Forecast" sub-tab
Action: Scroll to the citywide chart
Say: "ML-powered 30-day forecast with 90% confidence bands.
The shaded area is the conformal prediction interval —
we're 90% sure the true value falls within this range."

Action: Check the "Show per-hotspot details" checkbox
Say: "For deeper analysis, load per-hotspot forecasts —
select specific hotspots to compare their trajectories."

### 5c. Dispatch Recommendations
Action: Click "Action Recommendations" sub-tab
Action: Point to the summary metrics (Total Officers, Critical, High)
Say: "The system recommends specific actions: how many officers,
what resource type, ETA, and expected delay reduction."

Action: Scroll through the dispatch cards
Say: "Each card shows: cluster ID, area, severity, officers
needed, ETA, delay reduction %, and resource type —
Deploy Officers + Tow Truck, Patrol Unit, or Monitor."

Action: Scroll to the full recommendations table
Say: "Sortable table of all recommendations with every field."

---

## TAB 6: VALIDATION (30 seconds)

Action: Click "Validation" tab
Say: "Model health dashboard — ensemble R-squared, MAE, pipeline status.
We use time-series cross-validation, not random splits, to prevent
data leakage from future to past."

Action: Point to the model description
Say: "LightGBM + XGBoost + CatBoost stacking with a Ridge meta-learner.
30 Optuna trials per model, 5-fold time-series CV."

---

## CLOSING (30 seconds)
"To summarize: Trixie detects 1,263 hotspots from real police data,
predicts tomorrow's violations with a tuned ML ensemble, explains WHY
each hotspot exists using SHAP, simulates WHAT-IF scenarios with traffic
physics, propagates congestion cascades, generates 30-day forecasts,
and recommends specific dispatch actions — all in real-time.

The backend runs on HuggingFace Spaces, the dashboard on Streamlit Cloud,
and the entire pipeline processes 115,000+ records in under 3 minutes."

---

## CHEAT SHEET — Key Numbers to Remember
| Metric | Value |
|--------|-------|
| Raw records | 115,400 |
| Hotspots detected | 1,263 |
| Critical zones | 1,263 |
| Predictions | 1,260 |
| Ensemble R² | 0.177 |
| MAE | 2.09 |
| Conformal coverage | 90% |
| Optuna trials/model | 30 |
| CV folds | 5 |
| Forecast horizon | 30 days |
| Forecast rows | 37,800 |
| Features engineered | 66 |
| Backend endpoints | 16 |
| Dashboard tabs | 6 |
| Bengaluru pincodes mapped | 150+ |
| Response time (initial) | ~12s |
| Pipeline runtime | ~3 min |

## CHEAT SHEET — Tech Stack
| Layer | Tech |
|-------|------|
| Frontend | Streamlit (Python) |
| Backend | FastAPI on HuggingFace Spaces |
| ML | LightGBM + XGBoost + CatBoost + RidgeCV |
| Hyperparameter Tuning | Optuna (TPE sampler, 30 trials) |
| Feature Selection | Auto-correlation + permutation importance |
| Prediction Intervals | Conformal prediction (90% coverage) |
| Explainability | SHAP TreeExplainer |
| Traffic Model | Greenshields + M/M/1 queue |
| Spatial Clustering | HDBSCAN + KD-tree adjacency |
| Visualization | Plotly + Folium maps |
| Data | Bengaluru Police Violations (Jan-May) |

## CHEAT SHEET — Architecture
Browser (Streamlit Cloud) --> REST API --> HF Spaces (FastAPI)
                                              |
                                         JSON Cache (11.6MB)
                                              |
                                    Downloaded from HF Dataset Repo
                                              |
                                    Pre-computed by Pipeline (local)
```
