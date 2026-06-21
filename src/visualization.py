"""
Visualization module for Trixie-flipkartgridlock.
Folium maps + Plotly/matplotlib charts.
"""
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    HEATMAPS_DIR, BENGALURU_LAT, BENGALURU_LON
)


# ==================== FOLIUM MAPS ====================

def create_hotspot_heatmap(profiles: Dict[int, Dict], impact: Dict[int, Dict],
                           output_path: Path = None) -> str:
    """Create interactive Folium heatmap of violation density."""
    print("Creating hotspot heatmap...")
    
    if output_path is None:
        output_path = HEATMAPS_DIR / "hotspot_heatmap.html"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create map
    m = folium.Map(
        location=[BENGALURU_LAT, BENGALURU_LON],
        zoom_start=12,
        tiles="CartoDB positron",
    )
    
    # Add hotspots
    for cid, profile in profiles.items():
        lat = profile["centroid_lat"]
        lon = profile["centroid_lon"]
        violations = profile["total_violations"]
        severity = impact.get(cid, {}).get("severity_class", "LOW")
        
        # Color based on severity
        color_map = {
            "CRITICAL": "red",
            "HIGH": "orange",
            "MEDIUM": "yellow",
            "LOW": "green",
        }
        color = color_map.get(severity, "blue")
        
        # Circle size based on violations
        radius = max(5, min(30, violations / 50))
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>Cluster {cid}</b><br>"
                f"Violations: {violations}<br>"
                f"Severity: {severity}<br>"
                f"Area: {profile.get('area', 'Unknown')}",
                max_width=200,
            ),
        ).add_to(m)
    
    # Save
    m.save(str(output_path))
    print(f"  Saved to {output_path}")
    return str(output_path)


def create_impact_map(profiles: Dict[int, Dict], impact: Dict[int, Dict],
                      output_path: Path = None) -> str:
    """Create traffic impact severity map."""
    print("Creating impact map...")
    
    if output_path is None:
        output_path = HEATMAPS_DIR / "impact_map.html"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create map
    m = folium.Map(
        location=[BENGALURU_LAT, BENGALURU_LON],
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )
    
    # Add hotspots with impact data
    for cid, profile in profiles.items():
        lat = profile["centroid_lat"]
        lon = profile["centroid_lon"]
        impact_data = impact.get(cid, {})
        
        speed_drop = impact_data.get("worst_speed_drop_pct", 0)
        vhl = impact_data.get("total_vhl", 0)
        severity = impact_data.get("severity_class", "LOW")
        
        # Color gradient based on speed drop
        if speed_drop >= 40:
            color = "#FF0000"  # Red
        elif speed_drop >= 25:
            color = "#FF6600"  # Orange
        elif speed_drop >= 10:
            color = "#FFCC00"  # Yellow
        else:
            color = "#00CC00"  # Green
        
        # Marker with impact info
        folium.CircleMarker(
            location=[lat, lon],
            radius=max(5, min(25, speed_drop / 3)),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>Cluster {cid}</b><br>"
                f"Speed Drop: {speed_drop:.1f}%<br>"
                f"Vehicle-Hours Lost: {vhl:.1f}<br>"
                f"Severity: {severity}<br>"
                f"Worst Hour: {impact_data.get('worst_hour', 'N/A')}:00",
                max_width=250,
            ),
        ).add_to(m)
    
    # Save
    m.save(str(output_path))
    print(f"  Saved to {output_path}")
    return str(output_path)


def create_predictions_heatmap(predictions: pd.DataFrame,
                               output_path: Path = None) -> str:
    """Create predictions heatmap."""
    print("Creating predictions heatmap...")
    
    if output_path is None:
        output_path = HEATMAPS_DIR / "predictions_heatmap.html"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create map
    m = folium.Map(
        location=[BENGALURU_LAT, BENGALURU_LON],
        zoom_start=12,
        tiles="CartoDB positron",
    )
    
    # Add predictions
    for _, row in predictions.iterrows():
        lat = row.get("centroid_lat", BENGALURU_LAT)
        lon = row.get("centroid_lon", BENGALURU_LON)
        pred = row.get("predicted_violations", 0)
        confidence = row.get("confidence_pct", 80)
        area = row.get("area", "Unknown")
        
        # Color by predicted violations
        if pred >= 50:
            color = "#FF0000"
        elif pred >= 30:
            color = "#FF6600"
        elif pred >= 15:
            color = "#FFCC00"
        else:
            color = "#00CC00"
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=max(5, min(20, pred / 5)),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>Prediction</b><br>"
                f"Cluster: {row.get('cluster_id', 'N/A')}<br>"
                f"Predicted: {pred:.1f} violations<br>"
                f"Confidence: {confidence:.0f}%<br>"
                f"Area: {area}",
                max_width=200,
            ),
        ).add_to(m)
    
    # Save
    m.save(str(output_path))
    print(f"  Saved to {output_path}")
    return str(output_path)


# ==================== PLOTLY CHARTS ====================

def create_priority_chart(ranked: pd.DataFrame, output_path: Path = None) -> str:
    """Create priority score bar chart."""
    print("Creating priority chart...")
    
    if output_path is None:
        output_path = HEATMAPS_DIR / "priority_chart.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Top 10 hotspots
    top10 = ranked.head(10)
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Priority Score", "Component Breakdown"),
        column_widths=[0.5, 0.5],
    )
    
    # Bar chart
    fig.add_trace(
        go.Bar(
            x=top10["priority_score"],
            y=[f"C{cid}" for cid in top10["cluster_id"]],
            orientation="h",
            marker_color="coral",
            name="Priority Score",
        ),
        row=1, col=1,
    )
    
    # Component breakdown (stacked bar)
    for component in ["frequency", "impact", "urgency", "criticality"]:
        if component in top10.columns:
            fig.add_trace(
                go.Bar(
                    x=top10[component] if component in top10.columns else [0] * len(top10),
                    y=[f"C{cid}" for cid in top10["cluster_id"]],
                    orientation="h",
                    name=component.title(),
                ),
                row=1, col=2,
            )
    
    fig.update_layout(
        barmode="stack",
        height=500,
        title_text="Top 10 Hotspots by Priority Score",
        showlegend=True,
    )
    
    fig.write_image(str(output_path))
    print(f"  Saved to {output_path}")
    return str(output_path)


def create_temporal_heatmap(df: pd.DataFrame, output_path: Path = None) -> str:
    """Create day-of-week × hour heatmap."""
    print("Creating temporal heatmap...")
    
    if output_path is None:
        output_path = HEATMAPS_DIR / "temporal_heatmap.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter clustered data
    clustered = df[df["cluster_id"] != -1].copy()
    
    # Create pivot table
    pivot = clustered.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    
    # Day names
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot.index = [day_names[i] for i in pivot.index]
    
    # Plot
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt="d",
        linewidths=0.5,
        cbar_kws={"label": "Violations"},
    )
    plt.title("Violation Density by Day and Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"  Saved to {output_path}")
    return str(output_path)


def create_prediction_vs_actual_chart(backtest: pd.DataFrame,
                                       output_path: Path = None) -> str:
    """Create prediction vs actual line chart."""
    print("Creating prediction vs actual chart...")
    
    if output_path is None:
        output_path = HEATMAPS_DIR / "prediction_vs_actual.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=backtest["date"],
        y=backtest["mean_actual"],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#FF6B6B", width=2),
    ))
    
    fig.add_trace(go.Scatter(
        x=backtest["date"],
        y=backtest["mean_predicted"],
        mode="lines+markers",
        name="Predicted",
        line=dict(color="#4ECDC4", width=2),
    ))
    
    # Confidence band (if available)
    if "upper_bound" in backtest.columns and "lower_bound" in backtest.columns:
        fig.add_trace(go.Scatter(
            x=backtest["date"].tolist() + backtest["date"].tolist()[::-1],
            y=backtest["upper_bound"].tolist() + backtest["lower_bound"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(78, 205, 196, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval",
        ))
    
    fig.update_layout(
        title="Prediction vs Actual Violations",
        xaxis_title="Date",
        yaxis_title="Violations",
        height=400,
        template="plotly_dark",
    )
    
    fig.write_image(str(output_path))
    print(f"  Saved to {output_path}")
    return str(output_path)


# ==================== INTEGRATION ====================

def generate_all_visualizations(df: pd.DataFrame, profiles: Dict[int, Dict],
                                 impact: Dict[int, Dict], predictions: pd.DataFrame,
                                 ranked: pd.DataFrame) -> Dict[str, str]:
    """Generate all visualizations and return paths."""
    print("Generating all visualizations...")
    
    paths = {}
    
    paths["hotspot_heatmap"] = create_hotspot_heatmap(profiles, impact)
    paths["impact_map"] = create_impact_map(profiles, impact)
    paths["temporal_heatmap"] = create_temporal_heatmap(df)
    
    if predictions is not None and len(predictions) > 0:
        paths["predictions_heatmap"] = create_predictions_heatmap(predictions)
    
    if ranked is not None and len(ranked) > 0:
        paths["priority_chart"] = create_priority_chart(ranked)
    
    print(f"  Generated {len(paths)} visualizations")
    return paths


if __name__ == "__main__":
    from src.data_pipeline import run_pipeline
    from src.clustering import cluster_hotspots
    from src.traffic_impact import run_impact_analysis
    from src.scoring import rank_hotspots
    
    df = run_pipeline()
    df, profiles = cluster_hotspots(df)
    impact, ripple = run_impact_analysis(df, profiles)
    
    paths = generate_all_visualizations(df, profiles, impact, None, None)
    print(f"\nVisualization paths: {paths}")
