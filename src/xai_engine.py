"""
Explainable AI (XAI) engine for Trixie-flipkartgridlock.
True SHAP-based explanations from trained model (not heuristic weights).
"""
import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Optional, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import XAI_BACKGROUND_SAMPLES, XAI_MAX_DISPLAY_FEATURES


class XAIEngine:
    """SHAP-based explainable AI for parking prediction models."""
    
    def __init__(self, model: Any, X_background: pd.DataFrame, feature_names: List[str]):
        """
        Initialize XAI engine.
        
        Args:
            model: Trained LightGBM or XGBoost model
            X_background: Background dataset for SHAP (sample)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.X_background = X_background
        
        # Create SHAP explainer
        print("Initializing SHAP explainer...")
        try:
            self.explainer = shap.TreeExplainer(model, X_background)
            print("  SHAP TreeExplainer initialized")
        except Exception as e:
            print(f"  Warning: TreeExplainer failed ({e}), using KernelExplainer")
            # Fallback to KernelExplainer
            background_sample = shap.sample(X_background, min(XAI_BACKGROUND_SAMPLES, len(X_background)))
            self.explainer = shap.KernelExplainer(model.predict, background_sample)
    
    def explain_prediction(self, X_single: pd.DataFrame) -> Dict:
        """
        Explain a single prediction.
        
        Returns:
            Dict with feature contributions, base value, prediction
        """
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X_single)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        
        # Create contribution dict
        contributions = {}
        for i, name in enumerate(self.feature_names):
            contributions[name] = {
                "value": float(X_single.iloc[0][name]) if name in X_single.columns else 0,
                "shap_value": float(shap_values[i]),
                "direction": "positive" if shap_values[i] > 0 else "negative",
                "magnitude": abs(float(shap_values[i])),
            }
        
        # Sort by magnitude
        contributions = dict(
            sorted(contributions.items(), key=lambda x: x[1]["magnitude"], reverse=True)
        )
        
        # Prediction
        prediction = float(self.model.predict(X_single)[0])
        
        return {
            "prediction": prediction,
            "base_value": float(self.explainer.expected_value) if np.isscalar(self.explainer.expected_value) else float(self.explainer.expected_value[0]),
            "contributions": contributions,
            "top_features": dict(list(contributions.items())[:XAI_MAX_DISPLAY_FEATURES]),
        }
    
    def explain_cluster(self, cluster_features: pd.DataFrame) -> Dict:
        """
        Explain a cluster by aggregating SHAP values across all its data points.
        
        Args:
            cluster_features: DataFrame with all data points for a cluster
            
        Returns:
            Dict with aggregated explanations
        """
        if len(cluster_features) == 0:
            return {"error": "No data points in cluster"}
        
        # Compute SHAP values for all points
        shap_values = self.explainer.shap_values(cluster_features)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Aggregate: mean absolute SHAP
        mean_shap = np.mean(shap_values, axis=0)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        std_shap = np.std(shap_values, axis=0)
        
        # Create feature importance dict
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            feature_importance[name] = {
                "mean_shap": float(mean_shap[i]),
                "mean_abs_shap": float(mean_abs_shap[i]),
                "std_shap": float(std_shap[i]),
                "direction": "positive" if mean_shap[i] > 0 else "negative",
                "consistency": 1.0 - min(float(std_shap[i] / (abs(mean_shap[i]) + 1e-6)), 1.0),
            }
        
        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1]["mean_abs_shap"], reverse=True)
        )
        
        # Identify dominant factor
        dominant_factor = list(feature_importance.keys())[0] if feature_importance else "Unknown"
        
        # Compute percentage contributions
        total_abs = sum(v["mean_abs_shap"] for v in feature_importance.values())
        percentage_contributions = {}
        for name, data in feature_importance.items():
            pct = (data["mean_abs_shap"] / total_abs * 100) if total_abs > 0 else 0
            percentage_contributions[name] = round(pct, 2)
        
        return {
            "n_samples": len(cluster_features),
            "feature_importance": feature_importance,
            "percentage_contributions": percentage_contributions,
            "dominant_factor": dominant_factor,
            "top_5_features": dict(list(feature_importance.items())[:5]),
        }
    
    def explain_global(self, X: pd.DataFrame) -> Dict:
        """
        Global feature importance across entire dataset.
        
        Returns:
            Dict with global feature importance rankings
        """
        print("Computing global SHAP explanations...")
        
        # Sample for efficiency
        sample_size = min(XAI_BACKGROUND_SAMPLES * 2, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        shap_values = self.explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Global importance: mean absolute SHAP
        global_importance = {}
        for i, name in enumerate(self.feature_names):
            global_importance[name] = float(np.mean(np.abs(shap_values[:, i])))
        
        # Sort
        global_importance = dict(
            sorted(global_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Percentage
        total = sum(global_importance.values())
        global_percentage = {
            name: round(val / total * 100, 2) if total > 0 else 0
            for name, val in global_importance.items()
        }
        
        return {
            "global_importance": global_importance,
            "global_percentage": global_percentage,
            "top_10": dict(list(global_importance.items())[:10]),
        }
    
    def generate_root_breakdown(self, cluster_features: pd.DataFrame,
                                 factor_groups: Dict[str, List[str]] = None) -> Dict:
        """
        Generate human-readable root cause breakdown for a cluster.
        
        Args:
            cluster_features: Features for the cluster
            factor_groups: Optional mapping of factor name to feature names
            
        Returns:
            Dict with percentage breakdown by factor
        """
        if factor_groups is None:
            # Default factor groups
            factor_groups = {
                "Illegal Parking": ["severity_weight", "dominant_violation"],
                "Road Width": ["num_lanes", "road_type", "road_importance"],
                "Density": ["total_violations", "daily_rate"],
                "Time of Day": ["hour", "is_rush_hour", "time_bin"],
                "Junction Proximity": ["has_junction"],
            }
        
        # Get SHAP explanations
        explanation = self.explain_cluster(cluster_features)
        
        # Map features to factors
        factor_contributions = {}
        for factor, features in factor_groups.items():
            factor_shap = sum(
                explanation["feature_importance"].get(f, {}).get("mean_abs_shap", 0)
                for f in features
            )
            factor_contributions[factor] = factor_shap
        
        # Normalize to percentages
        total = sum(factor_contributions.values())
        factor_percentages = {
            factor: round(shap / total * 100, 1) if total > 0 else 0
            for factor, shap in factor_contributions.items()
        }
        
        # Sort
        factor_percentages = dict(
            sorted(factor_percentages.items(), key=lambda x: x[1], reverse=True)
        )
        
        dominant_factor = list(factor_percentages.keys())[0] if factor_percentages else "Unknown"
        
        return {
            "factor_percentages": factor_percentages,
            "dominant_factor": dominant_factor,
            "explanation": explanation,
        }


def create_xai_engine(model: Any, X_background: pd.DataFrame,
                       feature_names: List[str]) -> XAIEngine:
    """Factory function to create XAI engine."""
    return XAIEngine(model, X_background, feature_names)


if __name__ == "__main__":
    # Test with dummy data
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 200
    feature_names = [f"feature_{i}" for i in range(10)]
    X = pd.DataFrame(np.random.randn(n_samples, 10), columns=feature_names)
    y = np.random.randn(n_samples)
    
    # Train model
    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Create XAI engine
    engine = XAIEngine(model, X.iloc[:50], feature_names)
    
    # Explain single prediction
    result = engine.explain_prediction(X.iloc[[0]])
    print("Single prediction explanation:")
    print(f"  Prediction: {result['prediction']:.3f}")
    print(f"  Base value: {result['base_value']:.3f}")
    print(f"  Top feature: {list(result['top_features'].keys())[0]}")
    
    # Explain cluster
    cluster_result = engine.explain_cluster(X.iloc[:20])
    print(f"\nCluster explanation ({cluster_result['n_samples']} samples):")
    print(f"  Dominant factor: {cluster_result['dominant_factor']}")
