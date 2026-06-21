"""
Model monitoring for Trixie-flipkartgridlock.
Drift detection, performance tracking, backtesting.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, VALIDATION_DIR, DRIFT_THRESHOLD,
    RETRAIN_WINDOW_DAYS, MIN_SAMPLES_FOR_METRICS
)


class ModelMonitor:
    """Track model performance and detect drift."""
    
    def __init__(self, history_path: Path = None):
        """
        Initialize monitor.
        
        Args:
            history_path: Path to load/save prediction history
        """
        if history_path is None:
            history_path = VALIDATION_DIR / "prediction_history.pkl"
        
        self.history_path = history_path
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load prediction history from disk."""
        if self.history_path.exists():
            with open(self.history_path, "rb") as f:
                return pickle.load(f)
        return []
    
    def _save_history(self):
        """Save prediction history to disk."""
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "wb") as f:
            pickle.dump(self.history, f)
    
    def log_prediction(self, date: str, cluster_id: int, predicted: float,
                       actual: float = None):
        """
        Log a prediction (and optionally actual) for later evaluation.
        
        Args:
            date: Date string (YYYY-MM-DD)
            cluster_id: Cluster ID
            predicted: Predicted value
            actual: Actual value (if known)
        """
        entry = {
            "date": date,
            "cluster_id": cluster_id,
            "predicted": predicted,
            "actual": actual,
            "error": actual - predicted if actual is not None else None,
            "abs_error": abs(actual - predicted) if actual is not None else None,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.history.append(entry)
        self._save_history()
    
    def log_predictions_batch(self, date: str, predictions_df: pd.DataFrame,
                               actuals: Dict[int, float] = None):
        """
        Log a batch of predictions.
        
        Args:
            date: Date string
            predictions_df: DataFrame with 'cluster_id' and 'predicted_violations'
            actuals: Dict mapping cluster_id → actual violations
        """
        for _, row in predictions_df.iterrows():
            cid = int(row["cluster_id"])
            actual = actuals.get(cid) if actuals else None
            self.log_prediction(date, cid, row["predicted_violations"], actual)
    
    def compute_metrics(self, window_days: int = 7,
                         cluster_id: int = None) -> Optional[Dict]:
        """
        Compute performance metrics over a time window.
        
        Args:
            window_days: Number of days to look back
            cluster_id: Optional filter for specific cluster
            
        Returns:
            Dict with metrics or None if insufficient data
        """
        # Filter to entries with actuals
        with_actuals = [h for h in self.history if h["actual"] is not None]
        
        if len(with_actuals) < MIN_SAMPLES_FOR_METRICS:
            return None
        
        # Filter by date window
        cutoff_date = (datetime.now() - timedelta(days=window_days)).strftime("%Y-%m-%d")
        recent = [h for h in with_actuals if h["date"] >= cutoff_date]
        
        # Filter by cluster if specified
        if cluster_id is not None:
            recent = [h for h in recent if h["cluster_id"] == cluster_id]
        
        if len(recent) < MIN_SAMPLES_FOR_METRICS:
            return None
        
        actuals = np.array([h["actual"] for h in recent])
        preds = np.array([h["predicted"] for h in recent])
        errors = actuals - preds
        
        # Compute metrics
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mape = float(np.mean(np.abs(errors / (actuals + 1e-6))) * 100)
        
        # R² score
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-6))
        
        return {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "r2": round(r2, 4),
            "n_samples": len(recent),
            "window_days": window_days,
            "cluster_id": cluster_id,
        }
    
    def detect_drift(self, recent_window: int = 7, baseline_window: int = 30,
                      threshold: float = None) -> Dict:
        """
        Detect model drift by comparing recent vs baseline performance.
        
        Args:
            recent_window: Days for recent performance
            baseline_window: Days for baseline performance
            threshold: Drift threshold (default from config)
            
        Returns:
            Dict with drift detection results
        """
        if threshold is None:
            threshold = DRIFT_THRESHOLD
        
        recent_metrics = self.compute_metrics(recent_window)
        baseline_metrics = self.compute_metrics(baseline_window)
        
        if recent_metrics is None or baseline_metrics is None:
            return {
                "drift_detected": False,
                "reason": "Insufficient data for drift detection",
                "recent_mae": recent_metrics["mae"] if recent_metrics else None,
                "baseline_mae": baseline_metrics["mae"] if baseline_metrics else None,
            }
        
        # Compare MAE
        recent_mae = recent_metrics["mae"]
        baseline_mae = baseline_metrics["mae"]
        
        drift_ratio = recent_mae / (baseline_mae + 1e-6)
        drift_detected = drift_ratio > (1 + threshold)
        
        # Compare R²
        recent_r2 = recent_metrics["r2"]
        baseline_r2 = baseline_metrics["r2"]
        r2_change = recent_r2 - baseline_r2
        
        return {
            "drift_detected": drift_detected,
            "drift_ratio": round(drift_ratio, 3),
            "threshold": threshold,
            "recent_mae": round(recent_mae, 2),
            "baseline_mae": round(baseline_mae, 2),
            "recent_r2": round(recent_r2, 4),
            "baseline_r2": round(baseline_r2, 4),
            "r2_change": round(r2_change, 4),
            "recommendation": "Consider retraining" if drift_detected else "Model stable",
        }
    
    def get_history_summary(self) -> Dict:
        """Get summary statistics of prediction history."""
        if not self.history:
            return {"total_predictions": 0}
        
        with_actuals = [h for h in self.history if h["actual"] is not None]
        
        dates = set(h["date"] for h in self.history)
        clusters = set(h["cluster_id"] for h in self.history)
        
        return {
            "total_predictions": len(self.history),
            "predictions_with_actuals": len(with_actuals),
            "unique_dates": len(dates),
            "unique_clusters": len(clusters),
            "date_range": {
                "min": min(dates) if dates else None,
                "max": max(dates) if dates else None,
            },
        }
    
    def generate_backtest_report(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate backtest report for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with daily metrics
        """
        # Filter to date range
        relevant = [
            h for h in self.history
            if start_date <= h["date"] <= end_date and h["actual"] is not None
        ]
        
        if not relevant:
            return pd.DataFrame()
        
        # Group by date
        df = pd.DataFrame(relevant)
        
        daily_metrics = []
        for date, group in df.groupby("date"):
            actuals = group["actual"].values
            preds = group["predicted"].values
            errors = actuals - preds
            
            daily_metrics.append({
                "date": date,
                "mae": round(float(np.mean(np.abs(errors))), 2),
                "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 2),
                "mape": round(float(np.mean(np.abs(errors / (actuals + 1e-6))) * 100), 2),
                "r2": round(float(1 - np.sum(errors ** 2) / (np.sum((actuals - np.mean(actuals)) ** 2) + 1e-6)), 4),
                "n_clusters": len(group),
                "mean_predicted": round(float(np.mean(preds)), 1),
                "mean_actual": round(float(np.mean(actuals)), 1),
            })
        
        return pd.DataFrame(daily_metrics).sort_values("date")


def create_monitor() -> ModelMonitor:
    """Factory function to create model monitor."""
    return ModelMonitor()


if __name__ == "__main__":
    # Test with dummy data
    monitor = ModelMonitor()
    
    # Log some dummy predictions
    for i in range(10):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        for cid in range(5):
            predicted = np.random.poisson(10)
            actual = predicted + np.random.randint(-2, 3)
            monitor.log_prediction(date, cid, predicted, actual)
    
    # Compute metrics
    metrics = monitor.compute_metrics(7)
    print(f"Metrics: {metrics}")
    
    # Detect drift
    drift = monitor.detect_drift()
    print(f"Drift: {drift}")
    
    # Summary
    summary = monitor.get_history_summary()
    print(f"Summary: {summary}")
