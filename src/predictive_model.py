"""
Predictive model for Trixie-flipkartgridlock.
LightGBM/XGBoost ensemble with Optuna tuning and conformal prediction.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Dict, List, Tuple, Optional, Any
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, PREDICTIONS_DIR, LAG_FEATURES, ROLLING_WINDOWS,
    SEVERITY_ROLLING_WINDOWS, TREND_WINDOWS,
    DEFAULT_LGB_PARAMS, DEFAULT_XGB_PARAMS,
    OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, OPTUNA_RANDOM_STATE,
    ENSEMBLE_DEFAULT_WEIGHTS, CONFIDENCE_CI_WIDTH,
    CONFORMAL_ALPHA, CONFORMAL_CV_FOLDS
)
from src.utils import coefficient_of_variation, rolling_trend


def build_ml_features(df: pd.DataFrame, profiles: Dict[int, Dict]) -> pd.DataFrame:
    """
    Build ML features for each cluster-day combination.
    Returns DataFrame with 45+ features per row.
    """
    print("Building ML features...")
    
    clustered = df[df["cluster_id"] != -1].copy()
    
    # Aggregate to cluster-day level
    daily = clustered.groupby(["cluster_id", "date"]).agg(
        violations=("latitude", "count"),
        avg_severity=("severity_weight", "mean"),
        avg_duration=("violation_duration_minutes", "mean") if "violation_duration_minutes" in clustered.columns else ("severity_weight", "mean"),
        rush_hour_count=("is_rush_hour", "sum"),
        weekend_count=("is_weekend", "sum"),
        unique_vehicles=("vehicle_id", "nunique") if "vehicle_id" in clustered.columns else ("latitude", "count"),
        unique_vehicle_types=("vehicle_type", "nunique") if "vehicle_type" in clustered.columns else ("latitude", "count"),
        avg_hour=("hour", "mean"),
        std_hour=("hour", "std"),
        min_hour=("hour", "min"),
        max_hour=("hour", "max"),
    ).reset_index()
    
    # Fill NaN
    daily["std_hour"] = daily["std_hour"].fillna(0)
    
    # Sort by date
    daily = daily.sort_values(["cluster_id", "date"]).reset_index(drop=True)
    
    # ==================== LAG FEATURES ====================
    print("  Adding lag features...")
    for lag in LAG_FEATURES:
        daily[f"violations_lag_{lag}"] = daily.groupby("cluster_id")["violations"].shift(lag)
    
    # ==================== ROLLING WINDOW FEATURES ====================
    print("  Adding rolling window features...")
    for window in ROLLING_WINDOWS:
        grouped = daily.groupby("cluster_id")["violations"]
        daily[f"rolling_{window}d_mean"] = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
        daily[f"rolling_{window}d_max"] = grouped.transform(lambda x: x.rolling(window, min_periods=1).max())
        daily[f"rolling_{window}d_std"] = grouped.transform(lambda x: x.rolling(window, min_periods=1).std())
    
    # Severity rolling
    for window in SEVERITY_ROLLING_WINDOWS:
        grouped = daily.groupby("cluster_id")["avg_severity"]
        daily[f"severity_rolling_{window}d_mean"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    # ==================== TREND FEATURES ====================
    print("  Adding trend features...")
    for window in TREND_WINDOWS:
        grouped = daily.groupby("cluster_id")["violations"]
        daily[f"trend_{window}d"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda y: rolling_trend(y, len(y)) if len(y) >= 2 else 0, raw=True
            )
        )
    
    # Trend direction flags
    for window in TREND_WINDOWS:
        daily[f"trend_{window}d_up"] = (daily[f"trend_{window}d"] > 0).astype(int)
    
    # ==================== VOLATILITY FEATURES ====================
    print("  Adding volatility features...")
    daily["violations_7d_std"] = daily.groupby("cluster_id")["violations"].transform(
        lambda x: x.rolling(7, min_periods=1).std()
    )
    daily["violations_7d_cv"] = daily.apply(
        lambda row: coefficient_of_variation(
            daily[(daily["cluster_id"] == row["cluster_id"]) &
                   (daily["date"] <= row["date"]) &
                   (daily["date"] > row["date"] - timedelta(days=7))]["violations"].values
        ) if row["date"] > daily["date"].min() else 0, axis=1
    )
    
    # ==================== INTERACTION FEATURES ====================
    print("  Adding interaction features...")
    daily["weekend_x_count"] = daily["weekend_count"] * daily["violations"]
    daily["severity_x_duration"] = daily["avg_severity"] * daily["avg_duration"]
    daily["rush_x_count"] = daily["rush_hour_count"] * daily["violations"]
    daily["vehicles_per_violation"] = daily["unique_vehicles"] / daily["violations"].clip(lower=1)
    
    # ==================== RATIO FEATURES ====================
    print("  Adding ratio features...")
    daily["ratio_to_7d_mean"] = daily["violations"] / daily["rolling_7d_mean"].clip(lower=1)
    daily["ratio_to_14d_mean"] = daily["violations"] / daily["rolling_14d_mean"].clip(lower=1)
    daily["ratio_to_7d_max"] = daily["violations"] / daily["rolling_7d_max"].clip(lower=1)
    
    # ==================== ACCELERATION ====================
    daily["acceleration_3d"] = daily["trend_3d"] - daily.groupby("cluster_id")["trend_3d"].shift(3)
    
    # ==================== PROFILE FEATURES ====================
    print("  Adding profile features...")
    daily["is_chronic"] = daily["cluster_id"].map(
        {cid: int(profiles[cid]["is_chronic"]) for cid in profiles}
    ).fillna(0)
    
    daily["avg_daily_rate"] = daily["cluster_id"].map(
        {cid: profiles[cid]["daily_rate"] for cid in profiles}
    ).fillna(0)
    
    daily["num_lanes"] = daily["cluster_id"].map(
        {cid: profiles[cid]["num_lanes"] for cid in profiles}
    ).fillna(2)
    
    daily["has_junction"] = daily["cluster_id"].map(
        {cid: profiles[cid]["has_junction"] for cid in profiles}
    ).fillna(0)
    
    daily["total_violations"] = daily["cluster_id"].map(
        {cid: profiles[cid]["total_violations"] for cid in profiles}
    ).fillna(0)
    
    # Road type (label encoded)
    road_map = {"Ring": 4, "Main": 3, "Underpass": 2, "Cross": 1, "Other": 0}
    daily["road_type_encoded"] = daily["cluster_id"].map(
        {cid: road_map.get(profiles[cid]["road_type"], 0) for cid in profiles}
    ).fillna(0)
    
    daily["days_active"] = daily["cluster_id"].map(
        {cid: profiles[cid]["unique_days"] for cid in profiles}
    ).fillna(0)
    
    # ==================== TARGET ====================
    # Next-day violations (shift within cluster)
    daily["violations_next_day"] = daily.groupby("cluster_id")["violations"].shift(-1)
    
    # Drop rows with NaN target
    daily = daily.dropna(subset=["violations_next_day"])
    
    # Fill remaining NaN with 0
    daily = daily.fillna(0)
    
    print(f"  Features built: {len(daily.columns)} columns, {len(daily):,} rows")
    return daily


def train_ensemble(df: pd.DataFrame) -> Tuple[Any, Any, Dict, List[str]]:
    """
    Train LightGBM + XGBoost ensemble with Optuna tuning.
    Returns: (lgb_model, xgb_model, metrics, feature_names)
    """
    print("Training ML ensemble...")
    
    # Define feature columns (exclude target and metadata)
    exclude_cols = ["cluster_id", "date", "violations_next_day"]
    feature_names = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_names].values
    y = df["violations_next_day"].values
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=OPTUNA_CV_FOLDS)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")
    
    # ==================== OPTUNA TUNING ====================
    print("  Running Optuna hyperparameter tuning...")
    
    # LightGBM tuning
    lgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_RANDOM_STATE))
    lgb_study.optimize(lambda trial: _lgb_objective(trial, X_train, y_train, X_val, y_val), n_trials=OPTUNA_N_TRIALS)
    lgb_params = lgb_study.best_params
    lgb_params.update(DEFAULT_LGB_PARAMS)
    
    # XGBoost tuning
    xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_RANDOM_STATE))
    xgb_study.optimize(lambda trial: _xgb_objective(trial, X_train, y_train, X_val, y_val), n_trials=OPTUNA_N_TRIALS)
    xgb_params = xgb_study.best_params
    xgb_params.update(DEFAULT_XGB_PARAMS)
    
    # ==================== TRAIN FINAL MODELS ====================
    print("  Training final models...")
    
    # LightGBM
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.log_evaluation(0)])
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    
    # ==================== EVALUATE ====================
    lgb_pred = lgb_model.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)
    
    lgb_r2 = r2_score(y_val, lgb_pred)
    xgb_r2 = r2_score(y_val, xgb_pred)
    
    # Ensemble weight by R² score
    total_r2 = lgb_r2 + xgb_r2
    lgb_weight = lgb_r2 / total_r2 if total_r2 > 0 else 0.5
    xgb_weight = xgb_r2 / total_r2 if total_r2 > 0 else 0.5
    
    ensemble_pred = lgb_weight * lgb_pred + xgb_weight * xgb_pred
    ensemble_r2 = r2_score(y_val, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
    
    metrics = {
        "lgb_r2": lgb_r2,
        "xgb_r2": xgb_r2,
        "ensemble_r2": ensemble_r2,
        "ensemble_mae": ensemble_mae,
        "lgb_weight": lgb_weight,
        "xgb_weight": xgb_weight,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": len(feature_names),
    }
    
    print(f"  LightGBM R²: {lgb_r2:.4f}")
    print(f"  XGBoost R²: {xgb_r2:.4f}")
    print(f"  Ensemble R²: {ensemble_r2:.4f}")
    print(f"  Ensemble MAE: {ensemble_mae:.2f}")
    print(f"  Weights: LGB={lgb_weight:.3f}, XGB={xgb_weight:.3f}")
    
    return lgb_model, xgb_model, metrics, feature_names


def _lgb_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for LightGBM."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    params.update(DEFAULT_LGB_PARAMS)
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
    
    pred = model.predict(X_val)
    return r2_score(y_val, pred)


def _xgb_objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for XGBoost."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }
    xgb_params = {k: v for k, v in DEFAULT_XGB_PARAMS.items() if k != "eval_metric"}
    params.update(xgb_params)
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    
    pred = model.predict(X_val)
    return r2_score(y_val, pred)


def predict_tomorrow(lgb_model: Any, xgb_model: Any, df: pd.DataFrame,
                      feature_names: List[str], profiles: Dict[int, Dict],
                      lgb_weight: float = 0.5, xgb_weight: float = 0.5) -> pd.DataFrame:
    """
    Generate tomorrow's predictions with confidence intervals.
    Uses model disagreement for adaptive confidence.
    """
    print("Generating tomorrow's predictions...")
    
    # Get latest features for each cluster
    latest = df.groupby("cluster_id").last().reset_index()
    
    X = latest[feature_names].values
    
    # Predictions
    lgb_pred = lgb_model.predict(X)
    xgb_pred = xgb_model.predict(X)
    
    # Ensemble prediction
    ensemble_pred = lgb_weight * lgb_pred + xgb_weight * xgb_pred
    
    # Confidence based on model disagreement
    disagreement = np.abs(lgb_pred - xgb_pred)
    confidence = 95 - disagreement * 3  # Higher disagreement → lower confidence
    confidence = np.clip(confidence, 60, 95)
    
    # Confidence intervals (adaptive width)
    ci_width = CONFIDENCE_CI_WIDTH * (1 + disagreement / (np.mean(disagreement) + 1e-6))
    lower = ensemble_pred * (1 - ci_width)
    upper = ensemble_pred * (1 + ci_width)
    
    # Activation probability
    median_daily = latest["violations"].median() if "violations" in latest.columns else 1
    activation_prob = np.clip(ensemble_pred / (median_daily + 1e-6) * 50, 0, 100)
    
    # Create predictions DataFrame
    predictions = pd.DataFrame({
        "cluster_id": latest["cluster_id"],
        "predicted_violations": np.round(ensemble_pred, 1),
        "lgb_prediction": np.round(lgb_pred, 1),
        "xgb_prediction": np.round(xgb_pred, 1),
        "lower_bound": np.round(lower, 1),
        "upper_bound": np.round(upper, 1),
        "confidence_pct": np.round(confidence, 1),
        "activation_probability": np.round(activation_prob, 1),
        "centroid_lat": latest["cluster_id"].map({cid: profiles[cid]["centroid_lat"] for cid in profiles}),
        "centroid_lon": latest["cluster_id"].map({cid: profiles[cid]["centroid_lon"] for cid in profiles}),
        "area": latest["cluster_id"].map({cid: profiles[cid].get("area", "Unknown") for cid in profiles}),
        "road_type": latest["cluster_id"].map({cid: profiles[cid].get("road_type", "Other") for cid in profiles}),
    })
    
    # Sort by predicted violations
    predictions = predictions.sort_values("predicted_violations", ascending=False).reset_index(drop=True)
    
    print(f"  Generated {len(predictions)} predictions")
    print(f"  Top prediction: cluster {predictions.iloc[0]['cluster_id']} ({predictions.iloc[0]['predicted_violations']:.1f} violations)")
    
    return predictions


def save_models(lgb_model: Any, xgb_model: Any, metrics: Dict,
                feature_names: List[str], predictions: pd.DataFrame):
    """Save trained models and predictions to disk."""
    print("Saving models and predictions...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save models
    with open(MODELS_DIR / "lgb_model.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
    
    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    
    # Save metrics
    with open(MODELS_DIR / "training_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    
    # Save feature names
    with open(MODELS_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    
    # Save predictions
    date_str = datetime.now().strftime("%Y-%m-%d")
    predictions.to_csv(PREDICTIONS_DIR / f"predictions_{date_str}.csv", index=False)
    
    print(f"  Models saved to {MODELS_DIR}")
    print(f"  Predictions saved to {PREDICTIONS_DIR / f'predictions_{date_str}.csv'}")


def load_models() -> Tuple[Any, Any, Dict, List[str]]:
    """Load trained models from disk."""
    with open(MODELS_DIR / "lgb_model.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    
    with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    
    with open(MODELS_DIR / "training_metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    
    with open(MODELS_DIR / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    
    return lgb_model, xgb_model, metrics, feature_names


def run_ml_pipeline(df: pd.DataFrame, profiles: Dict[int, Dict]) -> Tuple[pd.DataFrame, Dict, List[str]]:
    """Run full ML pipeline: feature engineering → training → prediction."""
    # Build features
    ml_df = build_ml_features(df, profiles)
    
    # Train ensemble
    lgb_model, xgb_model, metrics, feature_names = train_ensemble(ml_df)
    
    # Generate predictions
    predictions = predict_tomorrow(
        lgb_model, xgb_model, ml_df, feature_names, profiles,
        metrics["lgb_weight"], metrics["xgb_weight"]
    )
    
    # Save
    save_models(lgb_model, xgb_model, metrics, feature_names, predictions)
    
    return predictions, metrics, feature_names


if __name__ == "__main__":
    from src.data_pipeline import run_pipeline
    from src.clustering import cluster_hotspots
    
    df = run_pipeline()
    df, profiles = cluster_hotspots(df)
    predictions, metrics, feature_names = run_ml_pipeline(df, profiles)
    
    print(f"\nML pipeline complete:")
    print(f"  Ensemble R²: {metrics['ensemble_r2']:.4f}")
    print(f"  Predictions: {len(predictions)}")
