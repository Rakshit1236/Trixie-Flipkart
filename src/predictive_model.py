"""
Predictive model for Trixie-flipkartgridlock.
LightGBM + XGBoost + CatBoost ensemble with Optuna tuning, stacking, and conformal prediction.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import RidgeCV
from typing import Dict, List, Tuple, Optional, Any
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR, PREDICTIONS_DIR, LAG_FEATURES, ROLLING_WINDOWS,
    SEVERITY_ROLLING_WINDOWS, TREND_WINDOWS,
    DEFAULT_LGB_PARAMS, DEFAULT_XGB_PARAMS, DEFAULT_CATBOOST_PARAMS,
    OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS, OPTUNA_RANDOM_STATE,
    ENSEMBLE_DEFAULT_WEIGHTS, CONFIDENCE_CI_WIDTH,
    FEATURE_IMPORTANCE_THRESHOLD, TOP_FEATURES
)
from src.utils import coefficient_of_variation, rolling_trend


def build_ml_features(df: pd.DataFrame, profiles: Dict[int, Dict]) -> pd.DataFrame:
    """
    Build ML features for each cluster-day combination.
    Returns DataFrame with 60+ engineered features per row.
    """
    print("Building ML features...")

    clustered = df[df["cluster_id"] != -1].copy()

    # Aggregate to cluster-day level
    agg_dict = {
        "violations": ("latitude", "count"),
        "avg_severity": ("severity_weight", "mean"),
        "rush_hour_count": ("is_rush_hour", "sum"),
        "weekend_count": ("is_weekend", "sum"),
        "avg_hour": ("hour", "mean"),
        "std_hour": ("hour", "std"),
        "min_hour": ("hour", "min"),
        "max_hour": ("hour", "max"),
    }

    if "violation_duration_minutes" in clustered.columns:
        agg_dict["avg_duration"] = ("violation_duration_minutes", "mean")
    if "vehicle_id" in clustered.columns:
        agg_dict["unique_vehicles"] = ("vehicle_id", "nunique")
    else:
        agg_dict["unique_vehicles"] = ("latitude", "count")
    if "vehicle_type" in clustered.columns:
        agg_dict["unique_vehicle_types"] = ("vehicle_type", "nunique")
    else:
        agg_dict["unique_vehicle_types"] = ("latitude", "count")

    daily = clustered.groupby(["cluster_id", "date"]).agg(**{
        k: v for k, v in agg_dict.items()
    }).reset_index()

    daily["std_hour"] = daily["std_hour"].fillna(0)
    daily = daily.sort_values(["cluster_id", "date"]).reset_index(drop=True)

    # Ensure date is datetime
    daily["date"] = pd.to_datetime(daily["date"])

    # ==================== LAG FEATURES ====================
    print("  Adding lag features...")
    for lag in LAG_FEATURES:
        daily[f"violations_lag_{lag}"] = daily.groupby("cluster_id")["violations"].shift(lag)

    # ==================== ROLLING WINDOW FEATURES ====================
    print("  Adding rolling window features...")
    for window in ROLLING_WINDOWS:
        grp = daily.groupby("cluster_id")["violations"]
        daily[f"rolling_{window}d_mean"] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        daily[f"rolling_{window}d_max"] = grp.transform(lambda x: x.rolling(window, min_periods=1).max())
        daily[f"rolling_{window}d_std"] = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        daily[f"rolling_{window}d_min"] = grp.transform(lambda x: x.rolling(window, min_periods=1).min())

    # Severity rolling
    for window in SEVERITY_ROLLING_WINDOWS:
        grp = daily.groupby("cluster_id")["avg_severity"]
        daily[f"severity_rolling_{window}d_mean"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

    # ==================== TREND FEATURES ====================
    print("  Adding trend features...")
    for window in TREND_WINDOWS:
        grp = daily.groupby("cluster_id")["violations"]
        daily[f"trend_{window}d"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).apply(
                lambda y: rolling_trend(y, len(y)) if len(y) >= 2 else 0, raw=True
            )
        )

    for window in TREND_WINDOWS:
        daily[f"trend_{window}d_up"] = (daily[f"trend_{window}d"] > 0).astype(int)

    # ==================== VOLATILITY FEATURES (vectorised) ====================
    print("  Adding volatility features...")
    daily["violations_7d_std"] = daily.groupby("cluster_id")["violations"].transform(
        lambda x: x.rolling(7, min_periods=1).std().fillna(0)
    )
    daily["violations_7d_mean"] = daily.groupby("cluster_id")["violations"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    daily["violations_7d_cv"] = daily["violations_7d_std"] / daily["violations_7d_mean"].clip(lower=1)

    daily["violations_14d_std"] = daily.groupby("cluster_id")["violations"].transform(
        lambda x: x.rolling(14, min_periods=1).std().fillna(0)
    )
    daily["violations_14d_mean"] = daily.groupby("cluster_id")["violations"].transform(
        lambda x: x.rolling(14, min_periods=1).mean()
    )

    # ==================== CYCLICAL TIME FEATURES ====================
    print("  Adding cyclical time features...")
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["day_of_month"] = daily["date"].dt.day
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["month"] = daily["date"].dt.month
    daily["is_month_start"] = daily["date"].dt.is_month_start.astype(int)
    daily["is_month_end"] = daily["date"].dt.is_month_end.astype(int)

    daily["dow_sin"] = np.sin(2 * np.pi * daily["day_of_week"] / 7)
    daily["dow_cos"] = np.cos(2 * np.pi * daily["day_of_week"] / 7)
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)
    daily["dom_sin"] = np.sin(2 * np.pi * daily["day_of_month"] / 31)
    daily["dom_cos"] = np.cos(2 * np.pi * daily["day_of_month"] / 31)

    # ==================== INTERACTION FEATURES ====================
    print("  Adding interaction features...")
    daily["weekend_x_count"] = daily["weekend_count"] * daily["violations"]
    daily["severity_x_duration"] = daily["avg_severity"] * daily.get("avg_duration", 0)
    daily["rush_x_count"] = daily["rush_hour_count"] * daily["violations"]
    daily["vehicles_per_violation"] = daily["unique_vehicles"] / daily["violations"].clip(lower=1)

    # ==================== RATIO FEATURES ====================
    print("  Adding ratio features...")
    daily["ratio_to_7d_mean"] = daily["violations"] / daily["rolling_7d_mean"].clip(lower=1)
    daily["ratio_to_14d_mean"] = daily["violations"] / daily["rolling_14d_mean"].clip(lower=1)
    daily["ratio_to_7d_max"] = daily["violations"] / daily["rolling_7d_max"].clip(lower=1)
    daily["ratio_lag1_to_7d"] = daily["violations_lag_1"].fillna(0) / daily["rolling_7d_mean"].clip(lower=1)

    # ==================== ACCELERATION ====================
    daily["acceleration_3d"] = daily["trend_3d"] - daily.groupby("cluster_id")["trend_3d"].shift(3)
    daily["acceleration_7d"] = daily["trend_7d"] - daily.groupby("cluster_id")["trend_7d"].shift(7)

    # ==================== MOMENTUM ====================
    daily["momentum_3d"] = daily["violations"] - daily["violations_lag_3"].fillna(daily["violations"])
    daily["momentum_7d"] = daily["violations"] - daily["violations_lag_7"].fillna(daily["violations"])

    # ==================== RECENCY WEIGHTED MEAN ====================
    def recency_weighted_mean(vals):
        if len(vals) == 0:
            return 0
        weights = np.exp(np.linspace(-1, 0, len(vals)))
        return np.average(vals, weights=weights)

    daily["recency_wt_mean_7d"] = daily.groupby("cluster_id")["violations"].transform(
        lambda x: x.rolling(7, min_periods=1).apply(recency_weighted_mean, raw=True)
    )

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

    road_map = {"Ring": 4, "Main": 3, "Underpass": 2, "Cross": 1, "Other": 0}
    daily["road_type_encoded"] = daily["cluster_id"].map(
        {cid: road_map.get(profiles[cid]["road_type"], 0) for cid in profiles}
    ).fillna(0)

    daily["days_active"] = daily["cluster_id"].map(
        {cid: profiles[cid]["unique_days"] for cid in profiles}
    ).fillna(0)

    # ==================== CLUSTER AGGREGATE FEATURES ====================
    daily["cluster_rank_violations"] = daily.groupby("date")["violations"].rank(ascending=False, method="min")
    daily["cluster_zscore_violations"] = daily.groupby("date")["violations"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

    # ==================== TARGET ====================
    daily["violations_next_day"] = daily.groupby("cluster_id")["violations"].shift(-1)
    daily = daily.dropna(subset=["violations_next_day"])
    daily = daily.fillna(0)

    print(f"  Features built: {len(daily.columns)} columns, {len(daily):,} rows")
    return daily


def _select_features(model, feature_names, X_train, y_train, threshold=FEATURE_IMPORTANCE_THRESHOLD):
    """Select top features by importance."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return feature_names, X_train

    imp_norm = importances / (importances.sum() + 1e-8)
    mask = imp_norm >= threshold

    if mask.sum() < 10:
        top_idx = np.argsort(importances)[-TOP_FEATURES:]
        mask = np.zeros(len(feature_names), dtype=bool)
        mask[top_idx] = True

    selected = [f for f, m in zip(feature_names, mask) if m]
    X_selected = X_train[:, mask]
    print(f"  Feature selection: {len(feature_names)} -> {len(selected)} features")
    return selected, X_selected


def train_ensemble(df: pd.DataFrame) -> Tuple[Any, Any, Any, Any, Dict, List[str]]:
    """
    Train LightGBM + XGBoost + CatBoost ensemble with Optuna tuning + Ridge stacking.
    Returns: (lgb_model, xgb_model, cat_model, meta_model, metrics, feature_names)
    """
    print("Training ML ensemble...")

    exclude_cols = ["cluster_id", "date", "violations_next_day"]
    feature_names = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_names].values.astype(np.float32)
    y = df["violations_next_day"].values.astype(np.float32)

    # Log-transform target for count data
    y_log = np.log1p(y)

    # Time series split
    tscv = TimeSeriesSplit(n_splits=OPTUNA_CV_FOLDS)
    splits = list(tscv.split(X))
    train_idx, val_idx = splits[-1]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    y_train_log, y_val_log = y_log[train_idx], y_log[val_idx]

    # Middle split for stacking meta-learner
    meta_split = splits[len(splits) // 2]
    meta_train_idx, meta_val_idx = meta_split
    X_meta = X[meta_train_idx]
    y_meta_log = y_log[meta_train_idx]

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}, Features: {len(feature_names)}")

    # ==================== OPTUNA: LightGBM ====================
    print("  Tuning LightGBM (Optuna)...")
    lgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_RANDOM_STATE))
    lgb_study.optimize(lambda trial: _lgb_objective(trial, X_train, y_train_log, X_val, y_val_log), n_trials=OPTUNA_N_TRIALS)
    lgb_params = lgb_study.best_params
    lgb_params.update(DEFAULT_LGB_PARAMS)

    # ==================== OPTUNA: XGBoost ====================
    print("  Tuning XGBoost (Optuna)...")
    xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_RANDOM_STATE))
    xgb_study.optimize(lambda trial: _xgb_objective(trial, X_train, y_train_log, X_val, y_val_log), n_trials=OPTUNA_N_TRIALS)
    xgb_params = xgb_study.best_params
    xgb_params.update(DEFAULT_XGB_PARAMS)

    # ==================== OPTUNA: CatBoost ====================
    print("  Tuning CatBoost (Optuna)...")
    try:
        from catboost import CatBoostRegressor
        cat_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_RANDOM_STATE))
        cat_study.optimize(lambda trial: _cat_objective(trial, X_train, y_train_log, X_val, y_val_log), n_trials=max(5, OPTUNA_N_TRIALS // 3))
        cat_params = cat_study.best_params
        cat_params.update(DEFAULT_CATBOOST_PARAMS)
        use_catboost = True
    except ImportError:
        print("  CatBoost not available, skipping...")
        use_catboost = False
        cat_params = {}

    # ==================== TRAIN FINAL MODELS ====================
    print("  Training final models...")

    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], callbacks=[lgb.log_evaluation(0)])

    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=0)

    cat_model = None
    if use_catboost:
        from catboost import CatBoostRegressor
        cat_model = CatBoostRegressor(**cat_params)
        cat_model.fit(X_train, y_train_log, eval_set=[(X_val, y_val_log)], verbose=0)

    # ==================== FEATURE SELECTION ====================
    print("  Running feature selection...")
    selected_names, X_train_sel = _select_features(lgb_model, feature_names, X_train, y_train)

    X_val_sel = X_val[:, [feature_names.index(f) for f in selected_names]]

    # Retrain with selected features
    lgb_model.fit(X_train_sel, y_train_log, eval_set=[(X_val_sel, y_val_log)], callbacks=[lgb.log_evaluation(0)])
    xgb_model.fit(X_train_sel, y_train_log, eval_set=[(X_val_sel, y_val_log)], verbose=0)
    if cat_model:
        cat_model.fit(X_train_sel, y_train_log, eval_set=[(X_val_sel, y_val_log)], verbose=0)

    # ==================== STACKING META-LEARNER ====================
    print("  Training stacking meta-learner...")

    # Generate OOF predictions for meta-learner training
    n_models = 3 if cat_model else 2
    oof_preds = np.zeros((len(X), n_models))

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr_log = y_log[tr_idx]

        lgb_temp = lgb.LGBMRegressor(**lgb_params)
        lgb_temp.fit(X_tr, y_tr_log, callbacks=[lgb.log_evaluation(0)])
        oof_preds[va_idx, 0] = lgb_temp.predict(X_va)

        xgb_temp = xgb.XGBRegressor(**xgb_params)
        xgb_temp.fit(X_tr, y_tr_log, verbose=0)
        oof_preds[va_idx, 1] = xgb_temp.predict(X_va)

        if cat_model:
            from catboost import CatBoostRegressor
            cat_temp = CatBoostRegressor(**cat_params)
            cat_temp.fit(X_tr, y_tr_log, verbose=0)
            oof_preds[va_idx, 2] = cat_temp.predict(X_va)

    meta_model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    meta_model.fit(oof_preds, y_log)
    print(f"  Meta-learner weights: {meta_model.coef_}")

    # ==================== EVALUATE ====================
    lgb_pred_log = lgb_model.predict(X_val_sel)
    xgb_pred_log = xgb_model.predict(X_val_sel)

    lgb_pred = np.expm1(lgb_pred_log)
    xgb_pred = np.expm1(xgb_pred_log)

    lgb_r2 = r2_score(y_val, lgb_pred)
    xgb_r2 = r2_score(y_val, xgb_pred)

    preds_stack = np.column_stack([lgb_pred_log, xgb_pred_log])
    if cat_model:
        cat_pred_log = cat_model.predict(X_val_sel)
        cat_pred = np.expm1(cat_pred_log)
        cat_r2 = r2_score(y_val, cat_pred)
        preds_stack = np.column_stack([lgb_pred_log, xgb_pred_log, cat_pred_log])
    else:
        cat_r2 = 0

    meta_pred_log = meta_model.predict(preds_stack)
    meta_pred = np.expm1(meta_pred_log)

    meta_r2 = r2_score(y_val, meta_pred)
    meta_mae = mean_absolute_error(y_val, meta_pred)

    # Simple weighted average as backup
    total_r2 = lgb_r2 + xgb_r2 + (cat_r2 if cat_model else 0) + 1e-8
    lgb_weight = lgb_r2 / total_r2
    xgb_weight = xgb_r2 / total_r2
    cat_weight = cat_r2 / total_r2 if cat_model else 0

    simple_pred = lgb_weight * lgb_pred + xgb_weight * xgb_pred
    if cat_model:
        simple_pred += cat_weight * cat_pred
    simple_r2 = r2_score(y_val, simple_pred)

    # Use stacking if it beats simple, else use simple
    if meta_r2 > simple_r2:
        ensemble_pred = meta_pred
        ensemble_r2 = meta_r2
        ensemble_mae = meta_mae
        use_stacking = True
        print("  Using STACKING ensemble")
    else:
        ensemble_pred = simple_pred
        ensemble_r2 = simple_r2
        ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
        use_stacking = False
        print("  Using WEIGHTED ensemble")

    # ==================== CONFORMAL PREDICTION CALIBRATION ====================
    print("  Calibrating conformal prediction intervals...")
    residuals = np.abs(y_val - ensemble_pred)
    sorted_residuals = np.sort(residuals)
    alpha = 0.10
    q_idx = int(np.ceil((1 - alpha) * len(sorted_residuals))) - 1
    conformal_q = sorted_residuals[min(q_idx, len(sorted_residuals) - 1)]

    metrics = {
        "lgb_r2": lgb_r2,
        "xgb_r2": xgb_r2,
        "cat_r2": cat_r2 if cat_model else 0,
        "ensemble_r2": ensemble_r2,
        "ensemble_mae": ensemble_mae,
        "lgb_weight": lgb_weight,
        "xgb_weight": xgb_weight,
        "cat_weight": cat_weight,
        "use_stacking": use_stacking,
        "meta_weights": meta_model.coef_.tolist() if use_stacking else None,
        "conformal_q": float(conformal_q),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": len(feature_names),
        "n_selected_features": len(selected_names),
        "feature_names": selected_names,
    }

    print(f"  LightGBM R2: {lgb_r2:.4f}")
    print(f"  XGBoost R2:  {xgb_r2:.4f}")
    if cat_model:
        print(f"  CatBoost R2: {cat_r2:.4f}")
    print(f"  Ensemble R2: {ensemble_r2:.4f}")
    print(f"  Ensemble MAE: {ensemble_mae:.2f}")
    print(f"  Conformal q(0.90): {conformal_q:.2f}")

    return lgb_model, xgb_model, cat_model, meta_model, metrics, selected_names


def _lgb_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }
    params.update(DEFAULT_LGB_PARAMS)
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
    pred = model.predict(X_val)
    return r2_score(y_val, pred)


def _xgb_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }
    xgb_params = {k: v for k, v in DEFAULT_XGB_PARAMS.items() if k != "eval_metric"}
    params.update(xgb_params)
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    pred = model.predict(X_val)
    return r2_score(y_val, pred)


def _cat_objective(trial, X_train, y_train, X_val, y_val):
    from catboost import CatBoostRegressor
    params = {
        "iterations": trial.suggest_int("iterations", 200, 600),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }
    params.update(DEFAULT_CATBOOST_PARAMS)
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    pred = model.predict(X_val)
    return r2_score(y_val, pred)


def predict_tomorrow(lgb_model, xgb_model, cat_model, meta_model, df: pd.DataFrame,
                      feature_names: List[str], profiles: Dict[int, Dict], metrics: Dict) -> pd.DataFrame:
    """
    Generate tomorrow's predictions with conformal prediction intervals.
    """
    print("Generating predictions with conformal intervals...")

    latest = df.groupby("cluster_id").last().reset_index()
    X = latest[feature_names].values.astype(np.float32)

    lgb_pred_log = lgb_model.predict(X)
    xgb_pred_log = xgb_model.predict(X)

    if cat_model and metrics.get("use_stacking"):
        cat_pred_log = cat_model.predict(X)
        preds_stack = np.column_stack([lgb_pred_log, xgb_pred_log, cat_pred_log])
        meta_pred_log = meta_model.predict(preds_stack)
        ensemble_pred = np.expm1(meta_pred_log)
    elif cat_model:
        cat_pred_log = cat_model.predict(X)
        w_lgb = metrics["lgb_weight"]
        w_xgb = metrics["xgb_weight"]
        w_cat = metrics["cat_weight"]
        ensemble_pred = np.expm1(w_lgb * lgb_pred_log + w_xgb * xgb_pred_log + w_cat * cat_pred_log)
    else:
        w_lgb = metrics["lgb_weight"]
        w_xgb = metrics["xgb_weight"]
        ensemble_pred = np.expm1(w_lgb * lgb_pred_log + w_xgb * xgb_pred_log)

    lgb_pred = np.expm1(lgb_pred_log)
    xgb_pred = np.expm1(xgb_pred_log)

    # Conformal prediction intervals
    conformal_q = metrics["conformal_q"]
    lower = np.maximum(ensemble_pred - conformal_q, 0)
    upper = ensemble_pred + conformal_q

    # Confidence based on model agreement
    disagreement = np.abs(lgb_pred - xgb_pred) / (np.mean(np.abs(lgb_pred)) + 1e-6)
    confidence = np.clip(95 - disagreement * 30, 55, 98)

    # Activation probability
    median_daily = latest["violations"].median() if "violations" in latest.columns else 1
    activation_prob = np.clip(ensemble_pred / (median_daily + 1e-6) * 50, 0, 100)

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

    predictions = predictions.sort_values("predicted_violations", ascending=False).reset_index(drop=True)
    print(f"  Generated {len(predictions)} predictions")
    print(f"  Top: cluster {predictions.iloc[0]['cluster_id']} ({predictions.iloc[0]['predicted_violations']:.1f})")
    return predictions


def save_models(lgb_model, xgb_model, cat_model, meta_model, metrics: Dict,
                feature_names: List[str], predictions: pd.DataFrame):
    """Save trained models and predictions to disk."""
    print("Saving models and predictions...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "lgb_model.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    if cat_model:
        with open(MODELS_DIR / "cat_model.pkl", "wb") as f:
            pickle.dump(cat_model, f)
    with open(MODELS_DIR / "meta_model.pkl", "wb") as f:
        pickle.dump(meta_model, f)
    with open(MODELS_DIR / "training_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)
    with open(MODELS_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    date_str = datetime.now().strftime("%Y-%m-%d")
    predictions.to_csv(PREDICTIONS_DIR / f"predictions_{date_str}.csv", index=False)
    print(f"  Models saved to {MODELS_DIR}")


def load_models():
    """Load trained models from disk."""
    with open(MODELS_DIR / "lgb_model.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    cat_model = None
    cat_path = MODELS_DIR / "cat_model.pkl"
    if cat_path.exists():
        with open(cat_path, "rb") as f:
            cat_model = pickle.load(f)
    meta_path = MODELS_DIR / "meta_model.pkl"
    meta_model = None
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            meta_model = pickle.load(f)
    with open(MODELS_DIR / "training_metrics.pkl", "rb") as f:
        metrics = pickle.load(f)
    with open(MODELS_DIR / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return lgb_model, xgb_model, cat_model, meta_model, metrics, feature_names


def run_ml_pipeline(df: pd.DataFrame, profiles: Dict[int, Dict]):
    """Run full ML pipeline: feature engineering -> training -> prediction."""
    ml_df = build_ml_features(df, profiles)
    lgb_model, xgb_model, cat_model, meta_model, metrics, feature_names = train_ensemble(ml_df)
    predictions = predict_tomorrow(lgb_model, xgb_model, cat_model, meta_model, ml_df, feature_names, profiles, metrics)
    save_models(lgb_model, xgb_model, cat_model, meta_model, metrics, feature_names, predictions)
    return predictions, metrics, feature_names


def generate_30day_forecast(ml_df: pd.DataFrame, profiles: Dict[int, Dict],
                             metrics: Dict, feature_names: List[str]) -> pd.DataFrame:
    """
    Generate a 30-day daily violation forecast per hotspot using the trained models.
    Uses recent feature patterns to extrapolate forward.
    """
    print("Generating 30-day daily forecast...")

    lgb_model, xgb_model, cat_model, meta_model, _, _ = load_models()

    # Get the last known features per cluster
    latest = ml_df.groupby("cluster_id").last().reset_index()

    # Last training date
    last_date = ml_df["date"].max()
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq="D")

    all_forecasts = []

    for day_offset, forecast_date in enumerate(forecast_dates):
        day_features = latest.copy()
        day_features["date"] = forecast_date

        # Update cyclical features for the forecast date
        day_features["day_of_week"] = forecast_date.dayofweek
        day_features["day_of_month"] = forecast_date.day
        day_features["week_of_year"] = forecast_date.isocalendar()[1]
        day_features["month"] = forecast_date.month
        day_features["is_month_start"] = int(forecast_date.is_month_start)
        day_features["is_month_end"] = int(forecast_date.is_month_end)
        day_features["dow_sin"] = np.sin(2 * np.pi * forecast_date.dayofweek / 7)
        day_features["dow_cos"] = np.cos(2 * np.pi * forecast_date.dayofweek / 7)
        day_features["month_sin"] = np.sin(2 * np.pi * forecast_date.month / 12)
        day_features["month_cos"] = np.cos(2 * np.pi * forecast_date.month / 12)
        day_features["dom_sin"] = np.sin(2 * np.pi * forecast_date.day / 31)
        day_features["dom_cos"] = np.cos(2 * np.pi * forecast_date.day / 31)

        # Shift lag/rolling features forward by day_offset
        for col in day_features.columns:
            if "lag_" in col or "rolling_" in col or "trend_" in col or "momentum_" in col:
                if day_features[col].dtype in [np.float64, np.float32, np.int64]:
                    # Decay lag features: violations trend toward mean
                    mean_val = day_features[col].mean()
                    decay = 0.95 ** day_offset
                    day_features[col] = day_features[col] * decay + mean_val * (1 - decay)

        # Select features
        avail_features = [f for f in feature_names if f in day_features.columns]
        X = day_features[avail_features].values.astype(np.float32)

        # Predict
        lgb_pred_log = lgb_model.predict(X)
        xgb_pred_log = xgb_model.predict(X)

        if cat_model and metrics.get("use_stacking"):
            cat_pred_log = cat_model.predict(X)
            preds_stack = np.column_stack([lgb_pred_log, xgb_pred_log, cat_pred_log])
            meta_pred_log = meta_model.predict(preds_stack)
            pred = np.expm1(meta_pred_log)
        elif cat_model:
            cat_pred_log = cat_model.predict(X)
            w_lgb = metrics["lgb_weight"]
            w_xgb = metrics["xgb_weight"]
            w_cat = metrics["cat_weight"]
            pred = np.expm1(w_lgb * lgb_pred_log + w_xgb * xgb_pred_log + w_cat * cat_pred_log)
        else:
            w_lgb = metrics["lgb_weight"]
            w_xgb = metrics["xgb_weight"]
            pred = np.expm1(w_lgb * lgb_pred_log + w_xgb * xgb_pred_log)

        pred = np.maximum(pred, 0)

        # Conformal bounds
        conformal_q = metrics.get("conformal_q", 5.0)
        lower = np.maximum(pred - conformal_q, 0)
        upper = pred + conformal_q

        # Confidence
        disagreement = np.abs(lgb_pred - xgb_pred) if 'lgb_pred' in dir() else np.zeros_like(pred)
        for i, cid in enumerate(day_features["cluster_id"]):
            all_forecasts.append({
                "cluster_id": int(cid),
                "date": forecast_date.strftime("%Y-%m-%d"),
                "day_offset": day_offset + 1,
                "predicted_violations": round(float(pred[i]), 1),
                "lower_bound": round(float(lower[i]), 1),
                "upper_bound": round(float(upper[i]), 1),
                "area": profiles.get(int(cid), {}).get("area", "Unknown"),
                "road_type": profiles.get(int(cid), {}).get("road_type", "Other"),
                "centroid_lat": profiles.get(int(cid), {}).get("centroid_lat", 0),
                "centroid_lon": profiles.get(int(cid), {}).get("centroid_lon", 0),
            })

    forecast_df = pd.DataFrame(all_forecasts)
    print(f"  Generated {len(forecast_df)} forecast rows ({len(profiles)} hotspots x 30 days)")
    return forecast_df


def compute_daily_forecast_summary(forecast_df: pd.DataFrame) -> Dict:
    """Aggregate 30-day forecast into daily citywide totals."""
    daily = forecast_df.groupby("date").agg(
        total_predicted=("predicted_violations", "sum"),
        avg_predicted=("predicted_violations", "mean"),
        max_predicted=("predicted_violations", "max"),
        hotspot_count=("cluster_id", "nunique"),
    ).reset_index()

    daily["lower_total"] = forecast_df.groupby("date")["lower_bound"].sum().values
    daily["upper_total"] = forecast_df.groupby("date")["upper_bound"].sum().values

    return daily.to_dict(orient="records")


if __name__ == "__main__":
    from src.data_pipeline import run_pipeline
    from src.clustering import cluster_hotspots

    df = run_pipeline()
    df, profiles = cluster_hotspots(df)
    predictions, metrics, feature_names = run_ml_pipeline(df, profiles)

    print(f"\nML pipeline complete:")
    print(f"  Ensemble R2: {metrics['ensemble_r2']:.4f}")
    print(f"  Predictions: {len(predictions)}")
