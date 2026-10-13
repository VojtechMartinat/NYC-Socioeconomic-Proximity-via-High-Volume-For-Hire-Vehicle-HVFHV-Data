from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.loader import TLCHVLoader


# Usage: from the project root (ADS-Coursework), run:
# python -m src.tip_prediction.train_xgboost --sample-rows 1000000

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "models" / "tip_prediction" / "xgboost"
TARGET_FEATURE = "tips"

NUMERIC_FEATURES = [
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "cbd_congestion_fee",
    "total_charge_excluding_tip",
    "fare_per_mile",
    "fare_per_minute",
    "trip_speed_mph",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]

CATEGORICAL_FEATURES = [
    "PULocationID",
    "DOLocationID",
    "hvfhs_license_num",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

COLUMNS_TO_LOAD = [
    "pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "hvfhs_license_num",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tolls",
    "bcf",
    "sales_tax",
    "congestion_surcharge",
    "airport_fee",
    "shared_request_flag",
    "shared_match_flag",
    "access_a_ride_flag",
    "wav_request_flag",
    "wav_match_flag",
    "cbd_congestion_fee",
    "tips",
]


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost classifier to predict whether an FHVHV trip receives a tip."
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=1_000_000,
        help="Rows to sample across 2025 files. Use 0 for all rows.",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of months to load, for example --months 1 2 3.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--prediction-sample-size", type=int, default=100_000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def resolve_xgboost_device(device_name: str) -> str:
    if device_name == "cpu":
        return "cpu"
    if device_name == "cuda":
        return "cuda"

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices and visible_devices.lower() != "none":
        return "cuda"
    return "cpu"


def load_xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is required for this trainer. Install the project requirements first.") from exc
    return XGBClassifier


def load_data(n_rows: int | None = None, seed: int = 42, months: list[int] | None = None) -> pd.DataFrame:
    log(f"Loading trip data (n_rows={n_rows}, months={months})...")
    loader = TLCHVLoader(year=2025)
    return loader.load_files(
        columns=COLUMNS_TO_LOAD,
        max_n_rows=n_rows,
        seed=seed,
        months=months,
    )


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    log("Preprocessing dataframe...")
    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")

    df["total_charge_excluding_tip"] = df[
        [
            "base_passenger_fare",
            "tolls",
            "bcf",
            "sales_tax",
            "congestion_surcharge",
            "airport_fee",
            "cbd_congestion_fee",
        ]
    ].sum(axis=1)

    safe_trip_miles = df["trip_miles"].clip(lower=1e-3)
    safe_trip_minutes = (df["trip_time"] / 60.0).clip(lower=1e-3)
    safe_trip_hours = (df["trip_time"] / 3600.0).clip(lower=1e-3)

    df["fare_per_mile"] = df["base_passenger_fare"] / safe_trip_miles
    df["fare_per_minute"] = df["base_passenger_fare"] / safe_trip_minutes
    df["trip_speed_mph"] = df["trip_miles"] / safe_trip_hours

    df = df[df[TARGET_FEATURE] >= 0]
    df = df[df["trip_time"] > 0]
    df = df[df["trip_miles"] >= 0]
    df = df[df["base_passenger_fare"] >= 0]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["pickup_datetime", TARGET_FEATURE] + ALL_FEATURES)
    df = df.sort_values("pickup_datetime").reset_index(drop=True)

    for feature in CATEGORICAL_FEATURES:
        df[feature] = df[feature].astype(str).astype("category")

    df["tip_happened"] = (df[TARGET_FEATURE] > 0).astype(np.int8)
    return df


def split_by_time(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    log(f"Train rows: {len(train_df):,}")
    log(f"Val rows:   {len(val_df):,}")
    log(f"Test rows:  {len(test_df):,}")

    return train_df, val_df, test_df


def build_model(args: argparse.Namespace, scale_pos_weight: float):
    XGBClassifier = load_xgb_classifier()
    device = resolve_xgboost_device(args.device)
    log(f"XGBoost device: {device}")
    log(f"scale_pos_weight: {scale_pos_weight:.4f}")

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric=["logloss", "aucpr", "auc"],
        random_state=args.seed,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        n_jobs=args.n_jobs,
        tree_method="hist",
        enable_categorical=True,
        device=device,
        verbosity=1,
    )


def compute_scale_pos_weight(labels: pd.Series) -> float:
    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)
    if positive_count == 0:
        return 1.0
    return negative_count / positive_count


def find_best_threshold(actual: np.ndarray, probabilities: np.ndarray) -> float:
    if len(np.unique(actual)) < 2:
        return 0.5

    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float32),
                np.linspace(0.05, 0.95, 37, dtype=np.float32),
                np.quantile(probabilities, np.linspace(0.0, 1.0, 41)).astype(np.float32),
            ]
        )
    )

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in candidate_thresholds:
        predictions = (probabilities >= float(threshold)).astype(np.int8)
        score = f1_score(actual, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def collect_metrics(actual: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int8)
    positive_rate = float(actual.mean())

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(actual, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predictions)),
        "precision": float(precision_score(actual, predictions, zero_division=0)),
        "recall": float(recall_score(actual, predictions, zero_division=0)),
        "f1": float(f1_score(actual, predictions, zero_division=0)),
        "positive_tip_rate": positive_rate,
        "predicted_positive_rate": float(predictions.mean()),
        "majority_baseline_accuracy": float(max(positive_rate, 1.0 - positive_rate)),
    }

    if len(np.unique(actual)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(actual, probabilities))
        metrics["average_precision"] = float(average_precision_score(actual, probabilities))
        metrics["log_loss"] = float(log_loss(actual, probabilities, labels=[0, 1]))

    return metrics


def save_outputs(
    args: argparse.Namespace,
    model,
    threshold: float,
    val_metrics: dict[str, float],
    test_df: pd.DataFrame,
    test_actual: np.ndarray,
    test_probabilities: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    test_metrics = collect_metrics(test_actual, test_probabilities, threshold)
    test_metrics.update(
        {
            "n_test_rows": int(len(test_actual)),
            "sample_rows": None if args.sample_rows == 0 else int(args.sample_rows),
            "validation_f1": float(val_metrics["f1"]),
        }
    )

    model_path = output_dir / "tip_occurrence_xgboost.json"
    metrics_path = output_dir / "tip_occurrence_test_metrics.json"
    val_metrics_path = output_dir / "tip_occurrence_val_metrics.json"
    predictions_path = output_dir / "tip_occurrence_test_predictions.csv"
    importances_path = output_dir / "tip_occurrence_feature_importances.csv"
    feature_config_path = output_dir / "tip_occurrence_feature_config.json"

    model.save_model(model_path)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    with open(val_metrics_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    feature_config = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
    }
    with open(feature_config_path, "w", encoding="utf-8") as f:
        json.dump(feature_config, f, indent=2)

    prediction_rows = min(args.prediction_sample_size, len(test_df))
    prediction_frame = test_df.iloc[:prediction_rows][
        [
            "pickup_datetime",
            "PULocationID",
            "DOLocationID",
            "hvfhs_license_num",
            "trip_miles",
            "trip_time",
            "base_passenger_fare",
            "total_charge_excluding_tip",
            "hour",
            "day_of_week",
            "month",
        ]
    ].copy()
    prediction_frame["actual_tip"] = test_df.iloc[:prediction_rows][TARGET_FEATURE].to_numpy(dtype=np.float32)
    prediction_frame["actual_tip_happened"] = test_actual[:prediction_rows]
    prediction_frame["predicted_tip_probability"] = test_probabilities[:prediction_rows]
    prediction_frame["predicted_tip_happened"] = (
        test_probabilities[:prediction_rows] >= threshold
    ).astype(np.int8)
    prediction_frame.to_csv(predictions_path, index=False)

    feature_importances = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importances.to_csv(importances_path, index=False)

    log(f"Saved model to: {model_path}")
    log(f"Saved test metrics to: {metrics_path}")
    log(f"Saved validation metrics to: {val_metrics_path}")
    log(f"Saved sampled test predictions to: {predictions_path}")
    log(f"Saved feature importances to: {importances_path}")
    log(f"Saved feature config to: {feature_config_path}")

    return test_metrics


def main() -> None:
    args = parse_args()
    requested_rows = None if args.sample_rows == 0 else args.sample_rows
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Outputs will be written to: {output_dir}")

    df = load_data(n_rows=requested_rows, seed=args.seed, months=args.months)
    df = preprocess_dataframe(df)
    train_df, val_df, test_df = split_by_time(df, train_frac=args.train_frac, val_frac=args.val_frac)

    y_train = train_df["tip_happened"].to_numpy(dtype=np.int8)
    y_val = val_df["tip_happened"].to_numpy(dtype=np.int8)
    y_test = test_df["tip_happened"].to_numpy(dtype=np.int8)

    model = build_model(args, scale_pos_weight=compute_scale_pos_weight(train_df["tip_happened"]))
    model.fit(
        train_df[ALL_FEATURES],
        y_train,
        eval_set=[(val_df[ALL_FEATURES], y_val)],
        verbose=100,
    )

    val_probabilities = model.predict_proba(val_df[ALL_FEATURES])[:, 1]
    threshold = find_best_threshold(y_val, val_probabilities)
    val_metrics = collect_metrics(y_val, val_probabilities, threshold)

    test_probabilities = model.predict_proba(test_df[ALL_FEATURES])[:, 1]
    test_metrics = save_outputs(
        args=args,
        model=model,
        threshold=threshold,
        val_metrics=val_metrics,
        test_df=test_df,
        test_actual=y_test,
        test_probabilities=test_probabilities,
        output_dir=output_dir,
    )

    log("Validation results")
    log("------------------")
    log(f"ROC AUC:            {val_metrics.get('roc_auc', float('nan')):.4f}")
    log(f"Average precision:  {val_metrics.get('average_precision', float('nan')):.4f}")
    log(f"F1:                 {val_metrics['f1']:.4f}")
    log(f"Recall:             {val_metrics['recall']:.4f}")
    log(f"Precision:          {val_metrics['precision']:.4f}")
    log(f"Threshold:          {val_metrics['threshold']:.4f}")

    log("Test results")
    log("------------")
    log(f"ROC AUC:            {test_metrics.get('roc_auc', float('nan')):.4f}")
    log(f"Average precision:  {test_metrics.get('average_precision', float('nan')):.4f}")
    log(f"F1:                 {test_metrics['f1']:.4f}")
    log(f"Recall:             {test_metrics['recall']:.4f}")
    log(f"Precision:          {test_metrics['precision']:.4f}")
    log(f"Balanced accuracy:  {test_metrics['balanced_accuracy']:.4f}")
    log(f"Majority baseline:  {test_metrics['majority_baseline_accuracy']:.4f}")


if __name__ == "__main__":
    main()
