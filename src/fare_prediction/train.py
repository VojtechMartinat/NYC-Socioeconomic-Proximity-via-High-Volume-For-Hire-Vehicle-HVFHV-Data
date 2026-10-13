import json
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.utils.loader import TLCHVLoader

# Usage: from the project root (ADS-Coursework), run:
# python -m src.fare_prediction.train

# We'll save all outputs in src/fare_prediction for now
OUTPUT_DIR = Path(__file__).parent.resolve()

# Target variable we want to predict
TARGET_FEATURE = "base_passenger_fare"

NUMERIC_FEATURES = [
    "trip_miles",
    "trip_time",
    "hour",
    "day_of_week",
    "airport_fee",
    "congestion_surcharge",
    "cbd_congestion_fee",
]

# The categorical features must be encoded during preprocessing
CATEGORICAL_FEATURES = [
    "PULocationID",
    "DOLocationID",
    "hvfhs_license_num",
    "shared_match_flag",
]

# All the features used for training
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

COLUMNS_TO_LOAD = [
    "pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "hvfhs_license_num",
    "airport_fee",
    "congestion_surcharge",
    "cbd_congestion_fee",
    "shared_match_flag",
    "base_passenger_fare",
]

def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)

def load_data(n_rows=None, seed=42):
    """
    Load raw trip data using the TLCHVLoader.
    Set n_rows=None to load everything the loader returns.
    """
    log(f"Loading data (n_rows={n_rows})...")
    loader = TLCHVLoader(year=2025)
    return loader.load_all_files_in_folder(
        columns=COLUMNS_TO_LOAD,
        max_n_rows=n_rows,
        seed=seed,
    )


def preprocess_dataframe(df):
    """
    Clean and prepare the raw dataframe before splitting.

    - parse pickup_datetime
    - extract hour and day_of_week
    - drop invalid / missing rows
    - sort chronologically for time-based splitting
    """
    log("Preprocessing dataframe...")
    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

    df = df[df[TARGET_FEATURE] > 0]
    df = df.dropna(subset=[TARGET_FEATURE] + ALL_FEATURES + ["pickup_datetime"])
    df = df.sort_values("pickup_datetime").reset_index(drop=True)

    return df


def split_by_time(df, train_frac=0.70, val_frac=0.15):
    """
    Split the dataframe chronologically into train / validation / test sets.
    """
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


def build_pipeline():
    """
    Build the modeling pipeline.

    The pipeline:
    - one-hot encodes categorical features
    - passes numeric features through unchanged
    - fits a XGBoost regressor to predict the fare
    """
    cat_transformer = ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
    num_transformer = ("num", "passthrough", NUMERIC_FEATURES)
    preprocessor = ColumnTransformer(transformers=[cat_transformer, num_transformer])

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=8,
        tree_method="hist",
        device="cuda",
        verbosity=1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def run_hyperparameter_sweep(train_df, val_df):
    """
    Run a grid search using a fixed validation split.

    Train rows are marked as training fold (-1).
    Validation rows are marked as validation fold (0).
    """
    log("Running hyperparameter sweep...")

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET_FEATURE]

    X_val = val_df[ALL_FEATURES]
    y_val = val_df[TARGET_FEATURE]

    # Combine train + val so PredefinedSplit can control which rows belong to the training portion versus the validation portion.
    X_search = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_search = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    test_fold = []
    for _ in range(len(X_train)):
        test_fold.append(-1)
    for _ in range(len(X_val)):
        test_fold.append(0)

    cv = PredefinedSplit(test_fold=test_fold)

    pipeline = build_pipeline()

    # Hyperparameter grid to search over.
    param_grid = {
        "model__n_estimators": [300, 500],
        "model__max_depth": [6, 8, 10],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__subsample": [0.8],
        "model__colsample_bytree": [0.8],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=cv,
        refit=True, # we want to refit the best model on train+val automatically
        n_jobs=1,
        verbose=3,
        return_train_score=True,
    )

    search.fit(X_search, y_search)

    log("\nBest params:")
    log(search.best_params_)
    log(f"Best validation MAE: {-search.best_score_:.4f}")

    return search


def evaluate_on_test(best_pipeline, test_df):
    """
    Evaluate the final refit model on the held-out test set.
    Saves predictions, residuals and errors to a CSV for later analysis.
    """
    log("Evaluating best model on held-out test set...")

    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET_FEATURE]

    preds = best_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)

    metrics = {
        "test_mae": float(mae),
        "test_r2": float(r2),
        "test_rmse": float(rmse),
        "n_test_rows": int(len(test_df)),
    }

    log("\nTest results")
    log("------------")
    log(f"MAE: ${mae:.4f}")
    log(f"R²:  {r2:.4f}")
    log(f"RMSE: ${rmse:.4f}")

    predictions_df = test_df[
        [
            "pickup_datetime",
            "PULocationID",
            "DOLocationID",
            "trip_miles",
            "trip_time",
            "hvfhs_license_num",
            "hour",
            "day_of_week",
            "shared_match_flag",
        ]
    ].copy()
    predictions_df["actual_fare"] = y_test.values
    predictions_df["predicted_fare"] = preds
    predictions_df["residual"] = predictions_df["actual_fare"] - predictions_df["predicted_fare"]
    predictions_df["abs_error"] = predictions_df["residual"].abs()
    predictions_df["squared_error"] = predictions_df["residual"] ** 2
    predictions_df["pct_error"] = (
        predictions_df["abs_error"] / predictions_df["actual_fare"].clip(lower=1e-6)
    ) * 100

    predictions_path = OUTPUT_DIR / "xgboost_regression" / "outputs" / "fare_xgb_test_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    log(f"Saved test predictions to: {predictions_path}")

    return metrics


def save_outputs(search, test_metrics, final_model):
    """
    Save all useful training artifacts to the current directory.

    Outputs:
    - best pipeline (.pkl)
    - best params (.json)
    - test metrics (.json)
    - full grid-search results (.csv)
    - feature importances (.csv)
    """
    log("Saving outputs...")

    model_path = OUTPUT_DIR / "xgboost_regression" / "outputs" / "fare_xgb_model.pkl"
    best_params_path = OUTPUT_DIR / "xgboost_regression" / "outputs" / "fare_xgb_best_params.json"
    test_metrics_path = OUTPUT_DIR / "xgboost_regression" / "outputs" / "fare_xgb_test_metrics.json"
    results_path = OUTPUT_DIR / "xgboost_regression" / "outputs" / "fare_xgb_gridsearch_results.csv"
    importances_path = OUTPUT_DIR / "xgboost_regression" / "outputs" / "fare_xgb_feature_importances.csv"

    joblib.dump(final_model, model_path)

    with open(best_params_path, "w") as f:
        json.dump(search.best_params_, f, indent=2)

    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Save the full search table, sorted by validation MAE.
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results["mean_val_mae"] = -cv_results["mean_test_score"]
    cv_results = cv_results.sort_values("mean_val_mae", ascending=True)
    cv_results.to_csv(results_path, index=False)

    # Extract feature importances from the best fitted pipeline.
    preprocessor = final_model.named_steps["preprocessor"] # type: ignore
    model = final_model.named_steps["model"] # type: ignore
 
    feature_names = preprocessor.get_feature_names_out()

    importances = pd.DataFrame()
    importances["feature"] = feature_names
    importances["importance"] = model.feature_importances_
    importances = importances.sort_values("importance", ascending=False)
    importances.to_csv(importances_path, index=False)

    log(f"Saved model to: {model_path}")
    log(f"Saved best params to: {best_params_path}")
    log(f"Saved test metrics to: {test_metrics_path}")
    log(f"Saved grid results to: {results_path}")
    log(f"Saved feature importances to: {importances_path}")

# Main runner to execute the entire training pipeline and save results
def main():
    # 1- Tune hyperparameters on subsample
    df_sample = load_data(n_rows=5_000_000, seed=42)
    df_sample = preprocess_dataframe(df_sample)
    train_df_sample, val_df_sample, _ = split_by_time(df_sample)
    search = run_hyperparameter_sweep(train_df_sample, val_df_sample)

    del df_sample, train_df_sample, val_df_sample

    # 2- Retrain on full data with best params
    df_full = load_data(n_rows=None, seed=42)
    df_full = preprocess_dataframe(df_full)
    train_df_full, val_df_full, test_df_full = split_by_time(df_full)

    train_val_full = pd.concat([train_df_full, val_df_full], axis=0).reset_index(drop=True)

    del df_full, train_df_full, val_df_full

    final_model = build_pipeline()
    log(f"Configured device: {final_model.named_steps['model'].get_params().get('device')}")

    final_model.set_params(**search.best_params_)
    final_model.fit(train_val_full[ALL_FEATURES], train_val_full[TARGET_FEATURE])

    del train_val_full

    test_metrics = evaluate_on_test(final_model, test_df_full)
    save_outputs(search, test_metrics, final_model)


if __name__ == "__main__":
    main()