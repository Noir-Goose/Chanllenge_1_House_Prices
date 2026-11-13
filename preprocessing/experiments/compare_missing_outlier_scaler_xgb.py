from datetime import datetime
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline as baseline_module
baseline_module = importlib.reload(baseline_module)
RANDOM_STATE = baseline_module.RANDOM_STATE
TRAIN_PATH = baseline_module.TRAIN_PATH
make_pipeline = baseline_module.make_pipeline
RESULTS_ROOT = baseline_module.RESULTS_ROOT

NO_FEATURE_CATEGORICALS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2",
    "BsmtExposure", "BsmtFinType1", "BsmtCond", "BsmtQual",
]
OUTLIER_COLUMNS = [
    "LotArea", "GrLivArea", "TotalBsmtSF", "LotFrontage", "GarageArea", "MasVnrArea"
]
SCALERS = {
    "none": None,
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler(),
    "maxabs": MaxAbsScaler(),
}


def handle_missing(X_train, X_valid):
    train, valid = X_train.copy(), X_valid.copy()
    for col in NO_FEATURE_CATEGORICALS:
        if col in train.columns:
            train[col] = train[col].fillna("None")
            valid[col] = valid[col].fillna("None")
    for col in ("GarageYrBlt", "MasVnrArea"):
        if col in train.columns:
            train[col] = train[col].fillna(0)
            valid[col] = valid[col].fillna(0)
    if "LotFrontage" in train.columns:
        median = train["LotFrontage"].median()
        train["LotFrontage"] = train["LotFrontage"].fillna(median)
        valid["LotFrontage"] = valid["LotFrontage"].fillna(median)
    if "Electrical" in train.columns:
        mode = train["Electrical"].mode(dropna=True).iloc[0]
        train["Electrical"] = train["Electrical"].fillna(mode)
        valid["Electrical"] = valid["Electrical"].fillna(mode)
    return train, valid


def cap_outliers(X_train, X_valid, method):
    train, valid = X_train.copy(), X_valid.copy()
    clipped = {}
    if method == "none":
        return train, valid, clipped
    for col in OUTLIER_COLUMNS:
        if col not in train.columns:
            continue
        values = train[col]
        if method == "IQR":
            q1, q3 = values.quantile([0.25, 0.75])
            spread = q3 - q1
            lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        else:
            mean, std = values.mean(), values.std()
            lower, upper = mean - 3 * std, mean + 3 * std
        clipped[col] = float(((values < lower) | (values > upper)).mean() * 100)
        train[col] = train[col].clip(lower, upper)
        valid[col] = valid[col].clip(lower, upper)
    return train, valid, clipped


def score(X_train, X_valid, y_train, y_valid, scaler):
    model = make_pipeline(X_train, numeric_scaler=scaler)
    model.fit(X_train, y_train)
    pred_log = model.predict(X_valid)
    pred_price = np.maximum(np.expm1(pred_log), 0)
    actual_price = np.expm1(y_valid)
    return {
        "rmsle": float(mean_squared_error(np.log1p(actual_price), np.log1p(pred_price)) ** 0.5),
        "rmse_log": float(mean_squared_error(y_valid, pred_log) ** 0.5),
        "mae": float(mean_absolute_error(actual_price, pred_price)),
    }


def main():
    data = pd.read_csv(TRAIN_PATH)
    y = np.log1p(data["SalePrice"])
    X = data.drop(columns=["SalePrice", "Id"], errors="ignore")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    baseline = score(X_train, X_valid, y_train, y_valid, None)
    missing_train, missing_valid = handle_missing(X_train, X_valid)
    experiments = []
    for outlier_method in ("none", "IQR", "Z-score"):
        train, valid, clipped = cap_outliers(missing_train, missing_valid, outlier_method)
        for scaler_name, scaler in SCALERS.items():
            metrics = score(train, valid, y_train, y_valid, scaler)
            row = {
                "missing_strategy": "semantic None / numeric 0 or train median / Electrical mode",
                "outlier_strategy": outlier_method,
                "scaler": scaler_name,
                "train_rows": int(len(train)),
                "outlier_percent_clipped_by_column": clipped,
                **metrics,
                "rmsle_change_vs_baseline": metrics["rmsle"] - baseline["rmsle"],
            }
            experiments.append(row)
            print(f"outlier={outlier_method:6s}, scaler={scaler_name:8s}: RMSLE {metrics['rmsle']:.5f}")

    results = {
        "model": "XGBoost Regressor",
        "validation_rows": int(len(X_valid)),
        "baseline": baseline,
        "experiments": experiments,
        "note": "Missing values are handled first, then outliers are capped using training-only bounds. Scalers apply only to numeric features. Lower RMSLE is better.",
    }
    run_dir = (RESULTS_ROOT.parent / "missing_outlier_scaler_xgb" /
               datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(experiments).drop(columns=["outlier_percent_clipped_by_column"]).sort_values(
        "rmsle"
    ).to_csv(run_dir / "summary.csv", index=False)
    shutil.copy2(__file__, run_dir / "run_script.py")
    print(f"\nBaseline RMSLE: {baseline['rmsle']:.5f}")
    print(f"Saved metrics, summary, and script copy to: {run_dir}")


if __name__ == "__main__":
    main()
