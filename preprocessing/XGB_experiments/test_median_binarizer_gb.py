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
from sklearn.base import BaseEstimator, TransformerMixin

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


class MedianBinarizer(BaseEstimator, TransformerMixin):
    """Convert each numeric feature to 0/1 using its training median."""

    def fit(self, X, y=None):
        self.thresholds_ = np.median(np.asarray(X, dtype=float), axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return (values > self.thresholds_).astype(np.float32)


def apply_missing(X_train, X_valid):
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


def apply_iqr_capping(X_train, X_valid):
    train, valid = X_train.copy(), X_valid.copy()
    bounds = {}
    for col in OUTLIER_COLUMNS:
        values = train[col]
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        train[col] = train[col].clip(lower, upper)
        valid[col] = valid[col].clip(lower, upper)
        bounds[col] = {"lower": float(lower), "upper": float(upper)}
    return train, valid, bounds


def evaluate(X_train, X_valid, y_train, y_valid, scaler):
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
    X_train, X_valid = apply_missing(X_train, X_valid)
    X_train, X_valid, bounds = apply_iqr_capping(X_train, X_valid)

    results = {
        "no_scaler": evaluate(X_train, X_valid, y_train, y_valid, None),
        "median_binarizer": evaluate(X_train, X_valid, y_train, y_valid, MedianBinarizer()),
    }
    metrics = {
        "model": "Gradient Boosting Regressor",
        "missing_strategy": "semantic None / numeric 0 or train median / Electrical mode",
        "outlier_strategy": "IQR capping using training-only bounds",
        "scaler_comparison": results,
        "binary_transform": "Each numeric feature is converted to 0/1 using the training median.",
        "validation_rows": int(len(X_valid)),
        "iqr_bounds": bounds,
        "rmsle_difference_binarizer_minus_none": results["median_binarizer"]["rmsle"] - results["no_scaler"]["rmsle"],
        "note": "Lower RMSLE is better.",
    }
    run_dir = (RESULTS_ROOT.parent / "scaler_comparison_gradient_boosting" / "median_binarizer" /
               datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    shutil.copy2(__file__, run_dir / "run_script.py")
    print(f"No scaler RMSLE: {results['no_scaler']['rmsle']:.5f}")
    print(f"Median binarizer RMSLE: {results['median_binarizer']['rmsle']:.5f}")
    print(f"Saved metrics and script copy to: {run_dir}")


if __name__ == "__main__":
    main()
