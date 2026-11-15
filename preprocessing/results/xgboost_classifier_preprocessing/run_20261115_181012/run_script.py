"""Compare four preprocessing configurations with a 3-class XGBoost model.

Run from any working directory. Results and a copy of this script are saved
under preprocessing/results/xgboost_classifier_preprocessing/run_*/.
"""

from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "input" / "train.csv"
RESULTS_ROOT = ROOT / "preprocessing" / "results" / "xgboost_classifier_preprocessing"
RANDOM_STATE = 42
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["Low (<150k)", "Medium (150k-250k)", "High (>=250k)"]

# These columns encode structural absence (for example, no garage/basement).
ABSENCE_CATEGORY_COLUMNS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2",
    "BsmtExposure", "BsmtFinType1", "BsmtCond", "BsmtQual",
]
ZERO_WHEN_ABSENT_COLUMNS = ["GarageYrBlt", "MasVnrArea"]
IQR_COLUMNS = [
    "LotArea", "GrLivArea", "TotalBsmtSF", "LotFrontage", "GarageArea", "MasVnrArea"
]

CONFIGURATIONS = [
    ("missing_only", "Missing handling", None, False),
    ("missing_outlier", "Missing + outlier capping", None, True),
    ("missing_outlier_minmax", "Missing + outlier + MinMaxScaler", MinMaxScaler(), True),
    ("missing_outlier_standard", "Missing + outlier + StandardScaler", StandardScaler(), True),
]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply domain-defined missing values that do not estimate from data."""
    result = frame.copy()
    for col in ABSENCE_CATEGORY_COLUMNS:
        if col in result.columns:
            result[col] = result[col].fillna("None").astype("string")
    for col in ZERO_WHEN_ABSENT_COLUMNS:
        if col in result.columns:
            result[col] = result[col].fillna(0)
    # Numeric-looking category codes must be encoded as categories.
    if "MSSubClass" in result.columns:
        result["MSSubClass"] = result["MSSubClass"].astype("string")
    return result


def fit_iqr_bounds(X_train: pd.DataFrame) -> dict:
    bounds = {}
    for col in IQR_COLUMNS:
        if col not in X_train:
            continue
        values = pd.to_numeric(X_train[col], errors="coerce").dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        spread = q3 - q1
        if spread > 0:
            bounds[col] = (float(q1 - 1.5 * spread), float(q3 + 1.5 * spread))
    return bounds


def apply_bounds(X: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    result = X.copy()
    for col, (lower, upper) in bounds.items():
        result[col] = pd.to_numeric(result[col], errors="coerce").clip(lower, upper)
    return result


def make_model(X_train: pd.DataFrame, scaler=None) -> Pipeline:
    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaler is not None:
        numeric_steps.append(("scaler", scaler))
    numeric = Pipeline(numeric_steps)
    preprocessing = ColumnTransformer(
        [("categorical", categorical, categorical_columns),
         ("numeric", numeric, numeric_columns)],
    )

    classifier = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", classifier)])


def evaluate(model: Pipeline, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    predictions = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)
    return {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "precision_macro": float(precision_score(y_valid, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_valid, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_valid, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_valid, probabilities, labels=CLASS_LABELS)),
    }


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")
    data = pd.read_csv(TRAIN_PATH)
    if "SalePrice" not in data.columns:
        raise ValueError("input/train.csv must include the SalePrice target column")

    y = pd.cut(
        data["SalePrice"],
        bins=[-np.inf, 150000, 250000, np.inf],
        labels=CLASS_LABELS,
        right=False,
    ).astype(int)
    X = prepare_features(data.drop(columns=["SalePrice", "Id"], errors="ignore"))
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    result_rows = []
    detailed_results = {}
    for key, label, scaler, cap_outliers in CONFIGURATIONS:
        if cap_outliers:
            bounds = fit_iqr_bounds(X_train)
            X_train_run = apply_bounds(X_train, bounds)
            X_valid_run = apply_bounds(X_valid, bounds)
        else:
            bounds = {}
            X_train_run, X_valid_run = X_train.copy(), X_valid.copy()

        model = make_model(X_train_run, scaler=scaler)
        model.fit(X_train_run, y_train)
        metrics = evaluate(model, X_valid_run, y_valid)
        detailed_results[key] = {
            "label": label,
            "validation": metrics,
            "iqr_bounds": {col: list(values) for col, values in bounds.items()},
        }
        result_rows.append({"configuration": label, **metrics})
        print(f"{label}: macro-F1={metrics['f1_macro']:.4f}, log-loss={metrics['log_loss']:.4f}")

    run_dir = RESULTS_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = pd.DataFrame(result_rows)
    summary.to_csv(run_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "model": "XGBoost Classifier",
        "target_classes": dict(zip(map(str, CLASS_LABELS), CLASS_NAMES)),
        "split": {"train_fraction": 0.8, "validation_fraction": 0.2, "random_state": RANDOM_STATE,
                  "stratified": True, "train_rows": int(len(X_train)), "validation_rows": int(len(X_valid))},
        "model_parameters": {
            "n_estimators": 300, "learning_rate": 0.05, "max_depth": 3,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "objective": "multi:softprob", "eval_metric": "mlogloss", "tree_method": "hist",
            "random_state": RANDOM_STATE,
        },
        "configurations": detailed_results,
        "note": "Kaggle test.csv has no labels; metrics are reported on the stratified validation split.",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    shutil.copy2(__file__, run_dir / "run_script.py")
    print(f"Saved comparison, metrics, and script copy to: {run_dir}")


if __name__ == "__main__":
    main()
