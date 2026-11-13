from pathlib import Path
from datetime import datetime
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "input" / "train.csv"
TEST_PATH = ROOT / "input" / "test.csv"
RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "gradient_boosting_baseline"
RANDOM_STATE = 42


def make_pipeline(X: pd.DataFrame, numeric_scaler=None) -> Pipeline:
    categorical_columns = X.select_dtypes(include=["object", "category", "string"]).columns
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if numeric_scaler is not None:
        numeric_steps.append(("scaler", numeric_scaler))
    numeric = Pipeline(steps=numeric_steps)
    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", categorical, categorical_columns),
            ("numeric", numeric, numeric_columns),
        ],
        sparse_threshold=0,
    )
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=2,
        loss="squared_error",
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Expected train.csv and test.csv in {TRAIN_PATH.parent}"
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = np.log1p(train["SalePrice"])
    X = train.drop(columns=["SalePrice", "Id"], errors="ignore")
    X_test = test.drop(columns=["Id"], errors="ignore")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    validation_model = make_pipeline(X_train)
    validation_model.fit(X_train, y_train)
    valid_log_predictions = validation_model.predict(X_valid)
    valid_predictions = np.maximum(np.expm1(valid_log_predictions), 0)
    valid_actual = np.expm1(y_valid)

    metrics = {
        "model": "Gradient Boosting Regressor",
        "validation_rows": int(len(X_valid)),
        "rmsle": float(mean_squared_error(np.log1p(valid_actual), np.log1p(valid_predictions)) ** 0.5),
        "rmse_log": float(mean_squared_error(y_valid, valid_log_predictions) ** 0.5),
        "mae": float(mean_absolute_error(valid_actual, valid_predictions)),
    }
    print("Baseline: Gradient Boosting Regressor (target = log1p(SalePrice))")
    print(f"Validation rows: {len(X_valid)}")
    print(f"RMSLE: {metrics['rmsle']:.5f}")
    print(f"RMSE (log): {metrics['rmse_log']:.5f}")
    print(f"MAE ($): {metrics['mae']:,.0f}")

    final_model = make_pipeline(X)
    final_model.fit(X, y)
    test_log_predictions = final_model.predict(X_test)
    submission = pd.DataFrame(
        {
            "Id": test["Id"],
            "SalePrice": np.maximum(np.expm1(test_log_predictions), 0),
        }
    )
    run_dir = RESULTS_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    submission.to_csv(run_dir / "submission_baseline.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    shutil.copy2(__file__, run_dir / "run_script.py")
    print(f"Saved run results and script copy to: {run_dir}")


if __name__ == "__main__":
    main()


