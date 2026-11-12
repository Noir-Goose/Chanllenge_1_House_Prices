from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "input" / "train.csv"
TEST_PATH = ROOT / "input" / "test.csv"
SUBMISSION_PATH = ROOT / "submission_baseline.csv"
RANDOM_STATE = 42


def make_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_columns = X.select_dtypes(include=["object", "category", "string"]).columns
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns

    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    preprocessing = ColumnTransformer(
        transformers=[
            ("categorical", categorical, categorical_columns),
            ("numeric", numeric, numeric_columns),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
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

    print("Baseline: Random Forest Regressor (target = log1p(SalePrice))")
    print(f"Validation rows: {len(X_valid)}")
    print(f"RMSLE: {mean_squared_error(np.log1p(valid_actual), np.log1p(valid_predictions)) ** 0.5:.5f}")
    print(f"RMSE (log): {mean_squared_error(y_valid, valid_log_predictions) ** 0.5:.5f}")
    print(f"MAE ($): {mean_absolute_error(valid_actual, valid_predictions):,.0f}")

    final_model = make_pipeline(X)
    final_model.fit(X, y)
    test_log_predictions = final_model.predict(X_test)
    submission = pd.DataFrame(
        {
            "Id": test["Id"],
            "SalePrice": np.maximum(np.expm1(test_log_predictions), 0),
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved test predictions to: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()


