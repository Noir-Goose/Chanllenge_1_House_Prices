from datetime import datetime
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline import RANDOM_STATE, TRAIN_PATH, make_pipeline, RESULTS_ROOT


# Mỗi cấu hình thử riêng một cột; các cột còn lại giữ cách điền của baseline.
STRATEGIES = {
    "PoolQC": ("constant", "None"),
    "MiscFeature": ("constant", "None"),
    "Alley": ("constant", "None"),
    "Fence": ("constant", "None"),
    "MasVnrType": ("constant", "None"),
    "FireplaceQu": ("constant", "None"),
    "LotFrontage": ("median", None),
    "GarageType": ("constant", "None"),
    "GarageYrBlt": ("constant", 0),
    "GarageFinish": ("constant", "None"),
    "GarageQual": ("constant", "None"),
    "GarageCond": ("constant", "None"),
    "BsmtFinType2": ("constant", "None"),
    "BsmtExposure": ("constant", "None"),
    "BsmtFinType1": ("constant", "None"),
    "BsmtCond": ("constant", "None"),
    "BsmtQual": ("constant", "None"),
    "MasVnrArea": ("constant", 0),
    "Electrical": ("most_frequent", None),
}


def score_model(X_train, X_valid, y_train, y_valid):
    model = make_pipeline(X_train)
    model.fit(X_train, y_train)
    pred_log = model.predict(X_valid)
    pred_price = np.maximum(np.expm1(pred_log), 0)
    actual_price = np.expm1(y_valid)
    return {
        "rmsle": float(mean_squared_error(np.log1p(actual_price), np.log1p(pred_price)) ** 0.5),
        "rmse_log": float(mean_squared_error(y_valid, pred_log) ** 0.5),
        "mae": float(mean_absolute_error(actual_price, pred_price)),
    }


def apply_strategy(column, strategy, value, X_train, X_valid):
    train = X_train.copy()
    valid = X_valid.copy()
    if strategy == "constant":
        train[column] = train[column].fillna(value)
        valid[column] = valid[column].fillna(value)
    elif strategy == "median":
        fill_value = train[column].median()
        train[column] = train[column].fillna(fill_value)
        valid[column] = valid[column].fillna(fill_value)
    elif strategy == "most_frequent":
        fill_value = train[column].mode(dropna=True).iloc[0]
        train[column] = train[column].fillna(fill_value)
        valid[column] = valid[column].fillna(fill_value)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return train, valid


def main():
    data = pd.read_csv(TRAIN_PATH)
    y = np.log1p(data["SalePrice"])
    X = data.drop(columns=["SalePrice", "Id"], errors="ignore")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    baseline = score_model(X_train, X_valid, y_train, y_valid)
    experiments = []
    for column, (strategy, value) in STRATEGIES.items():
        candidate_train, candidate_valid = apply_strategy(
            column, strategy, value, X_train, X_valid
        )
        scores = score_model(candidate_train, candidate_valid, y_train, y_valid)
        experiments.append({
            "column": column,
            "strategy": f"{strategy}: {value}" if value is not None else strategy,
            **scores,
            "rmsle_change_vs_baseline": scores["rmsle"] - baseline["rmsle"],
        })
        print(f"{column}: RMSLE {scores['rmsle']:.5f} (baseline {baseline['rmsle']:.5f})")

    metrics = {
        "model": "XGBoost Regressor",
        "validation_rows": int(len(X_valid)),
        "baseline": baseline,
        "experiments": experiments,
        "note": "Each experiment changes one missing-value strategy only; lower RMSLE is better.",
    }
    run_dir = (RESULTS_ROOT.parent / "missing_sweep_xgb" /
               datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(experiments).sort_values("rmsle").to_csv(run_dir / "summary.csv", index=False)
    shutil.copy2(__file__, run_dir / "run_script.py")
    print(f"\nBaseline RMSLE: {baseline['rmsle']:.5f}")
    print(f"Saved metrics, summary, and script copy to: {run_dir}")


if __name__ == "__main__":
    main()
