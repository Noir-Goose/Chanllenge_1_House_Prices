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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline as baseline_module
baseline_module = importlib.reload(baseline_module)
RANDOM_STATE = baseline_module.RANDOM_STATE
TRAIN_PATH = baseline_module.TRAIN_PATH
make_pipeline = baseline_module.make_pipeline
RESULTS_ROOT = baseline_module.RESULTS_ROOT

OUTLIER_COLUMNS = [
    "LotArea", "GrLivArea", "TotalBsmtSF", "LotFrontage", "GarageArea", "MasVnrArea"
]


def get_bounds(series, method):
    if method == "IQR":
        q1, q3 = series.quantile([0.25, 0.75])
        spread = q3 - q1
        return q1 - 1.5 * spread, q3 + 1.5 * spread
    mean, std = series.mean(), series.std()
    return mean - 3 * std, mean + 3 * std


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


def main():
    data = pd.read_csv(TRAIN_PATH)
    y = np.log1p(data["SalePrice"])
    X = data.drop(columns=["SalePrice", "Id"], errors="ignore")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    baseline = score_model(X_train, X_valid, y_train, y_valid)
    experiments = []

    for column in OUTLIER_COLUMNS:
        for method in ("IQR", "Z-score"):
            train = X_train.copy()
            valid = X_valid.copy()
            lower, upper = get_bounds(train[column], "IQR" if method == "IQR" else "Z-score")
            train[column] = train[column].clip(lower=lower, upper=upper)
            valid[column] = valid[column].clip(lower=lower, upper=upper)
            scores = score_model(train, valid, y_train, y_valid)
            clipped_pct = float(((X_train[column] < lower) | (X_train[column] > upper)).mean() * 100)
            experiments.append({
                "column": column,
                "strategy": f"{method} capping",
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "train_percent_clipped": clipped_pct,
                **scores,
                "rmsle_change_vs_baseline": scores["rmsle"] - baseline["rmsle"],
            })
            print(f"{column} / {method}: RMSLE {scores['rmsle']:.5f} (baseline {baseline['rmsle']:.5f})")

    metrics = {
        "model": "XGBoost Regressor",
        "validation_rows": int(len(X_valid)),
        "baseline": baseline,
        "experiments": experiments,
        "note": "Each experiment caps one numeric feature using training-only bounds; lower RMSLE is better.",
    }
    run_dir = (RESULTS_ROOT.parent / "outlier_sweep_xgb" /
               datetime.now().strftime("run_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(experiments).sort_values("rmsle").to_csv(run_dir / "summary.csv", index=False)
    shutil.copy2(__file__, run_dir / "run_script.py")
    print(f"\nBaseline RMSLE: {baseline['rmsle']:.5f}")
    print(f"Saved metrics, summary, and script copy to: {run_dir}")


if __name__ == "__main__":
    main()
