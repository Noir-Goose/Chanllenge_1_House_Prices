"""Compare Gradient Boosting feature-importance selection sizes.

Run from the project root:
    python feature_selection/gradient_boosting_importance.py

If the Jupyter working directory is ``preprocessing``:
    %run ../feature_selection/gradient_boosting_importance.py
"""

from datetime import datetime
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "input" / "train.csv"
RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "gradient_boosting_importance"
RANDOM_STATE = 42
CLASS_LABELS = [0, 1, 2]
TOP_K_VALUES = [10, 20, 30, "all"]
NO_FEATURE_CATEGORICALS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2",
    "BsmtExposure", "BsmtFinType1", "BsmtCond", "BsmtQual",
]
OUTLIER_COLUMNS = [
    "LotArea", "GrLivArea", "TotalBsmtSF", "LotFrontage", "GarageArea", "MasVnrArea"
]


def prepare_features(frame):
    frame = frame.copy()
    for column in NO_FEATURE_CATEGORICALS:
        if column in frame:
            frame[column] = frame[column].fillna("None")
    for column in ("GarageYrBlt", "MasVnrArea"):
        if column in frame:
            frame[column] = frame[column].fillna(0)
    return frame


def fit_iqr_bounds(frame):
    bounds = {}
    for column in OUTLIER_COLUMNS:
        q1, q3 = frame[column].quantile([0.25, 0.75])
        spread = q3 - q1
        bounds[column] = (q1 - 1.5 * spread, q3 + 1.5 * spread)
    return bounds


def apply_iqr_bounds(frame, bounds):
    frame = frame.copy()
    for column, (lower, upper) in bounds.items():
        frame[column] = frame[column].clip(lower=lower, upper=upper)
    return frame


def make_preprocessor(frame):
    categorical_columns = frame.select_dtypes(include=["object", "category", "string"]).columns
    numeric_columns = frame.select_dtypes(include=["number", "bool"]).columns
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MaxAbsScaler()),
    ])
    return ColumnTransformer(
        [("categorical", categorical, categorical_columns),
         ("numeric", numeric, numeric_columns)],
        sparse_threshold=0,
    )


def make_classifier():
    return GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
    )


def get_transformed_feature_names(preprocessor):
    try:
        return preprocessor.get_feature_names_out()
    except (AttributeError, ValueError):
        return np.array([f"feature_{i}" for i in range(len(preprocessor.transformers_[0][2]))])


def score_model(model, X, y):
    prediction = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision_macro": float(precision_score(y, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, prediction, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0)),
    }


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {TRAIN_PATH}")

    data = pd.read_csv(TRAIN_PATH)
    y = pd.cut(
        data["SalePrice"], [-np.inf, 150000, 250000, np.inf],
        labels=CLASS_LABELS, right=False,
    ).astype(int)
    X = prepare_features(data.drop(columns=["SalePrice", "Id"], errors="ignore"))
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=(0.15 / 0.85),
        random_state=RANDOM_STATE, stratify=y_train_valid,
    )

    # Learn IQR limits and feature rankings from training data only.
    bounds = fit_iqr_bounds(X_train)
    X_train_capped = apply_iqr_bounds(X_train, bounds)
    X_valid_capped = apply_iqr_bounds(X_valid, bounds)
    preprocessor = make_preprocessor(X_train_capped)
    train_encoded = preprocessor.fit_transform(X_train_capped)
    valid_encoded = preprocessor.transform(X_valid_capped)
    ranker = make_classifier().fit(train_encoded, y_train)
    importances = ranker.feature_importances_
    feature_names = get_transformed_feature_names(preprocessor)
    ranking = np.argsort(importances)[::-1]

    rows = []
    models = {}
    for requested_k in TOP_K_VALUES:
        k = len(ranking) if requested_k == "all" else min(requested_k, len(ranking))
        selected = ranking[:k]
        model = make_classifier().fit(train_encoded[:, selected], y_train)
        metrics = score_model(model, valid_encoded[:, selected], y_valid)
        rows.append({"selection": "all" if requested_k == "all" else f"top_{k}",
                     "feature_count": int(k), **metrics})
        models[requested_k] = (selected, metrics)
        print(f"{rows[-1]['selection']:>8}: validation macro-F1={metrics['f1_macro']:.4f}")

    # Pick k on validation macro-F1, then refit ranking and classifier on train+validation.
    best_k = max(TOP_K_VALUES, key=lambda key: models[key][1]["f1_macro"])
    best_count = len(ranking) if best_k == "all" else min(best_k, len(ranking))
    X_train_valid = pd.concat([X_train, X_valid])
    y_train_valid = pd.concat([y_train, y_valid])
    final_bounds = fit_iqr_bounds(X_train_valid)
    X_train_valid_capped = apply_iqr_bounds(X_train_valid, final_bounds)
    X_test_capped = apply_iqr_bounds(X_test, final_bounds)
    final_preprocessor = make_preprocessor(X_train_valid_capped)
    train_valid_encoded = final_preprocessor.fit_transform(X_train_valid_capped)
    test_encoded = final_preprocessor.transform(X_test_capped)
    final_ranker = make_classifier().fit(train_valid_encoded, y_train_valid)
    final_importances = final_ranker.feature_importances_
    final_ranking = np.argsort(final_importances)[::-1]
    selected = final_ranking[:best_count]
    final_model = make_classifier().fit(train_valid_encoded[:, selected], y_train_valid)
    test_metrics = score_model(final_model, test_encoded[:, selected], y_test)

    run_dir = RESULTS_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(run_dir / "validation_comparison.csv", index=False)
    final_names = final_preprocessor.get_feature_names_out()
    importance_frame = pd.DataFrame({
        "feature": final_names,
        "importance": final_importances,
    }).sort_values("importance", ascending=False)
    selected_feature_names = set(final_names[selected])
    importance_frame["selected"] = importance_frame["feature"].isin(selected_feature_names)
    importance_frame.to_csv(run_dir / "feature_importance.csv", index=False)
    importance_frame.loc[importance_frame["selected"], ["feature", "importance"]].to_csv(
        run_dir / "selected_features.csv", index=False
    )

    top = importance_frame.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"])
    ax.set(title="Top 20 Gradient Boosting Feature Importances", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(run_dir / "top_20_feature_importance.png", dpi=150)
    plt.close(fig)

    result = {
        "model": "Gradient Boosting Classifier",
        "target_classes": {"0": "SalePrice < 150000", "1": "150000 <= SalePrice < 250000", "2": "SalePrice >= 250000"},
        "preprocessing": "Semantic missing handling, IQR capping, median imputation, one-hot encoding, MaxAbsScaler.",
        "feature_selection": "Gradient Boosting impurity-based feature importance on training split; choose feature count by validation macro-F1.",
        "validation_comparison": rows,
        "chosen_feature_count": int(best_count),
        "chosen_feature_count_label": "all" if best_k == "all" else f"top_{best_count}",
        "test": test_metrics,
        "split_rows": {"train": int(len(X_train)), "validation": int(len(X_valid)), "test": int(len(X_test))},
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(__file__, run_dir / "run_script.py")
    print("\nSelected:", result["chosen_feature_count_label"])
    print("Test:", json.dumps(test_metrics, indent=2))
    print(f"Saved results and script copy to: {run_dir}")


if __name__ == "__main__":
    main()
