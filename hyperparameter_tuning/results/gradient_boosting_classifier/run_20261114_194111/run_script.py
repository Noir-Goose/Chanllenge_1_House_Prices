"""Tune Gradient Boosting hyperparameters for the three price classes.

Run from the project root:
    python hyperparameter_tuning/gradient_boosting_classifier.py

From a notebook whose working directory is preprocessing:
    %run ../hyperparameter_tuning/gradient_boosting_classifier.py
"""

from datetime import datetime
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "input" / "train.csv"
RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "gradient_boosting_classifier"
RANDOM_STATE = 42
CLASS_LABELS = [0, 1, 2]
NO_FEATURE_CATEGORICALS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2",
    "BsmtExposure", "BsmtFinType1", "BsmtCond", "BsmtQual",
]
OUTLIER_COLUMNS = [
    "LotArea", "GrLivArea", "TotalBsmtSF", "LotFrontage", "GarageArea", "MasVnrArea"
]


class IQRClipper(BaseEstimator, TransformerMixin):
    """Fit IQR caps on each training fold and apply them to that fold's data."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.bounds_ = {}
        for column in self.columns:
            if column in X.columns:
                q1, q3 = X[column].quantile([0.25, 0.75])
                spread = q3 - q1
                self.bounds_[column] = (q1 - 1.5 * spread, q3 + 1.5 * spread)
        return self

    def transform(self, X):
        X = X.copy()
        for column, (lower, upper) in self.bounds_.items():
            X[column] = X[column].clip(lower=lower, upper=upper)
        return X


def prepare_features(frame):
    frame = frame.copy()
    for column in NO_FEATURE_CATEGORICALS:
        if column in frame:
            frame[column] = frame[column].fillna("None")
    for column in ("GarageYrBlt", "MasVnrArea"):
        if column in frame:
            frame[column] = frame[column].fillna(0)
    return frame


def make_pipeline():
    # Column lists are learned from the DataFrame at fit time; keep DataFrame
    # through IQRClipper so its fold-specific quantiles do not see held-out rows.
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MaxAbsScaler()),
    ])
    # ColumnTransformer selectors are resolved from the DataFrame columns when
    # the pipeline is fit; use dtype selectors to keep the split fold-safe.
    from sklearn.compose import make_column_selector

    preprocessing = ColumnTransformer(
        [("categorical", categorical, make_column_selector(dtype_include=["object", "category", "string"])),
         ("numeric", numeric, make_column_selector(dtype_include=["number", "bool"]))],
        sparse_threshold=0,
    )
    selector = SelectFromModel(
        estimator=GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
        ),
        threshold=-np.inf,
        max_features=30,
    )
    classifier = GradientBoostingClassifier(random_state=RANDOM_STATE)
    return Pipeline([
        ("iqr", IQRClipper(OUTLIER_COLUMNS)),
        ("preprocessing", preprocessing),
        ("selection", selector),
        ("model", classifier),
    ])


def get_metrics(model, X, y):
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision_macro": float(precision_score(y, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, prediction, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=CLASS_LABELS)),
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

    parameter_space = {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [1, 2, 3],
        "model__min_samples_leaf": [1, 3, 5],
        "model__subsample": [0.8, 1.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=make_pipeline(),
        param_distributions=parameter_space,
        n_iter=15,
        scoring="f1_macro",
        cv=cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    validation_metrics = get_metrics(search.best_estimator_, X_valid, y_valid)

    # Final evaluation: after parameter selection, refit the selected pipeline
    # on train+validation and evaluate the held-out test split once.
    final_model = search.best_estimator_
    final_model.fit(X_train_valid, y_train_valid)
    test_metrics = get_metrics(final_model, X_test, y_test)

    run_dir = RESULTS_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cv_results.to_csv(run_dir / "cv_results.csv", index=False)
    feature_names = final_model.named_steps["preprocessing"].get_feature_names_out()
    selected_mask = final_model.named_steps["selection"].get_support()
    selected_features = pd.DataFrame({
        "feature": feature_names[selected_mask],
        "importance": final_model.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    selected_features.to_csv(run_dir / "selected_features.csv", index=False)

    best_rows = cv_results.head(10).sort_values("mean_test_score")
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = [str(row) for row in best_rows["params"]]
    ax.barh(labels, best_rows["mean_test_score"])
    ax.set(title="Top 10 hyperparameter configurations (CV macro-F1)", xlabel="Mean CV macro-F1")
    fig.tight_layout()
    fig.savefig(run_dir / "top_hyperparameters.png", dpi=150)
    plt.close(fig)

    result = {
        "model": "Gradient Boosting Classifier",
        "target_classes": {"0": "SalePrice < 150000", "1": "150000 <= SalePrice < 250000", "2": "SalePrice >= 250000"},
        "selection_metric": "macro-F1",
        "cross_validation": {"type": "StratifiedKFold", "folds": 3, "candidates": 15,
                             "best_macro_f1": float(search.best_score_)},
        "best_parameters": search.best_params_,
        "validation": validation_metrics,
        "test": test_metrics,
        "selected_feature_count": int(selected_mask.sum()),
        "preprocessing": "Fold-fitted IQR capping, semantic missing handling, median/mode imputation, One-Hot encoding, MaxAbsScaler, top 30 Gradient Boosting importance features.",
        "split_rows": {"train": int(len(X_train)), "validation": int(len(X_valid)), "test": int(len(X_test))},
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(__file__, run_dir / "run_script.py")
    print("Best CV macro-F1:", f"{search.best_score_:.4f}")
    print("Best parameters:", json.dumps(search.best_params_, indent=2))
    print("Validation:", json.dumps(validation_metrics, indent=2))
    print("Test:", json.dumps(test_metrics, indent=2))
    print(f"Saved results and script copy to: {run_dir}")


if __name__ == "__main__":
    main()
