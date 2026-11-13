from datetime import datetime
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT.parent / "input" / "train.csv"
RESULTS_ROOT = ROOT / "results" / "maxabs_classifier_gradient_boosting"
RANDOM_STATE = 42
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["Low (<150k)", "Medium (150k-250k)", "High (>=250k)"]
NO_FEATURE_CATEGORICALS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2",
    "BsmtExposure", "BsmtFinType1", "BsmtCond", "BsmtQual",
]
OUTLIER_COLUMNS = [
    "LotArea", "GrLivArea", "TotalBsmtSF", "LotFrontage", "GarageArea", "MasVnrArea"
]


def prepare_features(X):
    X = X.copy()
    for col in NO_FEATURE_CATEGORICALS:
        if col in X.columns:
            X[col] = X[col].fillna("None")
    for col in ("GarageYrBlt", "MasVnrArea"):
        if col in X.columns:
            X[col] = X[col].fillna(0)
    # LotFrontage and Electrical are imputed inside the pipeline using training data.
    return X


def make_pipeline(X):
    categorical_columns = X.select_dtypes(include=["object", "category", "string"]).columns
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MaxAbsScaler()),
    ])
    preprocessing = ColumnTransformer(
        [("categorical", categorical, categorical_columns),
         ("numeric", numeric, numeric_columns)],
        sparse_threshold=0,
    )
    classifier = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessing", preprocessing), ("model", classifier)])


def fit_iqr_bounds(X_train):
    bounds = {}
    for col in OUTLIER_COLUMNS:
        q1, q3 = X_train[col].quantile([0.25, 0.75])
        spread = q3 - q1
        bounds[col] = (q1 - 1.5 * spread, q3 + 1.5 * spread)
    return bounds


def apply_iqr_bounds(X, bounds):
    X = X.copy()
    for col, (lower, upper) in bounds.items():
        X[col] = X[col].clip(lower=lower, upper=upper)
    return X


def metrics_for(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_macro": float(precision_score(y, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=CLASS_LABELS)),
    }


def plot_confusion_matrix(y_true, y_pred, output_path):
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(
        xticks=CLASS_LABELS, yticks=CLASS_LABELS,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted price group", ylabel="Actual price group",
        title="Confusion Matrix (held-out test set)",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_and_pr(y_true, probabilities, roc_path, pr_path):
    y_binary = label_binarize(y_true, classes=CLASS_LABELS)
    fig, ax = plt.subplots(figsize=(7, 6))
    aucs = []
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_binary[:, i], probabilities[:, i])
        auc = roc_auc_score(y_binary[:, i], probabilities[:, i])
        aucs.append(auc)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="One-vs-Rest ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    average_precisions = []
    for i, name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y_binary[:, i], probabilities[:, i])
        ap = average_precision_score(y_binary[:, i], probabilities[:, i])
        average_precisions.append(ap)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="One-vs-Rest Precision-Recall Curves")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)
    return float(np.mean(aucs)), float(np.mean(average_precisions))


def plot_learning_curve(X_train, y_train, X_valid, y_valid, output_path):
    sizes = sorted(set(max(30, int(len(X_train) * fraction)) for fraction in (0.2, 0.4, 0.6, 0.8, 1.0)))
    rows = []
    for size in sizes:
        if size >= len(X_train):
            indices = np.arange(len(X_train))
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=RANDOM_STATE)
            indices, _ = next(splitter.split(X_train, y_train))
        X_sub = X_train.iloc[indices].copy()
        y_sub = y_train.iloc[indices].copy()
        bounds = fit_iqr_bounds(X_sub)
        X_sub_capped = apply_iqr_bounds(X_sub, bounds)
        X_valid_capped = apply_iqr_bounds(X_valid, bounds)
        model = make_pipeline(X_sub_capped)
        model.fit(X_sub_capped, y_sub)
        train_f1 = f1_score(y_sub, model.predict(X_sub_capped), average="macro", zero_division=0)
        valid_f1 = f1_score(y_valid, model.predict(X_valid_capped), average="macro", zero_division=0)
        rows.append({
            "training_rows": int(size),
            "train_macro_f1": float(train_f1),
            "validation_macro_f1": float(valid_f1),
        })
        print(f"Learning curve {size} rows: train F1={train_f1:.4f}, validation F1={valid_f1:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["training_rows"] for r in rows], [r["train_macro_f1"] for r in rows], marker="o", label="Train macro F1")
    ax.plot([r["training_rows"] for r in rows], [r["validation_macro_f1"] for r in rows], marker="o", label="Validation macro F1")
    ax.set(xlabel="Training rows", ylabel="Macro F1", title="Learning Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return rows


def main():
    data = pd.read_csv(TRAIN_PATH)
    y = pd.cut(
        data["SalePrice"],
        bins=[-np.inf, 150000, 250000, np.inf],
        labels=CLASS_LABELS,
        right=False,
    ).astype(int)
    X = prepare_features(data.drop(columns=["SalePrice", "Id"], errors="ignore"))

    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=(0.15 / 0.85),
        random_state=RANDOM_STATE, stratify=y_train_valid,
    )

    train_bounds = fit_iqr_bounds(X_train)
    X_train_capped = apply_iqr_bounds(X_train, train_bounds)
    X_valid_capped = apply_iqr_bounds(X_valid, train_bounds)
    validation_model = make_pipeline(X_train_capped)
    validation_model.fit(X_train_capped, y_train)
    validation_metrics = metrics_for(validation_model, X_valid_capped, y_valid)

    X_train_valid = pd.concat([X_train, X_valid])
    y_train_valid = pd.concat([y_train, y_valid])
    final_bounds = fit_iqr_bounds(X_train_valid)
    X_train_valid_capped = apply_iqr_bounds(X_train_valid, final_bounds)
    X_test_capped = apply_iqr_bounds(X_test, final_bounds)
    final_model = make_pipeline(X_train_valid_capped)
    final_model.fit(X_train_valid_capped, y_train_valid)
    test_predictions = final_model.predict(X_test_capped)
    test_probabilities = final_model.predict_proba(X_test_capped)
    test_metrics = metrics_for(final_model, X_test_capped, y_test)

    run_dir = RESULTS_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    plot_confusion_matrix(y_test, test_predictions, run_dir / "confusion_matrix.png")
    macro_auc, macro_ap = plot_roc_and_pr(
        y_test, test_probabilities, run_dir / "roc_curve.png", run_dir / "precision_recall_curve.png"
    )
    learning_curve_rows = plot_learning_curve(X_train, y_train, X_valid, y_valid, run_dir / "learning_curve.png")
    test_metrics["roc_auc_ovr_macro"] = macro_auc
    test_metrics["average_precision_macro"] = macro_ap

    metrics = {
        "model": "Gradient Boosting Classifier",
        "target_classes": {"0": "SalePrice < 150000", "1": "150000 <= SalePrice < 250000", "2": "SalePrice >= 250000"},
        "validation": validation_metrics,
        "test": test_metrics,
        "learning_curve": learning_curve_rows,
        "split_rows": {
            "train": int(len(X_train)),
            "validation": int(len(X_valid)),
            "test": int(len(X_test)),
        },
        "preprocessing": "Semantic missing handling, IQR capping, then MaxAbsScaler.",
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    shutil.copy2(__file__, run_dir / "run_script.py")
    print("\nValidation metrics:", json.dumps(validation_metrics, indent=2))
    print("Test metrics:", json.dumps(test_metrics, indent=2))
    print(f"Saved metrics, charts, and script copy to: {run_dir}")


if __name__ == "__main__":
    main()
