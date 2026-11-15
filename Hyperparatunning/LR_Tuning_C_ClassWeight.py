# House Prices - Logistic Regression - Missing + Outlier + StandardScaler
# Giữ nguyên chiến lược Missing và Outlier đã chọn trong thực nghiệm trước.
# Chú ý: LotFrontage và MasVnrArea bị xóa bởi chiến lược Missing,
# nên capping IQR của hai biến này sẽ không được áp dụng.
# Các bước fit (imputer, outlier, scaler) chỉ học từ train.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

from sklearn.base import BaseEstimator, TransformerMixin

OUTLIER_STRATEGIES = {'LotArea': 'zscore', 'GrLivArea': 'zscore', 'TotalBsmtSF': 'iqr', 'LotFrontage': 'iqr', 'GarageArea': 'iqr', 'MasVnrArea': 'iqr'}

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, strategies=None, iqr_factor=1.5, z_threshold=3.0):
        self.strategies = strategies
        self.iqr_factor = iqr_factor
        self.z_threshold = z_threshold

    def fit(self, X, y=None):
        self.bounds_ = {}
        strategies = self.strategies or {}
        for col, strategy in strategies.items():
            if strategy not in ("iqr", "zscore", "none"):
                raise ValueError(f"Chiến lược Outlier không hợp lệ: {col}={strategy}")
            if col not in X.columns or strategy == "none":
                continue
            values = pd.to_numeric(X[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if values.empty:
                continue
            if strategy == "iqr":
                q1, q3 = values.quantile([0.25, 0.75])
                spread = q3 - q1
                if spread == 0:
                    continue
                lower = q1 - self.iqr_factor * spread
                upper = q3 + self.iqr_factor * spread
            else:
                mean, std = values.mean(), values.std(ddof=0)
                if std == 0 or not np.isfinite(std):
                    continue
                lower = mean - self.z_threshold * std
                upper = mean + self.z_threshold * std
            self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            numeric = pd.to_numeric(X[col], errors="coerce")
            capped = numeric.clip(lower=lower, upper=upper)
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = capped
            else:
                # Giữ nguyên nhóm Missing; chỉ capping giá trị số thực sự.
                mask = numeric.notna()
                X.loc[mask, col] = capped.loc[mask].astype(str)
        return X


from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.metrics import precision_score, recall_score
from sklearn.base import clone
from pathlib import Path
from time import perf_counter
import warnings
warnings.filterwarnings("ignore", message="Features .* are constant", category=UserWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

class BestMissingTransformer(BaseEstimator, TransformerMixin):
    """Giữ nguyên drop/mode/Missing của LR_standard, fit mode trên từng train-fold."""
    def __init__(self, drop_cols=None, mode_cols=None, category_missing_cols=None):
        self.drop_cols = drop_cols
        self.mode_cols = mode_cols
        self.category_missing_cols = category_missing_cols

    def fit(self, X, y=None):
        X = X.drop(columns=self.drop_cols or [], errors="ignore")
        self.modes_ = {}
        for col in self.mode_cols or []:
            if col in X.columns:
                mode = X[col].mode(dropna=True)
                if not mode.empty:
                    self.modes_[col] = mode.iloc[0]
        return self

    def transform(self, X):
        X = X.copy().drop(columns=self.drop_cols or [], errors="ignore")
        for col, value in self.modes_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        for col in self.category_missing_cols or []:
            if col in X.columns:
                X[col] = X[col].astype("object").fillna("Missing")
        # LR_standard ép tất cả biến phân loại về string trước OneHot.
        for col in X.select_dtypes(include=["object", "category", "string"]).columns:
            X[col] = X[col].astype("string").fillna("Missing").astype(str)
        return X

# ============================================================
# FILTER FEATURE SELECTION - LOGISTIC REGRESSION + STANDARD SCALER
# Tiền xử lý theo LR_standard(3).py, thực nghiệm theo file mixed FS.
# Không dùng Mixed Scaler, Ordinal Encoder, Wrapper hoặc Embedded.
# ============================================================
SEED = 42
K_VALUES = [10, 20, 30, 50, 75, 100]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
OUTPUT_DIR = Path("LR_Tuning_C_ClassWeight_ANOVA_K100_Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = True

raw = pd.read_csv("../data/raw/train.csv", low_memory=False)
raw["PriceCategory"] = pd.cut(raw["SalePrice"],
    bins=[-np.inf, 150000, 250000, np.inf],
    labels=["Low", "Medium", "High"], right=False)
label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(raw["PriceCategory"])
X_all = raw.drop(columns=["SalePrice", "PriceCategory", "Id"], errors="ignore").copy()
X_all["MSSubClass"] = X_all["MSSubClass"].astype("object")
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)

# Chiến lược Missing giữ đúng LR_standard(3).py.
DROP_COLS = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'LotFrontage',
             'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure',
             'BsmtFinType2', 'BsmtFinType1', 'MasVnrArea']
MODE_COLS = ['BsmtQual', 'BsmtCond']
MISSING_CAT_COLS = ['FireplaceQu', 'GarageCond']

kept_cols = X_train.drop(columns=DROP_COLS, errors="ignore")
categorical_cols = kept_cols.select_dtypes(include=["object", "category", "string"]).columns.tolist()
numeric_cols = kept_cols.select_dtypes(include=["number"]).columns.tolist()

# Giữ đúng OneHot cho toàn bộ categorical, StandardScaler cho numeric.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer([
    ("categorical", categorical_pipeline, categorical_cols),
    ("numeric", numeric_pipeline, numeric_cols)
])


# ============================================================
# TUNING CLASS_WEIGHT - C=0.1 từ thí nghiệm trước.
# Penalty/solver giữ nguyên l2/lbfgs, không tuning penalty.
# ============================================================
import json
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score

BEST_C = 0.1
K_BEST = 100
CLASS_WEIGHTS = [None, "balanced"]


def build_pipeline(weight):
    return Pipeline([
        ("missing", clone(BestMissingTransformer(DROP_COLS, MODE_COLS, MISSING_CAT_COLS))),
        ("outlier", clone(OutlierCapper(strategies=OUTLIER_STRATEGIES))),
        ("preprocessor", clone(preprocessor)),
        ("feature_selection", SelectKBest(f_classif, k=K_BEST)),
        ("model", LogisticRegression(
            C=BEST_C, class_weight=weight, penalty="l2", solver="lbfgs",
            max_iter=2000, random_state=SEED
        ))
    ])


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


records = []
for weight in CLASS_WEIGHTS:
    name = "None" if weight is None else weight
    print(f"\n=== class_weight={name} | C={BEST_C} ===")
    start = perf_counter()
    scores = cross_validate(
        build_pipeline(weight), X_train, y_train, cv=CV,
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy", "neg_log_loss": "neg_log_loss"},
        return_train_score=True, n_jobs=1, error_score="raise"
    )
    elapsed = perf_counter() - start
    fitted = build_pipeline(weight).fit(X_train, y_train)
    pred_train, pred_val = fitted.predict(X_train), fitted.predict(X_val)
    prob_train, prob_val = fitted.predict_proba(X_train), fitted.predict_proba(X_val)
    row = {
        "C": BEST_C, "Class_Weight": name, "Penalty": "l2", "Solver": "lbfgs", "K": K_BEST,
        "CV_F1_Macro_Mean": scores["test_f1_macro"].mean(),
        "CV_F1_Macro_Std": scores["test_f1_macro"].std(),
        "CV_Train_F1_Macro": scores["train_f1_macro"].mean(),
        "CV_Accuracy_Mean": scores["test_accuracy"].mean(),
        "CV_Log_Loss_Mean": -scores["test_neg_log_loss"].mean(),
        "Train_F1_Macro": f1_score(y_train, pred_train, average="macro", zero_division=0),
        "Validation_F1_Macro": f1_score(y_val, pred_val, average="macro", zero_division=0),
        "Train_Classification_Error": 1 - accuracy_score(y_train, pred_train),
        "Validation_Classification_Error": 1 - accuracy_score(y_val, pred_val),
        "Train_Log_Loss": log_loss(y_train, prob_train, labels=np.arange(len(label_encoder.classes_))),
        "Validation_Log_Loss": log_loss(y_val, prob_val, labels=np.arange(len(label_encoder.classes_))),
        "CV_Time_Seconds": elapsed,
    }
    records.append(row)
    print(f"CV F1 Macro: {row['CV_F1_Macro_Mean']:.4f} ± {row['CV_F1_Macro_Std']:.4f}")

results = pd.DataFrame(records)
results.to_csv(OUTPUT_DIR / "01_class_weight_tuning_results.csv", index=False, encoding="utf-8-sig")
best_row = results.sort_values(
    ["CV_F1_Macro_Mean", "CV_F1_Macro_Std"], ascending=[False, True], kind="stable"
).iloc[0]
best_weight = None if best_row["Class_Weight"] == "None" else str(best_row["Class_Weight"])
print(f"\nClass weight tốt nhất theo 5-fold CV: {best_weight} | F1={best_row['CV_F1_Macro_Mean']:.4f}")

params = {
    "C": BEST_C, "class_weight": best_weight, "penalty": "l2", "solver": "lbfgs",
    "max_iter": 2000, "random_state": SEED,
    "feature_selection": "ANOVA_F", "k": K_BEST, "scaler": "StandardScaler",
    "selection_metric": "CV_F1_Macro_Mean"
}
(OUTPUT_DIR / "best_params.json").write_text(
    json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8"
)

x = np.arange(len(CLASS_WEIGHTS))
labels = ["None", "balanced"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, results.Train_Classification_Error, "o-", label="Train classification error")
ax.plot(x, results.Validation_Classification_Error, "s-", label="Validation (test) classification error")
ax.set_xticks(x, labels)
ax.set(xlabel="class_weight", ylabel="Error = 1 - Accuracy", title=f"Classification Error vs class_weight (C={BEST_C})")
ax.grid(alpha=.3); ax.legend()
save_plot(fig, "02_train_test_classification_error_vs_class_weight.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, results.Train_Log_Loss, "o-", label="Train Log Loss")
ax.plot(x, results.Validation_Log_Loss, "s-", label="Validation (test) Log Loss")
ax.set_xticks(x, labels)
ax.set(xlabel="class_weight", ylabel="Log Loss", title=f"Train / Validation Log Loss vs class_weight (C={BEST_C})")
ax.grid(alpha=.3); ax.legend()
save_plot(fig, "03_train_test_log_loss_vs_class_weight.png")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, results.CV_Train_F1_Macro, "o-", label="Train CV F1 Macro")
ax.errorbar(x, results.CV_F1_Macro_Mean, yerr=results.CV_F1_Macro_Std,
            fmt="s-", capsize=3, label="5-fold CV F1 Macro")
ax.set_xticks(x, labels)
ax.set(xlabel="class_weight", ylabel="F1 Macro", title=f"Train / CV F1 Macro vs class_weight (C={BEST_C})")
ax.grid(alpha=.3); ax.legend()
save_plot(fig, "04_cv_f1_macro_vs_class_weight.png")

final_pipe = build_pipeline(best_weight).fit(X_train, y_train)
y_pred = final_pipe.predict(X_val)
y_prob = final_pipe.predict_proba(X_val)
metrics = {
    "C": BEST_C, "Class_Weight": "None" if best_weight is None else best_weight,
    "Penalty": "l2", "Solver": "lbfgs", "K": K_BEST,
    "Accuracy": accuracy_score(y_val, y_pred),
    "Precision_Macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
    "Recall_Macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
    "F1_Macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
    "Log_Loss": log_loss(y_val, y_prob, labels=np.arange(len(label_encoder.classes_))),
    "ROC_AUC_Macro_OvR": roc_auc_score(y_val, y_prob, multi_class="ovr", average="macro")
}
pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "05_best_class_weight_validation_metrics.csv", index=False, encoding="utf-8-sig")
print("\n=== VALIDATION: CLASS_WEIGHT ĐƯỢC CHỌN ===")
for key, value in metrics.items():
    print(f"{key}: {value}")

names = final_pipe.named_steps["preprocessor"].get_feature_names_out()
selected = names[final_pipe.named_steps["feature_selection"].get_support()]
pd.DataFrame({"Selected_Feature": selected}).to_csv(
    OUTPUT_DIR / "06_anova_selected_features_k100.csv", index=False, encoding="utf-8-sig"
)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix(y_val, y_pred, labels=np.arange(len(label_encoder.classes_))),
    display_labels=label_encoder.classes_
).plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
ax.set_title(f"Confusion Matrix Heatmap - class_weight={best_weight} - C={BEST_C}")
save_plot(fig, "07_confusion_matrix_heatmap.png")

y_bin = label_binarize(y_val, classes=np.arange(len(label_encoder.classes_)))
fig, ax = plt.subplots(figsize=(8, 6))
roc_rows = []
for i, cls in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    roc_rows.append({"Class": cls, "ROC_AUC_OvR": roc_auc})
    ax.plot(fpr, tpr, label=f"{cls}: AUC={roc_auc:.3f}")
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
       title=f"ROC-AUC OvR - class_weight={best_weight} - C={BEST_C}")
ax.grid(alpha=.3); ax.legend()
save_plot(fig, "08_roc_auc_ovr.png")
pd.DataFrame(roc_rows).to_csv(OUTPUT_DIR / "09_roc_auc_per_class.csv", index=False, encoding="utf-8-sig")
print(f"\nĐã lưu kết quả tại: {OUTPUT_DIR.resolve()}")
print("Ghi chú: Validation (test) là tập hold-out 20%, không phải Kaggle test.csv.")
