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
OUTPUT_DIR = Path("LR_Filter_StandardScaler_Results")
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

base_steps = [
    ("missing", BestMissingTransformer(DROP_COLS, MODE_COLS, MISSING_CAT_COLS)),
    ("outlier", OutlierCapper(strategies=OUTLIER_STRATEGIES)),
    ("preprocessor", preprocessor)
]

def new_model():
    return LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")

def make_pipeline(selector=None, chi_square=False):
    steps = [(name, clone(estimator)) for name, estimator in base_steps]
    # Chi2 cần không âm; bổ sung MinMax chỉ sau StandardScaler ở nhánh Chi2.
    if chi_square:
        steps.append(("chi2_nonnegative", MinMaxScaler()))
    if selector is not None:
        steps.append(("feature_selection", selector))
    steps.append(("model", new_model()))
    return Pipeline(steps)

def make_selector(method, k):
    if method == "ANOVA_F":
        return SelectKBest(f_classif, k=k), False
    if method == "Mutual_Information":
        return SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=SEED), k=k), False
    if method == "Chi2":
        return SelectKBest(chi2, k=k), True
    raise ValueError(method)

def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)

print("\n=== BASELINE: STANDARD SCALER, KHÔNG FEATURE SELECTION ===")
start = perf_counter()
baseline_scores = cross_val_score(make_pipeline(), X_train, y_train,
                                  cv=CV, scoring="f1_macro", n_jobs=1)
baseline_seconds = perf_counter() - start
baseline_pipe = make_pipeline().fit(X_train, y_train)
n_features = len(baseline_pipe.named_steps["preprocessor"].get_feature_names_out())
ks = sorted({k for k in K_VALUES if 1 <= k <= n_features})
if not ks:
    ks = [n_features]
print(f"Số đặc trưng sau OneHot: {n_features}")
print(f"Baseline F1 Macro: {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}")

methods = ["ANOVA_F", "Mutual_Information", "Chi2"]
results = [{"Method": "Baseline", "K": n_features,
            "F1_CV_Mean": baseline_scores.mean(),
            "F1_CV_Std": baseline_scores.std(),
            "CV_Time_Seconds": baseline_seconds}]
for method in methods:
    for k in ks:
        selector, nonnegative = make_selector(method, k)
        start = perf_counter()
        scores = cross_val_score(make_pipeline(selector, nonnegative),
            X_train, y_train, cv=CV, scoring="f1_macro", n_jobs=1)
        elapsed = perf_counter() - start
        results.append({"Method": method, "K": k,
            "F1_CV_Mean": scores.mean(), "F1_CV_Std": scores.std(),
            "CV_Time_Seconds": elapsed})
        print(f"{method:20s} K={k:3d}: F1={scores.mean():.4f} ± {scores.std():.4f} | {elapsed:.1f}s")

result_df = pd.DataFrame(results).sort_values(
    ["F1_CV_Mean", "F1_CV_Std"], ascending=[False, True])
result_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False, encoding="utf-8-sig")
print("\n=== XẾP HẠNG 5-FOLD CV ===")
print(result_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
for method in methods:
    group = result_df[result_df.Method == method].sort_values("K")
    ax.plot(group.K, group.F1_CV_Mean, marker="o", label=method)
ax.axhline(baseline_scores.mean(), color="gray", linestyle="--", label="Baseline")
ax.set(xlabel="Số đặc trưng K", ylabel="F1 Macro (5-Fold CV)",
       title="Filter Feature Selection - Logistic Regression + StandardScaler")
ax.grid(alpha=.3)
ax.legend()
save_plot(fig, "01_cv_f1_comparison.png")

# Lưu các đặc trưng được chọn cho từng Filter và K (fit trên train đầy đủ).
all_feature_names = baseline_pipe.named_steps["preprocessor"].get_feature_names_out()
for method in methods:
    for k in ks:
        selector, nonnegative = make_selector(method, k)
        fitted = make_pipeline(selector, nonnegative).fit(X_train, y_train)
        mask = fitted.named_steps["feature_selection"].get_support()
        selected = all_feature_names[mask]
        pd.DataFrame({"Selected_Feature": selected}).to_csv(
            OUTPUT_DIR / f"selected_features_{method}_k{k}.csv",
            index=False, encoding="utf-8-sig")

# Chọn theo CV trên train; chỉ đánh giá validation một lần.
best = result_df.iloc[0]
method, k = best["Method"], int(best["K"])
if method == "Baseline":
    final_pipe = make_pipeline()
else:
    selector, nonnegative = make_selector(method, k)
    final_pipe = make_pipeline(selector, nonnegative)
final_pipe.fit(X_train, y_train)
y_pred = final_pipe.predict(X_val)
y_prob = final_pipe.predict_proba(X_val)
metrics = {"Method": method, "K": k,
    "Accuracy": accuracy_score(y_val, y_pred),
    "Precision_Macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
    "Recall_Macro": recall_score(y_val, y_pred, average="macro", zero_division=0),
    "F1_Macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
    "Log_Loss": log_loss(y_val, y_prob, labels=np.arange(len(label_encoder.classes_)))}
pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "validation_metrics.csv",
                                index=False, encoding="utf-8-sig")
print("\n=== VALIDATION ===")
for name, value in metrics.items():
    print(f"{name}: {value}")
selected_names = final_pipe.named_steps["preprocessor"].get_feature_names_out()
if method != "Baseline":
    selected_names = selected_names[final_pipe.named_steps["feature_selection"].get_support()]
pd.DataFrame({"Selected_Feature": selected_names}).to_csv(
    OUTPUT_DIR / "selected_features.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay(confusion_matrix(y_val, y_pred),
    display_labels=label_encoder.classes_).plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title(f"Confusion Matrix - {method}")
save_plot(fig, "02_confusion_matrix.png")

n_classes = len(label_encoder.classes_)
y_bin = label_binarize(y_val, classes=np.arange(n_classes))
fig, ax = plt.subplots(figsize=(7, 5))
for i, cls in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    ax.plot(fpr, tpr, label=f"{cls} AUC={auc(fpr, tpr):.3f}")
ax.plot([0, 1], [0, 1], "k--")
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC OvR")
ax.legend()
save_plot(fig, "03_roc_ovr.png")

fig, ax = plt.subplots(figsize=(7, 5))
for i, cls in enumerate(label_encoder.classes_):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
    ap = average_precision_score(y_bin[:, i], y_prob[:, i])
    ax.plot(recall, precision, label=f"{cls} AP={ap:.3f}")
ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall OvR")
ax.legend()
save_plot(fig, "04_precision_recall.png")

sizes, train_scores, cv_scores = learning_curve(
    final_pipe, X_train, y_train, cv=3, scoring="f1_macro",
    train_sizes=np.linspace(.2, 1., 4), n_jobs=1)
fig, ax = plt.subplots(figsize=(7, 5))
for scores, name in [(train_scores, "Train"), (cv_scores, "CV")]:
    mean, std = scores.mean(axis=1), scores.std(axis=1)
    ax.plot(sizes, mean, marker="o", label=name)
    ax.fill_between(sizes, mean-std, mean+std, alpha=.15)
ax.set(xlabel="Training size", ylabel="F1 Macro", title=f"Learning Curve - {method}")
ax.grid(alpha=.3)
ax.legend()
save_plot(fig, "05_learning_curve.png")
print(f"\nĐã lưu kết quả tại: {OUTPUT_DIR.resolve()}")
