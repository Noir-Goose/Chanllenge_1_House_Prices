# Cần thực nghiệm lại tổ hợp missing + outlier để xác nhận hiệu quả.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2, RFE, SelectFromModel
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score

import warnings

warnings.filterwarnings(
    "ignore",
    message="Features .* are constant",
    category=UserWarning
)

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning
)


# Xử lý outlier: IQR capping cho LotFrontage, z-score cho các biến khác.
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



# Các biến có thứ tự trong House Prices. Giá trị "Missing" được mã hóa -1.
# Mã ordinal biểu diễn thứ tự, KHÔNG chứng minh dữ liệu có phân phối Gaussian.
ORDINAL_LEVELS = {
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    "PavedDrive": ["N", "P", "Y"],
    "CentralAir": ["N", "Y"],
    "Street": ["Grvl", "Pave"],
    "Alley": ["Grvl", "Pave"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# 6 biến theo yêu cầu; những cột đã bị drop sẽ tự động bị bỏ qua.
FORCE_MINMAX_COLS = [
    "LotArea", "GrLivArea", "LotFrontage", "MasVnrArea", "TotalBsmtSF", "GarageArea"
]
SKEW_THRESHOLD = 1.0

class CategoricalToString(BaseEstimator, TransformerMixin):
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.DataFrame(X).fillna("Missing").astype(str).to_numpy()

class BestMissingTransformer(BaseEstimator, TransformerMixin):
    """Giữ chiến lược Missing gốc nhưng học mode từ train của mỗi lần fit."""
    def __init__(self, drop_cols=None, mode_cols=None, missing_category_cols=None):
        self.drop_cols = drop_cols
        self.mode_cols = mode_cols
        self.missing_category_cols = missing_category_cols

    def fit(self, X, y=None):
        X = X.drop(columns=self.drop_cols or [], errors="ignore")
        self.modes_ = {}

        for col in self.mode_cols or []:
            if col in X:
                mode = X[col].mode(dropna=True)
                self.modes_[col] = mode.iloc[0] if not mode.empty else "Missing"
        return self

    def transform(self, X):
        X = X.copy().drop(columns=self.drop_cols or [], errors="ignore")
        for col, mode in self.modes_.items():
            if col in X:
                X[col] = X[col].fillna(mode)

        for col in self.missing_category_cols or []:
            if col in X:
                X[col] = X[col].astype("object").fillna("Missing")
        return X

class SkewnessColumnScaler(BaseEstimator, TransformerMixin):
    """Sau median imputation: tự chọn MinMax/Z-score trên TRAIN của mỗi fold."""
    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.columns_, dtype=object)
    
    def __init__(self, columns=None, forced_minmax=None, threshold=1.0):
        self.columns = columns
        self.forced_minmax = forced_minmax
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.columns_ = list(self.columns or [])
        self.skewness_ = {}
        self.scalers_ = {}
        self.method_ = {}

        for j, col in enumerate(self.columns_):
            s = pd.Series(X[:, j]).skew()
            s = float(s) if np.isfinite(s) else 0.0
            method = "MinMaxScaler" if col in (self.forced_minmax or []) or abs(s) > self.threshold else "StandardScaler"
            scaler = MinMaxScaler() if method == "MinMaxScaler" else StandardScaler()
            scaler.fit(X[:, [j]])
            self.skewness_[col] = s
            self.method_[col] = method
            self.scalers_[col] = scaler
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for j, col in enumerate(self.columns_):
            X[:, j] = self.scalers_[col].transform(X[:, [j]]).ravel()
        return X



# ============================================================
# FEATURE SELECTION - sử dụng lại LR Mixed Scaling ban đầu
# Chạy: %run LR_mixed_scaler_feature_selection.py
# ============================================================
SEED = 42
K_VALUES = [10, 20, 30, 50, 75, 100]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

OUTPUT_PREFIX = "LR_FS"

# 1. Dữ liệu và nhãn: giữ đúng file gốc
raw = pd.read_csv("../data/raw/train.csv", low_memory=False)
raw["PriceCategory"] = pd.cut(
    raw["SalePrice"], bins=[-np.inf, 150000, 250000, np.inf],
    labels=["Low", "Medium", "High"], right=False
)

label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(raw["PriceCategory"])
X_all = raw.drop(columns=["SalePrice", "PriceCategory", "Id"], errors="ignore").copy()
X_all["MSSubClass"] = X_all["MSSubClass"].astype("object")
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED
)

# 2. Giữ chiến lược Missing / Outlier / Encoding / Scaling
DROP_COLS = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'LotFrontage',
             'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure',
             'BsmtFinType2', 'BsmtFinType1', 'MasVnrArea']

MODE_COLS = ['BsmtQual', 'BsmtCond']

MISSING_CAT_COLS = ['FireplaceQu', 'GarageCond']

kept_cols = X_train.drop(columns=DROP_COLS, errors="ignore")
ordinal_cols = [c for c in ORDINAL_LEVELS if c in kept_cols.columns]
nominal_cols = [c for c in kept_cols.select_dtypes(include=["object", "category", "string"]).columns
                if c not in ordinal_cols]
numeric_cols = kept_cols.select_dtypes(include=["number"]).columns.tolist()

nominal_pipeline = Pipeline([
    ("to_string", CategoricalToString()),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

ordinal_pipeline = Pipeline([
    ("to_string", CategoricalToString()),
    ("encoder", OrdinalEncoder(
        categories=[["Missing"] + ORDINAL_LEVELS[c] for c in ordinal_cols],
        handle_unknown="use_encoded_value", unknown_value=-1)),
    ("scaler", MinMaxScaler())
])

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", SkewnessColumnScaler(
        columns=numeric_cols, forced_minmax=FORCE_MINMAX_COLS,
        threshold=SKEW_THRESHOLD))
])

preprocessor = ColumnTransformer([
    ("categorical", nominal_pipeline, nominal_cols),
    ("ordinal", ordinal_pipeline, ordinal_cols),
    ("numeric", numeric_pipeline, numeric_cols)
])

base_steps = [
    ("missing", BestMissingTransformer(
        drop_cols=DROP_COLS, mode_cols=MODE_COLS,
        missing_category_cols=MISSING_CAT_COLS)),
    ("outlier", OutlierCapper(strategies=OUTLIER_STRATEGIES)),
    ("preprocessor", preprocessor)
]

def new_model():
    return LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")

def make_pipeline(selector=None, chi_square=False):
    steps = list(base_steps)
    # Chi2 cần tất cả đầu vào >= 0. Bước này CHỈ dùng cho nhánh Chi2,
    # fit riêng trên mỗi train-fold, không thay đổi baseline/method khác.
    if chi_square:
        steps.append(("chi2_nonnegative", MinMaxScaler()))
    if selector is not None:
        steps.append(("feature_selection", selector))
    steps.append(("model", new_model()))
    return Pipeline(steps)

# 3. Baseline: không Feature Selection
print("\n=== BƯỚC 1: BASELINE KHÔNG FEATURE SELECTION ===")
base = make_pipeline()
base_cv = cross_val_score(base, X_train, y_train, cv=CV, scoring="f1_macro", n_jobs=1)
print(f"Baseline CV F1 Macro: {base_cv.mean():.4f} ± {base_cv.std():.4f}")
base.fit(X_train, y_train)
n_features = len(base.named_steps["preprocessor"].get_feature_names_out())
print("Số đặc trưng sau Encoding:", n_features)
ks = sorted(set(k for k in K_VALUES if k <= n_features))
if not ks:
    ks = [n_features]

# 4. Filter và Wrapper / Embedded (tất cả nằm trong CV pipeline)
print("\n=== BƯỚC 2–5: THỬ CÁC PHƯƠNG PHÁP FEATURE SELECTION ===")
methods = {
    "ANOVA_F": lambda k: (SelectKBest(score_func=f_classif, k=k), False),
    "Mutual_Information": lambda k: (SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=SEED), k=k), False),
    "Chi2": lambda k: (SelectKBest(score_func=chi2, k=k), True),
    "RFE_LR": lambda k: (RFE(estimator=new_model(), n_features_to_select=k, step=0.2), False),
}
results = [{"Method": "Baseline", "K": n_features,
            "F1_CV_Mean": base_cv.mean(), "F1_CV_Std": base_cv.std()}]
for name, factory in methods.items():
    # RFE tương đối chậm: chỉ khảo sát 2 kích thước để chạy được trên máy cá nhân.
    method_ks = ks if name != "RFE_LR" else sorted(set([ks[0], ks[-1]]))
    for k in method_ks:
        selector, nonnegative = factory(k)
        pipe = make_pipeline(selector, chi_square=nonnegative)
        scores = cross_val_score(pipe, X_train, y_train, cv=CV,
                                 scoring="f1_macro", n_jobs=1)
        results.append({"Method": name, "K": k,
                        "F1_CV_Mean": scores.mean(), "F1_CV_Std": scores.std()})
        print(f"{name:20s} k={k:3d} | F1 Macro={scores.mean():.4f} ± {scores.std():.4f}")

result_df = pd.DataFrame(results).sort_values("F1_CV_Mean", ascending=False)
result_df.to_csv(f"{OUTPUT_PREFIX}_cv_results.csv", index=False, encoding="utf-8-sig")
print("\n=== BƯỚC 6: BẢNG XẾP HẠNG THEO 5-FOLD CV ===")
print(result_df.to_string(index=False))

# 5. Đồ thị F1 Macro theo số đặc trưng
fig, ax = plt.subplots(figsize=(9, 5))
for method, group in result_df[result_df.Method != "Baseline"].groupby("Method"):
    group = group.sort_values("K")
    ax.plot(group["K"], group["F1_CV_Mean"], marker="o", label=method)
ax.axhline(base_cv.mean(), linestyle="--", color="gray", label="Baseline")
ax.set(xlabel="Số đặc trưng được chọn (k)", ylabel="F1 Macro (5-fold CV)",
       title="Feature Selection - Logistic Regression + Mixed Scaling")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# 6. Chọn mô hình tốt nhất CHỈ theo CV trên Train, đánh giá 1 lần trên Validation
best = result_df.iloc[0]
method, k = best["Method"], int(best["K"])
if method == "Baseline":
    final_pipe = make_pipeline()
else:
    selector, nonnegative = methods[method](k)
    final_pipe = make_pipeline(selector, chi_square=nonnegative)
final_pipe.fit(X_train, y_train)
y_pred = final_pipe.predict(X_val)
y_prob = final_pipe.predict_proba(X_val)
print("\n=== BƯỚC 7: ĐÁNH GIÁ VALIDATION (CHỈ MÔ HÌNH TỐT NHẤT) ===")
print("Method:", method, "| k:", k)
print(f"Accuracy       : {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision Macro: {precision_score(y_val, y_pred, average='macro', zero_division=0):.4f}")
print(f"Recall Macro   : {recall_score(y_val, y_pred, average='macro', zero_division=0):.4f}")
print(f"F1 Macro       : {f1_score(y_val, y_pred, average='macro', zero_division=0):.4f}")
print(f"Log Loss       : {log_loss(y_val, y_prob, labels=np.arange(len(label_encoder.classes_))):.4f}")

names = final_pipe.named_steps["preprocessor"].get_feature_names_out()
if method != "Baseline":
    names = names[final_pipe.named_steps["feature_selection"].get_support()]
pd.DataFrame({"Selected_Feature": names}).to_csv(
    f"{OUTPUT_PREFIX}_selected_features.csv", index=False, encoding="utf-8-sig")
print("\nCác đặc trưng được chọn:", list(names))

cm = confusion_matrix(y_val, y_pred)
ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {method}")
plt.tight_layout(); plt.show()

n_classes = len(label_encoder.classes_)
y_bin = label_binarize(y_val, classes=np.arange(n_classes))
fig, ax = plt.subplots(figsize=(7, 5))
for i, cls in enumerate(label_encoder.classes_):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    ax.plot(fpr, tpr, label=f"{cls} AUC={auc(fpr, tpr):.3f}")
ax.plot([0, 1], [0, 1], "k--")
ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC OvR")
ax.legend(); plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(7, 5))
for i, cls in enumerate(label_encoder.classes_):
    p, r, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
    ap = average_precision_score(y_bin[:, i], y_prob[:, i])
    ax.plot(r, p, label=f"{cls} AP={ap:.3f}")
ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall OvR")
ax.legend(); plt.tight_layout(); plt.show()

sizes, train_sc, val_sc = learning_curve(
    final_pipe, X_train, y_train, cv=3, scoring="f1_macro",
    train_sizes=np.linspace(0.2, 1.0, 4), n_jobs=1)
fig, ax = plt.subplots(figsize=(7, 5))
for scores, label in [(train_sc, "Train"), (val_sc, "CV")]:
    mean, std = scores.mean(axis=1), scores.std(axis=1)
    ax.plot(sizes, mean, marker="o", label=label)
    ax.fill_between(sizes, mean-std, mean+std, alpha=0.15)
ax.set(xlabel="Training size", ylabel="F1 Macro", title=f"Learning Curve - {method}")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
