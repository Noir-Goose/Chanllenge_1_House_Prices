# Lưu ý: kết quả outlier từ Bang_so_sanh_outlier_da_sua.docx chưa có IQR hợp lệ
# cho LotFrontage. "none" = không capping; lựa chọn theo F1 Macro, rồi Recall Macro.
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
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.base import clone
from sklearn.metrics import precision_score, recall_score

import warnings
from pathlib import Path
import time

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

def make_pipeline(selector=None):
    steps = list(base_steps)
    if selector is not None:
        steps.append(("feature_selection", selector))
    steps.append(("model", new_model()))
    return Pipeline(steps)


# Wrapper-only experiment. Feature selection fitted INSIDE CV folds.
METHOD = "Forward"
OUTPUT_DIR = Path(f"LR_Wrapper_{METHOD}_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SFS is computationally intensive: default 1 k for each SFS direction.
# Edit K_VALUES_SFS to test more k values; full K_VALUES retained for RFE.
K_VALUES_SFS = [10] if METHOD == "Forward" else [190]
SFS_INNER_CV = 3


def selector_for(k):
    if METHOD == "RFE":
        return RFE(estimator=new_model(), n_features_to_select=k, step=0.2)
    return SequentialFeatureSelector(
        estimator=new_model(), n_features_to_select=k,
        direction="forward" if METHOD == "Forward" else "backward",
        scoring="f1_macro", cv=SFS_INNER_CV, n_jobs=1,
    )

print(f"\n=== {METHOD}: BASELINE ===")
baseline = make_pipeline()
t0 = time.perf_counter()
base_scores = cross_val_score(baseline, X_train, y_train, cv=CV, scoring="f1_macro", n_jobs=1)
base_seconds = time.perf_counter() - t0
baseline.fit(X_train, y_train)
feature_names = baseline.named_steps["preprocessor"].get_feature_names_out()
n_features = len(feature_names)
print(f"Number of encoded features: {n_features}")
print(f"Baseline: {base_scores.mean():.4f} ± {base_scores.std():.4f}")

candidate_ks = K_VALUES if METHOD == "RFE" else K_VALUES_SFS
candidate_ks = sorted(set(int(k) for k in candidate_ks if 0 < k < n_features))
if not candidate_ks:
    raise ValueError(f"No valid k values for {METHOD}; number of features = {n_features}")

records = [{"Method":"Baseline", "K":n_features,
            "F1_CV_Mean":base_scores.mean(), "F1_CV_Std":base_scores.std(),
            "Seconds":base_seconds}]
print(f"\n=== {METHOD}: 5-fold CV ===")
if METHOD != "RFE":
    print("SFS uses nested feature-selection CV and can be VERY slow.")
for k in candidate_ks:
    t0 = time.perf_counter()
    scores = cross_val_score(make_pipeline(selector_for(k)), X_train, y_train,
                             cv=CV, scoring="f1_macro", n_jobs=1)
    seconds = time.perf_counter()-t0
    records.append({"Method":METHOD,"K":k,"F1_CV_Mean":scores.mean(),
                    "F1_CV_Std":scores.std(), "Seconds":seconds})
    print(f"{METHOD} k={k}: F1={scores.mean():.4f} ± {scores.std():.4f}; {seconds:.1f}s")

result_df = pd.DataFrame(records).sort_values("F1_CV_Mean", ascending=False)
result_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False, encoding="utf-8-sig")
print(result_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9,5))
fs_rows = result_df[result_df.Method != "Baseline"].sort_values("K")
ax.plot(fs_rows.K, fs_rows.F1_CV_Mean, marker="o", label=METHOD)
ax.axhline(base_scores.mean(), linestyle="--", color="gray", label="Baseline")
ax.set(xlabel="Selected features (k)",ylabel="5-fold CV F1 Macro",title=f"{METHOD} vs Baseline")
ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_cv_f1_comparison.png", dpi=180, bbox_inches="tight")
plt.show()

# Select configuration using training CV only, then validate once.
best = result_df.iloc[0]
method = str(best.Method)
k = int(best.K)
final_pipe = make_pipeline() if method == "Baseline" else make_pipeline(selector_for(k))
final_pipe.fit(X_train, y_train)
y_pred = final_pipe.predict(X_val)
y_prob = final_pipe.predict_proba(X_val)
metrics = {
    "Method":method,"K":k,
    "Accuracy":accuracy_score(y_val,y_pred),
    "Precision_Macro":precision_score(y_val,y_pred,average="macro",zero_division=0),
    "Recall_Macro":recall_score(y_val,y_pred,average="macro",zero_division=0),
    "F1_Macro":f1_score(y_val,y_pred,average="macro",zero_division=0),
    "Log_Loss":log_loss(y_val,y_prob,labels=np.arange(len(label_encoder.classes_)))
}
pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "validation_metrics.csv",index=False,encoding="utf-8-sig")
print("\nValidation:",metrics)

names = final_pipe.named_steps["preprocessor"].get_feature_names_out()
if method != "Baseline":
    names = names[final_pipe.named_steps["feature_selection"].get_support()]
pd.DataFrame({"Selected_Feature":names}).to_csv(
    OUTPUT_DIR / "selected_features.csv",index=False,encoding="utf-8-sig")
print("Selected features:",list(names))

cm = confusion_matrix(y_val,y_pred)
fig,ax=plt.subplots(figsize=(7,5))
ConfusionMatrixDisplay(cm,display_labels=label_encoder.classes_).plot(ax=ax,cmap="Blues")
ax.set_title(f"Confusion Matrix - {method}")
fig.tight_layout();fig.savefig(OUTPUT_DIR / "02_confusion_matrix.png",dpi=180,bbox_inches="tight");plt.show()

n_classes=len(label_encoder.classes_)
y_bin=label_binarize(y_val,classes=np.arange(n_classes))
fig,ax=plt.subplots(figsize=(7,5))
for i,cls in enumerate(label_encoder.classes_):
    fpr,tpr,_=roc_curve(y_bin[:,i],y_prob[:,i])
    ax.plot(fpr,tpr,label=f"{cls} AUC={auc(fpr,tpr):.3f}")
ax.plot([0,1],[0,1],"k--")
ax.set(xlabel="False Positive Rate",ylabel="True Positive Rate",title=f"ROC OvR - {method}")
ax.legend();fig.tight_layout();fig.savefig(OUTPUT_DIR / "03_roc_ovr.png",dpi=180,bbox_inches="tight");plt.show()

fig,ax=plt.subplots(figsize=(7,5))
for i,cls in enumerate(label_encoder.classes_):
    p,r,_=precision_recall_curve(y_bin[:,i],y_prob[:,i])
    ap=average_precision_score(y_bin[:,i],y_prob[:,i])
    ax.plot(r,p,label=f"{cls} AP={ap:.3f}")
ax.set(xlabel="Recall",ylabel="Precision",title=f"Precision-Recall OvR - {method}")
ax.legend();fig.tight_layout();fig.savefig(OUTPUT_DIR / "04_precision_recall.png",dpi=180,bbox_inches="tight");plt.show()

sizes,train_sc,val_sc=learning_curve(
    final_pipe,X_train,y_train,cv=3,scoring="f1_macro",
    train_sizes=np.linspace(.2,1.,4),n_jobs=1)
fig,ax=plt.subplots(figsize=(7,5))
for scores,label in [(train_sc,"Train"),(val_sc,"CV")]:
    mean,std=scores.mean(axis=1),scores.std(axis=1)
    ax.plot(sizes,mean,marker="o",label=label)
    ax.fill_between(sizes,mean-std,mean+std,alpha=.15)
ax.set(xlabel="Training size",ylabel="F1 Macro",title=f"Learning Curve - {method}")
ax.legend();ax.grid(alpha=.3);fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_learning_curve.png",dpi=180,bbox_inches="tight");plt.show()
print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")
