# -*- coding: utf-8 -*-

# Chuyển trực tiếp từ notebook LR_RandomSearch_ANOVA_K100_StandardScaler.ipynb


# %% [markdown]
# # Randomized Search – Logistic Regression + StandardScaler + ANOVA F (K=100)
#
# Notebook từng bước, giữ nguyên hai lớp xử lý Missing/Outlier từ `LR_Filter_StandardScaler(6).py`. Chọn siêu tham số bằng 5-Fold CV trên Train; Validation chỉ đánh giá sau tuning. Chạy lần lượt từ trên xuống.


# %% [markdown]
# ## 1. Import thư viện và cấu hình


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import json
from pathlib import Path
from time import perf_counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, log_loss,
    classification_report, ConfusionMatrixDisplay, roc_curve, auc)

warnings.filterwarnings('ignore', category=FutureWarning, message=".*'penalty' was deprecated.*")
warnings.filterwarnings('ignore', message='Features .* are constant', category=UserWarning)
warnings.filterwarnings('ignore', message='invalid value encountered in divide', category=RuntimeWarning)
SEED = 42
OUTPUT_DIR = Path('LR_RandomSearch_ANOVA_K100_Results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% [markdown]
# ## 2. Đọc dữ liệu, tạo ba lớp giá nhà


# %%
raw = pd.read_csv('../data/raw/train.csv', low_memory=False)
raw['PriceCategory'] = pd.cut(raw['SalePrice'], bins=[-np.inf, 150000, 250000, np.inf],
                              labels=['Low', 'Medium', 'High'], right=False)
label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(raw['PriceCategory'])
X_all = raw.drop(columns=['SalePrice', 'PriceCategory', 'Id'], errors='ignore').copy()
X_all['MSSubClass'] = X_all['MSSubClass'].astype('object')
print('Dữ liệu:', X_all.shape)
print('Nhãn:', list(label_encoder.classes_))
print(raw['PriceCategory'].value_counts())


# %% [markdown]
# ## 3. Chia Train (80%) và Validation (20%)


# %%
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=SEED)
print('Train:', X_train.shape, 'Validation:', X_val.shape)


# %% [markdown]
# ## 4. Chiến lược Missing Values và lớp biến đổi từ file gốc


# %%
DROP_COLS = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'LotFrontage',
             'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure',
             'BsmtFinType2', 'BsmtFinType1', 'MasVnrArea']
MODE_COLS = ['BsmtQual', 'BsmtCond']
MISSING_CAT_COLS = ['FireplaceQu', 'GarageCond']


# %%
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


# %% [markdown]
# ## 5. Chiến lược Outlier và lớp biến đổi từ file gốc


# %%
OUTLIER_STRATEGIES = {
    'LotArea': 'zscore', 'GrLivArea': 'zscore', 'TotalBsmtSF': 'iqr',
    'LotFrontage': 'iqr', 'GarageArea': 'iqr', 'MasVnrArea': 'iqr'
}


# %%
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


# %% [markdown]
# ## 6. One-Hot Encoding và StandardScaler


# %%
kept_cols = X_train.drop(columns=DROP_COLS, errors='ignore')
categorical_cols = kept_cols.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
numeric_cols = kept_cols.select_dtypes(include=['number']).columns.tolist()

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer([
    ('categorical', categorical_pipeline, categorical_cols),
    ('numeric', numeric_pipeline, numeric_cols)
])
print('Cột phân loại:', len(categorical_cols), 'Cột số:', len(numeric_cols))


# %% [markdown]
# ## 7. Pipeline hoàn chỉnh: Missing → Outlier → Encoding/Scaling → ANOVA F (100) → Logistic Regression


# %%
pipeline = Pipeline([
    ('missing', BestMissingTransformer(DROP_COLS, MODE_COLS, MISSING_CAT_COLS)),
    ('outlier', OutlierCapper(strategies=OUTLIER_STRATEGIES)),
    ('preprocessor', preprocessor),
    ('feature_selection', SelectKBest(score_func=f_classif, k=100)),
    ('model', LogisticRegression(max_iter=2000, random_state=SEED))
])
print(pipeline)


# %% [markdown]
# ## 8. Không gian tìm kiếm siêu tham số
#
# Giữ L2/`lbfgs` như mặc định file gốc. Tìm `C`, `class_weight`, `tol`, `fit_intercept`.


# %%
param_distributions = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'model__class_weight': [None, 'balanced'],
    'model__tol': [1e-3, 1e-4, 1e-5],
    'model__fit_intercept': [True, False]
}
print('Tổng tổ hợp có thể:', 6*2*3*2)


# %% [markdown]
# ## 9. Stratified 5-Fold CV


# %%
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# %% [markdown]
# ## 10. Khởi tạo RandomizedSearchCV


# %%
random_search = RandomizedSearchCV(
    estimator=pipeline, param_distributions=param_distributions,
    n_iter=30, scoring='f1_macro', cv=CV,
    random_state=SEED, n_jobs=-1, verbose=1,
    return_train_score=True, refit=True
)


# %% [markdown]
# ## 11. Chạy tìm kiếm


# %%
start = perf_counter()
random_search.fit(X_train, y_train)
elapsed = perf_counter() - start
print(f'Thời gian: {elapsed:.2f} giây')
print(f'Best CV F1 Macro: {random_search.best_score_:.4f}')
print('Best Parameters:', random_search.best_params_)


# %% [markdown]
# ## 12. Bảng kết quả các tổ hợp


# %%
results_df = pd.DataFrame(random_search.cv_results_)
columns = ['param_model__C', 'param_model__class_weight', 'param_model__tol',
           'param_model__fit_intercept', 'mean_train_score', 'mean_test_score',
           'std_test_score', 'rank_test_score', 'mean_fit_time']
results_df = results_df[columns].sort_values('rank_test_score')
print(results_df.to_string(index=False))


# %% [markdown]
# ## 13. Đánh giá cấu hình tốt nhất trên Validation


# %%
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_val)
y_prob = best_model.predict_proba(X_val)
metrics = {
    'Accuracy': accuracy_score(y_val, y_pred),
    'Precision Macro': precision_score(y_val, y_pred, average='macro', zero_division=0),
    'Recall Macro': recall_score(y_val, y_pred, average='macro', zero_division=0),
    'F1 Macro': f1_score(y_val, y_pred, average='macro', zero_division=0),
    'Log Loss': log_loss(y_val, y_prob, labels=best_model.classes_)
}
for name, value in metrics.items():
    print(f'{name}: {value:.4f}')
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))


# %% [markdown]
# ## 14. Biểu đồ Train/CV F1 Error
#
# Đây là `1 - F1 Macro`, không phải Classification Error (`1 - Accuracy`).


# %%
plot_df = results_df.sort_values('mean_test_score').reset_index(drop=True)
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(1 - plot_df['mean_train_score'].to_numpy(), marker='o', label='Train F1 Error')
ax.plot(1 - plot_df['mean_test_score'].to_numpy(), marker='o', label='CV F1 Error')
ax.set(xlabel='Tổ hợp tham số (xếp theo CV F1)', ylabel='1 - F1 Macro',
       title='Random Search - Train vs CV F1 Error')
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / '01_train_cv_error.png', dpi=180)
plt.show()


# %% [markdown]
# ## 15. Confusion Matrix


# %%
fig, ax = plt.subplots(figsize=(7, 5))
ConfusionMatrixDisplay.from_predictions(
    y_val, y_pred, display_labels=label_encoder.classes_,
    cmap='Blues', values_format='d', ax=ax)
ax.set_title('Confusion Matrix - Best Random Search')
fig.tight_layout(); fig.savefig(OUTPUT_DIR / '02_confusion_matrix.png', dpi=180)
plt.show()


# %% [markdown]
# ## 16. ROC-AUC One-vs-Rest


# %%
classes = best_model.classes_
y_bin = label_binarize(y_val, classes=classes)
fig, ax = plt.subplots(figsize=(8, 6))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
    name = label_encoder.inverse_transform([cls])[0]
    ax.plot(fpr, tpr, label=f'{name} AUC={auc(fpr, tpr):.3f}')
ax.plot([0, 1], [0, 1], 'k--')
ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC-AUC OvR')
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUTPUT_DIR / '03_roc_auc.png', dpi=180)
plt.show()


# %% [markdown]
# ## 17. Lưu CSV, JSON và danh sách 100 đặc trưng


# %%
results_df.to_csv(OUTPUT_DIR / 'random_search_results.csv', index=False, encoding='utf-8-sig')
pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / 'validation_metrics.csv', index=False, encoding='utf-8-sig')
with open(OUTPUT_DIR / 'best_params.json', 'w', encoding='utf-8') as f:
    json.dump(random_search.best_params_, f, indent=4, ensure_ascii=False)

names = best_model.named_steps['preprocessor'].get_feature_names_out()
mask = best_model.named_steps['feature_selection'].get_support()
pd.DataFrame({'Selected_Feature': names[mask]}).to_csv(
    OUTPUT_DIR / 'selected_features_ANOVA_k100.csv', index=False, encoding='utf-8-sig')
print('Đã lưu tại:', OUTPUT_DIR.resolve())


# %% [markdown]
# **Lưu ý:** Chưa có kết quả huấn luyện sẵn trong notebook này. Chạy các cell theo thứ tự với `../data/raw/train.csv` để tạo kết quả. Chỉ chọn siêu tham số theo CV trên Train, không chọn theo Validation.
