# Lưu ý: kết quả outlier từ Bang_so_sanh_outlier_da_sua.docx chưa có IQR hợp lệ
# cho LotFrontage. "none" = không capping; lựa chọn theo F1 Macro, rồi Recall Macro.
# Cần thực nghiệm lại tổ hợp missing + outlier để xác nhận hiệu quả.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize, MinMaxScaler, StandardScaler, OrdinalEncoder
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


SEEDS = [42]

# 1. Đọc dữ liệu House Prices
# Nhãn phân loại: Low (<150000), Medium (150000-249999), High (>=250000)
df = pd.read_csv("../data/raw/train.csv", low_memory=False)
df["PriceCategory"] = pd.cut(
    df["SalePrice"], bins=[-np.inf, 150000, 250000, np.inf],
    labels=["Low", "Medium", "High"], right=False
)
label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(df["PriceCategory"])
X_all = df.drop(columns=["SalePrice", "PriceCategory", "Id"], errors="ignore").copy()
X_all["MSSubClass"] = X_all["MSSubClass"].astype("object")
categorical_id_cols = ["MSSubClass"]

print("HOUSE PRICES: BEST MISSING + OUTLIER + MIXED SCALING + ORDINAL MAPPING + 4 BIỂU ĐỒ")

def run_lr_pipeline(seed):
    # 2. Chia dữ liệu: train 80%, validation 20% (giống House Prices trước đó)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
    )

    X_train, X_val = X_train.copy(), X_val.copy()
    
    y_train = pd.Series(y_train, index=X_train.index)
    y_val = pd.Series(y_val, index=X_val.index)

    # ==========================================
    # 3. MISSING: GIỮ CÁC CHIẾN LƯỢC ĐÃ CHỌN
    # ==========================================
    drop_feature_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'LotFrontage', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'MasVnrArea']
    mode_cols = ['BsmtQual', 'BsmtCond']
    category_missing_cols = ['FireplaceQu', 'GarageCond']

    missing_transformer = BestMissingTransformer(
        drop_cols=drop_feature_cols,
        mode_cols=mode_cols,
        missing_category_cols=category_missing_cols
    )
    # Chỉ dùng để xác định tên cột (không học mode từ validation).
    kept_cols = X_train.drop(columns=drop_feature_cols, errors="ignore")
    for col in categorical_id_cols:
        if col in kept_cols:
            kept_cols[col] = kept_cols[col].astype("object")

    # ==========================================
    # 4. PIPELINE TIỀN XỬ LÝ + MÔ HÌNH
    # ==========================================
    ordinal_cols = [c for c in ORDINAL_LEVELS if c in kept_cols.columns]
    categorical_cols = [c for c in kept_cols.select_dtypes(include=["object", "category", "string"]).columns
                        if c not in ordinal_cols]
    numeric_cols = kept_cols.select_dtypes(include=["number"]).columns.tolist()

    # Đưa việc chuẩn hóa kiểu dữ liệu categorical vào Pipeline để dùng đúng trong CV.
    categorical_pipeline = Pipeline([
        ("to_string", CategoricalToString()),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Ordinal Mapping trước, MinMaxScaler sau; MinMax được chọn để đồng bộ thang đo,
    # KHÔNG PHẢI vì ordinal có phân phối Gaussian.
    ordinal_pipeline = Pipeline([
        ("to_string", CategoricalToString()),
        ("encoder", OrdinalEncoder(
            categories=[["Missing"] + ORDINAL_LEVELS[c] for c in ordinal_cols],
            handle_unknown="use_encoded_value", unknown_value=-1
        )),
        ("scaler", MinMaxScaler())
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", SkewnessColumnScaler(
            columns=numeric_cols,
            forced_minmax=FORCE_MINMAX_COLS,
            threshold=SKEW_THRESHOLD
        ))
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_cols),
        ("ordinal", ordinal_pipeline, ordinal_cols),
        ("numeric", numeric_pipeline, numeric_cols)
    ])

    model_name = "Logistic Regression (Best Missing + Outlier + Skew Mixed Scaling + Ordinal)"
    pipeline = Pipeline([
        ("missing", missing_transformer),
        ("outlier", OutlierCapper(strategies=OUTLIER_STRATEGIES)),
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced"))
    ])

    # ==========================================
    # 5. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH
    # ==========================================
    print(f"\nĐang huấn luyện mô hình {model_name}...")
    pipeline.fit(X_train, y_train)
    # Báo cáo scaler được chọn từ TRAIN sau Missing + Outlier + Median Imputation.
    fitted_scaler = pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps["scaler"]
    report = pd.DataFrame([
        {"Feature": col, "Skewness_train": fitted_scaler.skewness_[col],
         "Scaler": fitted_scaler.method_[col]}
        for col in numeric_cols
    ]).sort_values("Skewness_train", key=lambda s: s.abs(), ascending=False)
    print("\n--- SKEWNESS VÀ SCALER THEO TỪNG BIẾN SỐ (TRAIN) ---")
    print(report.to_string(index=False))
    print("\nOrdinal columns:", ordinal_cols)
    print("Nominal columns:", categorical_cols)
    report.to_csv("LR/report/LR_skewness_scaler_report.csv", index=False, encoding="utf-8-sig")


    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    loss = log_loss(y_val, y_proba, labels=np.arange(len(label_encoder.classes_)))

    print(f"\n--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP VALIDATION ---")
    from sklearn.metrics import precision_score, recall_score
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision Macro: {precision_score(y_val, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall Macro   : {recall_score(y_val, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 Macro       : {f1:.4f}")
    print(f"Log Loss       : {loss:.4f}")

    # ==========================================
    # 6. TRỰC QUAN HÓA 4 BIỂU ĐỒ ĐẦY ĐỦ
    # ==========================================
    
    # === Biểu đồ 1: Confusion Matrix ===
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.show()

    # === Biểu đồ 2: ROC Curve ===
    n_classes = len(label_encoder.classes_)
    y_val_bin = label_binarize(y_val, classes=np.arange(n_classes))
    
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_val_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc:.2f})')
    
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # === Biểu đồ 3: Precision-Recall Curve ===
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_val_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_val_bin[:, i], y_proba[:, i])
        ax.plot(recall, precision, label=f'Class {label_encoder.classes_[i]} (AP = {ap:.2f})')
        
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name}')
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

    # === Biểu đồ 4: Learning Curve ===
    print(f"Đang tính toán Learning Curve...")
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X_train, y_train, cv=3, scoring='f1_macro',
        train_sizes=np.linspace(0.2, 1.0, 4), n_jobs=-1, random_state=seed
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.15)
    
    ax.plot(train_sizes, val_mean, label='Cross-validation score', color='orange', marker='o')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.15)
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('F1 Macro Score')
    ax.set_title(f'Learning Curve - {model_name}')
    ax.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return len(X_train), len(X_val)

# Chạy chương trình
n_tr, n_va = run_lr_pipeline(SEEDS[0])
print(f"\nSố dòng train còn lại sau khi xử lý: {n_tr}")
print(f"Số dòng validation còn lại sau khi xử lý: {n_va}")