import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

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

print("ĐÁNH GIÁ MÔ HÌNH HOUSE PRICES VỚI CHIẾN LƯỢC XỬ LÝ MISSING + 4 BIỂU ĐỒ")

def run_lr_pipeline(seed):
    # 2. Chia dữ liệu: train 80%, validation 20% (giống House Prices trước đó)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
    )

    X_train, X_val = X_train.copy(), X_val.copy()
    
    y_train = pd.Series(y_train, index=X_train.index)
    y_val = pd.Series(y_val, index=X_val.index)

    # ==========================================
    # 3. XỬ LÝ MISSING VALUE ĐÚNG THEO YÊU CẦU
    # ==========================================

    # Chọn chiến lược tốt nhất riêng cho mô hình theo missing(1).docx.
    # Tiêu chí: F1 Macro cao nhất, sau đó Recall Macro cao nhất.
    # Các kết quả là thí nghiệm đơn biến, cần đánh giá lại khi kết hợp.
    drop_feature_cols = ['Fence', 'BsmtExposure']
    mode_cols = ['PoolQC', 'MiscFeature', 'GarageFinish', 'GarageCond', 'BsmtFinType1']
    category_missing_cols = ['Alley', 'MasVnrType', 'FireplaceQu', 'LotFrontage', 'GarageType', 'GarageYrBlt', 'GarageQual', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'MasVnrArea']

    # 1. Xóa cột (không xóa dòng dữ liệu)
    X_train = X_train.drop(columns=drop_feature_cols, errors="ignore")
    X_val = X_val.drop(columns=drop_feature_cols, errors="ignore")

    # 2. Điền mode từ tập train để tránh rò rỉ dữ liệu
    for col in mode_cols:
        if col in X_train.columns:
            train_mode = X_train[col].mode(dropna=True)
            if not train_mode.empty:
                X_train[col] = X_train[col].fillna(train_mode.iloc[0])
                X_val[col] = X_val[col].fillna(train_mode.iloc[0])

    # 3. Missing là một nhóm riêng, kể cả biến số như LotFrontage/MasVnrArea
    for col in category_missing_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("object").fillna("Missing")
            X_val[col] = X_val[col].astype("object").fillna("Missing")

    # ==========================================
    # 4. XÂY DỰNG PIPELINE TIỀN XỬ LÝ & MÔ HÌNH
    # ==========================================
    for col in categorical_id_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype("object")
            X_val[col] = X_val[col].astype("object")

    categorical_cols = X_train.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric_cols = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # OneHotEncoder yêu cầu mỗi cột phân loại có kiểu dữ liệu thống nhất.
    # Đặc biệt các cột số được điền "Missing" có thể trộn float và str.
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("string").fillna("Missing")
        X_val[col] = X_val[col].astype("string").fillna("Missing")

    # Pipeline phân loại: điền NaN còn sót và OneHotEncoder
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Pipeline biến số: điền NaN còn sót, không scale (giữ cấu trúc gốc)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", "passthrough")
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_cols),
        ("numeric", numeric_pipeline, numeric_cols)
    ])

    model_name = "Random Forest (Best Missing per Feature)"
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
             n_estimators=200, random_state=seed,n_jobs=-1, class_weight="balanced"
        ))
    ])

    # ==========================================
    # 5. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH
    # ==========================================
    print(f"\nĐang huấn luyện mô hình {model_name}...")
    pipeline.fit(X_train, y_train)

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
        pipeline, X_train, y_train, cv=3, scoring='accuracy',
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
    ax.set_ylabel('Accuracy Score')
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