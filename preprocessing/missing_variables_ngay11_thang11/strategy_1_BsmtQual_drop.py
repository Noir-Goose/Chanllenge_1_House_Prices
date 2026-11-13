import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, log_loss
)


SEEDS = [42]

# 1. Đọc dữ liệu House Prices; phân nhóm giá để dùng 4 mô hình phân loại
# Nhóm: 0 = dưới 150000; 1 = 150000 đến dưới 250000; 2 = từ 250000
# Các ngưỡng cố định, không học từ validation.
df = pd.read_csv("../data/raw/train.csv", low_memory=False)
df = df.replace("?", np.nan)
df = df.dropna(subset=["SalePrice"]).copy()

y_all = pd.cut(df["SalePrice"], bins=[-np.inf, 150000, 250000, np.inf],
               labels=[0, 1, 2], right=False).astype(int).to_numpy()
X_all = df.drop(columns=["SalePrice", "Id"], errors="ignore").copy()

categorical_id_cols = []

print("CHIẾN LƯỢC 1: XÓA CỘT BsmtQual CÓ MISSING")

def run_once(seed):
    # 2. Chia dữ liệu: train 80%, validation 20%
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
    )
    X_train, X_val = X_train.copy(), X_val.copy()

    # 3. BsmtQual thiếu gần 100%, xóa dòng sẽ không đủ mẫu để huấn luyện.
    # Vì vậy chiến thuật xóa được áp dụng cho CỘT BsmtQual.
    X_train = X_train.drop(columns=["BsmtQual"])
    X_val = X_val.drop(columns=["BsmtQual"])

    # 4. Xác định cột phân loại và cột số
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

    # 5. Tiền xử lý
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, categorical_cols),
        ("numeric", numeric_pipeline, numeric_cols)
    ])

    # 6. Khai báo mô hình
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, random_state=seed
        )
    }

    # 7. Huấn luyện và đánh giá trên validation
    results = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_val)
        y_proba = pipeline.predict_proba(X_val)

        results[name] = {
            "Accuracy       ": accuracy_score(y_val, y_pred),
            "Precision Macro": precision_score(y_val, y_pred, average="macro", zero_division=0),
            "Recall Macro   ": recall_score(y_val, y_pred, average="macro", zero_division=0),
            "F1 Macro       ": f1_score(y_val, y_pred, average="macro", zero_division=0),
            "Log Loss       ": log_loss(y_val, y_proba, labels=np.arange(3)),
        }
    return results, len(X_train), len(X_val)


all_results = []
for s in SEEDS:
    res, n_tr, n_va = run_once(s)
    all_results.append(res)

print("Số dòng train còn lại:", n_tr)
print("Số dòng validation còn lại:", n_va)

for name in all_results[0]:
    print(f"\n{name}")
    for metric in all_results[0][name]:
        vals = np.array([r[name][metric] for r in all_results])
        if len(vals) == 1:
            print(f"{metric}: {vals[0]:.4f}")
        else:
            print(f"{metric}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}   (n={len(vals)} seed)")
