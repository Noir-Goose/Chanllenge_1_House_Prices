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

# 1. Đọc dữ liệu
# Dữ liệu House Prices: train.csv có nhãn SalePrice, test.csv không có nhãn.
df = pd.read_csv("../data/raw/train.csv", low_memory=False)

# Phân loại giá nhà thành ba nhóm, cố định ngưỡng giữa các chiến lược.
def price_group(price):
    if price < 150000:
        return "Low"
    if price < 250000:
        return "Medium"
    return "High"

label_encoder = LabelEncoder()
y_all = label_encoder.fit_transform(df["SalePrice"].map(price_group))
X_all = df.drop(columns=["SalePrice", "Id"], errors="ignore").copy()

# MSSubClass là mã loại nhà, không phải đại lượng liên tục.
categorical_id_cols = ["MSSubClass"]

print("Z-SCORE CAPPING: TotalBsmtSF")

def run_once(seed):
    # 2. Chia dữ liệu: train 80%, validation 20% (cùng seed cho hai chiến lược)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all
    )
    X_train, X_val = X_train.copy(), X_val.copy()

    # 3. Z-score Capping: giới hạn outlier của biến TotalBsmtSF (mean/std tính trên tập train)
    outlier_col = "TotalBsmtSF"
    X_train[outlier_col] = pd.to_numeric(X_train[outlier_col], errors="coerce")
    X_val[outlier_col] = pd.to_numeric(X_val[outlier_col], errors="coerce")

    mean = X_train[outlier_col].mean()
    std = X_train[outlier_col].std()

    print(f"[seed={seed}] Biến được xử lý: {outlier_col}")
    print(f"Trung bình: {mean}  Độ lệch chuẩn: {std}")

    if pd.notna(std) and std > 0:
        lower = mean - 3 * std
        upper = mean + 3 * std
        pct_clipped = ((X_train[outlier_col] < lower) | (X_train[outlier_col] > upper)).mean() * 100

        print(f"Ngưỡng dưới: {lower}  Ngưỡng trên: {upper}")
        print(f"Tỷ lệ dòng train bị cắt: {pct_clipped:.2f}%")
        if pct_clipped == 0:
            print("LƯU Ý: không có giá trị nào bị cắt, kết quả sẽ y hệt baseline.")

        # Giới hạn giá trị ngoại lệ trên train và validation
        X_train[outlier_col] = X_train[outlier_col].clip(lower=lower, upper=upper)
        X_val[outlier_col] = X_val[outlier_col].clip(lower=lower, upper=upper)
    else:
        print("Không xử lý: độ lệch chuẩn không hợp lệ hoặc bằng 0.")

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
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=seed,
            n_jobs=-1, class_weight="balanced"
        ),
        
        "Tree": DecisionTreeClassifier(random_state=seed, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
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
            "Log Loss       ": log_loss(y_val, y_proba, labels=np.arange(len(label_encoder.classes_))),
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
