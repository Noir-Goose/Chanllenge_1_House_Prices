from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Dùng chung đúng pipeline XGBoost của baseline để kết quả so sánh được.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline import RANDOM_STATE, TRAIN_PATH, make_pipeline


def main() -> None:
    data = pd.read_csv(TRAIN_PATH)
    y = np.log1p(data["SalePrice"])
    X = data.drop(columns=["SalePrice", "Id"], errors="ignore").copy()

    # Với PoolQC, missing mang nghĩa căn nhà không có hồ bơi.
    X["PoolQC"] = X["PoolQC"].fillna("None")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = make_pipeline(X_train)
    model.fit(X_train, y_train)

    predicted_log_prices = model.predict(X_valid)
    predicted_prices = np.maximum(np.expm1(predicted_log_prices), 0)
    actual_prices = np.expm1(y_valid)
    rmsle = mean_squared_error(
        np.log1p(actual_prices), np.log1p(predicted_prices)
    ) ** 0.5

    print('Missing strategy: PoolQC missing -> "None" (no pool)')
    print(f"Validation RMSLE: {rmsle:.5f}")
    print(f"Validation RMSE (log): {mean_squared_error(y_valid, predicted_log_prices) ** 0.5:.5f}")
    print(f"Validation MAE ($): {mean_absolute_error(actual_prices, predicted_prices):,.0f}")


if __name__ == "__main__":
    main()
