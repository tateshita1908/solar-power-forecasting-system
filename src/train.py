import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

FEATURES = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]


def train_and_evaluate(df_master: pd.DataFrame, plant_label: str, split_date: str = "2020-06-11"):
    """Train a LightGBM regressor with a chronological (not random) train/test split."""
    X = df_master[FEATURES]
    y = df_master["DC_POWER"]
    dates = df_master["DATE_STR"]

    train_mask = dates <= split_date
    test_mask = (dates > split_date) & (dates <= "2020-06-17")

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{plant_label}: RMSE={rmse:.2f} kW, R2={r2:.4f} (train={len(X_train)}, test={len(X_test)})")
    return model, X_test, y_test, y_pred


def save_model(model: lgb.LGBMRegressor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.booster_.save_model(path)


if __name__ == "__main__":
    from src.dataset import preprocess_plant1_data, preprocess_plant2_data

    df_p1 = preprocess_plant1_data("data/raw/Plant_1_Generation_Data.csv", "data/raw/Plant_1_Weather_Sensor_Data.csv")
    df_p2 = preprocess_plant2_data("data/raw/Plant_2_Generation_Data.csv", "data/raw/Plant_2_Weather_Sensor_Data.csv")

    model_p1, *_ = train_and_evaluate(df_p1, "Plant 1")
    model_p2, *_ = train_and_evaluate(df_p2, "Plant 2")

    save_model(model_p1, "models/plant1_model.txt")
    save_model(model_p2, "models/plant2_model.txt")
    print("Saved models to models/plant1_model.txt and models/plant2_model.txt")
