import os

import pandas as pd


def preprocess_plant1_data(gen_path: str, weather_path: str) -> pd.DataFrame:
    """Load and merge Plant 1 generation + weather data into one feature table."""
    if not os.path.exists(gen_path) or not os.path.exists(weather_path):
        raise FileNotFoundError("Raw data files for Plant 1 could not be found.")

    df_gen = pd.read_csv(gen_path)
    df_weather = pd.read_csv(weather_path)

    df_gen["DATE_TIME"] = pd.to_datetime(df_gen["DATE_TIME"])
    df_weather["DATE_TIME"] = pd.to_datetime(df_weather["DATE_TIME"])

    # Median across the 22 inverters smooths out individual hardware dropouts
    df_gen_clean = df_gen.groupby("DATE_TIME")[["DC_POWER", "AC_POWER"]].median().reset_index()

    df_master = pd.merge(df_gen_clean, df_weather, on="DATE_TIME", how="inner")
    df_master["DATE_STR"] = df_master["DATE_TIME"].dt.strftime("%Y-%m-%d")

    target_features = ["DATE_TIME", "DATE_STR", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "DC_POWER"]
    return df_master[target_features]


def preprocess_plant2_data(gen_path: str, weather_path: str) -> pd.DataFrame:
    """Load and merge Plant 2 generation + weather data into one feature table."""
    if not os.path.exists(gen_path) or not os.path.exists(weather_path):
        raise FileNotFoundError("Raw data files for Plant 2 could not be found.")

    df_gen = pd.read_csv(gen_path)
    df_weather = pd.read_csv(weather_path)

    df_gen["DATE_TIME"] = pd.to_datetime(df_gen["DATE_TIME"])
    df_weather["DATE_TIME"] = pd.to_datetime(df_weather["DATE_TIME"])

    # DAILY_YIELD/TOTAL_YIELD are excluded: some inverters don't reset them at midnight,
    # so instantaneous DC_POWER/AC_POWER are used instead.
    df_gen_clean = df_gen.groupby("DATE_TIME")[["DC_POWER", "AC_POWER"]].median().reset_index()

    df_master = pd.merge(df_gen_clean, df_weather, on="DATE_TIME", how="inner")
    df_master["DATE_STR"] = df_master["DATE_TIME"].dt.strftime("%Y-%m-%d")

    target_features = [
        "DATE_TIME", "DATE_STR", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "DC_POWER", "AC_POWER",
    ]
    return df_master[target_features]
