import pandas as pd

from src.dataset import preprocess_plant1_data, preprocess_plant2_data


def _write_csv(path, df):
    df.to_csv(path, index=False)
    return str(path)


def test_preprocess_plant1_data_merges_generation_and_weather(tmp_path):
    gen_df = pd.DataFrame({
        "DATE_TIME": ["2020-05-15 00:00:00", "2020-05-15 00:00:00", "2020-05-15 00:15:00", "2020-05-15 00:15:00"],
        "SOURCE_KEY": ["INV_A", "INV_B", "INV_A", "INV_B"],
        "DC_POWER": [0.0, 10.0, 0.0, 20.0],
        "AC_POWER": [0.0, 9.0, 0.0, 18.0],
    })
    weather_df = pd.DataFrame({
        "DATE_TIME": ["2020-05-15 00:00:00", "2020-05-15 00:15:00"],
        "AMBIENT_TEMPERATURE": [25.0, 25.5],
        "MODULE_TEMPERATURE": [24.0, 24.5],
        "IRRADIATION": [0.0, 0.0],
    })

    gen_path = _write_csv(tmp_path / "gen.csv", gen_df)
    weather_path = _write_csv(tmp_path / "weather.csv", weather_df)

    result = preprocess_plant1_data(gen_path, weather_path)

    assert list(result.columns) == [
        "DATE_TIME", "DATE_STR", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "DC_POWER",
    ]
    assert len(result) == 2
    # Median of [0.0, 10.0] across the two inverters at the first timestamp
    assert result.loc[0, "DC_POWER"] == 5.0


def test_preprocess_plant2_data_keeps_ac_power_and_drops_yield_columns(tmp_path):
    gen_df = pd.DataFrame({
        "DATE_TIME": ["2020-05-15 00:00:00", "2020-05-15 00:00:00"],
        "SOURCE_KEY": ["INV_A", "INV_B"],
        "DC_POWER": [10.0, 30.0],
        "AC_POWER": [9.0, 27.0],
        "DAILY_YIELD": [100.0, 9525.0],
        "TOTAL_YIELD": [1000.0, 50000.0],
    })
    weather_df = pd.DataFrame({
        "DATE_TIME": ["2020-05-15 00:00:00"],
        "AMBIENT_TEMPERATURE": [25.0],
        "MODULE_TEMPERATURE": [24.0],
        "IRRADIATION": [0.0],
    })

    gen_path = _write_csv(tmp_path / "gen.csv", gen_df)
    weather_path = _write_csv(tmp_path / "weather.csv", weather_df)

    result = preprocess_plant2_data(gen_path, weather_path)

    assert "DAILY_YIELD" not in result.columns
    assert "TOTAL_YIELD" not in result.columns
    # Median of [10.0, 30.0] across the two inverters
    assert result.loc[0, "DC_POWER"] == 20.0
    assert result.loc[0, "AC_POWER"] == 18.0
