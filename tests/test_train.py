import numpy as np
import pandas as pd

from src.train import train_and_evaluate


def _synthetic_master_df(n_days=10, rows_per_day=6):
    rng = np.random.default_rng(42)
    dates = []
    for day in range(n_days):
        dates += [f"2020-06-{day + 1:02d}"] * rows_per_day

    ambient = rng.uniform(20, 35, size=len(dates))
    module = ambient + rng.uniform(0, 10, size=len(dates))
    irradiation = rng.uniform(0, 1000, size=len(dates))
    dc_power = irradiation * 0.02 + rng.normal(0, 1, size=len(dates))

    return pd.DataFrame({
        "DATE_STR": dates,
        "AMBIENT_TEMPERATURE": ambient,
        "MODULE_TEMPERATURE": module,
        "IRRADIATION": irradiation,
        "DC_POWER": dc_power,
    })


def test_train_and_evaluate_predicts_for_every_test_row():
    df_master = _synthetic_master_df()

    model, X_test, y_test, y_pred = train_and_evaluate(df_master, "Synthetic Plant", split_date="2020-06-07")

    assert len(y_pred) == len(X_test) == len(y_test)
    assert len(y_pred) > 0
