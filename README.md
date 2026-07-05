# Solar Power Generation Forecasting

This project predicts short-term solar power output (`DC_POWER`) for two solar plants, using historical generation logs and weather sensor data. It follows the CRISP-DM process: data exploration, cleaning, feature engineering, modeling, and evaluation.

## Results (Plant 1)
* **R² score:** 0.9894
* **RMSE:** 388.50 kW
* **Validation method:** Chronological split, not random shuffling — trained on May 15–June 11, tested on June 12–June 17. This avoids leaking future information into training, which is a common mistake with time-series data.

These numbers come from actually re-running [notebooks/01_solar_forecasting_plant1.ipynb](notebooks/01_solar_forecasting_plant1.ipynb) top to bottom — re-run it yourself to confirm.

## What was done
1. **Data understanding:** Explored the raw generation and weather data, and found sensor issues such as midday clipping (inverters flatlining around noon) and missing readings.
2. **Data preparation:** Cleaned the data by imputing missing/anomalous values using the median across inverters at the same timestamp, then merged generation and weather data on time.
3. **Modeling:** Trained a LightGBM regressor to predict `DC_POWER` from weather and time features.
4. **Evaluation:** Used a chronological (not random) train/test split to simulate real forecasting conditions.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone <this-repo>
cd solar-power-forecasting-system
uv sync
```

## Data

The raw CSVs are not included in this repository (see `.gitignore`). Place the following files under `data/raw/` before running the notebooks:

```
data/raw/Plant_1_Generation_Data.csv
data/raw/Plant_1_Weather_Sensor_Data.csv
data/raw/Plant_2_Generation_Data.csv
data/raw/Plant_2_Weather_Sensor_Data.csv
```

Source: [Solar Power Generation Data on Kaggle](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data/data)

## How to run

Explore and re-run the analysis in the notebooks:

```bash
uv run jupyter notebook notebooks/01_solar_forecasting_plant1.ipynb
```

Run the cells top to bottom. The final cells print the R² and RMSE shown above.

Train the models used by the Streamlit app (saves to `models/`):

```bash
uv run python -m src.train
```

Launch the Streamlit dashboard (requires the models above to exist):

```bash
uv run streamlit run app.py
```

Run the test suite:

```bash
uv run pytest
```

## Repository contents
* `notebooks/01_solar_forecasting_plant1.ipynb` — Plant 1: data prep, feature analysis, model training and evaluation.
* `notebooks/02_solar_forecasting_plant2.ipynb` — same pipeline applied to Plant 2.
* `notebooks/solar_plant_analytics.ipynb` — background notes on how PV plants and the dataset are structured.
* `src/dataset.py` — preprocessing functions shared by `src/train.py` and `app.py`.
* `src/train.py` — trains and saves the LightGBM models to `models/`.
* `app.py` — Streamlit dashboard: actual vs predicted power for the test window, plus a manual prediction form.
* `tests/` — unit tests for `src/dataset.py` and `src/train.py`.