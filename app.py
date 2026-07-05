import lightgbm as lgb
import pandas as pd
import streamlit as st

from src.dataset import preprocess_plant1_data, preprocess_plant2_data
from src.train import FEATURES

PLANTS = {
    "Plant 1": {
        "gen_path": "data/raw/Plant_1_Generation_Data.csv",
        "weather_path": "data/raw/Plant_1_Weather_Sensor_Data.csv",
        "model_path": "models/plant1_model.txt",
        "preprocess": preprocess_plant1_data,
    },
    "Plant 2": {
        "gen_path": "data/raw/Plant_2_Generation_Data.csv",
        "weather_path": "data/raw/Plant_2_Weather_Sensor_Data.csv",
        "model_path": "models/plant2_model.txt",
        "preprocess": preprocess_plant2_data,
    },
}
TEST_START_DATE = "2020-06-12"
TEST_END_DATE = "2020-06-17"


@st.cache_data
def load_features(plant_name: str) -> pd.DataFrame:
    config = PLANTS[plant_name]
    return config["preprocess"](config["gen_path"], config["weather_path"])


@st.cache_resource
def load_model(model_path: str) -> lgb.Booster:
    return lgb.Booster(model_file=model_path)


st.title("Solar Power Forecasting")
st.caption("Predicts DC_POWER from weather features using a LightGBM model trained with a chronological train/test split.")

plant_name = st.selectbox("Plant", list(PLANTS.keys()))
config = PLANTS[plant_name]

try:
    model = load_model(config["model_path"])
except lgb.basic.LightGBMError:
    st.error(
        f"No trained model found at `{config['model_path']}`. "
        f"Run `uv run python -m src.train` first to train and save the models."
    )
    st.stop()

st.header("Predicted vs actual power (test window)")

df = load_features(plant_name)
test_df = df[(df["DATE_STR"] >= TEST_START_DATE) & (df["DATE_STR"] <= TEST_END_DATE)].copy()
test_df["PREDICTED_DC_POWER"] = model.predict(test_df[FEATURES])

chart_df = test_df.set_index("DATE_TIME")[["DC_POWER", "PREDICTED_DC_POWER"]]
chart_df.columns = ["Actual", "Predicted"]
st.line_chart(chart_df)

st.header("Try a prediction")
st.caption("Enter weather conditions to get a predicted DC power output.")

col1, col2, col3 = st.columns(3)
ambient_temp = col1.number_input("Ambient temperature (°C)", value=25.0)
module_temp = col2.number_input("Module temperature (°C)", value=30.0)
irradiation = col3.number_input("Irradiation (W/m²)", value=500.0, min_value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([[ambient_temp, module_temp, irradiation]], columns=FEATURES)
    prediction = model.predict(input_df)[0]
    st.metric("Predicted DC Power", f"{prediction:,.1f} kW")
