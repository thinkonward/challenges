from datetime import datetime

import pandas as pd

from src.feature_extraction import prepare_dataset
from src.settings import HEATING_TOTAL_COL, PLUG_COL, ALL_TARGETS, BUILDING_ID_COL, DATA_DIR
from src.training import FEATURES_NAMES


def price(timestamp):
    if (datetime.strptime('18:15', '%H:%M').time() <= timestamp.time()
            <= datetime.strptime('22:00', '%H:%M').time()):
        return 0.35
    else:
        return 0.2


def prepare(models):
    scrooge_electricity_path = DATA_DIR / "scrooge_bldg.parquet"
    scrooge_weather_path = DATA_DIR / "scrooge_weather.parquet"
    scrooge_meta_path = DATA_DIR / "scrooge_metadata.parquet"

    weather_data = pd.read_parquet(scrooge_weather_path)
    electricity_data = pd.read_parquet(scrooge_electricity_path)
    metadata = pd.read_parquet(scrooge_meta_path)

    dataset = prepare_dataset(weather_data, electricity_data, metadata)

    dataset = dataset[FEATURES_NAMES]

    print(dataset.select_dtypes('object').columns.tolist())

    print(dataset.dtypes)

    dataset = dataset[(dataset["timestamp"] >= pd.Timestamp(2018, 12, 22, 0, 15)) & (
            dataset["timestamp"] <= pd.Timestamp(2019, 1, 1, 0, 0))].copy()
    timestamps = dataset["timestamp"]

    dataset[BUILDING_ID_COL] = dataset[BUILDING_ID_COL].astype('category')

    for target_col in [HEATING_TOTAL_COL, PLUG_COL]:
        dataset[target_col] = models[target_col].predict(dataset.drop(columns=[HEATING_TOTAL_COL, PLUG_COL]))

    dataset["timestamp"] = timestamps
    dataset["total_consumption"] = dataset[HEATING_TOTAL_COL] + dataset[PLUG_COL]
    dataset["party_consumption"] = (dataset["total_consumption"] * 0.3 * 4 + 2) / 4  # in kWh
    dataset["electricity_price"] = dataset["timestamp"].apply(price)
    dataset["party_cost"] = dataset["party_consumption"] * dataset["electricity_price"]

    party = dataset[(dataset["timestamp"] >= pd.Timestamp(2018, 12, 22, 17, 15)) & (
            dataset["timestamp"] <= pd.Timestamp(2019, 1, 1, 0, 0))].copy()

    party["total_party_cost"] = party["party_cost"].rolling(16).sum(numeric_only=True)

    party_totals = party[
        (party["timestamp"].dt.hour >= 21) | (party["timestamp"].dt.time == datetime.strptime('00:00', '%H:%M').time())]

    party_end_index = party_totals[party_totals["total_party_cost"] == party_totals["total_party_cost"].min()].index[0]

    submission = party.loc[party_end_index - 15: party_end_index][["timestamp", "party_cost"]]

    submission.to_csv("submission.csv", index=False)
