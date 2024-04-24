from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve().parent
DATA_DIR = ROOT_DIR / "data" / "data"
WEATHER_DIR = ROOT_DIR / "data" / "weather"

MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

HEATING_BKUP_COL = "out.electricity.heating_hp_bkup.energy_consumption"
HEATING_COL = "out.electricity.heating.energy_consumption"
PLUG_COL = "out.electricity.plug_loads.energy_consumption"
HEATING_TOTAL_COL = "heating_total"
HEATING_AND_PLUG_COL = "heating_and_plugs"


ALL_RAW_TARGETS = [HEATING_BKUP_COL, HEATING_COL, PLUG_COL]
ALL_TARGETS = ALL_RAW_TARGETS + [HEATING_TOTAL_COL, HEATING_AND_PLUG_COL]

BUILDING_ID_COL = "bldg_id"
LAT_COL = "in.weather_file_latitude"
LON_COL = "in.weather_file_longitude"
