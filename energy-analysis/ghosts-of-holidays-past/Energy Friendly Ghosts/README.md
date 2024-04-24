# Overview
This notebook presents a solution to the competition "The Energy-Efficient Ghosts of Holiday Past".
- Competition website: https://thinkonward.com/app/c/challenges/ghosts-of-holidays-past
- Team: Energy Friendly Ghosts

# Structure of this folder
- Notebook and solution details
  - 001-External-Data-Weather-Downloader.ipynb: Weather data downloader
  - 00m-All-in-one-notebook-v3-timezone.ipynb: Main notebook


- ghosts-of-holidays-past-train-data: provided train dataset

- weather: historical weather. The best way is the unzip the weather.zip into this folder. Otherwise, run the 001-External-Data-Weather-Downloader.ipynb notebook with a valid api_key to download weather information.

- workspace: output directory
  - sub_scrooge_bldg_party.csv: Submission file
  - Details

```console
workspace
├── buffer
├── models
│   ├── dict_foundation_quantile.pickle
│   └── dict_model_scrooge_bldg.pickle
└── subs
    └── sub_scrooge_bldg_party.csv
```

# Environment
## Conda Environment
- python: Python 3.9.18
- requirements file: requirements.txt
In case there are still missing packages, please use the requirements-longer-version.txt.
- java: 1.8.0_402

## Most Important libraries
- rtdip_sdk
- pyspark
- catboost
