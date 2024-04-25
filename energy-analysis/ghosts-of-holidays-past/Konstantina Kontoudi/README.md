## Environment set up
To set up the environment run from the project root:
```
virtualenv --python=/usr/bin/python3.9 venv
source venv/bin/activate
```
Then, install dev and pipeline requirements with:
```
pip install -r requirements.txt
```

To be able to use the jupyter notebook with this virtual environment run:
```
python -m ipykernel install --user --name=venv
```
and make sure to choose the `env` kernel in the jupyter notebook interface.
To launch the jupyter notebook just execute:
```
jupyter notebook
```

Download and extract the data in the `data` folder. After the extraction the folder should
have a `data` sub-folder with the following structure:
```
├── data
│   ├── 100.parquet
│   ├── .
│   ├── .
│   ├── .
│   ├── 98.parquet
│   ├── 99.parquet
│   ├── 9.parquet
│   ├── data_dictionary.tsv
│   ├── enumeration_dictionary.tsv
│   ├── metadata.parquet
│   ├── scrooge_bldg.parquet
│   └── scrooge_metadata.parquet
```



```
├── weather
│   ├── daily_weather_location_0.parquet
│   ├── daily_weather_location_10.parquet
│   ├── daily_weather_location_11.parquet
│   ├── daily_weather_location_1.parquet
│   ├── daily_weather_location_2.parquet
│   ├── daily_weather_location_3.parquet
│   ├── daily_weather_location_4.parquet
│   ├── daily_weather_location_5.parquet
│   ├── daily_weather_location_6.parquet
│   ├── daily_weather_location_7.parquet
│   ├── daily_weather_location_8.parquet
│   ├── daily_weather_location_9.parquet
│   ├── hourly_weather_location_0.parquet
│   ├── hourly_weather_location_10.parquet
│   ├── hourly_weather_location_11.parquet
│   ├── hourly_weather_location_1.parquet
│   ├── hourly_weather_location_2.parquet
│   ├── hourly_weather_location_3.parquet
│   ├── hourly_weather_location_4.parquet
│   ├── hourly_weather_location_5.parquet
│   ├── hourly_weather_location_6.parquet
│   ├── hourly_weather_location_7.parquet
│   ├── hourly_weather_location_8.parquet
│   ├── hourly_weather_location_9.parquet
│   └── weather_metadata.parquet
```