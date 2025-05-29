#  Project Summary: dynamic-rythms

A project developed to participate in the **Dynamic Rhythms Contest**, focused on predicting power outages caused by 
severe weather events, particularly storms.

![](assets/images/dynamic-rhythms.png)

## Objective: 
Develop a machine learning model that can forecast power outages in advance using historical and real-time meteorological data. 
The goal is to enable early, accurate, and geospatially precise predictions to support proactive responses in energy systems.

This project contributes to the advancement of rare-event forecasting and supports the development of more sustainable and resilient infrastructure.

## Project structure: 
```
dynamic-rythms/
├── assets/                               # Visual and media assets
│   ├── images/                           # Icons, logos, and UI visuals
│   └── time-organization/                # Time-based diagrams or organization figures
│
├── config/                               # Configuration and parameter files
│   ├── config.yaml                       # Global config definitions
│   ├── model_params.json                 # Model hyperparameters
│   └── POWER_Parameter_Manager.xlsx      # External parameter manager spreadsheet
│
├── data/                                 # Structured data lifecycle
│   ├── external/                         # Third-party raw data
│   ├── final/                            # Cleaned data ready for modeling
│   ├── temp_results/                     # Temporary analysis outputs
│   ├── interim/                          # Intermediate transformed data
│   └── raw/                              # Immutable raw source data
│       └── dynamic-rhythms-train-data/   # Contest-provided data
│           └── data/
│               ├── eaglei_data/
│               └── NOAA_StormEvents/
│
├── models/                               # Trained model artifacts
│   ├── conformal_model.pkl               # Conformal prediction model
│   └── model.pkl                         # Main classification model
│
├── notebooks/                            # Jupyter notebooks for development and EDA
│
├── src/                                  # Source code for the project (flat module structure)
│   ├── __init__.py
│   ├── cleaner.py
│   ├── conformal_predictions.py
│   ├── data_dataset_creation.py
│   ├── dataset_splitting.py
│   ├── downloader.py
│   ├── feature_generation.py
│   ├── meteorological_api.py
│   ├── model_metrics.py
│   ├── storm_outages.py
│   ├── training_model.py
│   ├── understanding_model.py
│   └── utils.py
│
├── model_pipeline.ipynb                 # Main and most important file. Contains the model pipeline. 
├── model_pipeline.html                  # Freezed pipeline file, explaining results, etc. 
├── general_pipeline.py                  # Script to orchestrate the whole pipeline
├── .gitignore                           # Git version control exclusions
├── LICENSE                              # License file (e.g., MIT, Apache)
├── README.md                            # Project overview and instructions
├── requirements.txt                     # Python dependencies
└── setup.py                             # Installation script (if packaging as module)
```

---

## Steps to Run This Project

Follow these steps to set up and run the pipeline end to end:

---

### 1. Prepare the Data
After cloning or downloading this repository, you need to manually place the contest data into the appropriate folder:
- Unzip the `dynamic-rhythms-train-data` archive into the `data/raw` directory.
- The extracted folder **must** be named exactly `dynamic-rhythms-train-data`.

Your folder structure should look like this:
```
    dynamic-rythms/
    ├── data/
    │   ├── external/                        # Data from third-party sources.
    │   ├── final/                           # Final version of data, ready for model, and test.
    │   ├── temp_results/                    # Temporary files that are generated from results.
    │   ├── interim/                         # Intermediate data files.
    │   └── raw/                             # Original, immutable data.
    │       └── dynamic-rhythms-train-data/  # Provided by the contest.
    │           └── data/
    │               ├── eaglei_data/
    │               └── NOAA_StormEvents/
    │       
```
- The expected path is: `data/raw/dynamic-rhythms-train-data/`

### 2. Create a Virtual Environment

Make sure you're using **Python version 3.10.17**.
Then, create and activate a virtual environment using your preferred method.  


### 3. Install Project Dependencies

From the project root, install the required packages: `pip install -r requirements.txt`


### 4. Run the Pipeline
Use Jupyter Notebook
  - Open and run the notebook `model_pipeline.ipynb` cell by cell. This will download external data, generate features, train the model and evaluate it.

 ### 5. Sit back and let it run
The pipeline will process the data, build features, train the model, and provide outputs.
Depending on your machine and internet speed, this may take some time.

---
## License
This project is licensed under the [MIT License](./LICENSE).  

## Contributor

- [REDACTED]
