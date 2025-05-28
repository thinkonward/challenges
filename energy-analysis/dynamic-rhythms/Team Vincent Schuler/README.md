# ThinkOnward Competition - Dynamic Rythms - [REDACTED] 2025.

This guide outlines the steps to build a global historical dataset, train a model and perform inference.

> ##### 🧠 **<ins>Training Data**:</ins> 2019-01-01 → 2023-10-31.
> ##### 🔍 **<ins>Inference Data**:</ins> 2023-11-01 → 2023-12-31.

___
## Preliminary Steps:

### Step A: Organize Input Data
- Place **EAGLE-I** data in: `Files/eaglei_data/` *(at least year 2023 for inference)*
- Place **NOAA Storm Events** data in: `Files/NOAA_StormEvents/` *(not needed for inference)*

### Step B: Download External Data *(not needed for inference)*
- Most external data is already included, except for **daily weather data (2014–2023)**.  
- Download the files you need from: [NOAA GSOD Portal](https://www.ncei.noaa.gov/data/global-summary-of-the-day/)
- For inference, no download is required.

___
## Follow-up Steps:

### Step 1: Generate Intermediate Datasets *(not needed for inference)*
> All steps below can be run in parallel.
- **1a.** Run the `1a. Create CSV` notebook to generate an **hourly-aggregated dataset** from EAGLE-I data.  
  <ins>Output:</ins> `df.csv` *(provided for inference)*
- **1b.** Run the `1b. STORMS` notebook to generate an **hourly-aggregated storm dataset** from NOAA Storm Events.  
  <ins>Output:</ins> `df_storms_by_fips.csv` *(provided for inference)*
- **1c.** Run the `1c. Prepare daily weather` notebook to process **daily weather data**.  
  <ins>Outputs:</ins>
  - `daily_weather_info.csv` *(provided for inference)*
  - `dict_closest_station_to_fips.joblib` *(provided for inference)*

> After that, ensure all outputs from Step 1 are placed in: `Files/Outputs/`

### Step 2: Merge and Enrich the Dataset
- Run the `2. Prepare Dataset` notebook.
- This enriches the hourly dataset created in step 1.a (`df.csv`) with additional features and conduct some data analysis.  
  <ins>Output:</ins> `df_hourly_outages.csv`
  

### Step 3: Model Training *(not needed for inference)*
- Run the `3. Training` notebook to train and save a **LightGBM model**. *(provided for inference)*

### Step 4: Model Inference
- Run the `4. Inference` notebook to generate predictions and perform analysis using the dataset created in Step 2.

---

**Note:**  
Each notebook includes visualizations and data explorations that are useful for understanding the process and results.
