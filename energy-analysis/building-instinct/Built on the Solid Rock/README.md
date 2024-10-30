# Solution Overview

## The Environment

- The Jupyter notebook runs in a Python 3.10.14 environment. The code runs on a CPU runtime type, lasting less than 15 hours (excluding hyperparameter search and feature selection).

- The details of the operating system on which the notebook run is Linux-5.15.154+-x86_64-with-glibc2.31

- The RAM usage recorded throughout the run of the notebook was less than 30gb.

## The Code

- This notebook takes input and uses only the data files provided in the competition resources.

- At the end of the processes, it outputs 'scv_catselect_2xtsf.parquet' as the submission file.

- The methodology used is to predict building features in two stages as required by the heirarchical nature of the task. Thus, an ensemble of [5] LightGBM classifiers (trained on different folds of the training data) is built to identify the building stock type and then depending on the identified building stock type, additional characteristics of each building are then predicted by a set of LightGBM classifier ensembles (5 classifiers per ensemble).

- The General Overview of the Notebook is as follows:

   1. Importing relevant methods from the following libraries:

      - scipy==1.14.1
      - numpy==1.26.4
      - pandas==2.2.2
      - lightgbm==4.2.0
      - scikit-learn==1.2.2
      - category-encoders==2.6.3
      - tsfresh==0.20.3

   2. Setting Paths

      - Indicating the location of the input files to be used in the notebook.

      - These paths are stored in the following variables:
          - `TEST_DIR` : path to the folder contain parquet files of the test data buildings.
          - `TRAIN_DIR` : path to the folder contain parquet files of the train data buildings
          - `SS_FILE` : path to the parquet file showing how a submission should be formatted.
          - `LABEL_FILE` : path to the parquet file containing attributes (meta data) of each building in the train dataset

   3. Data Preprocessing and Feature Engineering

      - Data is read, preprocessed and features engineered in four different phases.

      - Phase 1: This phase extracts statistics of electricity consumption based on datetime features. This includes hour, day and month of peak consumption. Statistics such as difference between average weekend and average weekday consumption is calculated in addition to rolling window statistics.

      - Phase 2: An automated feature engineering process applying tsfresh extraction to a quarter (full data was not used because of resource constraints) of the time series data of each train data building.

      - Phase 3: This is the second manual feature engineering process that creates a set of 96 features each to represent consumption during weekends and consumption during weekdays. The aggregation is achieved by averaging consumption across all the 96 15-minute time blocks for each day (weekend days in one set and weekdays in the other set). This phase also calculates additional statistics based on these aggregates such as the number of outliers per set of 15-minute consumption values.

      - Phase 4: This second automated feature engineering process applies the tsfresh extraction process to the two sets of timeseries data created in Phase 3 for each building.

   4. Tackling Heirarchy 1 - Building Stock Type Classification

      - A preprocessing pipeline is created to encode non-numeric features using a `TargetEncoder` object with the next step being the scaling of features using `MinMaxScaler`

      - Feature selection using `select_features` method of `CatboostClassifier` with the `RecursiveByShapValues` algorithm. This feature selection is omitted from this notebook but the results are applied to reduce the huge feature dimension and improve model performance.

      - Hyperparameter optimisation using `optuna` to select best hyperparameters of `LGBMClassifier` necessary for predicting building stock type. This step is also omitted from this notebook due to its time-consuming nature.

      - Fitting 5 different `LGBMClassifier` models one each on each of 5 folds of the training data. These 5 models will be ensembled during inference by selecting the mode of their individual predictions.

   5. Tackling Hierarchy 2 - Commercial Building Models

      - The preprocessed data with engineered features is filtered to create a dataframe of commercial buildings only.

      - This filtered dataframe is manipulated such that individual characteristics of commercial buildings to be predicted are transformed from being in individual target columns into a single target column along with a `target_type` column indicating the kind of characteristic being predicted. This is done using the `create_combo_task` function.

      - A dictionary is created to store the index of the features relevant for predicting each kind of characteristic (`target_type`). These indices are obtained from a feature selection process using the `select_features` method of `CatboostClassifier` with the `RecursiveByShapValues` algorithm. This feature selection is omitted from this notebook but the results are applied to reduce the huge feature dimension and improve model performance.

      - A dictionary is created to store the optimised hyperparameters of the `LGBMClassifier` model for predicting the values of each kind of characteristic (`target_type`). These hyperparameter values are obtained from an optuna hyperparameter search process which has been omitted from this notebook because it is time-intensive.

      - Finally we iterate through each commercial building characteristic's data, preprocess it, select relevant features, and fit one `LGBMClassifier` model on each of 5 folds of the data.

      - The model performance is evaluated and all the relevant objects are stored to be used in predicting on final test data.

   6. Tackling Hierarchy 2 - Residential Building Models

      - The preprocessed data with engineered features is filtered to create a dataframe of residential buildings only.

      - This filtered dataframe is manipulated such that individual characteristics of residential buildings to be predicted are transformed from being in individual target columns into a single target column along with a `target_type` column indicating the kind of characteristic whose values are being predicted. This is done using the `create_combo_task` function.

      - A dictionary is created to store the index of the features relevant for predicting each kind of characteristic (`target_type`). These indices are obtained from a feature selection process using the `select_features` method of `CatboostClassifier` with the `RecursiveByShapValues` algorithm. This feature selection is omitted from this notebook but the results are applied to reduce the huge feature dimension and improve model performance.

      - A dictionary is created to store the optimised hyperparameters of the `LGBMClassifier` model for predicting the values of each kind of characteristic (`target_type`). These hyperparameter values are obtained from an optuna hyperparameter search process which has been omitted from this notebook because it is time-intensive.

      - Finally we iterate through each residential building characteristic's data, preprocess it, select relevant features, and fit one `LGBMClassifier` model on each of 5 folds of the data.

      - The model performance is evaluated and all the relevant objects are stored to be used in predicting on final test data

   7. Predicting on Testset and Preparing Submission

      - The data for the test set of buildings is read, preprocessed and features engineered in the same fashion of 4 phases like the training data.

      - The 5 models developed for classifying the building stock type take turns in predicting the building stock type of the buildings in the test set with the modal prediction value being adopted as the final prediction.

      - Based upon each building's predicted stock type, it is taken through the set of ensemble models for predicting the additional characteristics of either residential or commercial buildings.

      - The prediction dataframe of the residential and commercial buildings are combined and manipulated to fit the format of the sample submission file.

      - The formatted prediction dataframe is exported to the `scv_catselect_2xtsf.parquet` parquet file for submission.
