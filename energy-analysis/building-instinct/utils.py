import os
import pandas as pd
import numpy as np
import random
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def calculate_average_hourly_energy_consumption(folder_path, season_months_dict):
    """
    Process multiple parquet files in a folder, calculate hourly average energy consumption,
    and return a pandas DataFrame with each row corresponding to one file in the folder.

    Parameters:
    - folder_path (str): Path to the folder containing parquet files.
    - season_months_dict (dict): A dictionary where keys are season names (strings) and values are lists
      of corresponding month numbers. For example, {'cold': [1, 2, 12], 'hot': [6, 7, 8], 'mild': [3, 4, 5, 9, 10, 11]}.

    Returns:
    - df_ave_hourly (pd.DataFrame): A pandas DataFrame with each row corresponding to one file in the folder (i.e. one building).
      The columns are multi-layer with the first layer being the season and the second layer the hour of the day
      Index ('bldg_id') contains building IDs. Column values are average hourly electricity energy consumption
    """
    # Initialize an empty list to store individual DataFrames for each file
    result_dfs = []

    # Iterate through all files in the folder_path
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".parquet"):
            # Extract the bldg_id from the file name
            bldg_id = int(file_name.split(".")[0])

            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)

            # Read the original parquet file
            df = pd.read_parquet(file_path)

            # Convert 'timestamp' column to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Extract month and hour from 'timestamp'
            df["month"] = df["timestamp"].dt.month
            df["hour"] = df["timestamp"].dt.hour

            # Create a mapping from month to the corresponding season
            month_to_season = {
                month: season
                for season, months_list in season_months_dict.items()
                for month in months_list
            }

            # Assign a season to each row based on the month
            df["season"] = df["month"].map(month_to_season)

            # Calculate hourly average energy consumption for each row
            df["hourly_avg_energy_consumption"] = 4 * df.groupby(["season", "hour"])[
                "out.electricity.total.energy_consumption"
            ].transform("mean")

            # Pivot the dataframe to create the desired output format
            result_df = df.pivot_table(
                values="hourly_avg_energy_consumption",
                index="bldg_id",
                columns=["season", "hour"],
            )

            # Reset the column names
            result_df.columns = pd.MultiIndex.from_tuples(
                [
                    (season, hour + 1)
                    for season, months_list in season_months_dict.items()
                    for hour in range(24)
                ]
            )

            # Add 'bldg_id' index with values corresponding to the names of the parquet files
            result_df["bldg_id"] = bldg_id
            result_df.set_index("bldg_id", inplace=True)

            # Append the result_df to the list
            result_dfs.append(result_df)

    # Concatenate all individual DataFrames into a single DataFrame
    df_ave_hourly = pd.concat(result_dfs, ignore_index=False)

    return df_ave_hourly


def train_model(X, y):
    """
    Train hierarchical classification models for predicting building stock types and their respective attributes.

    This function trains three separate models:
    1. A classifier to predict the 'building_stock_type' (either 'commercial' or 'residential').
    2. A classifier for predicting attributes of commercial buildings.
    3. A classifier for predicting attributes of residential buildings.

    The function preprocesses the input data using standard scaling and optional one-hot encoding before training the classifiers.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature dataframe used for training. Each row represents a building, and each column represents a feature.

    y : pd.DataFrame
        The target dataframe containing the labels. It includes the 'building_stock_type' column and other columns ending
        with '_com' for commercial attributes and '_res' for residential attributes.

    Returns:
    -------
    list
        A list of three trained classifiers:
        1. classifier_type: A RandomForestClassifier model for predicting 'building_stock_type'.
        2. classifier_residential: A RandomForestClassifier model for predicting residential attributes.
        3. classifier_commercial: A RandomForestClassifier model for predicting commercial attributes.

    """

    # Define column transformers for commercial and residential buildings
    transformer_commercial = ColumnTransformer(
        [("scaler", StandardScaler(), X.columns), ("encoder", OneHotEncoder(), [])]
    )

    transformer_residential = ColumnTransformer(
        [("scaler", StandardScaler(), X.columns), ("encoder", OneHotEncoder(), [])]
    )

    # Filter features and targets for commercial and residential buildings
    X_commercial = X[y["building_stock_type"] == "commercial"]
    X_residential = X[y["building_stock_type"] == "residential"]
    y_commercial = y[y["building_stock_type"] == "commercial"].filter(like="_com")
    y_residential = y[y["building_stock_type"] == "residential"].filter(like="_res")

    # Train classifier to predict 'building_stock_type'
    classifier_type = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        ("scaler", StandardScaler(), X.columns),
                        ("encoder", OneHotEncoder(), []),
                    ]
                ),
            ),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )
    classifier_type.fit(X, y["building_stock_type"])

    # Train separate classifiers for commercial and residential buildings
    classifier_commercial = Pipeline(
        [
            ("preprocessor", transformer_commercial),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    classifier_residential = Pipeline(
        [
            ("preprocessor", transformer_residential),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Train models
    classifier_commercial.fit(X_commercial, y_commercial)
    classifier_residential.fit(X_residential, y_residential)

    return [classifier_type, classifier_residential, classifier_commercial]


def get_pred(X, classifier_list, column_list):
    """
    Generate predictions for a hierarchical multi-output multi-class classification problem.

    This function takes in a feature dataframe and a list of trained classifiers to generate predictions
    for the 'building_stock_type' and its respective attributes based on the hierarchical structure. The predictions
    are populated in a new dataframe with the same index as the input features and columns specified in the column list.

    Parameters:
    ----------
    X : pd.DataFrame
        The feature dataframe used for making predictions. Each row represents a building, and each column represents a feature.

    classifier_list : list
        A list of three trained classifiers:
        1. classifier_type: A model for predicting 'building_stock_type'.
        2. classifier_residential: A model for predicting residential attributes.
        3. classifier_commercial: A model for predicting commercial attributes.

    column_list : list
        A list of column names to be included in the output predictions dataframe. This should include 'building_stock_type'
        and other columns ending with '_com' for commercial attributes and '_res' for residential attributes.

    Returns:
    -------
    pd.DataFrame
        A dataframe containing the predictions. The index matches the input feature dataframe's index, and the columns are
        specified by the column list. The values are populated based on the hierarchical structure of the predictions.

    """

    classifier_type, classifier_residential, classifier_commercial = classifier_list

    # Predict 'building_stock_type'
    y_pred_type = classifier_type.predict(X)

    # Predict relevant columns based on predicted 'building_stock_type'
    y_pred_commercial = classifier_commercial.predict(X[y_pred_type == "commercial"])
    y_pred_residential = classifier_residential.predict(X[y_pred_type == "residential"])

    y_pred = pd.DataFrame(index=X.index, columns=column_list)

    # Set all values in y_pred to np.nan
    y_pred[:] = np.nan

    # Ensure the index name is the same
    y_pred.index.name = X.index.name

    y_pred["building_stock_type"] = y_pred_type
    y_pred.loc[
        y_pred_type == "commercial", y_pred.columns.str.endswith("_com")
    ] = y_pred_commercial
    y_pred.loc[
        y_pred_type == "residential", y_pred.columns.str.endswith("_res")
    ] = y_pred_residential

    return y_pred


def calculate_hierarchical_f1_score(
    df_targets, df_pred, alpha=0.4, average="macro", F1_list=False
):
    """
    Calculate the hierarchical F1-score for a multi-level classification problem.

    This function computes the F1-score at two hierarchical levels:
    1. The 'building_stock_type' level, which is the first level of hierarchy.
    2. The second level, which is conditional on the 'building_stock_type' being either 'commercial' or 'residential'.

    The final F1-score is a weighted average of the first level and second level F1-scores.

    Parameters:
    ----------
    df_targets : pd.DataFrame
        The dataframe containing the true target values. It must include a column 'building_stock_type' and other
        columns ending with '_com' or '_res' representing the second level of classification.

    df_pred : pd.DataFrame
        The dataframe containing the predicted values. It must be structured similarly to `df_targets`.

    alpha : float, optional, default=0.3
        The weight given to the first level F1-score in the final score calculation. The weight for the second level
        F1-score will be (1 - alpha).

    average : str, optional, default='macro'
        The averaging method for calculating the F1-score. It is passed directly to the `f1_score` function from sklearn.

    F1_list : bool, optional, default=False
        If True, the function returns a dictionary of F1-scores for all individual columns along with the overall F1-score.

    Returns:
    -------
    float or tuple
        If `F1_list` is False, returns a single float representing the overall hierarchical F1-score.
        If `F1_list` is True, returns a tuple where the first element is the overall hierarchical F1-score and the second
        element is a dictionary containing the F1-scores for all individual columns.

    """

    def calculate_f1_l2(df_targets, df_pred, average):
        """
        Calculate the F1-score for the second level of hierarchy.

        Parameters:
        ----------
        df_targets : pd.DataFrame
            The dataframe containing the true target values for the second level of hierarchy.
        df_pred : pd.DataFrame
            The dataframe containing the predicted values for the second level of hierarchy.
        average : str
            The averaging method for calculating the F1-score.

        Returns:
        -------
        dict
            A dictionary where keys are column names and values are the corresponding F1-scores.
        """
        F1_l2_dict = {column: 0 for column in df_targets.columns}

        # Find the intersection of indices
        common_indices = df_targets.index.intersection(df_pred.index)

        # Check if the intersection is empty
        if common_indices.empty:
            return F1_l2_dict
        else:
            # Select only the rows with common indices
            df_targets_common = df_targets.loc[common_indices]
            df_pred_common = df_pred.loc[common_indices]

            # Calculate the F1-score for each column based on the common rows
            for column in df_targets.columns:
                F1_l2_dict[column] = f1_score(
                    df_targets_common[column], df_pred_common[column], average=average
                )

        return F1_l2_dict

    # Sort both dataframes based on index
    df_targets = df_targets.sort_index()
    df_pred = df_pred.sort_index()

    # Calculate F1 score for the first level of hierarchy
    F1_l1 = f1_score(
        df_targets["building_stock_type"],
        df_pred["building_stock_type"],
        average=average,
    )
    F1_dict = {"building_stock_type": F1_l1}

    # Calculate F1 score for the second level of hierarchy (commercial buildings)
    df_com_targets = df_targets[
        df_targets["building_stock_type"] == "commercial"
    ].filter(like="_com")
    df_com_pred = df_pred[df_pred["building_stock_type"] == "commercial"].filter(
        like="_com"
    )
    F1_l2_dict_com = calculate_f1_l2(df_com_targets, df_com_pred, average)
    F1_l2_com = sum(F1_l2_dict_com.values()) / len(F1_l2_dict_com.values())

    F1_l2_dict = {}
    F1_l2_dict.update(F1_l2_dict_com)

    # Calculate F1 score for the second level of hierarchy (residential buildings)
    df_res_targets = df_targets[
        df_targets["building_stock_type"] == "residential"
    ].filter(like="_res")
    df_res_pred = df_pred[df_pred["building_stock_type"] == "residential"].filter(
        like="_res"
    )
    F1_l2_dict_res = calculate_f1_l2(df_res_targets, df_res_pred, average)
    F1_l2_res = sum(F1_l2_dict_res.values()) / len(F1_l2_dict_res.values())

    F1_l2_dict.update(F1_l2_dict_res)
    F1_l2_dict_sorted = sorted(F1_l2_dict.items(), key=lambda x: x[1], reverse=True)
    F1_dict.update(F1_l2_dict_sorted)

    # Calculate F1 score for the second level of hierarchy
    F1_l2 = (F1_l2_com + F1_l2_res) / 2

    # Calculate overall F1 score
    F1 = alpha * F1_l1 + (1 - alpha) * F1_l2

    if F1_list:
        return F1, F1_dict

    return F1


def sample_submission_generator(bldg_id_list, df_targets, path_to_save):
    """
    Generate a sample submission dataframe with a specified distribution and save it as a .parquet file.

    This function creates a dataframe with the same columns as `df_targets` and populates it with values
    that resemble the distribution of values in `df_targets`. The index of the dataframe is given by
    `bldg_id_list`, and the column values are sampled to match the distribution of the corresponding columns
    in `df_targets`. The dataframe is then saved as a .parquet file to the specified path.

    Parameters:
    ----------
    bldg_id_list : list of int
        List of building IDs to be used as the index of the generated dataframe.

    df_targets : pd.DataFrame
        The metadata dataframe used to sample the column values. It provides the distribution of values for
        each column to be replicated in the generated dataframe.

    path_to_save : str
        The path where the generated dataframe will be saved as a .parquet file.

    Returns:
    -------
    pd.DataFrame
        The generated dataframe with the same columns as `df_targets` and index values as `bldg_id_list`,
        populated with values sampled from the distribution of `df_targets`.

    """

    # Create an empty dataframe with the same columns and index name as df_targets
    df = pd.DataFrame(index=bldg_id_list, columns=df_targets.columns)
    df.index.name = df_targets.index.name

    # Populate the first column 'building_stock_type'
    building_stock_type_distribution = df_targets["building_stock_type"].value_counts(
        normalize=True
    )
    df["building_stock_type"] = np.random.choice(
        building_stock_type_distribution.index,
        size=len(bldg_id_list),
        p=building_stock_type_distribution.values,
    )

    # Separate columns into residential and commercial
    res_columns = [col for col in df_targets.columns if col.endswith("_res")]
    com_columns = [col for col in df_targets.columns if col.endswith("_com")]

    # Populate the rest of the columns based on the value of 'building_stock_type'
    for bldg_id in df.index:
        if df.at[bldg_id, "building_stock_type"] == "residential":
            df.loc[bldg_id, com_columns] = np.nan
            for col in res_columns:
                distribution = df_targets[col].value_counts(normalize=True)
                df.at[bldg_id, col] = np.random.choice(
                    distribution.index, p=distribution.values
                )
        else:  # commercial
            df.loc[bldg_id, res_columns] = np.nan
            for col in com_columns:
                distribution = df_targets[col].value_counts(normalize=True)
                df.at[bldg_id, col] = np.random.choice(
                    distribution.index, p=distribution.values
                )

    # Save the dataframe as a parquet file
    df.to_parquet(path_to_save)
    return df
