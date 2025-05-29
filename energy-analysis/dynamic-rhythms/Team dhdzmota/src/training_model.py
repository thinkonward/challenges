import pickle

from xgboost import XGBClassifier

import src.dataset_splitting as dataset_splitting
import src.utils as utils


RANDOM_SEED = utils.RANDOM_SEED

INTERIM_DATA_PATH = utils.get_data_path('interim')
FINAL_DATA_PATH = utils.get_data_path('final')
GENERAL_PATH = utils.get_general_path()
MODEL_PATH = utils.join_paths(GENERAL_PATH, 'models')
MODEL_FILENAME = utils.join_paths(MODEL_PATH, 'model.pkl')


NO_FEATURE_COLS = [
    'time',
    'episode_fips_id',
    'meteorological_current_datetime_val',
    'hours_to_outage',
    'outage_in_an_hour',
    'storm_duration',
    'total_customers_out',
]

ADDITIONAL_INFO_COLUMNS = [
    'coord0',
    'coord1',
    'day_of_year',
    'month_of_year',
]

XGB_PARAMETERS = dict(
    n_estimators=1000000,
    learning_rate=0.1,
    max_depth=3,
    scale_pos_weight=None,
    use_label_encoder=False,
    eval_metric='aucpr',
    reg_lambda=1,
    alpha=1,
    subsample=0.75,
    colsample_bytree=0.75,
    early_stopping_rounds=25,
    gamma=1,
    seed=RANDOM_SEED,
)


def read_splitted_data():
    """
    Load and return the previously split datasets for training, calibration, and testing.

    :return: datasets: dict
        Dictionary containing split datasets, typically including keys like 'train', 'cal', 'test', and 'OOT'.
    """
    datasets = dataset_splitting.read_datasets()
    return datasets


def get_datasets_x_y_info(datasets):
    """
    Split each dataset into features (X), target variable (y), and additional information.

    :param datasets: dict
        Dictionary of datasets with keys like 'train', 'cal', 'test', etc., each containing a DataFrame.
    :return: x, y, info: tuple of dicts
        x: dict
            Feature matrices with non-feature columns dropped.
        y: dict
            Target labels (`outage_in_an_hour`) for each dataset split.
        info: dict
            Additional information columns retained from each dataset.
    """
    x = {}
    y = {}
    info = {}
    for key, data in datasets.items():
        x[key] = data.drop(NO_FEATURE_COLS, axis=1)
        y[key] = data.outage_in_an_hour
        info[key] = data[NO_FEATURE_COLS + ADDITIONAL_INFO_COLUMNS]
    return x, y, info


def save_x_y_info(x, info):
    """
    Save feature matrices and corresponding metadata for each dataset split.

    :param x: dict
        Dictionary containing feature DataFrames (X) for each dataset split.
    :param info: dict
        Dictionary containing metadata DataFrames (e.g., target, additional info) for each split.
    :return: None
    """
    for key in x.keys():
        x_path = utils.join_paths(FINAL_DATA_PATH, f'x_{key}.parquet')
        i_path = utils.join_paths(FINAL_DATA_PATH, f'i_{key}.parquet')
        print(f'Saving x for {key} data at: {x_path}')
        x[key].to_parquet(x_path)
        print(
            f'Saving info for {key} data at: {i_path} '
            f'(target and inference are contained).'
        )
        info[key].to_parquet(i_path)


def save_model(model):
    """
    Save a trained model object to disk using pickle.

    :param model: object
        Trained model to be saved.
    :return: None
    """
    print(f'Saving model at {MODEL_FILENAME}')
    with open(MODEL_FILENAME, "wb") as f:
        pickle.dump(model, f)
    print('Done.')


def get_model():
    """
    Load a trained model object from disk using pickle.

    :return: load_model: object
        The loaded model.
    """
    with open(MODEL_FILENAME, 'rb') as f:
        load_model = pickle.load(f)
    return load_model


def model_score(model, X):
    """
    Compute prediction scores (probabilities for the positive class) using a trained model.

    :param model: object
        Trained classification model with a `predict_proba` method.
    :param X: array-like or pandas DataFrame
        Feature data to score.
    :return: score: numpy array
        Array of predicted probabilities for the positive class.
    """
    score = model.predict_proba(X)[:, 1]
    return score


def train_model(force=False, save_data=False):
    """
    Train an XGBoost classifier on the prepared training data, and optionally save the model and data.

    :param force: bool, optional
        If True, retrains the model even if a saved model already exists. Default is False.
    :param save_data: bool, optional
        If True, saves the feature sets and prediction info after training. Default is False.
    :return: model: XGBClassifier
        The trained XGBoost classification model.
    """
    if not utils.check_if_filepath_exists(MODEL_FILENAME) or force:
        # Read data
        datasets = read_splitted_data()
        x, y, info = get_datasets_x_y_info(datasets)

        neg = (y['train'] == 0).sum()
        pos = (y['train'] == 1).sum()
        scale_pos_weight = neg / pos

        XGB_PARAMETERS['scale_pos_weight'] = scale_pos_weight
        print(f' The model parameters are: {XGB_PARAMETERS}')

        model = XGBClassifier(**XGB_PARAMETERS)
        print('Start with training...')
        model.fit(
            x['train'],
            y['train'],
            eval_set=[(x['train'], y['train']), (x['eval'], y['eval'])],
        )
        save_model(model)

        for key in info.keys():
            info[key]['y'] = y[key]
            info[key]['pred'] = model_score(model, x[key])
        if save_data:
            print('Saving data...')
            save_x_y_info(x, info)
    else:
        print(
            f'Model has already been trained and already exists'
            f', it is located at: {MODEL_FILENAME}'
        )
        model = get_model()
    return model


if __name__ == "__main__":
    train_model(force=False)
