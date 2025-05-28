import pandas as pd
import pickle

from mapie.classification import MapieClassifier

import src.utils as utils
import src.training_model as training_model
import src.understanding_model as understanding_model


GENERAL_PATH = utils.get_general_path()
MAPIE_MODEL_PATH = utils.join_paths(GENERAL_PATH, 'models')
MAPIE_MODEL_FILENAME = utils.join_paths(MAPIE_MODEL_PATH, 'conformal_model.pkl')


def save_mapie_model(model):
    """
    Save the MAPIE model object to a file using pickle.

    :param model: Trained MAPIE model object to be saved.
    :return: None
    """
    with open(MAPIE_MODEL_FILENAME, "wb") as f:
        pickle.dump(model, f)


def get_mapie_model():
    """
    Load the MAPIE model object from a file using pickle.

    :return: load_model
        The loaded MAPIE model object.
    """
    with open(MAPIE_MODEL_FILENAME, 'rb') as f:
        load_model = pickle.load(f)
    return load_model


def fit_mapie_classifier(model, x, y, save=True, force=False):
    """
    Fit a MAPIE classifier using a pre-trained model and training data.
    If a saved model exists and force is False, load the existing model instead.

    :param model: Pre-trained scikit-learn classifier to wrap with MAPIE.
    :param x: array-like, shape (n_samples, n_features)
        Training input samples.
    :param y: array-like, shape (n_samples,)
        Target values (class labels).
    :param save: bool, optional
        Whether to save the fitted MAPIE model. Default is True.
    :param force: bool, optional
        Whether to force retraining even if a saved model exists. Default is False.
    :return: mapie_clf
        A fitted MAPIE classifier object.
    """
    if not utils.check_if_filepath_exists(MAPIE_MODEL_FILENAME) or force:
        mapie_clf = MapieClassifier(estimator=model, cv='prefit', method='lac')
        mapie_clf.fit(x, y)
        if save:
            save_mapie_model(mapie_clf)
    else:
        print(
            f'Mapie Model has already been trained and already exists'
            f', it is located at: {MAPIE_MODEL_FILENAME}'
        )
        mapie_clf = get_mapie_model()
    return mapie_clf


def conformal_prediction(mapie_clf, x, alpha=0.10):
    """
    Perform conformal prediction using a fitted MAPIE classifier.

    :param mapie_clf: Fitted MAPIE classifier object.
    :param x: array-like, shape (n_samples, n_features)
        Input samples to predict.
    :param alpha: float, optional
        Desired error rate for the prediction sets. Default is 0.10.
    :return: y_pred_label, y_predicted_sets, class_certainty
        y_pred_label: array of shape (n_samples,)
            Predicted class labels with -1 for uncertain cases.
        y_predicted_sets: array of shape (n_samples, n_classes)
            Binary matrix indicating the prediction set for each sample.
        class_certainty: array of shape (n_samples,)
            1 if model is confident in the prediction, 0 if uncertain.
    """
    y_pred_label, y_predicted_sets = mapie_clf.predict(x, alpha=alpha)
    # For each sample, returns 1 if there is certainty regarding
    # the class with respect to the model and the allowed error rate, 0 if not.
    class_certainty = 1 - (y_predicted_sets.sum(axis=1).reshape(1, -1)[0] - 1)
    y_pred_label = pd.Series(y_pred_label)
    y_pred_label.loc[class_certainty==0] = -1
    y_pred_label = y_pred_label.to_numpy()
    return y_pred_label, y_predicted_sets, class_certainty


def conformal_prediction_projected(mapie_clf, x, alpha=0.10):
    """
    Get predicted class labels from conformal prediction, ignoring prediction sets and certainty.

    :param mapie_clf: Fitted MAPIE classifier object.
    :param x: array-like, shape (n_samples, n_features)
        Input samples to predict.
    :param alpha: float, optional
        Desired error rate for the prediction sets. Default is 0.10.
    :return: y_pred_label: array
        Predicted class labels with -1 for uncertain cases.
    """
    y_pred_label, _, _ = conformal_prediction(mapie_clf, x, alpha=alpha)
    return y_pred_label


def get_predicted_labels_and_set(mapie_clf, x, alpha=0.10, sets=['test'], ):
    """
    Apply conformal prediction using a fitted MAPIE classifier on specified dataset splits.

    :param mapie_clf: Fitted MAPIE classifier object.
    :param x: dict
        Dictionary containing input features for each dataset split (e.g., 'test', 'cal', 'OOT').
    :param alpha: float, optional
        Desired error rate for the prediction sets. Default is 0.10.
    :param sets: list of str, optional
        List of dataset split names to process. Default is ['test'].
    :return: results: dict
        Dictionary with each split containing:
            - 'y_pred_label': array of predicted class labels (with -1 for uncertain cases)
            - 'y_predicted_sets': array of prediction sets
            - 'class_certainty': array of certainty values per instance
    """
    print(f'The allowed error rate is {alpha}. This means a coverage of {1-alpha}.')
    results = {}
    for key in sets:
        x_key = x[key]
        results[key] = {}
        y_pred_label, y_predicted_sets, class_certainty = conformal_prediction(mapie_clf, x_key, alpha=alpha)
        results[key]['y_pred_label'] = y_pred_label
        results[key]['y_predicted_sets'] = y_predicted_sets
        results[key]['class_certainty'] = class_certainty
    return results


def integrate_certainty_into_info_df(info, results, sets=['test']):
    """
    Integrate class certainty values into the corresponding info DataFrames.

    :param info: dict
        Dictionary of DataFrames for each dataset split (e.g., 'test', 'train', 'OOT').
    :param results: dict
        Dictionary containing result DataFrames with 'class_certainty' column for each split.
    :param sets: list of str, optional
        List of keys in `info` and `results` to update. Default is ['test'].
    :return: info: dict
        Updated `info` dictionary with 'class_certainty' column added to each specified split.
    """
    for key in sets:
        info_key = info[key]
        results_key = results[key]
        info_key['class_certainty'] = results_key['class_certainty']
    return info


def get_certainty_frame(info, sets=['test']):
    """
    Compute the average class certainty for prediction score quantiles in each specified dataset split.

    :param info: dict
        Dictionary containing dataframes for each dataset split (e.g., 'train', 'test', 'OOT'),
        each with a 'pred' column (prediction score) and 'class_certainty' column.
    :param sets: list of str, optional
        List of keys in `info` to compute certainty frames for. Default is ['test'].
    :return: certainty_frames: dict
        Dictionary with the same keys as `sets`, containing pandas Series where
        index represents quantile intervals and values are average class certainty.
    """
    certainty_frames = {}
    for key in sets:
        info_key = info[key]
        q = pd.qcut(info_key['pred'], q=1000)
        info_key['q'] = q
        certainty_frame = info_key.groupby('q').class_certainty.mean()
        certainty_frames[key] = certainty_frame
    return certainty_frames


def get_certainty_quantiles_min_max(certainty_frame):
    """
    Get the minimum and maximum quantile bounds where certainty is less than 1.

    :param certainty_frame: pandas Series
        Series indexed by quantile intervals with certainty values.
    :return: q_min, q_max
        q_min: float
            Left bound of the first interval with certainty < 1.
        q_max: float
            Right bound of the last interval with certainty < 1.
    """
    possible_qs = certainty_frame[certainty_frame < 1]
    q_min = 0
    q_max = 0
    if possible_qs.shape[0]:
        q_min = possible_qs.index.min().left
        q_max = possible_qs.index.max().right
        return q_min, q_max
    print('Certainty frame does not exist. '
          'Change (lower) alpha in conformal prediction to get results.')
    return q_min, q_max

def train_conformal_predictor(show=True):
    """
    Train a conformal predictor using a MAPIE classifier and evaluate it on an out-of-time (OOT) dataset.

    This function fits the conformal model, integrates class certainty information,
    calculates certainty quantiles, and optionally visualizes the score distribution.

    :param show: bool, optional
        Whether to plot the general score distribution with quantiles. Default is True.
    :return: None
    """
    model = training_model.get_model()
    x, info = understanding_model.get_final_datasets()
    mapie_clf = fit_mapie_classifier(model, x['cal'], info['cal']['y'])
    mapie_results = get_predicted_labels_and_set(mapie_clf, x, alpha=0.05, sets=['OOT'])
    info = integrate_certainty_into_info_df(info, mapie_results, sets=['OOT'])
    certainty_frames = get_certainty_frame(info, sets=['OOT'])
    q_min, q_max = get_certainty_quantiles_min_max(certainty_frames['OOT'])
    if show:
        understanding_model.plot_general_score_distribution_w_qs(info, sets=['test'], q_min=q_min, q_max=q_max)


if __name__ == "__main__":
    train_conformal_predictor(show=False)