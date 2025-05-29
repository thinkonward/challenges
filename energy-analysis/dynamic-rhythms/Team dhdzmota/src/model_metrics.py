import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
from sklearn.utils import resample

import src.utils as utils


RANDOM_SEED = utils.RANDOM_SEED


def compute_brierscoreloss(df):
    """
    Compute the Brier Score Loss for given predictions.

    :param df: pandas DataFrame
        DataFrame containing the true labels under 'y' and predicted probabilities under 'pred'.
    :return: score: float
        The computed Brier Score Loss. Returns NaN if computation is not possible.
    """
    try:
        score = brier_score_loss(df['y'], df['pred'])
    except ValueError:
        score = np.nan
    return score


def compute_aucroc(df):
    """
    Compute the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) for given predictions.

    :param df: pandas DataFrame
        DataFrame containing the true labels under 'y' and predicted probabilities under 'pred'.
    :return: score: float
        The computed AUC-ROC score. Returns NaN if computation is not possible.
    """
    try:
        score = roc_auc_score(df['y'], df['pred'])
    except ValueError:
        score = np.nan
    return score


def compute_aucpr(df):
    """
    Compute the Area Under the Precision-Recall Curve (AUCPR) for given predictions.

    :param df: pandas DataFrame
        DataFrame containing the true labels under 'y' and predicted scores under 'pred'.
    :return: score: float
        The computed AUCPR score. Returns NaN if computation is not possible.
    """
    try:
        precision, recall, _ = precision_recall_curve(df['y'], df['pred'])
        score = auc(recall, precision)
    except ValueError:
        score=np.nan
    return score


def bootstrap_func(df, function, n_bootstraps=50, ci=0.95, seed=RANDOM_SEED):
    """
    Perform bootstrapping on a metric function to estimate confidence intervals.

    :param df: pandas DataFrame
        Input DataFrame to sample from.
    :param function: callable
        Function that takes a DataFrame and returns a single numeric score.
    :param n_bootstraps: int, optional
        Number of bootstrap iterations. Default is 50.
    :param ci: float, optional
        Confidence interval level (e.g., 0.95 for 95% CI). Default is 0.95.
    :param seed: int, optional
        Random seed for reproducibility. Default is RANDOM_SEED.
    :return: response: pandas Series
        Series containing the mean score, lower error bound (`err_lo`), and upper error bound (`err_hi`).
    """
    bootstraped_scores = []
    rng = np.random.RandomState(seed)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, df.shape[0], df.shape[0])
        score = function(df.iloc[indices])
        bootstraped_scores.append(score)
    bs = pd.Series(bootstraped_scores)
    ci_upper = bs.quantile(ci)
    ci_lower = bs.quantile(1-ci)
    bs_mean = bs.mean()
    response = pd.Series({'mean': bs_mean, 'err_lo': bs_mean-ci_lower, 'err_hi': ci_upper-bs_mean})
    return response
