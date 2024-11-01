import numpy as np
import lightgbm as lgb

from typing import Dict, Tuple
from scipy.special import softmax
from sklearn.metrics import f1_score

def lgb_binary_f1_score(y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    y_pred = y_pred >= 0.5
    return 'f1', f1_score(y_true=y_true, y_pred=y_pred, average='macro'), True

def lgb_multi_f1_score(y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    y_pred = y_pred.argmax(axis=1)
    return 'f1', f1_score(y_true=y_true, y_pred=y_pred, average='macro'), True


def lgb_regression_f1_score(y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_true = data.get_label()
    y_pred = np.clip(np.round(y_pred), 0, None)
    return 'f1', f1_score(y_true=y_true, y_pred=y_pred, average='macro'), True

def softmax_by_target(
        target_position_list: list[int], y_pred: np.ndarray
    ) -> np.ndarray:
    """Compute the softmax for multiple target in isolation"""
    for target_position in target_position_list:
        y_pred[:, target_position] = softmax(y_pred[:, target_position], axis=1)
    
    return y_pred

def f1_by_target(target_position_list: list[int], y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
    f1_score_list = [
        f1_score(
            y_true=y_true[:, position_target].argmax(axis=1),
            y_pred=y_pred[:, position_target].argmax(axis=1),
            average='macro'
        )
        for position_target in target_position_list
    ]
    return f1_score_list