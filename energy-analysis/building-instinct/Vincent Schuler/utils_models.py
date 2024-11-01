# ----------------------------------------------------------------------------
# USEFUL FUNCTIONS FOR MODELISATION

import numpy as np
from sklearn.metrics import f1_score

# ----------------------------------------------------------------------------

def get_n_splits(n_classes) :
    """
    Determines the number of splits (e.g., for cross-validation) based on the number of classes.

    Parameters:
    n_classes (int): The number of unique classes in the dataset.

    Returns:
    int: The recommended number of splits for cross-validation.
    """

    if n_classes <= 7 : return 10
    elif n_classes <= 9 : return 8
    elif n_classes <= 12 : return 7
    elif n_classes <= 15 : return 5
    else : return 4


# ----------------------------------------------------------------------------

import optuna
from warnings import filterwarnings

# Silent optuna trial results
optuna.logging.set_verbosity(optuna.logging.WARNING)
filterwarnings("ignore", category=DeprecationWarning)
filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)

def get_preds_from_thresholds(y_pred, labels, thresholds) :
    """
    Adjusts predictions based on specified thresholds, and assigns class labels accordingly.

    Parameters:
    y_pred (np.array): Array of predicted class probabilities, shape (n_samples, n_classes).
    labels (np.array): Array of unique class labels.
    thresholds (list): List of thresholds for each class, used to modify predictions.

    Returns:
    np.array: Final predicted class labels after applying thresholds.
    """
    
    # Evaluate diff between preds and thresholds + clip to avoid negative values (we keep only preds greater than thresholds)
    y_pred_custom = (y_pred - thresholds).clip(0, None)

    # Loop over each row in the predictions.
    y_pred_final = np.argmax(y_pred, axis=1)
    for row_idx in range(y_pred.shape[0]):
        # If no class probability exceeds the threshold, use the class with the max probability
        if y_pred_custom[row_idx].sum() == 0:
            y_pred_final[row_idx] = np.argmax(y_pred[row_idx])
        # Otherwise, use the class with the highest probability after threshold adjustment.
        else:
            y_pred_final[row_idx] = np.argmax(y_pred_custom[row_idx])

    # Convert predicted class indices to class labels.
    y_pred_final = labels[y_pred_final]
    
    # Return
    return y_pred_final


class OptunaRounder:
    """
    A class for optimizing thresholds of class predictions using Optuna to maximize the F1 score.

    Attributes:
    y_true (np.array): Ground truth class labels.
    y_pred (np.array): Predicted class probabilities, shape (n_samples, n_classes).
    class_names (list): List of class names.
    labels (np.array): Unique class labels (extracted from y_true).
    n_classes (int): Number of unique classes.

    Methods:
    __call__(trial): Suggests thresholds for each class and calculates the F1 score for evaluation.
    """


    def __init__(self, y_true, y_pred, class_names):
        """
        Initializes the OptunaRounder with the ground truth, predictions, and class names.

        Parameters:
        y_true (np.array): Ground truth class labels.
        y_pred (np.array): Predicted class probabilities, shape (n_samples, n_classes).
        class_names (list): List of class names.
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.class_names = class_names
        self.labels = np.unique(y_true)
        self.n_classes = len(class_names)
        
    def __call__(self, trial):
        """
        Executes a trial to suggest thresholds and computes the F1 score.

        Parameters:
        trial (optuna.Trial): Optuna trial object for hyperparameter tuning.

        Returns:
        float: Macro-averaged F1 score after applying optimized thresholds.
        """
        
        # Thresholds
        thresholds = [trial.suggest_float(f"w{i}", 0, 1) for i in range(self.n_classes)]
    
        # Get preds based on custom thresholds
        y_pred_final = get_preds_from_thresholds(self.y_pred, self.labels, thresholds)

        # Return f1 score        
        return f1_score(self.y_true, y_pred_final, average='macro')