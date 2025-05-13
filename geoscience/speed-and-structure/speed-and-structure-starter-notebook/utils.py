import numpy as np
from anytree import Node, RenderTree
from typing import Dict, List


def dummy_prediction(input_data, output_shape):
    """
    Generates a dummy prediction for velocity model.

    Parameters:
    input_data: list of 5 2D receiver data np.ndarrays

    Returns:
    prediction: A 2D np.ndarray with velocity model prediction
    """
    prediction = np.random.choice(np.linspace(1, 5, 20, dtype=np.float64), size=output_shape)
    
    return prediction


def create_submission(sample_id: str, prediction: np.ndarray, submission_path: str):
    """Function to create submission file out of one test prediction at time

    Parameters:
        sample_id: filename
        prediction: 2D np.ndarray of predicted velocity model
        submission_path: path to save submission

    Returns:
        None
    """

    try:
        submission = dict(np.load(submission_path))
    except:
        submission = dict({})

    submission.update(dict({sample_id: prediction}))

    np.savez(submission_path, **submission)

    return

def sketch_directory_tree():
    # Create nodes for the directory structure
    root = Node("current_directory")
    starter_notebook = Node("starter_notebook.ipynb", parent=root)
    utils = Node("utils.py", parent=root)
    data = Node("data", parent=root)
    train = Node("train", parent=data)
    test = Node("test", parent=data)
    
    # Render the directory tree
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))
    return


def calculate_mape(vel_gt: np.ndarray, vel_pred: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Percentage Error between two 2D NumPy arrays
    of the SAME size.

    The error for each element (i,j) is calculated as:
        abs(vel_gt[i,j] - vel_pred[i,j]) / (vel_gt[i,j] + EPSILON)
    EPSILON is added to the denominator to prevent division by zero.

    Args:
        vel_gt: A 2D NumPy array representing the ground truth values.
        vel_pred: A 2D NumPy array representing the predicted values.
                  Must have the same shape as vel_gt.

    Returns:
        The mean of the normalized absolute errors.
    """
    # Define a small epsilon to prevent division by zero
    EPSILON = 1e-8

    # Calculate the absolute difference element-wise
    absolute_difference = np.abs(vel_gt - vel_pred)

    # Calculate the normalized error element-wise
    # Add EPSILON to vel_gt in the denominator to avoid division by zero
    # and to handle cases where vel_gt[i,j] is 0.
    # If vel_gt[i,j] is 0 and vel_pred[i,j] is also 0, error is 0.
    # If vel_gt[i,j] is 0 and vel_pred[i,j] is non-zero, error is large.
    normalized_error = absolute_difference / (np.abs(vel_gt) + EPSILON)

    # Calculate the mean of the normalized errors
    mape = np.mean(normalized_error)

    return mape

def calculate_score(answerkey_file: str, submission_file: str) -> float:
    """
    Calculates the average mape across multiple samples.

    This function iterates through pairs of groundtruth and predicted velocity models
    stored in npz files. It calculates MAPE for each pair
    using the `calculate_mape` function.

    The final score is the average MAPE over all samples.

    Args:
        answerkey_file: an npz file where keys are string sample identifiers (e.g., IDs)
                   and values are 2D NumPy arrays representing the ground truth
                   velocity models.
        submission_file: an npz file with string keys, ideally matching `answerkey`,
                    where values are 2D NumPy arrays representing the predicted
                    velocity models.

    Returns:
        The average MAPE score across all samples.
    """
    answerkey =  dict(np.load(answerkey_file))
    submission =  dict(np.load(submission_file))
    
    mape_samples: List[float] = [] # List to store MAPE for samples

    # Iterate through the keys (sample IDs) in the answer key
    for key in answerkey.keys():

        # Retrieve the ground truth and predicted arrays for the current key
        vel_gt = answerkey[key]
        vel_pred = submission[key]

        # Calculate the MAPE for the current sample using the helper function
        mape_score = calculate_mape(vel_gt, vel_pred)
        mape_samples.append(mape_score)

    return float(np.mean(mape_samples))