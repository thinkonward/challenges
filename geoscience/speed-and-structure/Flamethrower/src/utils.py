import os
import gc
import sys
import csv
from glob import glob
import math
import shutil
import pickle
import random
import logging
import argparse
import warnings
import pprint
import numpy as np
from anytree import Node, RenderTree
from typing import Dict, List

from copy import deepcopy
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import yaml
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

from tqdm import tqdm
from scipy import stats as st
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

import torch



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


def mape_loss(y_pred, y_true):
    EPSILON = torch.tensor(1e-8, device=y_true.device, dtype=y_true.dtype)
    return torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + EPSILON)))



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count
            
            
def save_checkpoint(ema_model, output_folder, name, fold=None):
    checkpoint = {
                  "ema_model": ema_model.module.state_dict(),
                  }
    
    if not os.path.exists(os.path.join(output_folder, name)):
        os.makedirs(os.path.join(output_folder, name))


    torch.save(checkpoint, f"{output_folder}/{name}/checkpoint-{name}-trained-fold{fold}.pt")
    
    
    
    
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
        
    def __call__(self, val_score):

        score = val_score

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    

    
# Extracting train data
def create_ids(split):
    # List to store the names of subfolders (sample IDs)
    sample_paths = glob(f"{split}/*")
    # extract the name of samples, i.e. sample IDs
    sample_ids = [path.split("/")[-1] for path in sample_paths]
    return sample_ids


def create_df(data_path):
    data_ids = create_ids(data_path)
    df = pd.DataFrame(columns = ['image_id'])
    df['image_id'] = data_ids
    df['image_path'] = df['image_id'].apply(lambda x : os.path.join(data_path, x))
    return df


# Clustering Velocity Samples for representative data splitting
def extract_features(model):
    """
    Extract meaningful statistical features from the 2D velocity model array.
    
    Features include:
    - Mean velocity along depth (per row)
    - Std deviation along depth (per row)
    - Kurtosis along depth (per row)
    - Skewness along depth (per row)
    
    - Mean velocity along horizontal (per column)
    - Std deviation along horizontal (per column)
    - Kurtosis along horizontal (per column)
    - Skewness along horizontal (per column)
    
    Returns a 1D feature vector.
    """
    # Along depth (axis=1) features per row
    min_depth = np.min(model, axis=1)
    max_depth = np.max(model, axis=1)
    mean_depth = np.mean(model, axis=1)
    std_depth = np.std(model, axis=1)
    

    # Along horizontal (axis=0) features per column
    min_horz = np.min(model, axis=0)
    max_horz = np.max(model, axis=0)
    mean_horz = np.mean(model, axis=0)
    std_horz = np.std(model, axis=0)

    # Concatenate all features into a single vector
    features = np.concatenate([
        min_depth, max_depth, mean_depth, std_depth,
        min_horz, max_horz, mean_horz, std_horz,
    ])

    return features

    
def cluster_velocity_models(paths, seed=42):
    """
    Clusters 2D velocity models using meaningful statistics extracted along both axes with automatic cluster selection.

    Parameters:
        paths (list of str): List of paths to folders containing 'vp_model.npy'.
        n_components (int): Number of PCA components for dimensionality reduction.
        max_clusters (int): Maximum number of clusters to consider for KMeans.
        visualize (bool): If True, visualize clusters in 2D using t-SNE.
        seed (int): Random seed for reproducibility.

    Returns:
        best_labels (np.ndarray): Cluster labels for each model using the optimal number of clusters.
        X_reduced (np.ndarray): Reduced PCA features for each model.
        best_k (int): Optimal number of clusters.
    """
    # Step 1: Load models and extract features
    velocity_models = [np.load(os.path.join(path, "vp_model.npy")) for path in paths]
    features = [extract_features(model) for model in velocity_models]
    X_reduced = np.array(features)

    kmeans = KMeans(n_clusters=6, random_state=seed)
    labels = kmeans.fit_predict(X_reduced)
    

    return kmeans, labels


def predict_cluster_velocity_models(kmeans, paths, seed=42):
    """
    Clusters 2D velocity models using meaningful statistics extracted along both axes with automatic cluster selection.

    Parameters:
        paths (list of str): List of paths to folders containing 'vp_model.npy'.
        n_components (int): Number of PCA components for dimensionality reduction.
        max_clusters (int): Maximum number of clusters to consider for KMeans.
        visualize (bool): If True, visualize clusters in 2D using t-SNE.
        seed (int): Random seed for reproducibility.

    Returns:
        best_labels (np.ndarray): Cluster labels for each model using the optimal number of clusters.
        X_reduced (np.ndarray): Reduced PCA features for each model.
        best_k (int): Optimal number of clusters.
    """
    # Step 1: Load models and extract features
    velocity_models = [np.load(os.path.join(path, "vp_model.npy")) for path in paths]
    features = [extract_features(model) for model in velocity_models]
    X_reduced = np.array(features)

    labels = kmeans.predict(X_reduced)
    

    return labels


def visualise_velocity_group(df, seismic_group, num_samples=4):
    """
    Visualises velocity models (vp_model.npy) for multiple samples in a given seismic group.
    
    Args:
        df (pd.DataFrame): DataFrame containing at least 'image_path' and 'seismic_group' columns.
        seismic_group (int or str): The group to filter by.
        num_samples (int): Number of samples to display.
    """
    # Filter DataFrame for the selected group
    group_df = df[df['seismic_group'] == seismic_group]

    if group_df.empty:
        raise ValueError(f"No samples found for seismic group {seismic_group}")

    # Randomly select samples
    selected_paths = random.sample(list(group_df['image_path']), min(num_samples, len(group_df)))

    # Create subplots
    fig, axes = plt.subplots(1, len(selected_paths), figsize=(4 * len(selected_paths), 5))

    if len(selected_paths) == 1:
        axes = [axes]  # Make iterable if only 1 sample

    for ax, sample_path in zip(axes, selected_paths):
        # Load vp_model.npy (velocity model)
        vp_data = np.load(os.path.join(sample_path, "vp_model.npy"))

        ax.imshow(vp_data.T, cmap="jet", aspect="auto")
        ax.set_title(os.path.basename(sample_path))
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
    
def plot_seismic_group_distribution(df):
    """
    Plots the distribution of seismic groups from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'seismic_group' column.
    """
    if 'seismic_group' not in df.columns:
        raise ValueError("DataFrame must contain a 'seismic_group' column.")
    
    counts = df['seismic_group'].value_counts().sort_index()
    
    plt.figure(figsize=(8, 5))
    counts.plot(kind='bar', color='skyblue', edgecolor='black')
    
    plt.title("Seismic Group Distribution")
    plt.xlabel("Seismic Group")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
