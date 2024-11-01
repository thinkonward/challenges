# Essential Libraries for Data Manipulation
import os
import numpy as np
import pandas as pd

# Libraries for advanced data structures
from collections import defaultdict, Counter

# Fast Fourier Transform from SciPy
from scipy.fft import fft

# Progress bar for loops
from tqdm import tqdm

# Joblib for saving models or large data files efficiently
import joblib

# Libraries for Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn Model Selection
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

# LightGBM libraries for building models
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor

# Scikit-learn Metrics for Model Evaluation
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Suppressing warnings
import warnings
warnings.filterwarnings("ignore")

# Pandas display options
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
