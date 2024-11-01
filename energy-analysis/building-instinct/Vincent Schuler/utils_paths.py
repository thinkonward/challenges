import os

# Directory containing files (train labels ...)
filedir   = "Files/"

# Directory where the features of each model are saved
feats_dir = "Features/"

# Directory where the thresholds are stored
thr_dir   = "Thresholds/"

# Directory where the models are stored
model_dir = "Trained_models/"

# Directory containing the individual train and test data (one file per buliding)
train_filedir = filedir + "building-instinct-train-data/"
test_filedir  = filedir + "building-instinct-test-data/"

"""
# Create directory if they do not exist yet
for directory in [filedir, feats_dir, thr_dir, model_dir,]:
    if not os.path.exists(directory):
        os.makedirs(directory)
"""