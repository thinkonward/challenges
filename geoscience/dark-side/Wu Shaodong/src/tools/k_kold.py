import os
import numpy as np
from sklearn.model_selection import KFold

SRC_TRAIN_DATA_ROOT = r"/root/autodl-tmp/Dark_side_of_the_volume/data/train_data/"
KFOLD_TXT_SAVE_ROOT = r"/root/autodl-tmp/Dark_side_of_the_volume/data/train_txt/"

NUM_FOLD = 5
RANDOM_SEED=123
os.makedirs(KFOLD_TXT_SAVE_ROOT, exist_ok=True)

all_train_case = np.asarray(os.listdir(SRC_TRAIN_DATA_ROOT))
kf = KFold(n_splits=NUM_FOLD, random_state=RANDOM_SEED, shuffle=True)
for i, (train_index, valid_index) in enumerate(kf.split(all_train_case)):
    train_case = all_train_case[train_index]
    valid_case = all_train_case[valid_index]

    np.savetxt(f"{KFOLD_TXT_SAVE_ROOT}/train_f{i}.txt", train_case, fmt="%s")
    np.savetxt(f"{KFOLD_TXT_SAVE_ROOT}/val_f{i}.txt", valid_case, fmt="%s")