import os
import numpy as np
from sklearn.model_selection import KFold


if __name__ == "__main__":


    if True:
        SRC_TRAIN_DATA_ROOT = r"/root/autodl-tmp/Speed_and_Structure/data/train/"
        KFOLD_TXT_SAVE_ROOT = r"/root/autodl-tmp/Speed_and_Structure/data/train_txt/"
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


    if False:
        SRC_TRAIN_DATA_ROOT = r"/root/autodl-tmp/Speed_and_Structure/data/train/"
        KFOLD_TXT_SAVE_ROOT = r"/root/autodl-tmp/Speed_and_Structure/data/train_txt/"

        all_case = os.listdir(SRC_TRAIN_DATA_ROOT)
        all_cur_val_case_txt = f"{KFOLD_TXT_SAVE_ROOT}/val_f0.txt"
        fp = open(all_cur_val_case_txt, "r")
        all_cur_val_case = fp.readlines()
        all_cur_val_case = [case.strip() for case in all_cur_val_case]
        fp.close()

        new_train_case_list = []
        new_train_case_txt = f"{KFOLD_TXT_SAVE_ROOT}/new_train_f0.txt"
        for case in all_case:
            if case not in all_cur_val_case:
                new_train_case_list.append(case)
        with open(new_train_case_txt, "w") as fp:
            for case in new_train_case_list:
                fp.write(f"{case}\n")
