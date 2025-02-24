from engine import Inferencer25D, EnsembleInferencer25D
from data import rescale_volume
import glob
import os
import numpy as np


def create_submission(
    sample_id: str, prediction: np.ndarray, submission_path: str, append: bool = True
):
    """Function to create submission file out of one test prediction at time

    Parameters:
        sample_id: id of survey used for perdiction
        prediction: binary 3D np.ndarray of predicted faults
        submission_path: path to save submission
        append: whether to append prediction to existing .npz or create new one

    Returns:
        None
    """

    if append:
        try:
            submission = dict(np.load(submission_path))
        except:
            print("File not found, new submission will be created.")
            submission = dict({})
    else:
        submission = dict({})

    # Positive value coordinates
    coordinates = np.stack(np.where(prediction == 1)).T
    coordinates = coordinates.astype(np.uint16)

    submission.update(dict([[sample_id, coordinates]]))

    np.savez(submission_path, **submission)

if __name__ == "__main__":
    cfg_path = r"./src/configs/infer_config.py"
    ckpt = ["./my_checkpoints/effv2s_f0_cv09017_lb09415.pth", "./my_checkpoints/effv2s_f1_cv08901_lb09445.pth"]
    test_data_root = r"./data/test_data/"
    submission_file = r"./final_submission.npz"
    inferencer = EnsembleInferencer25D(cfg_path, ckpt)
    for case in os.listdir(test_data_root):
        data = glob.glob(f"{test_data_root}/{case}/*.npy")
        npy = np.load(data[0], allow_pickle=True, mmap_mode="r+")
        res = inferencer(npy)
        create_submission(case, 
                          res, 
                          submission_file,
                          append=True
        )
        
