import os
import numpy as np
import argparse
import torch
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.deterministic = True
import numpy as np
from detectron2.config import LazyConfig, instantiate
from tqdm import tqdm



class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.case_list = os.listdir(self.data_root)

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        case = self.case_list[idx]
        seis = []
        for receiver_idx in ["1", "75", "150", "151", "225", "300"]:
            if receiver_idx == "151":
                dt = np.zeros_like(seis[-1])
                seis.append(dt)
            else:
                dt = np.load(f"{self.data_root}/{case}/receiver_data_src_{receiver_idx}.npy")
                seis.append(dt[:-1, :])
        seis = np.stack(seis, axis=0)
        seis = seis * 1e4
        return dict(seis=seis, case=case)


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


@torch.no_grad()
def do_test(input_root, submission_path, model_list):
    test_ds = TestDataset(input_root)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False)
    # model.eval()
    prog_bar = tqdm(test_loader)
    for batch in prog_bar:
        seis = batch["seis"].cuda()
        case = batch["case"][0]
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            pred_vel = sum([md(seis) for md in model_list]) / len(model_list)
            pred_vel = pred_vel.float()

            seis = seis.flip(dims=(1, -1))
            pred_vel_hflp = sum([md(seis) for md in model_list]) / len(model_list)
            pred_vel_hflp = pred_vel_hflp.flip(dims=(-1,)).float()

            pred_vel = (pred_vel + pred_vel_hflp) / 2.0

        pred_vel = pred_vel.cpu().numpy().astype(np.float64).squeeze().transpose(1, 0)
        create_submission(case, pred_vel, submission_path)


def main():
    data_root = r"./data/test_data/"
    submission_path = r"./final_submission.npz"
    configs = [
        # # tiny s_at_8 ohem2
        r"./src/configs/eva02_tiny_split_at_8.py",
        r"./src/configs/eva02_tiny_split_at_8.py",
        r"./src/configs/eva02_tiny_split_at_8.py",
        r"./src/configs/eva02_tiny_split_at_8.py",
        r"./src/configs/eva02_tiny_split_at_8.py",
        r"./src/configs/eva02_tiny_split_at_8.py",

        # # base s_at_6
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",
        r"./src/configs/eva02_base_split_at_6.py",

        # # base s_at_8
        r"./src/configs/eva02_base_split_at_8.py",
        r"./src/configs/eva02_base_split_at_8.py",
        r"./src/configs/eva02_base_split_at_8.py",
        r"./src/configs/eva02_base_split_at_8.py",
        r"./src/configs/eva02_base_split_at_8.py",
        r"./src/configs/eva02_base_split_at_8.py",

    ]
    
    ckpt_paths = [
        # tiny s_at_8 ohem2
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed2.pth",
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed5.pth",
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed9.pth",
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed10.pth",
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed14.pth",
        r"./my_checkpoints/eva02_tiny_s_at_8/best_score_seed3487.pth",

        # base s_at_6
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed4.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed6.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed7.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed9.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed10.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed11.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed12.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed13.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed14.pth",
        r"./my_checkpoints/eva02_base_s_at_6/best_score_seed15.pth",


        # base s_at_8
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed2.pth",
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed3.pth",
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed4.pth",
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed5.pth",
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed7.pth",
        r"./my_checkpoints/eva02_base_s_at_8/best_score_seed10.pth",

    ]


    def create_model(cfg, ck_path):
        cfg = LazyConfig.load(cfg)
        model = instantiate(cfg.model)
        load_from = ck_path
        state_dict = torch.load(load_from, map_location="cpu", weights_only=False)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {
            k.replace("_orig_mod.module.", "").replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }
        load_result = model.load_state_dict(state_dict, strict=False)
        print(load_result)
        model = model.cuda()
        model.eval()
        return model

    
    model_list = []
    for cfg, ck_path in zip(configs, ckpt_paths):
        model = create_model(cfg, ck_path)
        model_list.append(model)

    do_test(data_root, submission_path, model_list)


if __name__ == "__main__":
    main()