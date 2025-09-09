import numpy as np
from torch.utils.data import Dataset


class SASDataset(Dataset):
    def __init__(self, data_root, data_txt, mode:str="train"):
        self.data_root = data_root
        self.mode = mode
        with open(data_txt, "r") as fp:
            self.case_list = fp.readlines()
        self.case_list = [case.strip() for case in self.case_list]

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
        vel = np.load(f"{self.data_root}/{case}/vp_model.npy").transpose(1, 0)
        if self.mode == "train":
            # Temporal flip augmentation
            if np.random.random() < 0.5:
                seis = seis[::-1, :, ::-1]
                vel = vel[:, ::-1]
                seis = seis.copy()
                vel = vel.copy()
        seis = seis * 1e4
        return dict(seis=seis, vel=vel, case_name=case)