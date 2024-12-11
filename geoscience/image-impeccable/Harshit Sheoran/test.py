import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

from torch.utils.data import Dataset, DataLoader

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import logging
import os
import shutil
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from accelerate.logging import get_logger
from accelerate import Accelerator

def seed_everything(seed=1234):
    import random, os, torch
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

#hydra_logger = logging.getLogger(__name__)
logger = get_logger(__name__)

accelerator = Accelerator()

from obsidian.tester.tester import Tester
from obsidian.datasets.ObsidianDataset import ImpeccableTestDataset
from obsidian import *

@hydra.main(config_path=None)
def main(cfg: DictConfig):

    if HydraConfig.initialized():
        hydra_run_dir = HydraConfig.get().run.dir
        os.chdir(hydra.utils.get_original_cwd())

    cfg.cfg = f"{os.getcwd()}/{cfg.cfg}"

    past_config = OmegaConf.load(cfg.cfg)
    cfg = OmegaConf.merge(past_config, cfg)

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    seed_everything(cfg.seed)

    data = pd.read_csv(cfg.data_info.data_csv)

    if cfg.mode=='test':
        valid_data = data[data.fold==cfg.data_info.fold]
        valid_data = valid_data[int(len(valid_data)//3 * 0.802)*3:]
        valid_data = valid_data[valid_data.idx.isin([75, 150, 225])].reset_index(drop=True)
    
    if cfg.mode=='predict':
        valid_data = data

    valid_dataset = instantiate(cfg.dataset)(cfg, valid_data, transforms=None, is_training=False)
    
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.dataloader.valid_batch_size, shuffle=False, 
                        num_workers=cfg.dataloader.num_workers)
    
    model = instantiate(cfg.model)
    
    #logger.info(model)

    tester = Tester(cfg, logger, model, valid_loader)
    tester.load_checkpoint()
    tester.test()

if __name__=='__main__':
    main()