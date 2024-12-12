import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import copy

from torch.utils.data import Dataset, DataLoader

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import logging
import os
import shutil
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import neptune
import torch

from accelerate.logging import get_logger
from accelerate import Accelerator

#We kill the cache on every new run
import redis
redis_cache = redis.StrictRedis(host='localhost', port=6379, db=0)
redis_cache.flushall()

def seed_everything(seed=1234):
    import random, os, torch
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

accelerator = Accelerator()

#hydra_logger = logging.getLogger(__name__)
logger = get_logger(__name__)

from obsidian.trainer.trainer import Trainer

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    folds_to_run = copy.deepcopy(cfg.data_info.fold)

    done_copy = False

    first_model = None
    for fold in folds_to_run:

        accelerator = Accelerator()

        if accelerator.is_main_process:
            if not done_copy:
                shutil.copytree('obsidian', cfg.output_dir.replace('/temp', f'/obisidan_copy'), dirs_exist_ok=True)
                shutil.copy('main.py', cfg.output_dir.replace('/temp', f'/main.py'))
                done_copy = True

            if os.path.exists(cfg.output_dir):
                open(f"{cfg.output_dir}/main.log", "w").close()
                for root, dirs, files in os.walk(cfg.output_dir):
                    if '.hydra' in root: continue

                    for file in files:
                        if not file.endswith('.log'):
                            os.remove(os.path.join(root, file))
        accelerator.wait_for_everyone()

        cfg.data_info.fold = fold

        if accelerator.is_main_process and cfg.neptune_run:
            run = neptune.init_run(
                project="obsidian-1/obsidian",
                api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhMmZhMDRmYy1lOTNiLTRkMzItYjU1ZC0wOGEyNzFhN2Q0NjcifQ==",
            )

            #run["config_file"].upload("./configs/config.yaml")
            run['config'] = OmegaConf.to_yaml(cfg, resolve=True)

            for key in cfg.keys():
                run[f"parameters/{key}"] = cfg[key]

        else:
            run = {}

        seed_everything(cfg.seed)

        data = pd.read_csv(cfg.data_info.data_csv)
        
        train_data = data[data.fold!=fold]
        valid_data = data[data.fold==cfg.data_info.fold]
        
        extra = valid_data[:int(len(valid_data)//3 * 0.802)*3]
        train_data = pd.concat([train_data, extra])

        valid_data = valid_data[int(len(valid_data)//3 * 0.802)*3:]
        valid_data = valid_data[valid_data.idx.isin([75, 150, 225])].reset_index(drop=True)

        train_dataset = instantiate(cfg.dataset)(cfg, train_data, transforms=None, is_training=True)
        valid_dataset = instantiate(cfg.dataset)(cfg, valid_data, transforms=None, is_training=False)

        train_loader = DataLoader(train_dataset, batch_size=cfg.dataloader.train_batch_size, shuffle=True, 
                                num_workers=cfg.dataloader.num_workers)
        
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.dataloader.valid_batch_size, shuffle=False, 
                                num_workers=cfg.dataloader.num_workers)

        if first_model is None:
            model = instantiate(cfg.model)
            first_model = copy.deepcopy(model)
        else:
            model = copy.deepcopy(first_model)

        #st = model.state_dict()
        #print(list(st.keys())[:5])
        #logger.info(st['segmentor.encoder.model.bn1.weight'])
        #logger.info(st['segmentor.segmentation_head.0.weight'])
        
        criterion = instantiate(cfg.criterion)

        optimizer = instantiate(cfg.optimizer)(model.parameters())

        steps_per_epoch = int(len(train_data) / cfg.dataloader.train_batch_size / cfg.training.accum_steps)
        cfg.dataloader.steps_per_epoch = steps_per_epoch

        scheduler = instantiate(cfg.scheduler)(optimizer, 
                                num_training_steps=steps_per_epoch * cfg.training.num_epochs * cfg.training.scheduler.upscale_steps,
                                num_warmup_steps=steps_per_epoch * cfg.training.scheduler.num_warmup_epoch)
        
        #logger.info(model)

        trainer = Trainer(cfg, run, logger, model, train_loader, valid_loader, criterion, optimizer, scheduler)
        trainer.fit()

        del model
        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            if cfg.neptune_run:
                run.stop()

            #os.rename(cfg.output_dir, cfg.output_dir.replace('/temp', f'/F{fold}'))
            shutil.copytree(cfg.output_dir, cfg.output_dir.replace('/temp', f'/F{fold}'), dirs_exist_ok=True)

if __name__=='__main__':
    main()