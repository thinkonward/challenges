import os
import numpy as np
import time

import torch
from torch import nn, optim
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage

from hydra.utils import instantiate

from accelerate import Accelerator

import logging
from obsidian.logging_helpers import helpers as logging_helpers

class Tester:
    def __init__(self, cfg, logger, model, valid_loader):
        self.cfg = cfg
        self.model = model
        self.valid_loader = valid_loader
        
        self.logger = logger
        self.log_stream = logging_helpers.LogStream()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.logger.logger.addHandler(self.log_handler)

        self.global_step = 0

        mp = 'bf16' if self.cfg.training.amp else 'no'
        self.accelerator = Accelerator(mixed_precision=mp)

        self.model.to(self.accelerator.device)

        self.valid_loader, self.model = self.accelerator.prepare(
            self.valid_loader, self.model)

    def load_checkpoint(self):
        state_dict = torch.load(f'{self.cfg.ckpt}', map_location='cpu')
        self.model.module.load_state_dict(state_dict)

    def post_process_to_submission(self, prediction_dict, save_path):
        submission = dict({})
        
        for key in prediction_dict:
            base = '_'.join(key.split('_')[:-2])
            ix = key.split('_')[-2]
            sl = (int(key.split('_')[-1]) // 75) - 1

            new_key = f"{base}_gt.npy-{ix}_{sl}"

            output = prediction_dict[key]['output'][0]
            output = (output*255).astype(np.uint8)

            submission.update({new_key: output})
        
        np.savez(save_path, **submission)
                
    def test(self):
        save_pred = False
        if 'save' in self.cfg or self.cfg.mode=='predict':
            save_pred = True
            prediction_dict = {}

        flip_tta = False
        if 'flip_tta' in self.cfg:
            flip_tta = True

        self.model.eval()

        cache_dir = f'{self.cfg.output_dir}/cache/'
        os.makedirs(cache_dir, exist_ok=1)

        Metrics = [instantiate(self.cfg[key]) for key in self.cfg.keys() if 'metric_' in key]
        
        if self.accelerator.is_main_process:
            bar = tqdm(self.valid_loader)
        else:
            bar = self.valid_loader

        for step, batch in enumerate(bar):
            step += 1
            images = batch['noised'].to(self.accelerator.device)
            if self.cfg.mode=='test':
                targets = batch['denoised'].to(self.accelerator.device)
            ids = batch['ids']

            with torch.no_grad():
                logits, logits2 = self.model(images)

            if flip_tta:
                images_flip = batch['noised_flip'].to(self.accelerator.device)
                
                logits_flip, logits2 = self.model(images_flip)
                
                logits_flip = torch.flip(logits_flip, (2,))
                logits = (logits + logits_flip) / 2
            
            logits = self.accelerator.gather(logits)
        
            outputs = logits.float().detach().cpu().numpy()

            if self.cfg.mode=='test':
                targets = self.accelerator.gather(targets)
                targets = targets.float().detach().cpu().numpy()

            self.accelerator.wait_for_everyone()
            rank = self.accelerator.process_index
            np.save(f'{cache_dir}/ids_{rank}.npy', ids)
            self.accelerator.wait_for_everyone()
            ids = np.concatenate([np.load(f"{cache_dir}/ids_{_}.npy") for _ in range(self.cfg.n_gpus)])
            self.accelerator.wait_for_everyone()
            
            if self.cfg.mode=='test':
                for Metric in Metrics:
                    Metric.accumulate(outputs, targets)

            if save_pred or self.cfg.mode=='predict':
                if self.cfg.mode=='test':
                    for iid, output, target in zip(ids, outputs, targets):
                        prediction_dict[iid] = {'output': output, 'target': target}
                else:
                    for iid, output in zip(ids, outputs):
                        prediction_dict[iid] = {'output': output}

        if self.cfg.mode=='test':
            metric_to_score = {}
            for Metric in Metrics:
                score = Metric.calculate_score()
                metric_to_score[Metric.name()] = score
        
        self.logger.info(f"Config: {self.cfg.cfg}")
        self.logger.info(f"Checkpoint: {self.cfg.ckpt}")
        if self.cfg.mode=='test':
            self.logger.info(f"Metrics:")
            for name in metric_to_score:
                self.logger.info(f"{name}: {metric_to_score[name]}")

            chosen_metric = self.cfg.validation.save_metric
            score = metric_to_score[chosen_metric]
        else:
            score = -1

        if save_pred or self.cfg.mode=='predict':
            base = '/'.join(self.cfg.ckpt.split('/')[:-1])
            if self.cfg.mode=='test':
                save_path = f'{base}/prediction_dict.npy'
            else:
                save_path = f'{base}/test_prediction_dict.npy'
            np.save(save_path, prediction_dict)
            self.logger.info(f"Saved Predictions at: {save_path}")
        
        if self.cfg.mode=='predict':
            base = '/'.join(self.cfg.ckpt.split('/')[:-1])
            try:
                name = self.cfg.sub_name
            except:
                name = 'sample_sub.npz'
            save_path = f"{base}/{name}"
            self.post_process_to_submission(prediction_dict, save_path)
            self.logger.info(f"Saved Submission at: {save_path}")

        return score