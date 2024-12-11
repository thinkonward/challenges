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

class Trainer:
    def __init__(self, cfg, run, logger, model, train_loader, valid_loader, criterion, optimizer, scheduler):
        self.cfg = cfg
        self.run = run
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.logger = logger
        self.log_stream = logging_helpers.LogStream()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.logger.logger.addHandler(self.log_handler)

        self.global_step = 0

        mp = 'bf16' if self.cfg.training.amp else 'no'
        self.accelerator = Accelerator(mixed_precision=mp)

        self.model.to(self.accelerator.device)

        self.train_loader, self.valid_loader, self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.train_loader, self.valid_loader, self.model, self.optimizer, self.scheduler)
        
        ema_decay_per_iter = self.cfg.training.ema.ema_decay_per_epoch ** (1 / self.cfg.dataloader.steps_per_epoch)
        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay_per_iter)

    def save_checkpoint(self, postfix='', save_ema=True):
        filename = f"{self.cfg.output_dir}/{self.cfg.data_info.fold}"
        if self.accelerator.is_main_process:
            if save_ema:
                with self.ema.average_parameters():
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    self.accelerator.save(unwrapped_model.state_dict(), 
                                      filename+postfix+'_EMA.pth')
            else:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                self.accelerator.save(unwrapped_model.state_dict(),
                                    filename+postfix+'.pth')
    
    def train_one_epoch(self):
        self.model.train()
        
        if self.accelerator.is_main_process:
            bar = tqdm(self.train_loader, bar_format='{desc}', leave=False)
        else:
            bar = self.train_loader

        running_loss = 0.
        start = time.time()

        for step, batch in enumerate(bar):
            step += 1

            images = batch['noised'].to(self.accelerator.device)
            targets = batch['denoised'].to(self.accelerator.device)

            logits, logits2 = self.model(images)
            
            loss = self.criterion(logits.float(), targets.float())

            running_loss += (loss - running_loss) * (1 / step)
            
            self.accelerator.backward(loss / self.cfg.training.accum_steps)

            if step % self.cfg.training.accum_steps == 0 or step == len(bar):
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.ema.update()
                self.global_step += 1

            end = time.time()

            avg_batch_time = (end-start) / step
            running_eta = avg_batch_time * (len(self.train_loader) - step)
            
            lr = "{:2e}".format(self.optimizer.param_groups[0]['lr'])

            if self.accelerator.is_main_process:
                if self.cfg.neptune_run:
                    self.run["train/loss"].append(loss)

                descp_dict = {
                    "mode": "train",
                    "epoch": self.epoch,
                    "iter": f"{step}/{len(bar)}",
                    "eta": int(running_eta),
                    "lr": lr,
                    "loss": running_loss.item(),
                }
                
                description = str(descp_dict)
                
                bar.set_description(description)
                
                if step%self.cfg.log_every==0 or step==len(self.train_loader):
                    bar.close()
                    self.logger.info(f"{description}")
                    self.log_stream.reset()
                    bar = tqdm(self.train_loader, total=len(self.train_loader)-step, 
                                bar_format='{desc}', leave=False)

            self.accelerator.wait_for_everyone()

            #if step==10: break
        
        self.save_checkpoint(postfix='')

    def validate(self):
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
            targets = batch['denoised'].to(self.accelerator.device)
            ids = batch['ids']

            with torch.no_grad():
                with self.ema.average_parameters():
                    logits, logits2 = self.model(images)
            
            logits = self.accelerator.gather(logits)
            targets = self.accelerator.gather(targets)
        
            outputs = logits.float().detach().cpu().numpy()
            targets = targets.float().detach().cpu().numpy()

            self.accelerator.wait_for_everyone()
            rank = self.accelerator.process_index
            np.save(f'{cache_dir}/ids_{rank}.npy', ids)
            self.accelerator.wait_for_everyone()
            ids = np.concatenate([np.load(f"{cache_dir}/ids_{_}.npy") for _ in range(self.cfg.n_gpus)])
            self.accelerator.wait_for_everyone()
            
            for Metric in Metrics:
                Metric.accumulate(outputs, targets)

            #if step==10: break
        
        metric_to_score = {}
        for Metric in Metrics:
            score = Metric.calculate_score()
            metric_to_score[Metric.name()] = score
        
        self.logger.info(f"Epoch {self.epoch} Metrics:")
        for name in metric_to_score:
            if self.accelerator.is_main_process and self.cfg.neptune_run:
                    self.run[f"valid/{name}"].append(metric_to_score[name])

            self.logger.info(f"{name}: {metric_to_score[name]}")

        chosen_metric = self.cfg.validation.save_metric
        score = metric_to_score[chosen_metric]

        return score

    def fit(self):
        save_best = self.cfg.validation.save_best
        if save_best=='min': best_score = float('inf')
        if save_best=='max': best_score = -float('inf')

        for epoch in range(self.cfg.training.num_epochs):
            self.epoch = epoch+1
            self.logger.info(f"[F{self.cfg.data_info.fold}] Training Epoch {self.epoch}")
            self.train_one_epoch()
            self.accelerator.wait_for_everyone()
            self.logger.info(f"[F{self.cfg.data_info.fold}] Validating Epoch {self.epoch}")
            score = self.validate()
            self.accelerator.wait_for_everyone()

            if save_best=='min':
                if score <= best_score:
                    self.logger.info(f"Saving Epoch {self.epoch} as best with new low score of {score:.4f}")
                    self.save_checkpoint(postfix='_best', save_ema=True)
                    best_score = score
            if save_best=='max':
                if score >= best_score:
                    self.logger.info(f"Saving Epoch {self.epoch} as best with new high score of {score:.4f}")
                    self.save_checkpoint(postfix='_best', save_ema=True)
                    best_score = score
            
            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                print()
                print()
                print()

            #break