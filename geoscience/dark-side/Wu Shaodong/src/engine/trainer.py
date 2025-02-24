import os
import time
import logging
import torch
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.cuda.amp import autocast, GradScaler
from mmengine.config import Config
from mmengine.config import ConfigDict
from segmentation_models_pytorch.losses import DiceLoss, SoftBCEWithLogitsLoss, FocalLoss
from timm.optim import Lookahead
import albumentations as A
from models import build_model
from data import build_dataset, train_data_trainsform, infer_data_transform
from utils import get_logger, get_dice
from mmengine.runner.checkpoint import load_state_dict


def set_seed(seed=3407):  # torch.manual_seed(3407) is all u need
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.deterministic = False
        #torch.backends.cudnn.benchmark = True

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

class Trainer:
    def __init__(self,
                 config_path: str):
        
        self.loss_factor = 0.5
        self.loss2_factor = 0.5
        self.loss_aux_factor = 0.3
        self._get_config(config_path)
        set_seed(self.training_config.seed)
        self._init_training_config(self.training_config)
        self._create_model(self.model_config)
        self._create_dataloader(self.dataset_config)
        self._create_optimizer(self.optimizer_config)
        self._create_scheduler(self.scheduler_config, self.optimizer)
        self._create_criterion(self.loss_config, self.loss2_config, self.aux_loss_config)
        self._create_scaler()
        self.infer_data_transform = A.Compose(infer_data_transform)

    def _get_config(self, config_path):
        self.config             = Config.fromfile(config_path)
        self.model_config       = self.config.model
        print(self.model_config)
        self.loss_config        = self.config.loss
        self.aux_loss_config    = self.config.loss_aux if "loss_aux" in self.config else None
        self.loss2_config       = self.config.loss2 if "loss2" in self.config else None
        self.dataset_config     = self.config.dataset
        self.optimizer_config   = self.config.optimizer
        self.scheduler_config   = self.config.scheduler
        self.training_config    = self.config.train_cfg
        self.val_cfg            = self.config.val_cfg
        self.device             = self.config.device

    def _init_training_config(self, training_config: ConfigDict):
        self.use_amp        = self.training_config.use_amp
        self.train_bs       = self.training_config.batch_size
        self.num_workers    = self.training_config.num_workers
        self.save_path      = training_config.save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.train_epochs   = training_config.train_epochs
        self.cur_iteration  = 0
        self.cur_epoch      = 0
        self.best_score     = 0
        self.val_iterations = training_config.val_iterations
        self.log_step       = training_config.log_step
        self.exp_name       = self.save_path.split("/")[-2] if self.save_path.split("/")[-1] == "" else self.save_path.split("/")[-1]
        self.logger         = get_logger(f"Dark-Side-Seg_{self.exp_name}",
                                        f"{self.save_path}/log.file",
                                        logging.INFO,
                                        'w')
        self.logger.info(self.config)
    def _create_model(self, model_cfg: ConfigDict):
        self.model = build_model(model_cfg)
        if (self.model_config.pretrained_path != None):
            if self.model_config.type=="MMSegModel":
                # self.model.load_state_dict(, strict=False)
                load_state_dict(self.model, 
                                torch.load(self.model_config.pretrained_path)["state_dict"], 
                                strict=False, 
                                logger=None)
            else:
                st_dict = torch.load(self.model_config.pretrained_path)
                st_dict.pop("conv_stem.weight")
                self.model.encoder.model.load_state_dict(st_dict, strict=False)
        self.model.to(self.device)

    def _create_dataloader(self, dataset_cfg: ConfigDict):
        self.train_dataset  = build_dataset(dataset_cfg.train)
        self.val_dataset    = build_dataset(dataset_cfg.val)
        self.train_loader   = DataLoader(self.train_dataset,
                                         batch_size        = self.train_bs,
                                         shuffle           = True,
                                         num_workers       = self.num_workers, 
                                         pin_memory        = True, 
                                         drop_last         = False,
                                         worker_init_fn    = worker_init_fn
                             )
        self.val_loader     = DataLoader(self.val_dataset,
                                         batch_size        = 1,
                                         shuffle           = False,
                                         num_workers       = 1, 
                                         pin_memory        = True, 
                                         drop_last         = False,
                                         worker_init_fn    = worker_init_fn,
                             )

    def _create_optimizer(self, optimizer_cfg: ConfigDict):
        optimizer_name = optimizer_cfg.type
        optimizer_cfg.pop("type")
        if optimizer_name == "Adam":
            self.optimizer = Adam(self.model.parameters(), **optimizer_cfg)
        elif optimizer_name == "SGD":
            self.optimizer = SGD(self.model.parameters(), **optimizer_cfg)
        elif optimizer_name == "AdamW":
            self.optimizer = AdamW(self.model.parameters(), **optimizer_cfg)
        else:
            raise NotImplementedError(f"Only Adam, SGD and AdamW supported now, \
                                      {optimizer_name} is not supported now.")

    def _create_scheduler(self, 
                          scheduler_cfg: ConfigDict,
                          optimizer: torch.optim.Optimizer):
        scheduler_name = scheduler_cfg.type
        scheduler_cfg.pop("type")
        if scheduler_name == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                        **scheduler_cfg
            )
        elif scheduler_name == "StepLR":
            self.scheduler =  torch.optim.lr_scheduler.StepLR(optimizer,
                                                              **scheduler_cfg
            )
        elif scheduler_name == "PolyLR":
            self.scheduler =  torch.optim.lr_scheduler.LambdaLR(optimizer, 
                                                                lr_lambda=lambda epoch: (1 - epoch / self.train_epochs) ** 0.9
            )
        else:
            raise ValueError(f"{scheduler_name} is not implemented")


    def _create_criterion(self, loss_cfg: ConfigDict, loss2_cfg: ConfigDict, aux_loss_cfg: ConfigDict):
        loss_name = loss_cfg.type
        loss_cfg.pop("type")
        if loss_name == "SMP_BCE":
            self.criterion = SoftBCEWithLogitsLoss(**loss_cfg)
        elif loss_name == "SMP_DICE":
            self.criterion = DiceLoss(**loss_cfg)
        elif loss_name == "SMP_Focal":
            self.criterion = FocalLoss(**loss_cfg)
        else:
            raise NotImplementedError(f"Only SMP_BCE and SMP_DICE supported now, \
                                      {loss_name} is not supported now.")

        if aux_loss_cfg is not None:
            loss_name = aux_loss_cfg.type
            aux_loss_cfg.pop("type")
            if loss_name == "SMP_BCE":
                self.aux_criterion = SoftBCEWithLogitsLoss(**aux_loss_cfg)
            elif loss_name == "SMP_DICE":
                self.aux_criterion = DiceLoss(**aux_loss_cfg)
            elif loss_name == "SMP_Focal":
                self.aux_criterion = FocalLoss(**aux_loss_cfg)
            else:
                raise NotImplementedError(f"Only SMP_BCE and SMP_DICE supported now, \
                                        {loss_name} is not supported now.")
        if loss2_cfg is not None:
            loss_name = loss2_cfg.type
            loss2_cfg.pop("type")
            if loss_name == "SMP_BCE":
                self.criterion2 = SoftBCEWithLogitsLoss(**loss2_cfg)
            elif loss_name == "SMP_DICE":
                self.criterion2 = DiceLoss(**loss2_cfg)
            elif loss_name == "SMP_Focal":
                self.criterion2 = FocalLoss(**loss2_cfg)
            else:
                raise NotImplementedError(f"Only SMP_BCE and SMP_DICE supported now, \
                                        {loss_name} is not supported now.")            

            
    def _create_scaler(self,):
        self.scaler = GradScaler(enabled = self.use_amp)

    def _val_preprocess(self, inputs: np.ndarray):
        if inputs.ndim == 2:
            inp = inputs[..., None]
            inp = self.infer_data_transform(image=inp)["image"]
            inp = inp[None, ...]
            in_data = inp.to(self.device)
        elif inputs.ndim == 3:
            # HWC to CHW
            inp = self.infer_data_transform(image=inputs)["image"]
            inp = inp[None, ...]
            in_data = inp.to(self.device)
        return in_data
    
    def _val_postprocess(self, inputs: torch.Tensor, ori_shape):
        return inputs
    
    def _inference_slice(self, inputs: np.ndarray):
        ori_shape = inputs.shape
        # infer normal
        in_data = self._val_preprocess(inputs)
        seg_logits = self.model(in_data)["seg_logits"].squeeze()
        pred_logits = self._val_postprocess(seg_logits, ori_shape)
        return pred_logits

    def _inference_volume(self, inputs):
        h, w, z = inputs.shape
        result = torch.zeros(inputs.shape, dtype=torch.float32, device=self.device)
        for i in range(h):
            if i == 0: inp = inputs[:, [i, i, i+1], :]
            elif i == (h - 1): inp = inputs[:, [i-1, i, i], :]
            else: inp = inputs[:, [i-1, i, i + 1], :]
            inp = inp.transpose(0, 2, 1)
            slice_msk = self._inference_slice(inp)
            result[:, i, :] += slice_msk
        # argmax
        volume_seg_logits = result.sigmoid().cpu().numpy()
        volume_mask_pred = (volume_seg_logits >
                            self.val_cfg.pos_thr).astype(np.uint8)
        del result
        return volume_mask_pred

    def _val_once(self,):
        global_scores = []
        self.model.eval()
        with torch.no_grad():
            for step, data in enumerate(self.val_loader):
                volume = data["volume"].cpu().numpy().squeeze()
                labels = data["label"].cpu().numpy().squeeze()
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    volume_mask_pred = self._inference_volume(volume)
                    sc = get_dice(labels, volume_mask_pred)
                    global_scores.append(sc)
        sub_score = sum(global_scores) / len(global_scores)
        self.model.train()
        return sub_score
    
    def _get_learning_rate(self,):
        return self.optimizer.param_groups[0]['lr']

    def _train_one_epoch(self,):
        self.model.train()
        for step, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            with autocast(self.use_amp):
                images = data["slice"].to(self.device)
                labels = data["label"].to(self.device)
                outputs = self.model(images)
                preds = outputs["seg_logits"]
                loss_decode = self.criterion(preds, labels)

                loss2 = self.criterion2(preds, labels)
                log_loss_decode = loss_decode.item()
                log_loss2_decode = loss2.item()
                log_loss_aux = 0
                if "seg_logits_aux" in outputs:
                    aux_preds = outputs["seg_logits_aux"]
                    loss_aux = self.aux_criterion(aux_preds, labels)
                    log_loss_aux = loss_aux.item()
                    total_loss = self.loss_factor * loss_decode + self.loss_aux_factor * loss_aux
                else:
                    total_loss = self.loss_factor * loss_decode + self.loss2_factor * loss2
            self.scaler.scale(total_loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.cur_iteration += 1

            if 0 == (self.cur_iteration % self.log_step):
                lr = self._get_learning_rate()
                self.logger.info(f"cur_epoch {self.cur_epoch}  cur_iteration {self.cur_iteration}  cur_lr {lr}  total_loss {total_loss.item()}  loss_decode {self.loss_factor * log_loss_decode}  loss2 {self.loss2_factor * log_loss2_decode}  loss_aux {self.loss_aux_factor * log_loss_aux}  best_val_score {self.best_score}")
            # val
            if 0 == (self.cur_iteration % self.val_iterations):
                start_time = time.time()
                score = self._val_once()
                elapsed = time.time() - start_time
                # save best model
                torch.save(self.model.state_dict(), 
                     f"{self.save_path}/model_iteration{self.cur_iteration}.pth")
                if score >= self.best_score:
                    torch.save(self.model.state_dict(), 
                               f"{self.save_path}/best_model_score{score}.pth")
                self.best_score = max(self.best_score, score)
                self.logger.info(f"\n\nValidate Result:")
                self.logger.info(f"cur_epoch {self.cur_epoch}  cur_iteration {self.cur_iteration}  cur_score {score}  best_val_score {self.best_score}  spend_time: {elapsed:.0f}s")
        self.scheduler.step()

    def train(self,):
        for cur_epoch in range(self.train_epochs):
            self.cur_epoch += 1
            start_time = time.time()
            self._train_one_epoch()