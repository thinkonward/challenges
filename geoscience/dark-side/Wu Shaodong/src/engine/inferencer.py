import torch
import torch.nn.functional as F
import numpy as np
from mmengine.config import Config
from mmengine.config import ConfigDict
import albumentations as A
from models import build_model
from data import rescale_volume, infer_data_transform
import gc

class Inferencer25D:
    def __init__(self,
                 config_path: str,
                 model_path: str):
        self._init_config(config_path, model_path)
        self._init_model(self.model_config)
        self.infer_data_transform = A.Compose(infer_data_transform)

    def _init_config(self, config_path, model_path):
        self.config             = Config.fromfile(config_path)
        self.model_config       = self.config.model
        self.model_path         = model_path
        self.device             = self.config.device
        self.test_cfg           = self.config.test_cfg
        self.use_amp            = self.config.train_cfg.use_amp

    def _init_model(self, model_cfg: ConfigDict):
        self.model = build_model(model_cfg)
        print("loading ckpt")
        self.model.load_state_dict(torch.load(self.model_path), strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("model eval")

    def _preprocess(self, inputs: np.ndarray):
        if inputs.ndim == 2:
            inp = inputs[..., None]
            inp = self.infer_data_transform(image=inp)["image"]
            inp = inp[None, ...]
            in_data = inp.to(self.device)
        # 针对通道不为1的情况，还有别的地方需要修改
        elif inputs.ndim == 3:
            # HWC to CHW
            # HWC to CHW
            inp = self.infer_data_transform(image=inputs)["image"]
            inp = inp[None, ...]
            in_data = inp.to(self.device)
        return in_data
    
    def _postprocess(self, inputs: torch.Tensor, ori_shape):
        inputs = inputs.sigmoid()
        return inputs
        

    def _inference_slice(self, inputs: np.ndarray):
        slice_mask = torch.zeros(inputs.shape[:2], dtype=torch.float32, device=self.device)
        cnt = torch.zeros(inputs.shape[:2], dtype=torch.float32, device=self.device)
        ori_shape = inputs.shape

        # infer normal
        if "normal" in self.test_cfg.test_flip_tta:
            in_data = self._preprocess(inputs)
            seg_logits = self.model(in_data)["seg_logits"].squeeze()
            mask = self._postprocess(seg_logits, ori_shape)
            slice_mask += mask
            cnt += 1.0
        # hflip tta
        if "h_flip_tta" in self.test_cfg.test_flip_tta:
            inp = inputs.copy()[:, ::-1, :]
            in_data = self._preprocess(inp)
            seg_logits = self.model(in_data)["seg_logits"].squeeze()
            mask = self._postprocess(seg_logits, ori_shape)
            mask = mask.flip(-1)
            slice_mask += mask
            cnt += 1.0
        # vflip tta
        if "v_flip_tta" in self.test_cfg.test_flip_tta:
            inp = inputs.copy()[::-1, :, :]
            in_data = self._preprocess(inp)
            seg_logits = self.model(in_data)["seg_logits"].squeeze()
            mask = self._postprocess(seg_logits, ori_shape)
            mask = mask.flip(0)
            slice_mask += mask
            cnt += 1.0
        slice_mask /= cnt
        torch.cuda.empty_cache()
        gc.collect()
        return slice_mask

    def __call__(self, inputs):
        # 数据归一化在内部做
        inputs = rescale_volume(inputs)
        inputs = inputs.astype(np.float32)
        h, w, z = inputs.shape
        result = torch.zeros(inputs.shape, dtype=torch.float32, device=self.device)
        cnt = torch.zeros(inputs.shape, dtype=torch.float32, device=self.device)
        if "h" in self.test_cfg.test_dim_tta:
            for i in range(h):
                if i == 0: inp = inputs[:, [i, i, i+1], :]
                elif i == (h - 1): inp = inputs[:, [i-1, i, i], :]
                else: inp = inputs[:, [i-1, i, i + 1], :]
                inp = inp.transpose(0, 2, 1)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        slice_msk = self._inference_slice(inp)
                result[:, i, :] += slice_msk
                cnt[:, i, :] += 1.0

        if "w" in self.test_cfg.test_dim_tta:
            for i in range(w):
                inp = inputs[i, :, :]
                if i == 0: inp = inputs[[i, i, i+1], :, :]
                elif i == (h - 1): inp = inputs[[i-1, i, i], :, :]
                else: inp = inputs[[i-1, i, i + 1], :, :]
                inp = inp.transpose(1, 2, 0)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        slice_msk = self._inference_slice(inp)
                result[i, :, :] += slice_msk
                cnt[i, :, :] += 1.0
        
        result /= cnt
        # argmax
        volume_seg_logits = result.cpu().numpy()
        volume_mask_pred = (volume_seg_logits >
                            self.test_cfg.pos_thr).astype(np.uint8)
        # del result
        # del cnt
        return volume_mask_pred


class EnsembleInferencer25D:
    def __init__(self,
                 config_path: str,
                 model_paths: str):
        self._init_config(config_path, model_paths)
        self._init_model(self.model_config)
        self.infer_data_transform = A.Compose(infer_data_transform)

    def _init_config(self, config_path, model_paths):
        self.config             = Config.fromfile(config_path)
        self.model_config       = self.config.model
        self.model_paths         = model_paths
        self.device             = self.config.device
        self.test_cfg           = self.config.test_cfg
        self.use_amp            = self.config.train_cfg.use_amp

    def _init_model(self, model_cfg: ConfigDict):
        self.models = [build_model(md_cfg) for md_cfg in model_cfg]
        print("loading ckpt")
        for i in range(len(self.models)):
            self.models[i].load_state_dict(torch.load(self.model_paths[i]), strict=True)
            self.models[i].to(self.device)
            self.models[i].eval()
        print("model eval")

    def _preprocess(self, inputs: np.ndarray):
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
    
    def _postprocess(self, inputs: torch.Tensor, ori_shape):
        inputs = inputs.sigmoid()
        return inputs
        

    def _inference_slice(self, inputs: np.ndarray):
        slice_mask = torch.zeros(inputs.shape[:2], dtype=torch.float32, device=self.device)
        cnt = torch.zeros(inputs.shape[:2], dtype=torch.float32, device=self.device)
        ori_shape = inputs.shape

        # infer normal
        if "normal" in self.test_cfg.test_flip_tta:
            in_data = self._preprocess(inputs)
            seg_logits = [self._postprocess(md(in_data)["seg_logits"].squeeze(), 
                                            ori_shape) for md in self.models]
            mask = torch.stack(seg_logits, dim=0).mean(dim=0)
            slice_mask += mask
            cnt += 1.0
        # hflip tta
        if "h_flip_tta" in self.test_cfg.test_flip_tta:
            inp = inputs.copy()[:, ::-1, :]
            in_data = self._preprocess(inp)
            seg_logits = [self._postprocess(md(in_data)["seg_logits"].squeeze(), 
                                            ori_shape) for md in self.models]
            mask = torch.stack(seg_logits, dim=0).mean(dim=0)
            mask = mask.flip(-1)
            slice_mask += mask
            cnt += 1.0
        # vflip tta
        if "v_flip_tta" in self.test_cfg.test_flip_tta:
            inp = inputs.copy()[::-1, :, :]
            in_data = self._preprocess(inp)
            seg_logits = [self._postprocess(md(in_data)["seg_logits"].squeeze(), 
                                            ori_shape) for md in self.models]
            mask = torch.stack(seg_logits, dim=0).mean(dim=0)
            mask = mask.flip(0)
            slice_mask += mask
            cnt += 1.0
        slice_mask /= cnt
        torch.cuda.empty_cache()
        gc.collect()
        return slice_mask

    def __call__(self, inputs):
        # 数据归一化在内部做
        inputs = rescale_volume(inputs)
        inputs = inputs.astype(np.float32)
        h, w, z = inputs.shape
        result = torch.zeros(inputs.shape, dtype=torch.float32, device=self.device)
        cnt = torch.zeros(inputs.shape, dtype=torch.float32, device=self.device)
        if "h" in self.test_cfg.test_dim_tta:
            for i in range(h):
                if i == 0: inp = inputs[:, [i, i, i+1], :]
                elif i == (h - 1): inp = inputs[:, [i-1, i, i], :]
                else: inp = inputs[:, [i-1, i, i + 1], :]
                inp = inp.transpose(0, 2, 1)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        slice_msk = self._inference_slice(inp)
                result[:, i, :] += slice_msk
                cnt[:, i, :] += 1.0

        if "w" in self.test_cfg.test_dim_tta:
            for i in range(w):
                inp = inputs[i, :, :]
                if i == 0: inp = inputs[[i, i, i+1], :, :]
                elif i == (h - 1): inp = inputs[[i-1, i, i], :, :]
                else: inp = inputs[[i-1, i, i + 1], :, :]
                inp = inp.transpose(1, 2, 0)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        slice_msk = self._inference_slice(inp)
                result[i, :, :] += slice_msk
                cnt[i, :, :] += 1.0
        
        result /= cnt
        # argmax
        volume_seg_logits = result.cpu().numpy()
        volume_mask_pred = (volume_seg_logits >
                            self.test_cfg.pos_thr).astype(np.uint8)
        # del result
        # del cnt
        return volume_mask_pred
