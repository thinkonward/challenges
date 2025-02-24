import torch
import torch.nn.functional as F
import numpy as np
from mmengine.config import Config
from mmengine.config import ConfigDict
from models import build_model
from data import rescale_volume, infer_data_transform
from utils import get_dice
import gc
import glob
import albumentations as A

class Validator:
    def __init__(self,
                 config_path: str,
                 model_path: str,
                 root_dir: str,
                 txt_file: str):
        
        self.root_dir          = root_dir
        self.txt_file          = txt_file
        self._get_data_list()
        self._init_config(config_path, model_path)
        self._init_model(self.model_config)
        self.infer_data_transform = A.Compose(infer_data_transform)

    def _get_data_list(self,):
        self.volume_ids = np.loadtxt(self.txt_file, dtype=str).tolist()
        self.data_file_paths = []
        self.label_file_paths = []
        for volume_id in self.volume_ids:
            self.data_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/seismicCubes_RFC_fullstack_*.npy")[0])
            self.label_file_paths.append(glob.glob(f"{self.root_dir}/{volume_id}/fault_segments_*.npy")[0])

    def _init_config(self, config_path, model_path):
        self.config             = Config.fromfile(config_path)
        self.model_config       = self.config.model
        self.model_path         = model_path
        self.device             = self.config.device
        self.val_cfg           = self.config.val_cfg
        self.use_amp            = self.config.train_cfg.use_amp

    def _init_model(self, model_cfg: ConfigDict):
        self.model = build_model(model_cfg)
        print("loading ckpt")
        self.model.load_state_dict(torch.load(self.model_path), strict=True)
        self.model.to(self.device)
        self.model.eval()
        print("model eval")

    def _preprocess(self, inputs: np.ndarray):
        # 交换维度+to tensor + pad to 32
        if self.val_cfg.pad_to_32:
            h1, w1 = inputs.shape
            h2, w2 = 32 * (h1 // 32), 32 * (w1 // 32)
            padding_w = w2 - w1
            padding_h = h2 - h1
            inputs = np.pad(inputs, pad_width=[(0,padding_h),(0,padding_w)], mode='edge')
            inp = inputs[..., None]
            inp = self.infer_data_transform(image=inp)["image"]
            inp = inp[None, ...]
            in_data = inp.to(self.device)
        if inputs.ndim == 2:
            inp = inputs[..., None]
            inp = self.infer_data_transform(image=inp)["image"]
            inp = inp[None, ...]
            in_data = inp.to(self.device)
        # 针对通道不为1的情况，还有别的地方需要修改
        elif inputs.ndim == 3:
            # HWC to CHW
            inputs = inputs.transpose(2, 0, 1)
            inp = inputs[:, None, ...]
            # in_data = torch.tensor(inp, dtype=torch.float32, device=self.device)
            inp = torch.tensor(inp, dtype=torch.float32, device=self.device)
            in_data = inp.to(self.device)
        return in_data
    
    def _postprocess(self, inputs: torch.Tensor, ori_shape):
        # move padding, 需要在cfg中指定pad_to_32参数， 只有smp_model需要设置为True
        if self.val_cfg.pad_to_32:
            inputs = inputs[:ori_shape[0], :ori_shape[1]]
        inputs = inputs.sigmoid()
        return inputs
        

    def _inference_slice(self, inputs: np.ndarray):
        slice_mask = torch.zeros(inputs.shape, dtype=torch.float32, device=self.device)
        cnt = torch.zeros(inputs.shape, dtype=torch.float32, device=self.device)
        ori_shape = inputs.shape
        # infer normal
        if "normal" in self.val_cfg.test_flip_tta:
            in_data = self._preprocess(inputs)
            seg_logits = self.model(in_data)["seg_logits"].squeeze()
            # print(seg_logits.max())
            mask = self._postprocess(seg_logits, ori_shape)
            slice_mask += mask
            cnt += 1.0

        # hflip tta
        if "h_flip_tta" in self.val_cfg.test_flip_tta:
            inp = inputs.copy()[:, ::-1]
            in_data = self._preprocess(inp)
            seg_logits = self.model(in_data)["seg_logits"].squeeze()
            mask = self._postprocess(seg_logits, ori_shape)
            # mask = mask[:, ::-1]
            mask = mask.flip(-1)
            slice_mask += mask
            cnt += 1.0
        
        # vflip tta
        if "v_flip_tta" in self.val_cfg.test_flip_tta:
            inp = inputs.copy()[::-1, :]
            in_data = self._preprocess(inp)
            seg_logits = self.model(in_data)["seg_logits"].squeeze()
            mask = self._postprocess(seg_logits, ori_shape)
            # mask = mask[::-1, :]
            mask = mask.flip(0)
            slice_mask += mask
            cnt += 1.0
        slice_mask /= cnt
        torch.cuda.empty_cache()
        gc.collect()
        return slice_mask

    def _inference_volume(self, inputs):
        # 所有训练数据归一化已经做了
        h, w, z = inputs.shape
        result = torch.zeros(inputs.shape, dtype=torch.float32, device=self.device)
        cnt = torch.zeros(inputs.shape, dtype=torch.float32, device=self.device)
        if "h" in self.val_cfg.test_dim_tta:
            for i in range(h):
                inp = inputs[:, i, :]
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        slice_msk = self._inference_slice(inp)
                result[:, i, :] += slice_msk
                cnt[:, i, :] += 1.0

        if "w" in self.val_cfg.test_dim_tta:
            for i in range(w):
                inp = inputs[i, :, :]
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        slice_msk = self._inference_slice(inp)
                result[i, :, :] += slice_msk
                cnt[i, :, :] += 1.0

        if "z" in self.val_cfg.test_dim_tta:
            for i in range(z):
                inp = inputs[:, :, i]
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        slice_msk = self._inference_slice(inp)
                result[:, :, i] += slice_msk
                cnt[:, :, i] += 1.0
        
        result /= cnt

        # argmax
        volume_seg_logits = result.cpu().numpy()
        volume_mask_pred = (volume_seg_logits >
                            self.val_cfg.pos_thr).astype(np.uint8)
        # del result
        # del cnt
        print(volume_mask_pred.shape, volume_mask_pred.dtype, volume_mask_pred.sum())
        return volume_mask_pred
    
    def evaluate(self,):
        global_scores = []
        self.model.eval()
        with torch.no_grad():
            for step, (volume, label) in enumerate(zip(self.data_file_paths,
                                                        self.label_file_paths)):
                volume = np.load(volume, allow_pickle=True, mmap_mode="r+")
                label = np.load(label, allow_pickle=True, mmap_mode="r+")
                with torch.cuda.amp.autocast(enabled=self.use_amp):  # torch 自动混合精度推理,后面可尝试apex
                    volume_mask_pred = self._inference_volume(volume)
                    sc = get_dice(label, volume_mask_pred)
                    global_scores.append(sc)
                    print(sc)
        sub_score = sum(global_scores) / len(global_scores)

        return sub_score
        
    
        

    


        
