
import torch


# This class is defined for wandb compatibility
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
cfg = dotdict()

cfg.fold_info_file = "./fold_info_all_with_synth.csv"
cfg.load_model_path = None # in case you want to initialize with a pretrained model

# --- Dataset Params ---
cfg.target_len = 2048 
# These values are calculated from the a subset of training data
cfg.x_mean = -5.26e-6
cfg.x_std = 0.0155
cfg.y_min = 1.5
cfg.y_max = 4.5
cfg.y_mean = 2.78
cfg.y_std = 0.93
cfg.y_median = 2.93
cfg.holdout_idx = 3 # which fold to hold out for validation. This is the fold column from fold_info csv file
cfg.one_channel = True # True: Have the seismic readings side by side in one channel with shape (B, 1, T, 31*5). False: Treating them as separate channels with shape (B, 5, T, 31)

# --- Normalization Params ---
cfg.y_norm = False # z-norm the labels
cfg.y_min_max_norm = True # min-max norm the labels with a final activation of sigmoid
cfg.x_norm = True # z-norm the input

# --- Augmentation Params ---
cfg.horizontal_flip = True # Whether to randomly flip the input and output horizontally for data augmentation
cfg.hflip_prob = 0.5  # Probability of horizontal flip
cfg.horizontal_tta = True # Whether to apply horizontal TTA

# --- Model params ---
cfg.ema = True # Model EMA (Exponential Moving Average)
cfg.ema_decay = 0.99
cfg.backbone = "caformer_b36.sail_in22k_ft_in1k" 
cfg.backbone_pretrained = True
cfg.fuse_ch = 64 
cfg.norm_layer = "identity"
cfg.dropout = -1

# --- Training params ---
cfg.batch_size = 1 # 8
cfg.n_epochs = 20 # 10
cfg.num_workers = 2
cfg.lr = 1e-4
cfg.weight_decay = 1e-5
cfg.scheduler = "cosine"
cfg.eta_min = 0 # 1e-6
cfg.loss = "mape"
cfg.optimizer = "adamw"
cfg.grad_accum = 8
cfg.grad_clip = 1.0
cfg.print_every = 10 * cfg.grad_accum

# --- Misc ---
cfg.seed = 42 + cfg.holdout_idx
cfg.use_cuda = torch.cuda.is_available()
cfg.device = torch.device("cuda" if cfg.use_cuda else "cpu")
