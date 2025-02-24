import os,gc,sys,cv2,math,random, pickle, glob, time
from datetime import datetime, timedelta
import numpy as np, pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import h5py
import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch_3d as smp

import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
# from monai.networks.nets import UNETR, SwinUNETR

RANDOM_STATE = 41
FOLD = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
fix_seed(RANDOM_STATE)

DEPTH = 64
HEIGHT=WIDTH=128
PATCH_SIZE = (DEPTH,WIDTH,HEIGHT)
STRIDE_SIZE = (32,32,32)

dice_loss = smp.losses.DiceLoss(
                        mode="binary",
                        classes=None,
                        log_loss=False,             # Do not use log version of Dice loss
                        from_logits=True,           # Model outputs are raw logits
                        smooth=1e-5,                # A small smoothing factor for stability
                        ignore_index=None,          # Don't ignore any classes
                        eps=1e-7                    # Epsilon for numerical stability
                    )


class SeisDatasetVal(Dataset):
    def __init__(self, path, patch_size, stride, test_mode=False):
        """
        PyTorch Dataset for generating patch coordinates dynamically.

        Args:
            path (str): Volume path
            patch_size (tuple): The shape of the patches (Depth, Height, Width).
            stride (int): The stride to use for patch extraction.
        """

        #x = 3D volume of shape (Depth, Height, Width)
        #y = 3D volume of shape (Depth, Height, Width)
        with h5py.File(f'{path}/x.h5', 'r') as h5file:
            self.X = h5file['x'][:].transpose(2,0,1)

        if test_mode:
          self.Y = None
        else:
          self.Y = np.load(f'{path}/y.npz')['arr_0'].transpose(2,0,1)
        self.volume_shape = self.X.shape

        self.patch_size = patch_size
        self.stride = stride
        self.coords = self._generate_coords()

    def _generate_coords(self):
        """
        Generate patch starting coordinates dynamically based on volume shape, patch size, and stride.
        """
        depth, height, width = self.volume_shape
        p_d, p_h, p_w = self.patch_size
        coords = []

        for z in range(0, depth, self.stride[0]):
            for x in range(0, width, self.stride[1]):
                for y in range(0, height, self.stride[2]):
                    # Adjust starting indices near boundaries
                    if z + p_d > depth:
                        z = depth - p_d
                    if x + p_h > width:
                        x = width - p_h
                    if y + p_w > height:
                        y = height - p_w
                    coords.append((z, x, y))

        return coords

    def _reconstruct_volume_dynamic(self, patches):
        """
        Reconstructs the volume from patches with averaging for overlaps.

        Args:
            patches (list): List of patch arrays.

        Returns:
            np.ndarray: Reconstructed volume of shape `output_shape`.
        """
        output_shape = self.volume_shape
        reconstructed = np.zeros(output_shape, dtype=np.float32)
        patch_size = self.patch_size

        counts = np.zeros(output_shape, dtype=np.float32)
        for patch, (z, x, y) in zip(patches, self.coords):
            reconstructed[z:z+patch_size[0], x:x+patch_size[1], y:y+patch_size[2]] += patch
            counts[z:z+patch_size[0], x:x+patch_size[1], y:y+patch_size[2]] += 1
        reconstructed /= counts  # Average overlapping areas

        return reconstructed

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        """
        Dynamically extract a patch at runtime using the stored coordinates.
        """
        z, x, y = self.coords[idx]
        p_d, p_h, p_w = self.patch_size
        x_ = self.X[z:z + p_d, x:x + p_h, y:y + p_w]
        x_ = torch.tensor(x_,dtype=torch.float32).unsqueeze(0)

        if self.Y is not None:
          y_ = self.Y[z:z + p_d, x:x + p_h, y:y + p_w]
          y_ = torch.tensor(y_).int().unsqueeze(0)
          return x_,y_
        else:
          return x_


class PLModel(L.LightningModule):
    def __init__(self, learning_rate=1e-4,bs=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.bs = bs

        encoder_name = 'tu-ecaresnet26t.ra2_in1k'
        self.model = smp.Unet(
          encoder_name=encoder_name,
          in_channels=1,
          classes=1
          )

        self.start_time = datetime.now()
        self.elapsed_time = 0


        self.validation_outputs = []
        self.training_outputs = []

    def forward(self, x):
        logits = self.model(x)
        return logits



    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)
        loss = dice_loss(logits,y)

        self.training_outputs.append([loss.item()])

        self.log("train_loss",loss,prog_bar=True,batch_size=self.bs)
        return loss


    def on_train_epoch_end(self):

        end_time = datetime.now()
        elapsed_time = end_time - self.start_time
        self.elapsed_time = elapsed_time
        formatted_time = str(timedelta(seconds=elapsed_time.total_seconds()))

        current_epoch = self.current_epoch+1
        max_epochs = self.trainer.max_epochs
        eta = self.elapsed_time * (max_epochs - current_epoch)/current_epoch
        formatted_eta = str(timedelta(seconds=eta.total_seconds()))

        trn_dice = np.mean([o[0] for o in self.training_outputs]).round(4)
        trn_dice = (1-trn_dice).round(4)

        print(f'EP: {self.current_epoch},dice: {trn_dice}, Elapsed: {formatted_time} ETA: {formatted_eta}')

        self.log('trn_dice',trn_dice,on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        self.training_outputs = []


    def validation_step(self, batch, batch_idx):
        gc.collect()
        if batch_idx==0:
          assert len(self.validation_outputs)==0

        path = batch[0]
        ds = SeisDatasetVal(path,patch_size=PATCH_SIZE, stride=PATCH_SIZE)
        patches = np.empty([len(ds)] + list(PATCH_SIZE),dtype=np.float32)

        dl = DataLoader(ds, batch_size=self.bs*2, shuffle=False,drop_last=False)
        i0 = 0; i1=0
        for i,(x,y) in enumerate(dl):
          with torch.no_grad():
              logits = self(x.to(device)).cpu().numpy()

          i0 = i1
          i1 += len(logits)
          logits = logits.squeeze(1)
          patches[i0:i1] = logits


        logits = ds._reconstruct_volume_dynamic(patches)
        y = torch.tensor(ds.Y,dtype=torch.int)

        del patches, ds; gc.collect(); torch.cuda.empty_cache()


        logits = torch.tensor(logits,dtype=torch.float32)

        loss = dice_loss(logits,y).item()
        pred_mask = logits.sigmoid()


        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask, y, mode='binary',threshold=0.5)
        f1_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro').item()

        self.validation_outputs.append([loss,f1_score])

        self.log("val_loss", loss, prog_bar=True,batch_size=1)



    def on_validation_epoch_start(self):
        self.model.eval()
        torch.enable_grad(False)
        gc.collect(); torch.cuda.empty_cache()


    def on_validation_epoch_end(self):

        val_dice = np.mean([o[0] for o in self.validation_outputs]).round(4)
        val_f1 = np.mean([o[1] for o in self.validation_outputs]).round(4)
        val_dice = (1-val_dice).round(4)
        self.log('val_dice',val_dice,on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.log('val_f1',val_f1,on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        print('val_ep: ',self.current_epoch,val_dice,val_f1)

        self.validation_outputs = []

    def configure_optimizers(self):
        print('estimated_stepping_batches', self.trainer.estimated_stepping_batches)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate,weight_decay=1e-2)
        print(optimizer)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches
        )

        return {"optimizer": optimizer, "lr_scheduler": {'scheduler': scheduler, 'interval':'step'}}



def create_submission(
    sample_id: str, prediction: np.ndarray, submission_path: str, append: bool = True
):
    """Function to create submission file out of one test prediction at time

    Parameters:
        sample_id: id of survey used for perdiction
        prediction: binary 3D np.ndarray of predicted faults
        submission_path: path to save submission
        append: whether to append prediction to existing .npz or create new one

    Returns:
        None
    """

    if append:
        try:
            submission = dict(np.load(submission_path))
        except:
            print("File not found, new submission will be created.")
            submission = dict({})
    else:
        submission = dict({})

    # Positive value coordinates
    coordinates = np.stack(np.where(prediction == 1)).T
    coordinates = coordinates.astype(np.uint16)

    submission.update(dict([[sample_id, coordinates]]))

    np.savez(submission_path, **submission)
model_path = 'output/model-p1-epoch=999.ckpt'
model = PLModel.load_from_checkpoint(model_path)
model.to(device);model.eval();

paths_test = glob.glob('data/test_norm/*')
paths_test = sorted(paths_test)
print(len(paths_test))

for path in tqdm(paths_test):
    ds = SeisDatasetVal(path,patch_size=PATCH_SIZE, stride=STRIDE_SIZE, test_mode=True)

    patches = np.empty([len(ds)] + list(PATCH_SIZE),dtype=np.float32)
    print(len(ds))
    dl = DataLoader(ds, batch_size=8, shuffle=False,drop_last=False)
    i0 = 0; i1=0
    for i,x in enumerate(dl):
      with torch.inference_mode():
          logits = model(x.to(device)).cpu().numpy()

      i0 = i1
      i1 += len(logits)
      logits = logits.squeeze(1)
      patches[i0:i1] = logits


    logits = ds._reconstruct_volume_dynamic(patches)
    del patches, ds; gc.collect(); torch.cuda.empty_cache()

    logits = torch.tensor(logits,dtype=torch.float32)

    pred_mask = logits.sigmoid()
    pred_mask = pred_mask.numpy().transpose(1,2,0)
    pred_mask = pred_mask.round().astype(np.uint8)
    create_submission(path.split('/')[-1],pred_mask,f'submission.npz')

