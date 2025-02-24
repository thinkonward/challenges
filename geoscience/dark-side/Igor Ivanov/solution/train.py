#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import GroupKFold
import segmentation_models_pytorch as smp
import albumentations as A
from argparse import ArgumentParser

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

parser = ArgumentParser()
parser.add_argument('--input_dir', default='train_img_axis_*', type=str, help='Directory with training images')
parser.add_argument('--output_dir', default='checkpoints', type=str, help='Directory to save model checkpoints')
parser.add_argument('--encoder_name', default='timm-efficientnet-b8', type=str, help='Encoder architecture')
parser.add_argument('--encoder_weights', default='imagenet', type=str, help='Encoder pretrained weights')
parser.add_argument('--in_channels', default=1, type=int, help='Number of input channels')
parser.add_argument('--classes', default=2, type=int, help='Number of classes')
parser.add_argument('--n_epochs', default=20, type=int, help='Number of epochs to train')
parser.add_argument('--batch_size', default=5, type=int, help='Batch size')
parser.add_argument('--accum', default=8, type=int, help='Number of steps for gradient accumulation')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--vertical', default=0, type=int, choices=[0, 1], help='Image orientation. 0 - horizontal 300x1259, 1 - vertical 1259x300')
parser.add_argument('--aug', default=2, type=int, choices=[0, 1, 2], help='Type of augmentation. 0 - no transformation, 1 - vertical flip, 2 - vertical and horizontal flip')
parser.add_argument('--num_workers', default=os.cpu_count(), type=int, help='Number of workers')
parser.add_argument('--use_amp', default=1, type=int, choices=[0, 1], help='Whether to use auto mixed precision')
parser.add_argument('--ckpt', default=None, type=str, help='Path to checkpoint to continue training from')
parser.add_argument('--initial_fold', default=0, type=int, help='Initial fold index (0 to 4)')
parser.add_argument('--final_fold', default=1, type=int, help='Final fold index (1 to 5)')
parser.add_argument('--reduce_p', default=1, type=int, help='Patience for learning rate reduction')
parser.add_argument('--reduce_f', default=0.5, type=float, help='Factor for learning rate reduction')
parser.add_argument('--reduce_mode', default='min', type=str, help='Mode (min/max) for learning rate reduction')
parser.add_argument('--save_seconds', default=3600, type=int, help='Save ckpt each n seconds')
parser.add_argument('--device', default='cuda:0', type=str, help='Device')
args = parser.parse_args() # pass empty list to run in notebook using default arg values: parser.parse_args([])
for a in [a for a in vars(args) if '__' not in a]: print('%-25s %s' % (a, vars(args)[a]))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class ImageDataset(torch.utils.data.Dataset):
    """
    Image dataset.

    Parameters:
    df : pd.DataFrame
        DataFrame with "image" and "label" columns 
        representing paths to each image and corresponding label
    transforms : 
        Albumentation transformations
    vertical : bool
        Image orientation. 0 - horizontal 300x1259, 1 - vertical 1259x300
    """
    def __init__(self, df, transforms=None, vertical=False):
        self.df = df
        self.transforms = transforms
        self.vertical = vertical

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        series = self.df.iloc[idx]
        f_image = series['image']
        f_label = series['label']
        example_id = '_'.join(f_image.split('/')[-1].split('_')[1:-1])
        # load
        image = np.array(Image.open(f_image))
        if os.path.exists(f_label):
            label = np.array(Image.open(f_label))
        else:
            # if label has no positive pixels we don't save in during data creation as .png
            # instead we just create it here on the fly
            label = np.zeros((300, 1259))
        # pad
        image = np.pad(image, ((10, 10), (10, 11)), 'constant', constant_values=(0, 0))
        label = np.pad(label, ((10, 10), (10, 11)), 'constant', constant_values=(0, 0))
        # vertical orientation - 90 deg clockwise
        if self.vertical:            
            image = np.rot90(image, k=1, axes=(1, 0))
            label = np.rot90(label, k=1, axes=(1, 0))
        # aug
        if self.transforms:
            trans = self.transforms(image=image, mask=label)
            image = trans['image']
            label = trans['mask']
        # norm
        image = (image / 255).astype(np.float32)
        label = label.astype(np.int64)
        # channels first
        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)
        # sample
        sample = {'id': example_id, 'image': image, 'label': label}
        return sample


def keep_n_last_ckpt(wildcard, n=1):
    """
    Sort and remove all checkpoint files except n last.
    If during training we save only checkpoints which improved the score 
    this function retains n best checkpoints.

    Parameters:
    wildcard : str
        Checkpoint path wildcard e.g. 'model-f%d-*' % fold_id
    n : int
        Number of last checkpoints to retain
    """
    assert n > 0, 'Number of files to keep must be > 0'
    files = sorted(glob.glob(wildcard))
    if len(files):
        for file in files[:-n]:
            os.remove(file)


def create_aug(aug):
    """
    Compose augmentation

    Parameters:
    aug : int
        Type of augmentation. 
        0 - no transformation
        1 - vertical flip
        2 - vertical and horizontal flip

    Returns:
    T : A.Compose
        Composed augmentation
    """
    if aug == 0:
        # proba 0 means it will never happen
        T = A.Compose([
            A.VerticalFlip(p=0.0),
        ])
    elif aug == 1:
        T = A.Compose([
            A.VerticalFlip(p=0.5),
        ])
    elif aug == 2:
        T = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])
    return T


#------------------------------------------------------------------------------
# train/val split
#------------------------------------------------------------------------------

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device(args.device)

images = sorted(glob.glob(os.path.join(args.input_dir, '*/*_input.png')))
labels = [x.replace('_input.png', '_label.png') for x in images]
volume_ids = [image.split('/')[-2] for image in images]

print('N volumes:', len(np.unique(volume_ids)))
print('N images:', len(images))
print('N labels:', len(labels))

train_df = pd.DataFrame()
train_df['image'] = images
train_df['label'] = labels
train_df['volume_id'] = volume_ids

# split
train_df['fold_id'] = 0
train_df = train_df.reset_index(drop=True)
kf = GroupKFold(n_splits=10)
for fold_id, (train_index, val_index) in enumerate(kf.split(train_df, groups=train_df['volume_id'].values)):
    train_df.loc[train_df.index.isin(val_index), 'fold_id'] = fold_id
train_df = train_df.sample(frac=1.0, random_state=34)
train_df = train_df.reset_index(drop=True)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

for fold_id in range(args.initial_fold, args.final_fold):
    start_save = time.time()
    print('Init timer for saving ckpt: %d' % start_save)

    print('Fold:', fold_id)    
    tr_df = train_df[train_df['fold_id'] != fold_id]
    val_df = train_df[train_df['fold_id'] == fold_id]
    
    print('Init model...')
    model = smp.Unet(
                encoder_name=args.encoder_name,
                encoder_weights=args.encoder_weights,
                in_channels=args.in_channels,
                classes=args.classes,)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = smp.losses.DiceLoss(mode='multiclass').to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode=args.reduce_mode, factor=args.reduce_f, 
                    patience=args.reduce_p, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp))
    best_score = 100
    
    # load model from ckpt and continue training
    if args.ckpt:
        checkpoint = torch.load(args.ckpt)
        #
        full_epoch = checkpoint['full_epoch']
        last_epoch = checkpoint['epoch']
        last_batch = checkpoint['batch']
        last_avg_loss = checkpoint['avg_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        #
        # to avoid increase in GPU memory usage
        del checkpoint
        torch.cuda.empty_cache()
        #
        vals = (args.ckpt, full_epoch, last_epoch, last_batch, last_avg_loss)
        print('Continue training from checkpoint: [%s] \n is complete [%d] \n last epoch [%d] \n last batch [%d] \n last avg loss [%.6f]' % vals)
        #
        # load only once
        args.ckpt = None
    else:        
        full_epoch = 1
        last_epoch = -1
        last_batch = -1
        last_avg_loss = 0.0
        #
        vals = (full_epoch, last_epoch, last_batch, last_avg_loss)
        print('Starting from scratch: \n is complete [%d] \n last epoch [%d] \n last batch [%d] \n last avg loss [%.6f]' % vals)
    
    print('Start training...')    
    for epoch_id in range(args.n_epochs):
        #
        # skip epochs completed in previous run
        if epoch_id < last_epoch + full_epoch:
            continue

        # seeding DataLoader
        random.seed(epoch_id)
        np.random.seed(epoch_id)
        torch.manual_seed(epoch_id)

        print('Init datasets...')
        train_dataset = ImageDataset(tr_df, transforms=create_aug(args.aug), vertical=args.vertical)
        val_dataset = ImageDataset(val_df, transforms=None, vertical=args.vertical)
        
        train_loader = torch.utils.data.DataLoader(
                            train_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True,
                            pin_memory=True,)
        val_loader = torch.utils.data.DataLoader(
                            val_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            pin_memory=True,)

        print('Epoch: %d' % epoch_id)
        start = time.time()
        model.train()
        torch.set_grad_enabled(True)
        optimizer.zero_grad()
        avg_loss = 0 if full_epoch else last_avg_loss
        #
        for batch_id, batch in enumerate(train_loader):
            # skip batches completed in previous run
            if batch_id < last_batch + 1:
                continue
            x = batch['image']
            y = batch['label']
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                logits = model(x)
                loss = criterion(logits, y)
                loss = loss / args.accum
            avg_loss += (loss.item() * args.accum) / len(train_loader)
            # log
            if not batch_id % 1:
                print('Batch: %04d    Loss: %.4f    Time: %d' % 
                      (batch_id, avg_loss, (time.time() - start)), end='\r')

            scaler.scale(loss).backward()
            if (batch_id+1) % args.accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # save by time
                if time.time() - start_save > args.save_seconds:
                    start_save = time.time()
                    state = {
                        'full_epoch': 0,
                        'epoch': epoch_id,
                        'batch': batch_id,
                        'avg_loss': avg_loss,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        'scaler' : scaler.state_dict(),
                    }
                    p = 'ckpt-f%d-e%03d-b%06d.bin' % (fold_id, epoch_id, batch_id)
                    p = os.path.join(args.output_dir, p)
                    torch.save(state, p)
                    print('\nSaved ckpt by time:', p)

        print('\nEval: %d' % epoch_id)
        model.eval()
        torch.set_grad_enabled(False)
        avg_loss = 0
        #
        for batch_id, batch in enumerate(val_loader):
            x = batch['image']
            y = batch['label']
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=bool(args.use_amp)):
                logits = model(x)
                loss = criterion(logits, y)
            avg_loss += loss.item() / len(val_loader)
            # log
            if not batch_id % 1:
                print('Val batch: %04d    Val loss: %.4f    Time: %d' % 
                      (batch_id, avg_loss, (time.time() - start)), end='\r')

        # update LR
        scheduler.step(avg_loss)

        # save if loss improved
        # if avg_loss < best_score:
        #     best_score = avg_loss

        # save any        
        if 1:
            state = {
                'full_epoch': 1,
                'epoch': epoch_id,
                'batch': -1, # always -1 for full epoch
                'avg_loss': avg_loss,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'scaler' : scaler.state_dict(),
            }
            p = 'model-f%d-e%03d-%.4f.bin' % (fold_id, epoch_id, avg_loss)
            p = os.path.join(args.output_dir, p)
            torch.save(state, p)
            print('\nSaved model:', p)
        else:
            print('\nScore is not better: not saving the model')

        # at the end of each epoch
        full_epoch = 1
        last_epoch = epoch_id
        last_batch = -1
        last_avg_loss = 0.0

        keep_n_last_ckpt(os.path.join(args.output_dir, 'ckpt-f%d-*' % fold_id))
        keep_n_last_ckpt(os.path.join(args.output_dir, 'model-f%d-*' % fold_id))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
