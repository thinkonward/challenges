#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

import os
import glob
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from utils import create_submission, get_submission_score
from argparse import ArgumentParser

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

class ImageDataset(torch.utils.data.Dataset):
    """
    Image dataset.

    Parameters:
    files : list
        List of images
    transforms : 
        Albumentation transformations
    """
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f_image = self.files[idx]
        example_id = '_'.join(f_image.split('/')[-1].split('_')[1:-1])
        # load
        image = np.array(Image.open(f_image))
        # pad
        image = np.pad(image, ((10, 10), (10, 11)), 'constant', constant_values=(0, 0))
        # aug
        if self.transforms:
            trans = self.transforms(image=image)
            image = trans['image']
        # norm
        image = (image / 255).astype(np.float32)
        # channels first
        image = np.expand_dims(image, axis=0) # (1, 320, 1280)
        # sample
        sample = {'id': example_id, 'image': image}
        return sample


def inference(dirs_ax0, dirs_ax1, models, device, tta_list, rot_list, batch_size, num_workers):
    """
    Run inference and ensemble.

    Ensembling logic.
    Iterate over test volumes and slices across axis 0. For a given slice (image) 
    predict it using 5 models, then flip slice up-down and predict again, 
    then average 10 predicted logits. Recreate volumes from averaged slices (logits). 
    Repeat for axis 1. Now we have two list of predicted volumes 
    obtained from predictions across each of 2 axes.
    Average pairs of corresponding volumes and convert to binary masks.

    Parameters:
    dirs_ax0 : list
        List of the directories containing images taken from axis 0
    dirs_ax1 : list
        List of the directories containing images taken from axis 1
    models : list
        List of models
    device
        torch.device
    tta_list
        List of augmentations
    rot_list
        List where each item (0 or 1) corresponds to a model in "models" list 
        and indicates which image orientation model needs.
        0 - horizontal orientation of the image (default)
        1 - vertical orientation of the image (i.e. default with 90 deg rotation)
    batch_size : int
        Batch size
    num_workers : int
        Number of workers for dataloader

    Returns:
    prediction : list of 2 lists
        Predicted volume masks
    seismic_filenames : list
        List of volume names required for submission
    """    
    start = time.time()

    prediction = []
    seismic_filenames = []
    dir_pairs = list(zip(dirs_ax0, dirs_ax1))
    
    for counter_dirs, dir_pair in enumerate(dir_pairs):

        volume_per_axis = []
        for ax_id, d in enumerate(dir_pair):
    
            volume_id = d.split('/')[-2]
            if volume_id not in seismic_filenames:
                seismic_filenames.append(volume_id)
            files = sorted(glob.glob(os.path.join(d, '*_input.png')))
            #
            preds_per_volume = []
            test_dataset = ImageDataset(files)
            test_loader = torch.utils.data.DataLoader(
                                test_dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=True,)

            for batch_id, batch in enumerate(test_loader):
                image = batch['image'].to(device)
                all_logits = []
                for T in tta_list:
                    # tta
                    if T == 'flipud':
                        image = torch.flip(image, dims=(2,))
             
                    for model_id in range(len(models)):
                        if rot_list[model_id]:
                            image = torch.rot90(image, k=1, dims=(3, 2))

                        logits = models[model_id](image)

                        if rot_list[model_id]:
                            logits = torch.rot90(logits, k=1, dims=(2, 3))
                            image = torch.rot90(image, k=1, dims=(2, 3))

                        # tta-backward because we need same orientation for averaging
                        if T == 'flipud':
                            logits = torch.flip(logits, dims=(2,))

                        logits = logits[:, :, 10:-10, 10:-11]
                        logits = logits.detach().cpu().numpy()
                        all_logits.append(logits)
                        print('Volume: %03d    Axis: %d    Model: %03d    Batch: %03d    Time: %d' % (
                                counter_dirs, ax_id, model_id, batch_id, (time.time() - start)), end='\r')

                preds = np.mean(all_logits, axis=0)
                if ax_id == 0:
                    preds = np.transpose(preds, [0, 2, 3, 1])
                else:
                    preds = np.transpose(preds, [2, 0, 3, 1])
                preds_per_volume.append(preds)
    
            if ax_id == 0:
                volume = np.vstack(preds_per_volume)
            else:
                volume = np.hstack(preds_per_volume)
            volume_per_axis.append(volume)           

        preds = np.mean(volume_per_axis, axis=0)
        preds = np.argmax(preds, axis=-1).astype(np.uint8)
        prediction.append(preds)

    return prediction, seismic_filenames

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input_dir_axis_0', default='test_img_axis_0', type=str, help='Directory containing test images taken from axis 0')
    parser.add_argument('--input_dir_axis_1', default='test_img_axis_1', type=str, help='Directory containing test images taken from axis 1')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=os.cpu_count(), type=int, help='Number of workers')
    parser.add_argument('--model_dir', default='models', type=str, help='Directory with model weights')
    parser.add_argument('--submission_path', default='submission.npz', type=str, help='Submission path')
    args = parser.parse_args() # pass empty list to run in notebook using default arg values: parser.parse_args([])
    for a in [a for a in vars(args) if '__' not in a]: print('%-25s %s' % (a, vars(args)[a]))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
    # collect test dirs
    dirs_ax0 = sorted(glob.glob(os.path.join(args.input_dir_axis_0, '*/')))
    print('N dirs axis 0:', len(dirs_ax0)) # 50        
    dirs_ax1 = sorted(glob.glob(os.path.join(args.input_dir_axis_1, '*/')))
    print('N dirs axis 1:', len(dirs_ax1)) # 50
    
    # load models    
    encoder_names = [
        'tu-tf_efficientnet_l2.ns_jft_in1k',
        'timm-efficientnet-b8',        
        'tu-tf_efficientnet_b8.ap_in1k',
        'tu-tf_efficientnetv2_l.in1k',
        'tu-tf_efficientnetv2_xl.in21k_ft_in1k',
    ]
    weights = [
        os.path.join(args.model_dir, 'model-0-efl2-f0-e006-0.0633-ready.bin'),
        os.path.join(args.model_dir, 'model-1-ef8-f0-e015-0.0606-ready.bin'),
        os.path.join(args.model_dir, 'model-2-ef82vert-f0-e019-0.0587-ready.bin'),
        os.path.join(args.model_dir, 'model-3-efv2l-f0-e008-0.0621-ready.bin'),
        os.path.join(args.model_dir, 'model-4-efv2xl-f0-e010-0.0576-ready.bin'),
    ]
    rot_list = [0, 0, 1, 0, 0] # which models needs rotated image
    tta_list = ['same', 'flipud'] # same and flip up-down
    models = []
    for encoder_name, weight in zip(encoder_names, weights):
        model = smp.Unet(
                    encoder_name=encoder_name,
                    encoder_weights=None,
                    in_channels=1,
                    classes=2,)
        model.to(device)
        model.eval()
        torch.set_grad_enabled(False)
        model.load_state_dict(torch.load(weight, map_location=device))
        models.append(model)
        print('Loaded model: [%s], %s' % (encoder_name, weight))

    print('N models:', len(models))
    print('N TTA:', len(tta_list))

    # run inference
    prediction, seismic_filenames = inference(dirs_ax0, dirs_ax1, models, device, tta_list, rot_list, args.batch_size, args.num_workers)

    # create submission
    for sample_id, pred in zip(seismic_filenames, prediction):
        create_submission(sample_id, pred, args.submission_path, append=True)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
