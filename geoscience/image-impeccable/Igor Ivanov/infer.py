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
from utils import create_submission
from argparse import ArgumentParser

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def inference(dirs_ax0, dirs_ax1, models, device):
    """
    Run inference and ensembling.

    Ensembling logic.
    Each image in a given axis is predicted by all models from the "models" list,
    predictions are averaged and appended to the "preds_per_volume" list.
    After all 300 images per given volume are predicted, we create volume and 
    append it to an axis-specific list "prediction[0]" or "prediction[1]".
    In the end function returns a list with 2 inner lists 
    containing volumes for each of 2 axes.
    Outside this function we need to average volumes from axis-specific
    lists and convert the result to 0-255 range (uint8).

    Parameters:
    dirs_ax0 : list
        List of the directories containing images taken from axis 0
    dirs_ax1 : list
        List of the directories containing images taken from axis 1
    models : list
        List of model instances

    Returns:
    prediction : list of 2 lists
        Predicted volumes based on images from each of 2 axes       
    seismic_filenames : list
        List of volume names required for submission
    """

    prediction = [[], []]
    seismic_filenames = []    

    for ax_id, dirs in enumerate([dirs_ax0, dirs_ax1]):
    
        for counter_dirs, d in enumerate(dirs):
    
            volume_id = d.split('/')[-2]
            if volume_id not in seismic_filenames:
                seismic_filenames.append(volume_id)
            files = sorted(glob.glob(os.path.join(d, '*_noisy.png')))
            #
            preds_per_volume = []
            for counter_files, file in enumerate(files):
                # load
                image = np.array(Image.open(file)) # (300, 1259)
                # pad
                image = np.pad(image, ((10, 10), (10, 11)), 'constant', constant_values=(0, 0)) # (320, 1280)
                # norm
                image = (image / 255).astype(np.float32)
                # channels first
                image = np.expand_dims(image, axis=0) # (1, 320, 1280)
                # imitate batch
                image = np.expand_dims(image, axis=0) # (1, 1, 320, 1280)
                # tensor
                image = torch.tensor(image).to(device)
                #
                all_logits = []
                for model_id in range(len(models)):
                    #
                    logits = models[model_id](image) # torch.Size([1, 1, 320, 1280])
                    #
                    # unpad
                    logits = logits[:, :, 10:-10, 10:-11].squeeze() # torch.Size([300, 1259])
                    #
                    all_logits.append(logits.detach().cpu().numpy())
                    #
                    print('Axis: %d    Volume: %03d    Image: %03d    Model: %03d' % (ax_id, counter_dirs, counter_files, model_id), end='\r')
    
                preds = np.mean(all_logits, axis=0)
                preds_per_volume.append(preds)
    
            volume = np.stack(preds_per_volume, axis=ax_id) # (300, 300, 1259)
            prediction[ax_id].append(volume)

    return prediction, seismic_filenames

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input_dir_axis_0', default='test_img_axis_0', type=str, help='Directory containing test images taken from axis 0')
    parser.add_argument('--input_dir_axis_1', default='test_img_axis_1', type=str, help='Directory containing test images taken from axis 1')
    parser.add_argument('--encoder_name', default='timm-efficientnet-b0', type=str, help='Encoder architecture')
    parser.add_argument('--encoder_weights', default=None, type=str, help='Encoder pretrained weights')
    parser.add_argument('--in_channels', default=1, type=int, help='Number of input channels')
    parser.add_argument('--classes', default=1, type=int, help='Number of classes')
    parser.add_argument('--model_dir', default='models', type=str, help='Directory containing checkpoints')
    parser.add_argument('--submission_path', default='submission.npz', type=str, help='Submission path')
    args = parser.parse_args() # pass empty list to run in notebook using default arg values: parser.parse_args([])
    for a in [a for a in vars(args) if '__' not in a]: print('%-25s %s' % (a, vars(args)[a]))

    #----

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dirs_ax0 = sorted(glob.glob(os.path.join(args.input_dir_axis_0, '*/')))
    print('N dirs axis 0:', len(dirs_ax0)) # 15
    
    dirs_ax1 = sorted(glob.glob(os.path.join(args.input_dir_axis_1, '*/')))
    print('N dirs axis 1:', len(dirs_ax1)) # 15
    
    #----
    
    # load models from all folds
    models = []    
    for fold_id in range(5):
        model_file = sorted(glob.glob(os.path.join(args.model_dir, 'model-f%d-*.bin' % fold_id)))[-1]
        model = smp.Unet(
                    encoder_name=args.encoder_name,
                    encoder_weights=args.encoder_weights,
                    in_channels=args.in_channels,
                    classes=args.classes,
                    activation='sigmoid',)
        model.to(device)
        model.eval()
        torch.set_grad_enabled(False)
        model.load_state_dict(torch.load(model_file, map_location=device)['state_dict'])
        models.append(model)
        print('Loaded model:', model_file)
    print('N models:', len(models))

    #----

    # run inference
    prediction, seismic_filenames = inference(dirs_ax0, dirs_ax1, models, device)    

    # compute ensemble of exes
    prediction_ens = []
    for i in range(len(prediction[0])):
        pred = (((prediction[0][i] + 
                  prediction[1][i]) / 2) * 255).astype(np.uint8)
        prediction_ens.append(pred)

    # create submission file
    create_submission(seismic_filenames,    
                      prediction_ens, 
                      args.submission_path)
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
