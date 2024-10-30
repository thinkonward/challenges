import os
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from types import SimpleNamespace

from train_stage1 import Stage1BuildingDataset,Stage1SimpleCls
from train_stage3 import Stage2BuildingDataset
from train_stage2 import Stage2SimpleMLCls

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

   
def load_s1models(ckpts):
    models = []
    for ckpt in ckpts:
        model = Stage1SimpleCls(ckpt['model_name'])
        model.load_state_dict(torch.load(ckpt['path'], map_location=device)['model_state_dict'],strict=True)
        model.to(device)
        model.eval()
        models.append(model)
    return models

def load_s2models(ckpts,num_classes):
    models = []
    for ckpt in ckpts:
        model = Stage2SimpleMLCls(ckpt['model_name'], num_classes)
        model.load_state_dict(torch.load(ckpt['path'], map_location=device)['model_state_dict'],strict=True)
        model.to(device)
        model.eval()
        models.append(model)
    return models

def predict_stage1(df,stage1_ckpts):
    cfg = SimpleNamespace(**{})
    cfg.features_type = "weekly"
    dataset = Stage1BuildingDataset(df,"test",cfg)
    loader = DataLoader(dataset, batch_size=32, shuffle=False,drop_last=False, num_workers=4)
    models =load_s1models(stage1_ckpts)
    preproces = [ckpt['preprocess'] for ckpt in stage1_ckpts]
    all_preds = []
    with torch.no_grad():
        for i, (images,states,_) in enumerate(loader):
            images = images.to(device)
            states = states.to(device)
            preds = []
            for model,prep in zip(models,preproces):
                if prep == "daily":
                    images = images.flatten(1)[:,:365*96].reshape(-1,1,365,96)
                pred = model(images,states)
                preds.append(pred)
            preds = torch.stack(preds).mean(0)
            all_preds.append(preds)
    all_preds = torch.cat(all_preds).cpu().numpy()

    return all_preds.argmax(1)

def inference_ensemble_multilabel(models, preproces, dataloader, num_labels):
    """
    Perform inference using an ensemble of models for a multilabel classification task.
    Args:
        models (list): List of trained models.
        dataloader (torch.utils.data.DataLoader): DataLoader for inference data.
        num_labels (int): Number of independent labels.
    
    Returns:
        np.ndarray: Averaged predictions for each label.
    """
    all_predictions = []

    with torch.no_grad():
        for images,states ,_ in dataloader:
            images = images.cuda()
            states = states.cuda()
            
            # Initialize list to accumulate predictions for each label
            ensemble_logits = [None] * num_labels

            # Loop through each model in the ensemble
            for model,prep in zip(models,preproces):
                if prep == "daily":
                    images = images.flatten(1)[:,:365*96].reshape(-1,1,365,96)
                logits = model(images,states)  # The output is a list of logits, one per label
                
                # Loop through each label and accumulate logits
                for i in range(num_labels):
                    if ensemble_logits[i] is None:
                        ensemble_logits[i] = logits[i]
                    else:
                        ensemble_logits[i] += logits[i]
            
            for i in range(num_labels):
                ensemble_logits[i] /= len(models)
            preds = []
            for i in range(num_labels):
                probs = torch.softmax(ensemble_logits[i], dim=1)  # Softmax for this label's logits
                preds.append(torch.argmax(probs, dim=1))  # Get predicted class for this label
            
            # Stack predictions across all labels (each is [batch_size] shape)
            batch_predictions = torch.stack(preds, dim=1)  # Shape: [batch_size, num_labels]
            all_predictions.append(batch_predictions.cpu().numpy())
    
    # Concatenate predictions from all batches
    return np.concatenate(all_predictions)   
def predict_stage2(df,stage2_ckpts,type="com"):
    features_type = "weekly"
    target_cols = [col for col in df.columns if col.endswith(f"_{type}")]
    dataset = Stage2BuildingDataset(df=df,mode="test",
                                    target_cols=target_cols,
                                    features_type=features_type
                                    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=4)
    models = load_s2models(stage2_ckpts, dataset.num_classes)
    preproces = [ckpt['preprocess'] for ckpt in stage2_ckpts]
    predictions = inference_ensemble_multilabel(models,preproces, loader, len(target_cols))
    all_classes = []
    for i, tcol in enumerate(target_cols):
        with open(f'../data/Label_encoded/{tcol}_label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
            all_classes.append(le.inverse_transform(predictions[:,i]))
    return np.stack(all_classes).T
df=pd.read_parquet('../data/sample_submission.parquet',engine='pyarrow')

df_pred =df.copy()
with open(f'../data/Label_encoded/building_stock_type_label_encoder.pkl', 'rb') as f:
    s1_le = pickle.load(f)


print(f"stage 1 ")
s1_ckpts = [{"preprocess":"weekly","model_name":"tf_efficientnetv2_s_in21k","path":f"../model_checkpoints/stage1_v8_fold{fold}_last.pth"} for fold in range(10)]
s1_ckpts += [{"preprocess":"weekly","model_name": "seresnext26t_32x4d", "path": f"../model_checkpoints/stage1_v9_fold{fold}_last.pth"} for fold in range(10)]
s1_ckpts += [{"preprocess":"daily","model_name": "tf_efficientnetv2_s_in21k", "path": f"../model_checkpoints/stage1_v1_fold{fold}_last.pth"} for fold in range(10)]
s1_ckpts += [{"preprocess":"daily","model_name": "seresnext26t_32x4d", "path": f"../model_checkpoints/stage1_v2_fold{fold}_last.pth"} for fold in range(10)]
       
s1predictions = predict_stage1(df_pred,s1_ckpts)
df_pred.loc[:,"building_stock_type"]=s1_le.inverse_transform(s1predictions)


df_pred2 =df_pred.copy()
res_cols = [col for col in df_pred2.columns if col.endswith("_res")]
com_cols = [col for col in df_pred2.columns if col.endswith("_com")]

print(f"stage 2 fold ")
s2_com_ckpts = [{"preprocess":"weekly","model_name": "tf_efficientnetv2_s_in21k", "path": f"models/stage3_v1_com_fold{fold}_last.pth"} for fold in range(10)]
s2_com_ckpts += [{"preprocess":"weekly","model_name": "seresnext26t_32x4d", "path": f"models/stage3_v4_com_fold{fold}_last.pth"} for fold in range(10)]
             
s2_res_ckpts = [{"preprocess":"weekly","model_name": "tf_efficientnetv2_s_in21k", "path": f"models/stage3_v1_res_fold{fold}_last.pth"} for fold in range(10)]
s2_res_ckpts += [{"preprocess":"weekly","model_name": "seresnext26t_32x4d", "path": f"models/stage3_v2_res_fold{fold}_last.pth"} for fold in range(10)]
                 
df_com = df_pred2[(df_pred2.building_stock_type=="commercial")]
df_res = df_pred2[(df_pred2.building_stock_type=="residential")]
df_com = predict_stage2(df_com,s2_com_ckpts,type="com")
df_res = predict_stage2(df_res,s2_res_ckpts,type="res")
df_pred2.loc[df_pred2.building_stock_type=="commercial",com_cols] = df_com
df_pred2.loc[df_pred2.building_stock_type=="residential",res_cols] = df_res
df_pred2.loc[df_pred2.building_stock_type=="commercial",res_cols] = None
df_pred2.loc[df_pred2.building_stock_type=="residential",com_cols] = None

df_pred2.to_parquet('final_submission.parquet',engine='pyarrow')
df_pred2.to_csv('final_submission.csv')


    
    