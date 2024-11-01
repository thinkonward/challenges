import os
import gc
import time
import timm
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchaudio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
# import wandb

DEBUG = False
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Training Script for Stage 1 Model')

    parser.add_argument('--exp_id', type=str, default='', help='Path to load kernel')
    parser.add_argument('--freq_mask', type=int, default=10, help='freq maskings')
    parser.add_argument('--time_mask', type=int, default=16, help='time maskings')
    parser.add_argument('--p_mask', type=float, default=0.5, help='probability of applying mask')
    parser.add_argument('--load_last', type=bool, default=True, help='Load last checkpoint')
    parser.add_argument('--features_type', type=str, default="stand_features", help='Type of features')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', help='Model backbone')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    parser.add_argument('--init_lr', type=float, default=3e-4, help='Initial learning rate')
    parser.add_argument('--eta_min', type=float, default=3e-5, help='Minimum learning rate for scheduler')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.3, help='Drop path rate')
    parser.add_argument('--p_mixup', type=float, default=0.5, help='Probability of applying mixup')
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory for the dataset')
    parser.add_argument('--use_amp', type=bool, default=False, help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--n_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for logging')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory to save models')

    args = parser.parse_args()

    
    # Add the rest of your code to train the model, etc.
    return args

ustates =['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
       'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
       'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
       'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
       'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
s2id ={c:i for i,c in enumerate(ustates)}
id2s = {i:c for i,c in enumerate(ustates)}

class Stage1BuildingDataset(Dataset):
    def __init__(self, df, mode, cfg):
        self.df = df.reset_index()
        self.features_type=cfg.features_type
        self.mode = mode
        self.cfg = cfg
        with open(f'{self.cfg.data_dir}/Label_encoded/building_stock_type_label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        if self.mode == 'train':
            self.freq_m = torchaudio.transforms.FrequencyMasking(cfg.freq_mask)  # 10
            self.time_m = torchaudio.transforms.TimeMasking(cfg.time_mask)  # 16
        self.num_classes = len(le.classes_)
        if self.mode == 'test':
            self.targets = np.zeros(len(df))
        else:
            self.targets = le.transform(df["building_stock_type"])
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        id = row.bldg_id
        if self.mode == 'test':
            df = pd.read_parquet(f'{self.cfg.data_dir}/test/{id}.parquet',engine='pyarrow')
        else:
            df = pd.read_parquet(f'{self.cfg.data_dir}/train/{id}.parquet',engine='pyarrow')
        x = df["out.electricity.total.energy_consumption"].values
        if self.features_type=="weekly":
            v=np.zeros(576)
            x = np.concatenate([x,v])
            x= x.reshape(53,96*7)
        else:
            x= x.reshape(365,96)
        x = x/x.max()  
        state = s2id[df["in.state"].values[0]]
        x=x[np.newaxis,:,:]
        if self.mode == 'train' and random.random() < self.cfg.p_mask:
            x = torch.from_numpy(x).float()
            x = x.unsqueeze(-1)
            x = self.time_m(self.freq_m(x.permute(0, 3, 1, 2)))[0]
            
        else:
            x = torch.from_numpy(x).float()
        
        target = int(self.targets[index])
        return x,torch.tensor(state).long(), torch.tensor(target)   

class Stage1SimpleCls(nn.Module):
    def __init__(self,
                 base_model_name,
                 in_chans=1,
                 num_classes=2,
                 dropout=0,
                 state_emb_size=128,
                 drop_path_rate=0,
                 pretrained=True
                 ):
        super().__init__()

        self.dropout = dropout
        self.base_model = timm.create_model(base_model_name,
                                            in_chans=in_chans,
                                            features_only=True,
                                            drop_rate=dropout,
                                            drop_path_rate=drop_path_rate,
                                            pretrained=pretrained)
        
        self.state_embed = nn.Embedding(len(ustates), state_emb_size)
        self.backbone_depths = list(self.base_model.feature_info.channels())
        print(f'{base_model_name}')
        self.fc = nn.Linear(self.backbone_depths[-1] * 2+state_emb_size, num_classes)

    def forward(self, inputs,states):
        inputs = inputs.contiguous()
        output = self.base_model(inputs)
        features = output[-1]
        
        avg_pool = F.avg_pool2d(features, features.shape[2:])
        max_pool = F.max_pool2d(features, features.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        x = avg_max_pool.view(avg_max_pool.size(0), -1)
        state_emb = self.state_embed(states)
        x = torch.cat([x, state_emb], dim=1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)
        x = self.fc(x)
        return x

def mixup(input, truth, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, truth, shuffled_labels, lam


def train_func(model, loader_train, optimizer,criterion,args, scaler=None):
    model.train()
    train_loss = []
    bar = tqdm(loader_train)
    for x,state, targets in bar:
        optimizer.zero_grad()
        x = x.cuda()
        state = state.cuda()
        targets = targets.cuda()
        do_mixup = random.random() < args.p_mixup
        if do_mixup:
            x, targets, targets_mix, lam = mixup(x, targets)
        with amp.autocast():
            logits = model(x,state)
            if do_mixup:
                # Calculate CrossEntropyLoss for mixed targets
                loss = lam * criterion(logits, targets) + (1 - lam) * criterion(logits, targets_mix)
            else:
                loss = criterion(logits, targets)  # Normal loss calculation without Mixup

        train_loss.append(loss.item())
        if scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bar.set_description(f'smth: {np.mean(train_loss[-30:]):.4f}')
        # wandb.log({"train_loss": np.mean(train_loss[-30:])})
    return np.mean(train_loss)


def valid_func(model, loader_valid,criterion):
    model.eval()
    valid_loss = []

    all_targets = []
    all_predictions = []

    bar = tqdm(loader_valid)
    with torch.no_grad():
        for x,state, targets in bar:
            x = x.cuda()
            state = state.cuda()
            targets = targets.cuda()
            logits = model(x,state)

            loss = criterion(logits, targets)
            valid_loss.append(loss.item())

            # Apply softmax to the logits and get class predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)  # Get the class with the highest probability

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

            bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')

    # Calculate F1 score
    f1 = f1_score(all_targets, all_predictions, average='macro')  # Use average='macro' for multi-class
    print(f'Validation F1 Score: {f1:.4f}')
    # wandb.log({"valid_loss": np.mean(valid_loss)})
    # wandb.log({"f1 score": f1})

    return np.mean(valid_loss), f1  

def run(fold,df_targets,criterion,kernel_type,args):
    # wandb.init(project="building_power",name=f'{kernel_type}_fold{fold}',group="building_stock_type")
    # wandb.config.update({
    # "fold":fold,
    # **vars(args),
    # })
    log_file = os.path.join(args.log_dir, f'{kernel_type}.txt')
    model_file = os.path.join(args.model_dir, f'{kernel_type}_fold{fold}_best.pth')

    train_ = df_targets[df_targets['fold'] != fold]
    valid_ = df_targets[df_targets['fold'] == fold]
    dataset_train = Stage1BuildingDataset(train_, 'train',args)
    dataset_valid = Stage1BuildingDataset(valid_, 'valid',args)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    loader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = Stage1SimpleCls(args.backbone,
                      in_chans=1,
                      num_classes=dataset_train.num_classes,
                      dropout=args.drop_rate,
                     drop_path_rate=args.drop_path_rate,
                     pretrained=True
                      )
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.init_lr,weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    from_epoch = 0
    metric_best = 0
    loss_min = np.inf

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs, eta_min=args.eta_min)

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, args.n_epochs+1):
        scheduler_cosine.step(epoch-1)
        if epoch < from_epoch + 1:
            print(logs[epoch-1])
            continue

        print(time.ctime(), 'Epoch:', epoch)

        train_loss = train_func(model, loader_train, optimizer,criterion,args, scaler)
        valid_loss,f1_score = valid_func(model, loader_valid,criterion)
        metric = f1_score

        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, metric: {(metric):.6f}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if metric > metric_best:
            print(f'metric_best ({metric_best:.6f} --> {metric:.6f}). Saving model ...')
#             if not DEBUG:
            torch.save(model.state_dict(), model_file)
            metric_best = metric

        # Save Last
        if not DEBUG:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'score_best': metric_best,
                },
                model_file.replace('_best', '_last')
            )

    del model
    torch.cuda.empty_cache()
    # wandb.finish()
    gc.collect()
def main():
    args=get_args()
    kernel_type = f'stage1_{args.exp_id}'
    
    print(f'Kernel Type: {kernel_type}')
    print(f'Logging Directory: {args.log_dir}')
    print(f'Model Directory: {args.model_dir}')
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    df_targets = pd.read_parquet(args.data_dir+'/labels/train_label_fold10.parquet',engine='pyarrow')
    criterion = nn.CrossEntropyLoss(reduction='mean')
    for fold in range(args.n_folds):
        run(fold,df_targets,criterion,kernel_type,args)
    

if __name__ == "__main__":
    main()
