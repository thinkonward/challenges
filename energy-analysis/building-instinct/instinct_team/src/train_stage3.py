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
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import f1_score

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True
import wandb

DEBUG = False
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Training Script for Stage 3 Model')

    parser.add_argument('--exp_id', type=str, default='', help='Path to load kernel')
    parser.add_argument('--freq_mask', type=int, default=10, help='freq maskings')
    parser.add_argument('--state_embed_size', type=int, default=128, help='State embedding size')
    parser.add_argument('--time_mask', type=int, default=16, help='time maskings')
    parser.add_argument('--p_mask', type=float, default=0.5, help='probability of applying mask')
    parser.add_argument('--load_last', type=bool, default=True, help='Load last checkpoint')
    parser.add_argument('--target_type', type=str, default="res", help='Type of target columns for training')
    parser.add_argument('--features_type', type=str, default="stand_features", help='Type of features')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0', help='Model backbone')
    parser.add_argument('--ckpt', type=str, default='', help='Path to load checkpoint')
    
    parser.add_argument('--pooling',type=str,default='avg_max',help='Pooling type')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    parser.add_argument('--init_lr', type=float, default=3e-4, help='Initial learning rate')
    parser.add_argument('--eta_min', type=float, default=3e-5, help='Minimum learning rate for scheduler')
    parser.add_argument('--label_smoothing', type=float, default=0., help='Label smoothing')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--drop_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.3, help='Drop path rate')
    parser.add_argument('--p_mixup', type=float, default=0.5, help='Probability of applying mixup')
    parser.add_argument('--image_size', type=tuple, default=(365, 96), help='Input image size')
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

def standardize(x,mean=7.6492750738675905,std=3.8021495908327183):
    return x -mean/std

class Stage2BuildingDataset(Dataset):
    def __init__(self, df, mode,target_cols,features_type,args=None):
        self.df = df.reset_index()
        self.features_type=features_type
        self.target_cols=target_cols
        self.mode = mode
        self.args = args
        
        self.num_classes = []
        self.targets = []
        for target_col in self.target_cols:
            with open(f'{self.args.data_dir}/Label_encoded/{target_col}_label_encoder.pkl', 'rb') as f:
                le = pickle.load(f)
        
                self.num_classes.append(len(le.classes_))
                if self.mode == 'test':
                    self.targets.append(np.zeros(len(df)))
                else:
                    self.targets.append(le.transform(df[target_col]))
        if self.mode == 'train':
            self.freq_m = torchaudio.transforms.FrequencyMasking(self.args.freq_mask)  # 10
            self.time_m = torchaudio.transforms.TimeMasking(self.args.time_mask)  # 16
        self.targets = np.array(self.targets).T
        
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        id = row.bldg_id
        if self.mode == 'test':
            df = pd.read_parquet(f'{self.args.data_dir}/test/{id}.parquet',engine='pyarrow')
        else:
            df = pd.read_parquet(f'{self.args.data_dir}/train/{id}.parquet',engine='pyarrow')
        x = df["out.electricity.total.energy_consumption"].values
        if self.features_type == "weekly":
            v=np.zeros(576)
            x = np.concatenate([x,v])
            x= x.reshape(53,96*7)
        else:
            x= x.reshape(365,96)
        if self.target_cols[0].endswith("res"):
            x = standardize(x)
        else:
            x =x/x.max()
        
        state = s2id[df["in.state"].values[0]]
        x=x[np.newaxis,:,:]
        if self.mode == 'train' and random.random() < self.args.p_mask:
            x = torch.from_numpy(x).float()
            x = x.unsqueeze(-1)
            x = self.time_m(self.freq_m(x.permute(0, 3, 1, 2)))[0]
        else:
            x = torch.from_numpy(x).float()
        target = (self.targets[index])
        return x.float(), torch.tensor(state).long(),torch.tensor(target).long()  

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")

class Stage2SimpleMLCls(nn.Module):
    def __init__(self,
                 base_model_name,
                 num_classes,
                 in_chans=1,
                 dropout=0,
                 drop_path_rate=0,
                 state_embed_size=128,
                 pretrained=True,
                 pooling='avg_max'
                 ):
        super().__init__()

        self.dropout = dropout
        self.base_model = timm.create_model(base_model_name,
                                            in_chans=in_chans,
                                            features_only=True,
                                            drop_rate=dropout,
                                            drop_path_rate=drop_path_rate,
                                            pretrained=pretrained,
                                            )
        self.backbone_depths = list(self.base_model.feature_info.channels())
        self.state_embed = nn.Embedding(len(ustates), state_embed_size)
        self.pooling = pooling
        self.backbone_outsize = self.backbone_depths[-1] * 2
        if 'gem' in self.pooling :
            self.gem_pool = GeM(p_trainable=True)
            self.backbone_outsize = self.backbone_depths[-1] *3
        self.fc =  nn.ModuleList([nn.Linear(self.backbone_outsize+state_embed_size, num_class) for num_class in num_classes])

    def forward(self, inputs,states):
        inputs = inputs.contiguous()
        output = self.base_model(inputs)
        features = output[-1]

        avg_pool = F.avg_pool2d(features, features.shape[2:])
        max_pool = F.max_pool2d(features, features.shape[2:])
        avg_max_pool = torch.cat((avg_pool, max_pool), 1)
        x = avg_max_pool.view(avg_max_pool.size(0), -1)
        if 'gem' in self.pooling :
            gem = self.gem_pool(features)
            gem = gem[:,:,0,0]
            x = torch.cat([x,gem],dim=1)
       
            
        state_emb = self.state_embed(states)
        x = torch.cat([x, state_emb], dim=1)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, self.training)
        x =  [fc(x) for fc in self.fc] 
        return x
def mixup(input, truth, clip=[0, 1]):
    indices = torch.randperm(input.size(0))
    shuffled_input = input[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    input = input * lam + shuffled_input * (1 - lam)
    return input, truth, shuffled_labels, lam



class MultiLabelLoss(nn.Module):
    def __init__(self, num_labels, label_smoothing=0.0):
        super().__init__()
        # Apply label smoothing to each criterion
        self.criterions = nn.ModuleList(
            [nn.CrossEntropyLoss(label_smoothing=label_smoothing) for _ in range(num_labels)]
        )

    def forward(self, outputs, targets):
        losses = []
        # Loop through each label's output and corresponding target
        for output, target, criterion in zip(outputs, targets.T, self.criterions):
            losses.append(criterion(output, target))
        return sum(losses) / len(losses)

def train_func(model, loader_train, optimizer,criterion,args, scaler=None):
    model.train()
    train_loss = []
    bar = tqdm(loader_train)
    for images,states, targets in bar:
        optimizer.zero_grad()
        images = images.cuda()
        states = states.cuda()
        targets = targets.cuda()
        do_mixup = random.random() < args.p_mixup
        if do_mixup:
            images, targets, targets_mix, lam = mixup(images, targets)
        with amp.autocast():
            logits = model(images,states)
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

    
    all_targets = [[] for _ in range(len(criterion.criterions))]
    all_predictions = [[] for _ in range(len(criterion.criterions))]
    bar = tqdm(loader_valid)
    with torch.no_grad():
        for images,states, targets in bar:
            images = images.cuda()
            states = states.cuda()
            targets = targets.cuda()
            logits = model(images,states)

            loss = criterion(logits, targets)
            valid_loss.append(loss.item())
            for i, (logit, target) in enumerate(zip(logits, targets.T)):
                probs = torch.softmax(logit, dim=1)
                preds = torch.argmax(probs, dim=1)

                # Convert tensors to numpy for metrics calculation
                all_targets[i].extend(target.cpu().numpy())
                all_predictions[i].extend(preds.cpu().numpy())

            bar.set_description(f'smth:{np.mean(valid_loss[-30:]):.4f}')
    f1_scores = []
    for i in range(len(criterion.criterions)):
        f1 = f1_score(all_targets[i], all_predictions[i], average='macro')  # or 'micro'
        f1_scores.append(f1)
        print(f'Validation F1 Score for label {i}: {f1:.4f}')    
    # Calculate F1 score
    avg_f1_score = np.mean(f1_scores)
    print(f'Validation F1 Score: {f1:.4f}')
    # wandb.log({"valid_loss": np.mean(valid_loss)})
    # wandb.log({"f1 score": avg_f1_score})

    return np.mean(valid_loss), avg_f1_score 

def run(fold,df_targets,target_cols,criterion,kernel_type,args):
    # wandb.init(project="building_power",name=f'{kernel_type}_fold{fold}',group=args.target_type)
    # wandb.config.update({
    # "fold":fold,
    # **vars(args),
    # })
    log_file = os.path.join(args.log_dir, f'{kernel_type}.txt')
    model_file = os.path.join(args.model_dir, f'{kernel_type}_fold{fold}_best.pth')

    train_ = df_targets[df_targets['fold'] != fold]
    valid_ = df_targets[df_targets['fold'] == fold]
    dataset_train = Stage2BuildingDataset(train_, 'train',target_cols,args.features_type,args=args)
    num_classes = dataset_train.num_classes 
    dataset_valid = Stage2BuildingDataset(valid_, 'valid',target_cols,args.features_type,args=args)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    loader_valid = DataLoader(dataset_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model = Stage2SimpleMLCls(base_model_name=args.backbone, 
                        num_classes=num_classes, 
                        pretrained=args.pretrained, 
                        state_embed_size=args.state_embed_size,
                        dropout=args.drop_rate, 
                        drop_path_rate=args.drop_path_rate,
                        pooling=args.pooling)
    if args.ckpt!='':
        ckpt_path = f"{args.model_dir}/{args.ckpt}_{args.target_type}_fold0_last.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'],strict=True) 
        print(f"Loaded model from {ckpt_path}")   
        
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
    
    kernel_type = f'stage3_{args.exp_id}_{args.target_type}'
    
    print(f'Kernel Type: {kernel_type}')
    print(f'Logging Directory: {args.log_dir}')
    print(f'Model Directory: {args.model_dir}')
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    df_targets = pd.read_parquet(args.data_dir+f'/labels/train_label_fold{args.n_folds}.parquet',engine='pyarrow')
    if args.target_type =='res':
        df_targets = df_targets[df_targets["building_stock_type"]=="residential"]
        target_cols = [col for col in df_targets.columns if col.endswith("res") ]
    else:
        df_targets = df_targets[df_targets["building_stock_type"]=="commercial"]
        target_cols = [col for col in df_targets.columns if col.endswith("com") ]
    criterion = MultiLabelLoss(len(target_cols))
    for fold in range(10):
        run(fold,df_targets,target_cols,criterion,kernel_type,args)
    

if __name__ == "__main__":
    main()
