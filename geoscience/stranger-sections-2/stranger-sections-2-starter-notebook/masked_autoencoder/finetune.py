import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import CyclicLR, ConstantLR, SequentialLR, LinearLR
from lightly.models import utils
from lightly.models.modules import masked_autoencoder
from lightly.transforms.mae_transform import MAETransform
from masked_autoencoder.dataset import ImageDataset
from timm.models.vision_transformer import Block
from torchvision.transforms import v2
import sys
import os
import click
import json
import matplotlib.pyplot as plt

class SegmentationHead(nn.Module):
    '''
    Segmentation head for segmentation task
    
    Args:
        channels: int, number of channels in the input tensor
        num_classes: int, number of classes in the segmentation task
        num_features: int, number of features to be concatenated
    
    Returns:
        x: tensor, output tensor of the segmentation head
    '''
    def __init__(self, channels, num_classes, num_features=1):
        super().__init__()
        self.embedding_layer = nn.Linear(self.head_embed_dim, self.patch_shape[0]*self.patch_shape[1], bias=True)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * num_features, channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(channels)
            )
        self.predict = nn.Conv2d(channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        return x

class SegmentationViT(nn.Module):
    def __init__(self, 
                 backbone: nn.Module,  
                 freeze_projection: bool = True,
                 freeze_embeddings: bool = True, 
                 decoder_embed_dim: int = 1024,
                 norm_layer: nn.Module = nn.LayerNorm,
                 decoder_depth: int = 8,
                 decoder_num_heads: int = 16,
                 mlp_ratio: int = 4.,
                 in_channels: int = 3,
                 classes: int = 4
                 ):
        
        super().__init__()

        self.img_size = [backbone.img_size, backbone.img_size]
        self.embed_dim = backbone.embed_dim
        self.patch_size = backbone.patch_size
        self.num_patches = backbone.num_patches
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.sequence_length = backbone.seq_length
        self.in_channels = in_channels
        self.classes = classes
        
        #-------------------------------------------------------------------------
        # Encoder parameters taken from preloaded ViT model checkpoint
        
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(backbone)

        #-------------------------------------------------------------------------
        # Decoder parameters taken from Meta VitMAE research repo
        # https://github.com/facebookresearch/mae/blob/main/models_mae.py

        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True) #embedding layer acts as connection between encoder and decoder
        self.mask_token = nn.Parameter(torch.full([1, 1, decoder_embed_dim], -1.0))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # basic segmentation head based on SegFormer implementation: https://github.com/FrancescoSaverioZuppichini/SegFormer
        self.decoder_pred = SegmentationHead(self.embed_dim, num_classes=self.classes, num_features=1)
        # -----------------------------------------------------------------------
        
        self.freeze_projection = freeze_projection
        self.freeze_embeddings = freeze_embeddings
        
        # freeze conv projection and positional embedding layers
        if self.freeze_projection:
            for param in list(self.backbone.parameters())[:3]:
                param.requires_grad = False
                
        if self.freeze_embeddings:
            list(self.backbone.parameters())[3].requires_grad = False
        
        
        
    def forward_encoder(self, x, idx_keep=None):
        '''
        Encoder forward pass. Taken directly from pretrained ViT model. Set idx_keep to None for finetune training.
        '''
        x = self.backbone.encode(x, idx_keep)
        return x

    def forward_decoder(self, x):
        '''
        Modified forward pass for the decoder from Meta's VitMAE research repo.
        https://github.com/facebookresearch/mae/blob/main/models_mae.py
        '''

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], 160 + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # use the segmentation head to predict the segmentation mask
        x = self.decoder_pred(x)

        return x
    
    def forward(self, image, label):
        '''
        Forward pass for the model. Returns the prediction and loss.
        '''
        latent = self.forward_encoder(image)
        pred = self.forward_decoder(latent)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        loss = criterion(pred, label)

        return pred, loss

    
def finetune_vit(dataset: str,
                 output: str = 'checkpoints/ViT_L_16_pretrained.pth',
                 transform_kwargs: dict = {'min_scale': 0.2, 'normalize': False},
                 vit_model: str = 'ViT_L_16', 
                 starting_weights: str = "ViT_L_16_Weights.DEFAULT", 
                 local_checkpoint: bool = False,
                 batch_size: int=64,
                 n_workers: int = 4,
                 optimizer: str = 'SGD',
                 optimizer_kwargs: dict = {'lr': 5e-5, 'momentum': 0.0},
                 warmup_epochs: int = 10,
                 start_factor: float = 0.2,
                 linear_schedule: bool = True,
                 end_factor: float = 0.5,
                 n_epochs: int = 50,
                 cyclic_schedule: bool = False,
                 cyclic_step_size: int = 1000,
                 decoder_dim: int = 1024,
                 freeze_embeddings: bool = True,
                 freeze_projection: bool = True,
                 shell_call: bool = False) -> dict:
    """
    Fine-tune a Vision Transformer (ViT) model on a custom image dataset using a Masked Autoencoder (MAE) backbone.

    Args:
        dataset (str): Path to the seismic dataset directory.

    Options:
        output (str): Path to output checkpoint file. .pth'.
        transform_kwargs (dict): Dictionary containing transformation arguments.
            - min_scale (float): Minimum scale for data transformation. Default is 0.2.
            - normalize (bool): Normalize the data during transformation. Default is False.
        vit_model (str): ViT model type. Default is 'ViT_L_16'.
        starting_weights (str): ViT starting weights. Default is 'ViT_L_16_Weights.DEFAULT'.
        local_checkpoint (bool): load weights from local checkpoint.
        batch_size (int): Batch size. Default is 64.
        n_workers (int): Number of workers for data loader. Default is 4.
        optimizer (str): Optimizer type. Default is 'SGD'.
        optimizer_kwargs (dict): Dictionary containing additional keyword arguments passed to optimizer init.
            Default is {'lr': 5e-5, 'momentum': 0.0}.
        warmup_epochs (int): Number of warmup epochs. Default is 10.
        start_factor (float): Initial LR decrease for warmup. Default is 0.2.
        linear_schedule (bool): Use linear LR schedule after warmup period. Default is True.
        end_factor (float): Final decay factor for linear LR schedule. Default is 0.5.
        n_epochs (int): Number of training epochs. Default is 50.
        cyclic_schedule (bool): Use pre-optimized cyclic LR schedule. Default is False.
        cyclic_step_size (int): Number for iterations for half of LR cycle. Default is 1000.
        masking_rate (float): Masking rate for pretraining. Default is 0.75.
        decoder_dim (int): Dimension of the decoder tokens. Default is 1024.
        freeze_projection (bool): Freeze convolutional projection layer of ViT. Default is True.
        freeze_embeddings (bool): Freeze embedding layer of ViT. Default is True.
       
    Returns:
        Dictionary containing loss for each epoch of training
    """
    
    def get_schedulers(optimizer):
        base_lr = optimizer_kwargs.get('lr')
        warmup = warmup_epochs > 0  
        if linear_schedule:
            linear_scheduler = LinearLR(optimizer, start_factor=1.0, total_iters=n_epochs, end_factor=end_factor)

        else:
            # placeholder
            linear_scheduler = ConstantLR(optimizer, factor=1.0)
        
        cyclic_scheduler = None
        if cyclic_schedule:
            cyclic_scheduler = CyclicLR(optimizer, base_lr=base_lr/4.0, max_lr=base_lr,
                                        step_size_up=cyclic_step_size, mode='exp_range', gamma=0.99994)

        if warmup:
            warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warmup_epochs)
            sequential_scheduler = SequentialLR(optimizer, [warmup_scheduler, linear_scheduler], milestones=[warmup_epochs])
            return sequential_scheduler, cyclic_scheduler
        
        return linear_scheduler, cyclic_scheduler
    
    if local_checkpoint:
        backbone = torchvision.models.get_model(vit_model)
        backbone.load_state_dict(torch.load(starting_weights))
        print(f'Local checkpoint loaded: {starting_weights}')
    else:
        weights = torchvision.models.get_weight(starting_weights)
        backbone = torchvision.models.get_model(vit_model, weights=weights)
        print(f'Pretrained checkpoint loaded: {starting_weights}')
    
    model = SegmentationViT(backbone, 
                decoder_dim=decoder_dim, 
                freeze_embeddings=freeze_embeddings,
                freeze_projection=freeze_projection)
    
    transform = MAETransform(**transform_kwargs)
    
    dataset = ImageDataset(root=dataset, img_size=model.img_size, transform=transform, target_transform=transform)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_kwargs)
    elif optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    total_epochs = n_epochs if warmup_epochs < 1 else n_epochs + warmup_epochs
    main_scheduler, cyclic_scheduler = get_schedulers(optimizer)
    loss_history = dict()
    print("Entering Training Loop")
    for epoch in range(total_epochs):
        total_loss = .0
        for batch in dataloader:
            views = batch[0]
            images = views[0].to(device)  # views contains only a single view
            loss, pred = model(images)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if cyclic_schedule:
                cyclic_scheduler.step()
        avg_loss = total_loss / len(dataloader)
        current_lr = main_scheduler.get_last_lr()[-1]
        print(f"epoch: {epoch:>03}, loss: {avg_loss:.5f}, base_lr: {current_lr:.7f}")
        main_scheduler.step()
        loss_history.update({epoch: {'loss': avg_loss.item(), 'base_lr': current_lr}})
    print("Training Completed")
    torch.save(model.backbone.state_dict(), output)
    if shell_call:
        checkpoint_dir, checkpoint_name = os.path.split(output)
        report_path = checkpoint_dir + '/' + checkpoint_name.split('.')[0] + '_report.json'
        with open(report_path, 'w+') as f:
            json.dump(loss_history, f)
        return
    return loss_history
    
    
@click.command(context_settings={'show_default': True})
@click.argument('dataset', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), default='ViT_L_16_SEISMIC.pth', help='Path to output checkpoint')
@click.option('--transform-min-scale', type=float, default=0.2, help='Minimum scale for data transformation')
@click.option('--transform-normalize', type=bool, is_flag=True, default=False, help='Normalize the data during transformation')
@click.option('--vit-model', type=str, default='ViT_L_16', help='ViT model type')
@click.option('--local-checkpoint', type=bool, default=False, is_flag=True, help='Use local checkpoint of ViT')
@click.option('--starting-weights', type=str, default='ViT_L_16_Weights.DEFAULT', help='ViT starting weights or path to local checpoint')
@click.option('--batch-size', type=int, default=64, help='Batch size')
@click.option('--n-workers', type=int, default=4, help='Number of workers for data loader')
@click.option('--optimizer', type=str, default='SGD', help='Optimizer type')
@click.option('--lr', type=float, default=5e-5, help='Learning rate for the optimizer')
@click.option('--optimizer-params', type=(str, float), multiple=True, default=[['momentum', 0.0]], help='Additional keyword arguments passed to optimizer init')
@click.option('--n-epochs', type=int, default=50, help='Number of training epochs')
@click.option('--warmup-epochs', type=int, default=10, help='Number of warmup epochs')
@click.option('--start-factor', type=float, default=0.2, help='Initial LR decrease for warmup')
@click.option('--linear-schedule', type=bool, default=False, is_flag=True, help='Use linear LR schedule after warmup period')
@click.option('--end-factor', type=float, default=0.5, help='Final decay factor for linear LR schedule')
@click.option('--cyclic-schedule', type=bool, default=False, is_flag=True, help='Use pre-optimized cyclic LR schedule')
@click.option('--cyclic-step-size', type=int, default=1000, help='Number for iterations for half of LR cycle')
@click.option('--decoder-dim', type=int, default=1024, help='Dimension of the decoder tokens')       
@click.option('--freeze-projection', type=bool, default=False, is_flag=True, help='Freeze class token and convolutional projection layer of ViT') 
@click.option('--freeze-embeddings', type=bool, default=False, is_flag=True, help='Freeze positional embedding layer of ViT')  
def main(dataset, output, transform_min_scale, transform_normalize, vit_model,
         local_checkpoint, starting_weights, batch_size, n_workers, optimizer, 
         lr, optimizer_params, n_epochs, warmup_epochs, start_factor, linear_schedule,
         end_factor, cyclic_schedule, cyclic_step_size, decoder_dim, 
         freeze_projection, freeze_embeddings):

    transform_kwargs = {'min_scale': transform_min_scale, 'normalize': transform_normalize}
    optimizer_kwargs = {'lr': lr, **dict(optimizer_params)}
       
    finetune_vit(dataset, 
                 output, 
                 transform_kwargs=transform_kwargs,
                 vit_model=vit_model,
                 starting_weights=starting_weights,
                 local_checkpoint=local_checkpoint,
                 batch_size=batch_size, 
                 n_workers=n_workers, 
                 optimizer=optimizer, 
                 optimizer_kwargs=optimizer_kwargs, 
                 n_epochs=n_epochs, 
                 warmup_epochs=warmup_epochs,
                 start_factor=start_factor,
                 linear_schedule=linear_schedule,
                 end_factor=end_factor,
                 cyclic_schedule=cyclic_schedule,
                 cyclic_step_size=cyclic_step_size,
                 decoder_dim = decoder_dim,
                 freeze_projection=freeze_projection,
                 freeze_embeddings=freeze_embeddings,
                 shell_call=True)
              
if __name__ == '__main__':
    main()