import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch import nn
from torch.optim.lr_scheduler import CyclicLR, ConstantLR, SequentialLR, LinearLR
from lightly.models import utils
from lightly.models.modules import masked_autoencoder
from lightly.transforms.mae_transform import MAETransform
import sys
import os
from dataset import ImageDataset
import click
import json
import matplotlib.pyplot as plt

class MAE(nn.Module):
    def __init__(self, 
                 backbone: nn.Module, 
                 masking_rate: float = 0.75, 
                 freeze_projection: bool = True,
                 freeze_embeddings: bool = True, 
                 decoder_dim: int = 1024):
        
        super().__init__()

        self.decoder_dim = decoder_dim
        self.mask_ratio = masking_rate
        self.patch_size = backbone.patch_size
        self.sequence_length = backbone.seq_length
        self.mask_token = nn.Parameter(torch.full([1, 1, decoder_dim], -1.0))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(backbone)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=backbone.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=backbone.hidden_dim,
            hidden_dim=self.decoder_dim,
            mlp_dim=self.decoder_dim * 4,
            out_dim=backbone.patch_size**2 * 3,
            dropout=0,
            attention_dropout=0,
        )
        
        self.freeze_projection = freeze_projection
        self.freeze_embeddings = freeze_embeddings
        
        # freeze conv projection and positional embedding layers
        if self.freeze_projection:
            for param in list(self.backbone.parameters())[:3]:
                param.requires_grad = False
                
        if self.freeze_embeddings:
            list(self.backbone.parameters())[3].requires_grad = False
        
        
        
    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        return x_pred, target

    
def pretrain_mae(dataset: str,
                 output: str = 'checkpoints/ViT_L_16_SEISMIC.pth',
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
                 masking_rate: float = 0.75,
                 decoder_dim: int = 1024,
                 freeze_embeddings: bool = True,
                 freeze_projection: bool = True,
                 shell_call: bool = False) -> dict:
    """
    Pre-train-tune a Vision Transformer (ViT) model on a seismic dataset.

    Args:
        dataset (str): Path to the seismic dataset directory.

    Options:
        output (str): Path to output checkpoint file. Default is 'checkpoints/ViT_L_16_SEISMIC_PRETRAINED.pth'.
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
    
    transform = MAETransform(**transform_kwargs)
    
    # Loading unlabeled image dataset from folder
    dataset = torchvision.datasets.ImageFolder(root=dataset, transform=transform)
    
    if local_checkpoint:
        backbone = torchvision.models.get_model(vit_model)
        backbone.load_state_dict(torch.load(starting_weights))
        print(f'Local checkpoint loaded: {starting_weights}')
    else:
        weights = torchvision.models.get_weight(starting_weights)
        backbone = torchvision.models.get_model(vit_model, weights=weights)
        print(f'Pretrained checkpoint loaded: {starting_weights}')
    
    model = MAE(backbone, 
                masking_rate=masking_rate, 
                decoder_dim=decoder_dim, 
                freeze_embeddings=freeze_embeddings,
                freeze_projection=freeze_projection)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
    )

    criterion = nn.MSELoss()
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
            predictions, targets = model(images)
            loss = criterion(predictions, targets)
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
@click.option('--masking-rate', type=float, default=0.75, help='Masking rate for pretraining')
@click.option('--decoder-dim', type=int, default=1024, help='Dimension of the decoder tokens')       
@click.option('--freeze-projection', type=bool, default=False, is_flag=True, help='Freeze class token and convolutional projection layer of ViT') 
@click.option('--freeze-embeddings', type=bool, default=False, is_flag=True, help='Freeze positional embedding layer of ViT')  
def main(dataset, output, transform_min_scale, transform_normalize, vit_model,
         local_checkpoint, starting_weights, batch_size, n_workers, optimizer, 
         lr, optimizer_params, n_epochs, warmup_epochs, start_factor, linear_schedule,
         end_factor, cyclic_schedule, cyclic_step_size, masking_rate, decoder_dim, 
         freeze_projection, freeze_embeddings):

    transform_kwargs = {'min_scale': transform_min_scale, 'normalize': transform_normalize}
    optimizer_kwargs = {'lr': lr, **dict(optimizer_params)}
       
    pretrain_mae(dataset, 
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
                 masking_rate=masking_rate,
                 decoder_dim = decoder_dim,
                 freeze_projection=freeze_projection,
                 freeze_embeddings=freeze_embeddings,
                 shell_call=True)
              
if __name__ == '__main__':
    main()