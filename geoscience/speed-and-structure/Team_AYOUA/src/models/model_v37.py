import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def process_input(x,image_size):
    # x: (N, 5, 10001, 31)
    h, w = image_size
    N = x.size(0)
    x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
    return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_patches_h, num_patches_w, target_h, target_w):
        super().__init__()
        self.target_h = target_h
        self.target_w = target_w
        
       

        self.initial_channels = 512 # Number of channels after initial projection
        self.proj_linear = nn.Linear(embed_dim, self.initial_channels)
        
        
        self.up1 = nn.ConvTranspose2d(self.initial_channels, 256, kernel_size=4, stride=2, padding=1) 
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)   
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)   
        
        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x_tokens, num_patches_h, num_patches_w):
        # x_tokens: (N, num_patches, embed_dim) -- assuming CLS token is removed
        bs, num_patches, embed_dim = x_tokens.shape

        # 1. Project each token to initial_channels
        # Result: (N, num_patches, initial_channels)
        x = self.proj_linear(x_tokens)

        # 2. Reshape into a spatial feature map
        # Result: (N, initial_channels, num_patches_h, num_patches_w)
        x = x.transpose(1, 2).reshape(bs, self.initial_channels, num_patches_h, num_patches_w)

        # 3. Upsample through ConvTranspose2d layers
        x = F.relu(self.up1(x)) # (N, 256, 2*num_patches_h, 2*num_patches_w)
        x = F.relu(self.up2(x)) # (N, 128, 4*num_patches_h, 4*num_patches_w)
        x = F.relu(self.up3(x)) # (N, 64, 8*num_patches_h, 8*num_patches_w)
        x = F.relu(self.up4(x)) # (N, 32, 16*num_patches_h, 16*num_patches_w)
        
        
        # 4. Final convolution to get 1 channel
        x = self.final_conv(x) 

        # 5. Final interpolation to target size
        x = F.interpolate(x, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)
        return x

class FWIModel(nn.Module):
    def __init__(self,base_model,image_size,patch_size, 
                 pretrained: bool = True, split_at=4, fusion_method='mean'):
        super().__init__()
        self.output_h = 1259
        self.output_w = 300
        self.output_tokens = patch_size * patch_size
        self.fusion_method = fusion_method
        

        backbone = timm.create_model(
            base_model,
            pretrained=pretrained,
            dynamic_img_size=True,
            in_chans=1,
        )
        self.split_at = split_at
        self.image_size = image_size
        self.patch_size = patch_size
        self.backbone = backbone

        # Calculate num_patches_h, num_patches_w based on image_size and patch_size
        self.num_patches_h = image_size[0] // patch_size 
        self.num_patches_w = image_size[1] // patch_size 
        if  fusion_method == 'weighted':
            self.channel_weights = nn.Parameter(torch.ones(5)/5)
        
        # Get the embedding dimension from the backbone
        embed_dim = backbone.embed_dim # For "eva02_base_patch16_clip_224", embed_dim is 768

        # Pass num_patches_h and num_patches_w to the decoder
        self.decoder = Decoder(embed_dim, self.num_patches_h, self.num_patches_w, self.output_h, self.output_w)
    def _fuse_channels(self, x):
        """Apply the selected fusion method to the channel dimension"""
        N = x.shape[0]  # Batch size
        
        if self.fusion_method == 'mean':
            return x.mean(1)
        
        elif self.fusion_method == 'weighted':
            weights = F.softmax(self.channel_weights, dim=0)
            return (x * weights.view(1, 5, 1, 1)).sum(1)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def forward(self, x):
        x = process_input(x,self.image_size) # (N, 1, H, W)
        N, C, H, W = x.size()
        x = self.backbone.patch_embed(x.reshape(N * C, 1, H, W))  # Patchify
        x, rot_pos_embed = self.backbone._pos_embed(x)
        for blk in self.backbone.blocks[: self.split_at]:
            x = blk(x, rope=rot_pos_embed)
        x = x.reshape(N, C, *x.shape[1:])  # Reshape: (N, 5, num_tokens, dim)
        x = self._fuse_channels(x)

        for blk in self.backbone.blocks[self.split_at:]:
            x = blk(x, rope=rot_pos_embed)

        x = self.backbone.norm(x)  # (N, num_tokens, dim)
        
        # Remove CLS token for the decoder
        x_tokens = x[:, 1:, :] # (N, num_patches, embed_dim)

        # Pass tokens to the decoder
        masks = self.decoder(x_tokens, self.num_patches_h, self.num_patches_w)

        # Apply final activation and scaling
        masks = F.sigmoid(masks)
        masks = (masks*3)+1.5

        return masks.squeeze(1).transpose(1, 2)  # (N, H, W)

