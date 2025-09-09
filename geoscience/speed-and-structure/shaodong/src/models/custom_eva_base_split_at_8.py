import torch.nn as nn
import torch.nn.functional as F
import timm

def process_input(x):
    N, C, H, W = x.size()
    x = x.reshape(N, C, 1000, 10, 31)
    x = x.permute(0, 1, 2, 4, 3)
    x = x.reshape(N, C, 1000, 310)
    # 14*90, 14*22
    x = F.interpolate(x, size=(1260, 308), mode="bilinear", align_corners=False)
    return x


class SASModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = timm.create_model(
            "eva02_base_patch14_448.mim_in22k_ft_in22k",
            pretrained=True,
            dynamic_img_size=True,
            in_chans=1,
            patch_size=14,
        )
        self.backbone = backbone
        self.r = 14
        self.head = nn.Linear(768, self.r * self.r)
        self.pixel_shuffle = nn.PixelShuffle(self.r)
        self.split_at = 8

    def forward(self, x):
        x = process_input(x)
        N, C, H, W = x.size()
        x = self.backbone.patch_embed(x.reshape(N * C, 1, H, W))
        x, rot_pos_embed = self.backbone._pos_embed(x)
        for blk in self.backbone.blocks[: self.split_at]:
            x = blk(x, rope=rot_pos_embed)
        x = x.reshape(N, C, *x.size()[1:])
        x = x.mean(1)
        for blk in self.backbone.blocks[self.split_at :]:
            x = blk(x, rope=rot_pos_embed)
        x = self.backbone.norm(x)

        x = x[:, 1:]
        x = self.head(x)
        x = x.permute(0, 2, 1).reshape(N, self.r * self.r, 90, 22)
        x = self.pixel_shuffle(x)
        x = x[:, :, :-1, 4:-4]
        x = F.sigmoid(x.float()) * 3.0 + 1.5
        return x