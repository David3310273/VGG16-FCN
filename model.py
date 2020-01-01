import torchvision.models as models
import torch.nn as nn
import torch


class MyFCN(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.vgg16(pretrained=True)
        self.backbone_third = model.features[:17]  # (256, 28, 28) third pooling before conv layer
        self.backbone_fourth = model.features[:24] # (512, 14, 14) fourth pooling before conv layer
        self.backbone_fifth = model.features[:31]  # (512, 7, 7) final pooling before conv layer

        self.fcn_block_1 = nn.Sequential(          # (1, 7, 7)
            nn.Conv2d(512, 1, (1, 1))
        )

        self.fcn_block_1_1 = nn.Sequential(  # (1, 7, 7)
            nn.Conv2d(256, 1, (1, 1))
        )

        self.fcn_block_2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(1, 1, (3, 3), (1, 1)),
            nn.ReLU()
        )

        self.fcn_block_8 = nn.Sequential(
            nn.Upsample(scale_factor=8),
            nn.ConvTranspose2d(1, 1, (3, 3), (1, 1), 1),
            nn.ReLU()
        )

        self.fcn_block_16 = nn.Sequential(
            nn.Upsample(scale_factor=16),
            nn.ConvTranspose2d(1, 1, (3, 3), (1, 1), 1),
            nn.ReLU()
        )

        self.fcn_block_32 = nn.Sequential(
            nn.Upsample(scale_factor=32),
            nn.ConvTranspose2d(1, 1, (3, 3), (1, 1), 1),
            nn.ReLU()
        )

        self.seg_head = nn.Linear(3, 1)

    def forward(self, x):
        # Get basic feature map
        x_1 = self.backbone_fifth(x)       # (512, 7, 7)
        x_1 = self.fcn_block_1(x_1)        # (1, 7, 7)
        x_1 = self.fcn_block_32(x_1)       # (1, 224, 224)

        x_2 = self.backbone_fourth(x)      # (512, 14, 14)
        x_2 = self.fcn_block_1(x_2)        # (1, 14, 14)
        x_2 = self.fcn_block_16(x_2)       # (1, 224, 224)

        x_3 = self.backbone_third(x)       # (256, 28, 28)
        x_3 = self.fcn_block_1_1(x_3)      # (1, 28, 28)
        x_3 = self.fcn_block_8(x_3)        # (1, 224, 224)

        # # Fuse the feature map
        x_cat = torch.cat([x_1, x_2, x_3], dim=0)   # (3, 1, 224, 224)
        x_fused = torch.sum(x_cat, 0).unsqueeze(1)    # (1, 1, 224, 224)

        return x_fused
