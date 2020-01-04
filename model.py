import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from visualization import *

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.ini"))
is_debug = bool(config["app"]["debug"])

# referred to this site: https://github.com/Gurupradeep/FCN-for-Semantic-Segmentation
class MyFCN(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.vgg16(pretrained=True)
        self.backbone_third = model.features[:17]  # (256, 28, 28) third pooling before conv layer
        self.backbone_fourth = model.features[:24] # (512, 14, 14) fourth pooling before conv layer
        self.backbone_fifth = model.features[:31]  # (512, 7, 7) final pooling before conv layer

        self.conv_256_1 = nn.Sequential(
            nn.Conv2d(256, 1, (1, 1), 1),
        )

        self.conv_512_1 = nn.Sequential(
            nn.Conv2d(512, 1, (1, 1), 1),
        )

        # fc6
        self.conv_512_4096 = nn.Sequential(
            nn.Conv2d(512, 4096, (7, 7), 1, 3),
            nn.ReLU(inplace=True),
        )

        # fc7
        self.conv_4096_4096 = nn.Sequential(
            nn.Conv2d(4096, 4096, (1, 1), 1),
            nn.ReLU(inplace=True),
        )

        # score_fr
        self.conv_4096_1 = nn.Sequential(
            nn.Conv2d(4096, 1, (1, 1), 1),
            nn.ReLU(inplace=True),
        )

        # score_2 for 7=>14 and 14=>28
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(1, 1, (4, 4), 2),
        )

        # final upsample
        self.conv_transpose_8 = nn.Sequential(
            nn.ConvTranspose2d(1, 1, (16, 16), 8),
        )


    def forward(self, x):
        x_from_pooling_3 = self.backbone_third(x)
        x_from_pooling_4 = self.backbone_fourth(x)
        x_from_pooling_5 = self.backbone_fifth(x)

        # pooling 3
        x_3 = self.conv_256_1(x_from_pooling_3)

        # pooling 4
        x_4 = self.conv_512_1(x_from_pooling_4)     # (1, 1, 14, 14)

        # pooling 5
        x_5 = self.conv_512_4096(x_from_pooling_5)  # (1, 4096, 7, 7)
        x_5 = self.conv_4096_4096(x_5)              # (1, 4096, 7, 7)
        x_5 = self.conv_4096_1(x_5)                 # (1, 1, 7, 7)
        x_5 = self.conv_transpose(x_5)              # (1, 1, 16, 16)
        x_5 = F.pad(x_5, (-1, -1, -1, -1))          # crop layer, (1, 1, 14, 14)

        # fusing x_4
        x_fused_1 = x_4 + x_5                       # (1, 1, 14, 14)
        x_fused_1 = self.conv_transpose(x_fused_1)  # (1, 1, 30, 30)
        x_fused_1 = F.pad(x_fused_1, (-1, -1, -1, -1))  # crop layer, (1, 1, 28, 28)

        # fusing x_3
        x_fused_2 = x_3 + x_fused_1
        x_fused_2 = self.conv_transpose_8(x_fused_2)    # (1, 1, 232, 232)
        x_fused_2 = F.pad(x_fused_2, (-4, -4, -4, -4))  # crop layer (1, 1, 224, 224)

        return x_fused_2
