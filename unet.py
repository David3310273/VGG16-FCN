import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pdb


class MyUNet(nn.Module):

    def __init__(self):
        super(MyUNet, self).__init__()

        # self.dropout_input = nn.Dropout2d(p=0.01)

        self.backbone = list(models.vgg16(pretrained=True).children())
        main_block = list(self.backbone[0].children())
        self.scale_factor = 32  # 4x1x2x2x2
        self.backbone_layer0 = nn.Sequential(*main_block[:5])
        self.backbone_layer1 = nn.Sequential(*main_block[5:10])
        self.backbone_layer2 = nn.Sequential(*main_block[10:17])
        self.backbone_layer3 = nn.Sequential(*main_block[17:24])
        self.backbone_layer4 = nn.Sequential(*main_block[24:31])

        # self.dropout_feature = nn.Dropout2d(p=0.05)

        self.seg_block1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(512, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.seg_block2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(576, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.seg_block3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(288, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.seg_block4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(144, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.seg_block5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(80, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.seg_head = nn.Linear(11, 1)

    def forward(self, x_0):  # torch.Size([1, 3, 224, 224])
        assert (x_0.shape[2] % self.scale_factor == 0)
        assert (x_0.shape[3] % self.scale_factor == 0)  # torch.Size([4, 3, 288, 480])

        ##############################################

        x_1 = self.backbone_layer0(x_0)  # torch.Size([1, 64, 112, 112])
        x_2 = self.backbone_layer1(x_1)  # torch.Size([1, 128, 56, 56])
        x_3 = self.backbone_layer2(x_2)  # torch.Size([1, 256, 28, 28])
        x_4 = self.backbone_layer3(x_3)  # torch.Size([1, 512, 14, 14])
        x_5 = self.backbone_layer4(x_4)  # torch.Size([1, 512, 7, 7])

        # x_5 = self.dropout_feature(x_5)

        x_4_u = self.seg_block1(x_5)  # torch.Size([1, 64, 14, 14])
        x_4_c = torch.cat((x_4_u, x_4), 1)  # torch.Size([1, 576, 14, 14])

        x_3_u = self.seg_block2(x_4_c)  # torch.Size([1, 32, 28, 28])
        x_3_c = torch.cat((x_3_u, x_3), 1)  # torch.Size([1, 288, 28, 28])

        x_2_u = self.seg_block3(x_3_c)  # torch.Size([1, 16, 56, 56])
        x_2_c = torch.cat((x_2_u, x_2), 1)  # torch.Size([1, 144, 56, 56])

        x_1_u = self.seg_block4(x_2_c)  # torch.Size([1, 16, 112, 112])
        x_1_c = torch.cat((x_1_u, x_1), 1)  # torch.Size([4, 80, 112, 112])

        x_0_u = self.seg_block5(x_1_c)  # torch.Size([1, 8, 224, 224])
        x_0_c = torch.cat((x_0_u, x_0), 1)  # torch.Size([1, 11, 224, 224])

        x_seg = x_0_c.permute(0, 2, 3, 1)  # torch.Size([1, 224, 224, 11])
        x_seg = self.seg_head(x_seg)  # torch.Size([1, 224, 224, 1])

        ##############################################

        x_seg = x_seg.permute(0, 3, 1, 2)

        return x_seg
