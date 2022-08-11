import torch
import torch.nn as nn
from .AttnBlock import getAttentionBlock


class Encoder(nn.Module):
    def __init__(self, attnType='SELayer'):
        super(Encoder, self).__init__()
        # Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.attn2 = getAttentionBlock(attnType, 32)
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.attn3 = getAttentionBlock(attnType, 32)

        # Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.attn6 = getAttentionBlock(attnType, 64)
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.attn7 = getAttentionBlock(attnType, 64)

        # Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.attn10 = getAttentionBlock(attnType, 128)
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.attn11 = getAttentionBlock(attnType, 128)

    def forward(self, x):
        # Conv1
        x = self.layer1(x)
        x = self.attn2(self.layer2(x)) + x
        x = self.attn3(self.layer3(x)) + x
        # Conv2
        x = self.layer5(x)
        x = self.attn6(self.layer6(x)) + x
        x = self.attn7(self.layer7(x)) + x
        # Conv3
        x = self.layer9(x)
        x = self.attn10(self.layer10(x)) + x
        x = self.attn11(self.layer11(x)) + x
        return x


class Decoder(nn.Module):
    def __init__(self, attnType='SELayer'):
        super(Decoder, self).__init__()
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.attn13 = getAttentionBlock(attnType, 128)
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.attn14 = getAttentionBlock(attnType, 128)
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        # Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.attn17 = getAttentionBlock(attnType, 64)
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.attn18 = getAttentionBlock(attnType, 64)
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

        # Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.attn21 = getAttentionBlock(attnType, 32)
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.attn22 = getAttentionBlock(attnType, 32)
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Deconv3
        x = self.attn13(self.layer13(x)) + x
        x = self.attn14(self.layer14(x)) + x
        x = self.layer16(x)
        # Deconv2
        x = self.attn17(self.layer17(x)) + x
        x = self.attn18(self.layer18(x)) + x
        x = self.layer20(x)
        # Deconv1
        x = self.attn21(self.layer21(x)) + x
        x = self.attn22(self.layer22(x)) + x
        x = self.layer24(x)
        return x


class model(nn.Module):

    def __init__(self, attnType='SELayer'):
        super().__init__()

        self.encoder_lv1 = Encoder(attnType)
        self.encoder_lv2 = Encoder(attnType)
        self.encoder_lv3 = Encoder(attnType)

        self.decoder_lv1 = Decoder(attnType)
        self.decoder_lv2 = Decoder(attnType)
        self.decoder_lv3 = Decoder(attnType)

        self.atte1 = getAttentionBlock(attnType, 128)
        self.atte2 = getAttentionBlock(attnType, 128)
        self.atte3 = getAttentionBlock(attnType, 128)

    def forward(self, x):
        H = x.size(2)
        W = x.size(3)

        images_lv1 = x
        images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
        images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]
        images_lv3_1 = images_lv2_1[:, :, :, 0:int(W / 2)]
        images_lv3_2 = images_lv2_1[:, :, :, int(W / 2):W]
        images_lv3_3 = images_lv2_2[:, :, :, 0:int(W / 2)]
        images_lv3_4 = images_lv2_2[:, :, :, int(W / 2):W]

        feature_lv3_1 = self.encoder_lv3(images_lv3_1)
        feature_lv3_2 = self.encoder_lv3(images_lv3_2)
        feature_lv3_3 = self.encoder_lv3(images_lv3_3)
        feature_lv3_4 = self.encoder_lv3(images_lv3_4)

        feature_lv3_top = self.atte1(torch.cat((feature_lv3_1, feature_lv3_2), 3))
        feature_lv3_bot = self.atte1(torch.cat((feature_lv3_3, feature_lv3_4), 3))
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        residual_lv3_top = self.decoder_lv3(feature_lv3_top)
        residual_lv3_bot = self.decoder_lv3(feature_lv3_bot)

        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = self.atte2(torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3)
        residual_lv2 = self.decoder_lv2(feature_lv2)

        feature_lv1 = self.atte3(self.encoder_lv1(images_lv1 + residual_lv2) + feature_lv2)
        deblur_image = self.decoder_lv1(feature_lv1)
        return deblur_image


def get_Generator(attnType='SELayer'):
    return model(attnType)
