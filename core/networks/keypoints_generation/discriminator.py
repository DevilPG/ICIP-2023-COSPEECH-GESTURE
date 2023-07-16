from torch import nn
import torch

from ..building_blocks import ConvNormRelu


class PoseSequenceDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        leaky = self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.LEAKY_RELU

        self.seq = nn.Sequential(
            ConvNormRelu('1d', cfg.DATASET.NUM_LANDMARKS*2, 256, downsample=True, leaky=leaky),  # B, 256, 64
            ConvNormRelu('1d', 256, 512, downsample=True, leaky=leaky),  # B, 512, 32
            ConvNormRelu('1d', 512, 1024, kernel_size=3, stride=1, padding=1, leaky=leaky),  # B, 1024, 16
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1, bias=True)  # B, 1, 16
        )

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2)
        x = self.seq(x)
        x = x.squeeze(1)
        return x

class LipSequenceDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        leaky = self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.LEAKY_RELU

        self.seq = nn.Sequential(
            ConvNormRelu('1d', cfg.DATASET.NUM_LIP_KPS*2, 64, downsample=True, leaky=leaky),  # B, 64, 64
            ConvNormRelu('1d', 64, 128, downsample=True, leaky=leaky),  # B, 128, 32
            ConvNormRelu('1d', 128, 256, kernel_size=3, stride=1, padding=1, leaky=leaky),  # B, 256, 16
            nn.Conv1d(256, 1, kernel_size=3, stride=1, padding=1, bias=True)  # B, 1, 16
        )
        outchannel = 16
        if self.cfg.VOICE2POSE.POSE_DISCRIMINATOR.MOTION:
            outchannel = 15
        self.mlp = nn.Linear(outchannel, 1)

    def forward(self, kp):
        x = (kp[...,57:77] - kp[...,0:1]).clone() # get lip landmarks
        x = x.reshape(x.size(0), x.size(1), -1).transpose(1, 2) # bs, 40, 64
        x = self.seq(x)
        x = x.squeeze(1)
        x = self.mlp(x)
        x = torch.sigmoid(x)
        return x


class HandDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.left_hand_mlp = nn.Sequential(
            nn.Linear(cfg.DATASET.NUM_HAND_LANDMARKS, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        self.right_hand_mlp = nn.Sequential(
            nn.Linear(cfg.DATASET.NUM_HAND_LANDMARKS, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, kp):
        # get hand landmarks and normalize
        left_hand = kp[...,79:100] - kp[...,6:7]
        left_hand = left_hand.reshape(-1, self.cfg.DATASET.NUM_HAND_LANDMARKS)
        right_hand = kp[...,100:121] - kp[...,3:4]
        right_hand = right_hand.reshape(-1, self.cfg.DATASET.NUM_HAND_LANDMARKS) # B*64, 42

        left_res = torch.sigmoid(self.left_hand_mlp(left_hand)) #B*64, 1
        right_res = torch.sigmoid(self.right_hand_mlp(right_hand)) #B*64, 1
        res = torch.cat([left_res,right_res], dim=0)
        # print(res.shape)
        return res

