import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from ..building_blocks import ConvNormRelu


class PoseSeqEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.POSE2POSE.AUTOENCODER.LEAKY_RELU
        norm = cfg.POSE2POSE.AUTOENCODER.NORM
        out_channels = cfg.POSE2POSE.AUTOENCODER.CODE_DIM * 2  # 32 * 2
        in_channels = cfg.DATASET.NUM_LANDMARKS * 2  # 121 * 2

        self.blocks = nn.Sequential(
            ConvNormRelu('1d', in_channels, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, out_channels, downsample=True, norm=norm, leaky=leaky),
        )

    def forward(self, x):
        x = x.reshape(list(x.shape[:2]) + [-1]).permute([0, 2, 1])  # [B, 242, 64]

        x = self.blocks(x)  # [B, 64, 2]

        x = F.interpolate(x, 1).squeeze(-1)  # [B,64]

        mu = x[:, 0::2]  # [B,32]
        logvar = x[:, 1::2]  # [B,32]
        return mu, logvar


class PoseSeqDecoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.POSE2POSE.AUTOENCODER.LEAKY_RELU
        norm = cfg.POSE2POSE.AUTOENCODER.NORM
        in_channels = cfg.POSE2POSE.AUTOENCODER.CODE_DIM

        self.d5 = ConvNormRelu('1d', in_channels, 256, downsample=False, norm=norm, leaky=leaky)
        self.d4 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d3 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d2 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)

        self.blocks = nn.Sequential(
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_LANDMARKS * 2, kernel_size=1, bias=True)
        )

    def forward(self, x):
        x = F.interpolate(x.unsqueeze(-1), 2)

        x = self.d5(F.interpolate(x, x.shape[-1] * 2, mode='linear'))
        x = self.d4(F.interpolate(x, x.shape[-1] * 2, mode='linear'))
        x = self.d3(F.interpolate(x, x.shape[-1] * 2, mode='linear'))
        x = self.d2(F.interpolate(x, x.shape[-1] * 2, mode='linear'))
        x = self.d1(F.interpolate(x, x.shape[-1] * 2, mode='linear'))

        x = self.blocks(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=1024, num_layers=1):
        super(Encoder, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)

    def forward(self, x):
        outputs, h_n = self.rnn(x)
        ## TODO include attention
        return outputs


class DecoderCell(nn.Module):
    def __init__(self, hidden_size=1024, output_size=None, use_h=False, use_lang=False):
        super(DecoderCell, self).__init__()
        self.use_h = 1 if use_h else 0
        self.use_lang = 1 if use_lang else 0

        self.rnn = nn.GRUCell(input_size=output_size,
                              hidden_size=hidden_size)

        self.tp = nn.Linear(hidden_size, output_size)

        if self.use_lang:
            self.lin = nn.Linear(hidden_size + output_size, output_size)

    def forward(self, x, h):
        x = x.to(torch.float32)
        h = h.to(torch.float32)

        if self.use_h:
            x_ = torch.cat([x, h], dim=-1)
        else:
            x_ = x

        h_n = self.rnn(x_, h)
        ## TODO add attention
        tp_n = self.tp(h_n)
        if self.use_lang:
            y = self.lin(x) + tp_n
        else:
            y = x + tp_n
        return y, h_n


class TeacherForcing():
    '''
    Sends True at the start of training, i.e. Use teacher forcing maybe.
    Progressively becomes False by the end of training, start using gt to train
    '''

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch

    def __call__(self, epoch, batch_size=1):
        p = epoch * 1. / self.max_epoch
        random = torch.rand(batch_size)
        return (p < random).double()


class Decoder(nn.Module):
    def __init__(self, hidden_size, input_size=None,
                 use_h=False, start_zero=False,
                 use_lang=False, use_attn=False):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.cell = DecoderCell(hidden_size, input_size,
                                use_h=use_h, use_lang=use_lang)
        ## Hardcoded to reach 0% Teacher forcing in 10 epochs
        self.tf = TeacherForcing(0.1)
        self.start_zero = start_zero
        self.use_lang = use_lang
        self.use_attn = use_attn

    def forward(self, h, time_steps, gt, epoch=np.inf, attn=None):
        if self.use_lang:
            lang_z = h
        if self.start_zero:
            x = h.new_zeros(h.shape[0], self.input_size)
            x = h.new_tensor(torch.rand(h.shape[0], self.input_size))
        else:
            x = gt[:, 0, :]  ## starting point for the decoding

        Y = []
        for t in range(time_steps):
            if self.use_lang:
                if self.use_attn:  ### calculate attention at each time-step
                    lang_z = attn(h)
                x, h = self.cell(torch.cat([x, lang_z], dim=-1), h)
            else:
                x, h = self.cell(x, h)
            Y.append(x.unsqueeze(1))
            if t > 0:
                mask = self.tf(epoch, h.shape[0]).double().view(-1, 1).to(x.device)
                x = mask * gt[:, t - 1, :] + (1 - mask) * x
        return torch.cat(Y, dim=1)


# class Autoencoder(nn.Module):
#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.cfg = cfg
#
#         self.encoder = PoseSeqEncoder(cfg)
#         self.decoder = PoseSeqDecoder(cfg)
#
#     def forward(self, x, num_frames, mel=None, external_code=None):
#
#         if external_code is not None:
#             x = self.decoder(external_code)
#             x = x.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)
#             return x, external_code, torch.zeros_like(external_code)
#
#         mu, logvar = self.encoder(x)
#
#         eps = torch.randn(logvar.shape, device=logvar.device)
#         code = mu + torch.exp(0.5*logvar) * eps   # [B,32]
#
#
#         x = self.decoder(code)   # [B, 242, 64]
#
#         x = x.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)
#
#         return x, mu.squeeze(-1), logvar.squeeze(-1)


# class Autoencoder(nn.Module):
#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.cfg = cfg
#
#         self.encoder = Encoder(input_size=242, hidden_size=1024)
#         self.decoder = Decoder(hidden_size=1024, input_size=242)
#
#     def forward(self, pose, num_frames, mel=None, external_code=None):
#         pose = pose.reshape(list(pose.shape[:2]) + [-1])  # (B, 64,242)
#
#         pose_feat = self.encoder(pose)  # (B, 64, 1024)
#
#         x = self.decoder(pose_feat[:, -1, :], pose.shape[1], gt=pose)  # (B, 64, 242)
#
#         x = x.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)
#
#         return x


class Autoencoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM

        self.encoder = Encoder(input_size=242, hidden_size=512)
        self.decoder = nn.Sequential(
            ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 512, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_LANDMARKS*2, kernel_size=1, bias=True)
            )

    def forward(self, pose, num_frames, mel=None, external_code=None):
        pose = pose.reshape(list(pose.shape[:2]) + [-1])  # (B, 64,242)

        pose_feat = self.encoder(pose).permute([0,2,1])  # (B, 512, 64)

        x = self.decoder(pose_feat).permute([0,2,1])  # (B, 64, 242)

        x = x.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)

        return x
