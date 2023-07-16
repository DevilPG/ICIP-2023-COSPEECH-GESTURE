from faulthandler import cancel_dump_traceback_later
from turtle import forward
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import librosa
from torch.cuda.amp import autocast
import random

from .building_blocks import ConvNormRelu
from core.deepspeech import DeepSpeech, SpectrogramParser, load_model, TranscribeConfig


class GenerateDeepspeechScores(nn.Module):
    def __init__(self, s2g_cfg) -> None:
        super().__init__()
        self.cfg = TranscribeConfig()
        self.device = torch.device("cuda" if self.cfg.model.cuda else "cpu")
        print(self.cfg.model.model_path)
        self.Deepspeech_Model = load_model(
            device=self.device,
            model_path=s2g_cfg.DEEPSPEECH.CKPT_PATH
        )
        audio_conf = self.Deepspeech_Model.spect_cfg
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window.value
        self.normalize = True
        self.precision = self.cfg.model.precision

    def forward(self, audio):
        bs = audio.shape[0]
        mel = self.get_audio_feature(audio.cpu().numpy()).contiguous()
        mel = mel.view(bs, 1, mel.size(1), mel.size(2)).to(self.device)
        input_sizes = torch.IntTensor([mel.size(3)]).repeat(bs).int()
        with autocast(enabled=self.precision == 16):
            out, deepspeech_feature, output_sizes = self.Deepspeech_Model(mel, input_sizes)
        # semantic_feat = out.transpose(1,2)
        return deepspeech_feature, output_sizes

    def get_audio_feature(self, y):
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        return spect


class AudioEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM

        down_sample_block_1 = nn.Sequential(
            ConvNormRelu('2d', 1, 64, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('2d', 64, 64, downsample=True, norm=norm, leaky=leaky),
        )
        down_sample_block_2 = nn.Sequential(
            ConvNormRelu('2d', 64, 128, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('2d', 128, 128, downsample=True, norm=norm, leaky=leaky),  # downsample
        )
        down_sample_block_3 = nn.Sequential(
            ConvNormRelu('2d', 128, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('2d', 256, 256, downsample=True, norm=norm, leaky=leaky),  # downsample
        )
        down_sample_block_4 = nn.Sequential(
            ConvNormRelu('2d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('2d', 256, 256, kernel_size=(6, 3), stride=1, padding=0, norm=norm, leaky=leaky),  # downsample
        )

        self.specgram_encoder_2d = nn.Sequential(
            down_sample_block_1,
            down_sample_block_2,
            down_sample_block_3,
            down_sample_block_4
        )

    def forward(self, x, num_frames):
        x = self.specgram_encoder_2d(x.unsqueeze(1))
        x = F.interpolate(x, (1, num_frames), mode='bilinear')
        x = x.squeeze(2)
        return x


class DeepSpeechScoreEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.gru = nn.GRU(1024, 256, 2, batch_first=True)

    def forward(self, x, num_frames):  # x: B, 214, 1024
        x, _ = self.gru(x)  # B, 214, 256
        x = x.transpose(1, 2)  # B, 256, 214
        x = F.interpolate(x, (num_frames,), mode='linear')  # B, 256, 64
        # x = x.squeeze(2)
        return x


class UNet_1D(nn.Module):
    def __init__(self, cfg, in_feature=256) -> None:
        super().__init__()

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM

        # if cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:
        #     self.e0 = ConvNormRelu('1d', 256+cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION, 256, downsample=False, norm=norm, leaky=leaky)
        # elif cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
        #     self.e0 = ConvNormRelu('1d', 256+cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE, 256, downsample=False, norm=norm, leaky=leaky)
        # else:
        #     self.e0 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.e0 = ConvNormRelu('1d', in_feature, 512, downsample=False, norm=norm, leaky=leaky)

        self.e1 = ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky)
        self.e2 = ConvNormRelu('1d', 512, 512, downsample=True, norm=norm, leaky=leaky)
        self.e3 = ConvNormRelu('1d', 512, 512, downsample=True, norm=norm, leaky=leaky)
        self.e4 = ConvNormRelu('1d', 512, 512, downsample=True, norm=norm, leaky=leaky)
        self.e5 = ConvNormRelu('1d', 512, 512, downsample=True, norm=norm, leaky=leaky)
        self.e6 = ConvNormRelu('1d', 512, 512, downsample=True, norm=norm, leaky=leaky)

        self.d5 = ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky)
        self.d4 = ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky)
        self.d3 = ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky)
        self.d2 = ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky)
        self.d1 = ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)

        d5 = self.d5(F.interpolate(e6, e5.size(-1), mode='linear') + e5)
        d4 = self.d4(F.interpolate(d5, e4.size(-1), mode='linear') + e4)
        d3 = self.d3(F.interpolate(d4, e3.size(-1), mode='linear') + e3)
        d2 = self.d2(F.interpolate(d3, e2.size(-1), mode='linear') + e2)
        d1 = self.d1(F.interpolate(d2, e1.size(-1), mode='linear') + e1)

        return d1


class SequenceGeneratorCNN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM

        unet_in_feature = 256
        if cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
            self.deepspeech_encoder = DeepSpeechScoreEncoder(cfg)
            self.deepspeech_feature_size = 256
            unet_in_feature += self.deepspeech_feature_size

        self.audio_encoder = AudioEncoder(cfg)
        self.unet = UNet_1D(cfg, unet_in_feature)
        self.decoder = nn.Sequential(
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_LANDMARKS * 2, kernel_size=1, bias=True)
        )

    def forward(self, x, num_frames, code=None, deepspeech_feat=None):
        x = self.audio_encoder(x, num_frames)  # (B, C, num_frame)

        if self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:
            code = code.unsqueeze(2).repeat([1, 1, x.shape[-1]])
            x = torch.cat([x, code], 1)
        if self.cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
            deepspeech_feat = self.deepspeech_encoder(deepspeech_feat, num_frames)
            x = torch.cat([x, deepspeech_feat], 1)
        # print(x.shape)
        x = self.unet(x)
        x = self.decoder(x)

        x = x.permute([0, 2, 1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)
        return x

class PoseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=1024, num_layers=1):
        super(PoseEncoder, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)

    def forward(self, x):
        outputs, h_n = self.rnn(x)
        ## TODO include attention
        return outputs

class FaceBodyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.face_body_pre_gru = nn.GRU(1024, 512, num_layers=1, batch_first=True)
        self.face_body_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, face_feat, body_feat):
        joint_feat = torch.cat([face_feat, body_feat], dim=1).permute([0, 2, 1]) # B, Seq, 1024
        classify_feat, _ = self.face_body_pre_gru(joint_feat)
        classify_feat = classify_feat[:,-1,:] #B, 512
        classify_res = self.face_body_classifier(classify_feat) #B ,2

        return classify_res

class Jointencoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.audio_encoder_face = AudioEncoder(cfg)
        self.audio_encoder_body = AudioEncoder(cfg)
        self.deepspeech_encoder = DeepSpeechScoreEncoder(cfg)

        face_unet_in_feature = 256
        body_unet_in_feature = 256
        if cfg.VOICE2POSE.GENERATOR.SEED_POSE:
            face_unet_in_feature += cfg.DATASET.NUM_FACE_LANDMARKS * 2
            body_unet_in_feature += cfg.DATASET.NUM_BODY_LANDMARKS * 2
        if cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
            face_unet_in_feature += cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE
            body_unet_in_feature += cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM
        self.face_unet = UNet_1D(cfg, face_unet_in_feature)
        self.body_unet = UNet_1D(cfg, body_unet_in_feature)
        self.face_decoder = nn.Sequential(
            ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 512, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_FACE_LANDMARKS * 2, kernel_size=1, bias=True)
        )
        self.body_decoder = nn.Sequential(
            ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 512, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 128, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 128, 128, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(128, cfg.DATASET.NUM_BODY_LANDMARKS * 2, kernel_size=1, bias=True)
        )

        # pose_gt
        self.pose_encoder = PoseEncoder(input_size=102, hidden_size=512)
        self.pose_decoder = nn.Sequential(
            ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 512, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_LANDMARKS * 2, kernel_size=1, bias=True)
            )

        # face-body self-supervised module
        self.face_body_classifier = FaceBodyClassifier()
        self.cross_entrophy_F = nn.CrossEntropyLoss()

    def forward(self, x, num_frames, deepspeech_feat=None, seed_pose=None, pose_gt=None ):

        bs = x.shape[0]
        x_face = self.audio_encoder_face(x, num_frames)  # (B, C, num_frame)
        x_body = self.audio_encoder_body(x, num_frames)

        deepspeech_feat = self.deepspeech_encoder(deepspeech_feat, num_frames)  # B, 256, 64
        x_face = torch.cat([x_face, deepspeech_feat], 1)  # B, 512, 64
        x_body = torch.cat([x_body, deepspeech_feat], 1)  # B, 512, 64
        if self.cfg.VOICE2POSE.GENERATOR.SEED_POSE:
            face_indices = list(range(9, 79))
            body_indices = list(range(0, 9)) + list(range(79, 121))
            face_seed_pose = seed_pose[..., :2, face_indices]
            body_seed_pose = seed_pose[..., :2, body_indices]
            face_seed_pose = face_seed_pose.unsqueeze(1).repeat(1, 64, 1, 1).reshape(bs, 64, -1).transpose(1,
                                                                                                           2)  # B 70*2 64
            body_seed_pose = body_seed_pose.unsqueeze(1).repeat(1, 64, 1, 1).reshape(bs, 64, -1).transpose(1,
                                                                                                           2)  # B 51*2 64
            x_face = torch.cat([x_face, face_seed_pose], 1)
            x_body = torch.cat([x_body, body_seed_pose], 1)

        body_feat = self.body_unet(x_body)  # B, 512, 64
        body_output = self.body_decoder(body_feat)  # B, 51*2, 64
        body = body_output.permute([0, 2, 1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_BODY_LANDMARKS)

        face_feat = self.face_unet(x_face)
        face_output = self.face_decoder(face_feat)
        face = face_output.permute([0, 2, 1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_FACE_LANDMARKS)

        # face-body self-supervised correlation
        face_body_correlation_loss = self.calculate_face_body_correlation_loss(face_feat,body_feat)
    
        # pose_gt
        pose_gt = pose_gt[:,:,:,body_indices]
        pose_gt = pose_gt.reshape(list(pose_gt.shape[:2]) + [-1])  # (B, 64,102)
        pose_body_feat = self.pose_encoder(pose_gt).permute([0,2,1])  # (B,512,64)
        pose_body = self.body_decoder(pose_body_feat).permute([0,2,1])  # (B, 64, 102)
        pose_body = pose_body.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_BODY_LANDMARKS)
        # pose_body = pose_body.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)

        return face, body, pose_body, body_feat, pose_body_feat, face_body_correlation_loss

    def calculate_face_body_correlation_loss(self, face_feat, body_feat):
        bs = face_feat.shape[0]
        face_feat_sample1 = face_feat[...,:16]
        body_feat_sample1 = body_feat[...,:16]
        random_start = random.randint(8, 48)
        face_feat_sample2 = face_feat[...,random_start:random_start+16]
        body_feat_sample2 = body_feat[...,random_start:random_start+16]

        positive_classify_res1 = self.face_body_classifier(face_feat_sample1, body_feat_sample1) # B, 2
        positive_classify_res2 = self.face_body_classifier(face_feat_sample2, body_feat_sample2)
        negative_classify_res1 = self.face_body_classifier(face_feat_sample1, body_feat_sample2)
        negative_classify_res2 = self.face_body_classifier(face_feat_sample2, body_feat_sample1)

        positive_label = torch.ones(bs, dtype=torch.int64).cuda()
        negative_label = torch.zeros(bs, dtype=torch.int64).cuda()

        loss = self.cross_entrophy_F(positive_classify_res1,positive_label)
        loss += self.cross_entrophy_F(positive_classify_res2,positive_label)
        loss += self.cross_entrophy_F(negative_classify_res1,negative_label)
        loss += self.cross_entrophy_F(negative_classify_res2,negative_label)

        return loss


