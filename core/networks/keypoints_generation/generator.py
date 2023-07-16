from turtle import forward
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import librosa
from torch.cuda.amp import autocast

from ..building_blocks import ConvNormRelu
from core.deepspeech import DeepSpeech, SpectrogramParser, load_model, TranscribeConfig


class GenerateDeepspeechScores(nn.Module):
    def __init__(self, s2g_cfg) -> None:
        super().__init__()
        self.cfg = TranscribeConfig()
        self.device = torch.device("cuda" if self.cfg.model.cuda else "cpu")
        print(self.cfg.model.model_path)
        self.Deepspeech_Model = load_model(
            device = self.device,
            model_path = s2g_cfg.DEEPSPEECH.CKPT_PATH
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
        self.gru = nn.GRU(1024, 256, 2, batch_first = True)
        
    def forward(self, x, num_frames):  #x: B, 214, 1024
        x, _ = self.gru(x)  # B, 214, 256
        x = x.transpose(1,2) # B, 256, 214
        x = F.interpolate(x, (num_frames,), mode='linear') # B, 256, 64
        # x = x.squeeze(2)
        return x

class originalUNet_1D(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM
        
        if cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:
            self.e0 = ConvNormRelu('1d', 256+cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION, 256, downsample=False, norm=norm, leaky=leaky)
        else:
            self.e0 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        
        self.e1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.e2 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky)
        self.e3 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky)
        self.e4 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky)
        self.e5 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky)
        self.e6 = ConvNormRelu('1d', 256, 256, downsample=True, norm=norm, leaky=leaky)

        self.d5 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d4 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d3 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d2 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)
        self.d1 = ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky)

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

class regressor_fcn_bn_32(nn.Module):
	def __init__(self, feature_in_dim, feature_out_dim, default_size=256):
		super(regressor_fcn_bn_32, self).__init__()
		self.default_size = default_size
		self.use_resnet = True
				
		embed_size = default_size

		self.encoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(feature_in_dim,256,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(256),
			nn.MaxPool1d(kernel_size=2, stride=2),
		)

		self.conv5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv6 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv7 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,5,stride=2,padding=2),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv8 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv9 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.conv10 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)

		self.skip1 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)
		self.skip2 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)
		self.skip4 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)
		self.skip5 = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(embed_size,embed_size,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(embed_size),
		)
		# self.decoder = nn.Sequential(
		# 	nn.Dropout(0.5),
		# 	nn.Conv1d(embed_size,embed_size,3,padding=1),
		# 	nn.LeakyReLU(0.2, True),
		# 	nn.BatchNorm1d(embed_size),

		# 	nn.Dropout(0.5),
		# 	nn.ConvTranspose1d(embed_size, feature_out_dim, 7, stride=2, padding=3, output_padding=1),
		# 	nn.ReLU(True),
		# 	nn.BatchNorm1d(feature_out_dim),

		# 	nn.Dropout(0.5),
		# 	nn.Conv1d(feature_out_dim, feature_out_dim, 7, padding=3),
		# )

	## utility upsampling function
	def upsample(self, tensor, shape):
		return tensor.repeat_interleave(2, dim=2)[:,:,:shape[2]] 

	## forward pass through generator
	def forward(self, input_, percent_rand_=0.7):
		B, T = input_.shape[0], input_.shape[2]

		fourth_block = self.encoder(input_)

		fifth_block = self.conv5(fourth_block)
		sixth_block = self.conv6(fifth_block)
		seventh_block = self.conv7(sixth_block)
		eighth_block = self.conv8(seventh_block)
		ninth_block = self.conv9(eighth_block)
		tenth_block = self.conv10(ninth_block)

		ninth_block = tenth_block + ninth_block
		ninth_block = self.skip1(ninth_block)

		eighth_block = ninth_block + eighth_block
		eighth_block = self.skip2(eighth_block)

		sixth_block = self.upsample(seventh_block, sixth_block.shape) + sixth_block
		sixth_block = self.skip4(sixth_block)

		fifth_block = sixth_block + fifth_block
		fifth_block = self.skip5(fifth_block)

		# output = self.decoder(fifth_block)
		return fifth_block

class SequenceGeneratorCNN(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM

        self.audio_encoder = AudioEncoder(cfg)
        self.unet = originalUNet_1D(cfg)
        self.decoder = nn.Sequential(
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_LANDMARKS*2, kernel_size=1, bias=True)
            )

    def forward(self, x, num_frames, code=None, deepspeech_feat=None):
        x = self.audio_encoder(x, num_frames)  # (B, C, num_frame)

        if self.cfg.VOICE2POSE.GENERATOR.CLIP_CODE.DIMENSION is not None:
            code = code.unsqueeze(2).repeat([1, 1, x.shape[-1]])
            x = torch.cat([x, code], 1)

        x = self.unet(x)
        x = self.decoder(x)

        x = x.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_LANDMARKS)
        return x


# class HierarchicalPoseGenerator(nn.Module):
#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.cfg = cfg

#         self.audio_encoder = AudioEncoder(cfg)
#         self.deepspeech_encoder = DeepSpeechScoreEncoder(cfg)
#         self.face_predictor = nn.LSTM(input_size=256,
#                                   hidden_size=256,
#                                   num_layers=3,
#                                   dropout=0,
#                                   bidirectional=True,
#                                   batch_first=True)
#         self.face_decoder = nn.Sequential(
#             nn.Linear(in_features=512, out_features=512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, cfg.DATASET.NUM_FACE_LANDMARKS*2),
#         )

#         self.hidden_size = 200
#         self.body_predictor = nn.GRU(256+cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE, hidden_size=self.hidden_size, num_layers=2, 
#                                     batch_first=True,bidirectional=True, dropout=0.3)

#         self.body_encoder = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size//2),
#             nn.LeakyReLU(True),
#             nn.Linear(self.hidden_size//2, cfg.DATASET.NUM_BODY_LANDMARKS*2)
#         )
    
#     def forward(self, x, num_frames, deepspeech_feat=None):
#         x = self.audio_encoder(x, num_frames)  # (B, C, num_frame)
#         x = x.transpose(1,2) # B, 64, 256
#         face, _ = self.face_predictor(x) # B, 64, 256
#         face = self.face_decoder(face.reshape(-1, face.shape[2])) # B, 64, 72*2
#         face = face.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_FACE_LANDMARKS)

#         deepspeech_feat = self.deepspeech_encoder(deepspeech_feat, num_frames).transpose(1,2) # B, 64, 256
#         x = torch.cat([x,deepspeech_feat], 2) # B, 64, 512
#         output, _ = self.body_predictor(x) # B, 64, 400
#         output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # B, 64, 200
#         output = self.body_encoder(output.reshape(-1, output.shape[2])) # B*64, 49*2
#         body = output.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_BODY_LANDMARKS)

#         return face, body

# class HierarchicalPoseGenerator(nn.Module):
#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.cfg = cfg

#         self.audio_encoder_face = AudioEncoder(cfg)
#         self.audio_encoder_body = AudioEncoder(cfg)
#         self.deepspeech_encoder = DeepSpeechScoreEncoder(cfg)

#         face_unet_in_feature = 256
#         body_unet_in_feature = 256
#         if cfg.VOICE2POSE.GENERATOR.SEED_POSE:
#             face_unet_in_feature += cfg.DATASET.NUM_FACE_LANDMARKS*2
#             body_unet_in_feature += cfg.DATASET.NUM_BODY_LANDMARKS*2
#         if cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
#             face_unet_in_feature += cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE
#             body_unet_in_feature += cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE
        
#         leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
#         norm = cfg.VOICE2POSE.GENERATOR.NORM
#         self.face_unet = UNet_1D(cfg, face_unet_in_feature)
#         self.body_unet = UNet_1D(cfg, body_unet_in_feature)
#         self.face_decoder = nn.Sequential(
#             ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky),
#             ConvNormRelu('1d', 512, 256, downsample=False, norm=norm, leaky=leaky),
#             ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
#             ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
#             nn.Conv1d(256, cfg.DATASET.NUM_FACE_LANDMARKS*2, kernel_size=1, bias=True)
#             )
#         self.body_decoder = nn.Sequential(
#             ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky),
#             ConvNormRelu('1d', 512, 256, downsample=False, norm=norm, leaky=leaky),
#             ConvNormRelu('1d', 256, 128, downsample=False, norm=norm, leaky=leaky),
#             ConvNormRelu('1d', 128, 128, downsample=False, norm=norm, leaky=leaky),
#             nn.Conv1d(128, cfg.DATASET.NUM_BODY_LANDMARKS*2, kernel_size=1, bias=True)
#             )
    
#     def forward(self, x, num_frames, deepspeech_feat=None, seed_pose=None):
#         bs = x.shape[0]
#         x_face = self.audio_encoder_face(x, num_frames)  # (B, C, num_frame)
#         x_body = self.audio_encoder_body(x, num_frames)

#         deepspeech_feat = self.deepspeech_encoder(deepspeech_feat, num_frames) # B, 256, 64
#         x_face = torch.cat([x_face,deepspeech_feat], 1) # B, 512, 64
#         x_body = torch.cat([x_body,deepspeech_feat], 1) # B, 512, 64
#         if self.cfg.VOICE2POSE.GENERATOR.SEED_POSE:
#             face_indices = list(range(9, 79))
#             body_indices = list(range(0,9)) + list(range(79,121))
#             face_seed_pose = seed_pose[..., :2, face_indices]
#             body_seed_pose = seed_pose[..., :2, body_indices]
#             face_seed_pose = face_seed_pose.unsqueeze(1).repeat(1,64,1,1).reshape(bs,64,-1).transpose(1,2) #B 70*2 64
#             body_seed_pose = body_seed_pose.unsqueeze(1).repeat(1,64,1,1).reshape(bs,64,-1).transpose(1,2) #B 51*2 64
#             x_face = torch.cat([x_face,face_seed_pose], 1)
#             x_body = torch.cat([x_body,body_seed_pose], 1)

#         body_output = self.body_unet(x_body)  # B, 512, 64
#         body_output = self.body_decoder(body_output)  # B, 51*2, 64
#         body = body_output.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_BODY_LANDMARKS)

#         face_input = x_face
#         face_output = self.face_unet(face_input)
#         face_output = self.face_decoder(face_output)
#         face = face_output.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_FACE_LANDMARKS)

#         return face, body


# class HierarchicalPoseGenerator(nn.Module):
#     def __init__(self, cfg) -> None:
#         super().__init__()
#         self.cfg = cfg

#         # hand_unet_in_feature = 256
#         # if cfg.VOICE2POSE.GENERATOR.SEED_POSE:
#         #     hand_unet_in_feature += cfg.DATASET.NUM_HAND_LANDMARKS*2
#         # if cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
#         #     hand_unet_in_feature += cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE

#         # face_in_feature = 256
#         # if cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
#         #     face_in_feature += cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE
#         # face_decoder_in_feature = 512
#         # if cfg.VOICE2POSE.GENERATOR.SEED_POSE:
#         #     face_decoder_in_feature += cfg.DATASET.NUM_FACE_LANDMARKS*2

#         torso_in_feature = 256
#         if cfg.VOICE2POSE.GENERATOR.SEED_POSE:
#             torso_in_feature += cfg.DATASET.NUM_TORSO_LANDMARKS*2
#         if cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
#             torso_in_feature += cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE

#         # self.face_audio_encoder = AudioEncoder(cfg)
#         self.torso_audio_encoder = AudioEncoder(cfg)
#         # self.hand_audio_encoder = AudioEncoder(cfg)

#         self.deepspeech_encoder = DeepSpeechScoreEncoder(cfg)

#         # self.face_predictor = nn.LSTM(input_size=face_in_feature,
#         #                           hidden_size=256,
#         #                           num_layers=2,
#         #                           dropout=0.3,
#         #                           bidirectional=True,
#         #                           batch_first=True,)
#         # self.face_decoder = nn.Sequential(
#         #     nn.Linear(in_features=face_decoder_in_feature, out_features=512),
#         #     nn.BatchNorm1d(512),
#         #     nn.LeakyReLU(0.2),
#         #     nn.Linear(512, 256),
#         #     nn.BatchNorm1d(256),
#         #     nn.LeakyReLU(0.2),
#         #     nn.Linear(256, cfg.DATASET.NUM_FACE_LANDMARKS*2),
#         # )

#         self.hidden_size = 200
#         self.torso_predictor = nn.GRU(torso_in_feature, hidden_size=self.hidden_size, num_layers=2, 
#                                     batch_first=True,bidirectional=True, dropout=0.3)

#         self.torso_encoder = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size//2),
#             nn.LeakyReLU(True),
#             nn.Linear(self.hidden_size//2, cfg.DATASET.NUM_TORSO_LANDMARKS*2)
#         )

#         # self.hand_unet = UNet_1D(cfg, hand_unet_in_feature)
#         # leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
#         # norm = cfg.VOICE2POSE.GENERATOR.NORM
#         # self.hand_decoder = nn.Sequential(
#         #     ConvNormRelu('1d', 256, 128, downsample=False, norm=norm, leaky=leaky),
#         #     ConvNormRelu('1d', 128, 128, downsample=False, norm=norm, leaky=leaky),
#         #     ConvNormRelu('1d', 128, 128, downsample=False, norm=norm, leaky=leaky),
#         #     ConvNormRelu('1d', 128, 128, downsample=False, norm=norm, leaky=leaky),
#         #     nn.Conv1d(128, cfg.DATASET.NUM_HAND_LANDMARKS*2, kernel_size=1, bias=True)
#         # )
        
    
#     def forward(self, x, num_frames, deepspeech_feat=None, seed_pose=None):
#         bs = x.shape[0]
#         # x_face = self.face_audio_encoder(x, num_frames)
#         x_torso = self.torso_audio_encoder(x, num_frames)  # (B, C, num_frame)
#         # x_hand = self.hand_audio_encoder(x, num_frames)  # (B, C, num_frame)

#         # x_face = x_face.transpose(1,2) # B, 64, 256
#         x_torso = x_torso.transpose(1,2) # B, 64, 256
#         # x_hand = x_hand.transpose(1,2) # B, 64, 256

#         # face_indices = list(range(9, 79))
#         torso_indices = list(range(0,9))
#         # hand_indices = list(range(79,121))
#         # face_seed_pose = seed_pose[..., :2, face_indices]
#         torso_seed_pose = seed_pose[..., :2, torso_indices]
#         # hand_seed_pose = seed_pose[..., :2, hand_indices]
#         # face_seed_pose = face_seed_pose.unsqueeze(1).repeat(1,64,1,1).reshape(bs,64,-1) #B 64 69*2
#         torso_seed_pose = torso_seed_pose.unsqueeze(1).repeat(1,64,1,1).reshape(bs,64,-1) #B 64 9*2
#         # hand_seed_pose = hand_seed_pose.unsqueeze(1).repeat(1,64,1,1).reshape(bs,64,-1) #B 64 52*2

#         deepspeech_feat = self.deepspeech_encoder(deepspeech_feat, num_frames).transpose(1,2) # B, 64, 256
#         if self.cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
#             # x_face = torch.cat([x_face, deepspeech_feat], 2)
#             x_torso = torch.cat([x_torso, deepspeech_feat], 2)
#             # x_hand = torch.cat([x_hand, deepspeech_feat], 2)

#         # face, _ = self.face_predictor(x_face) # B, 64, 512
#         # if self.cfg.VOICE2POSE.GENERATOR.SEED_POSE:
#         #     face = torch.cat([face,face_seed_pose], 2)  #  B,64,512+70*2
#         # face = self.face_decoder(face.reshape(-1, face.shape[2])) # B, 64, 69*2
#         # face = face.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_FACE_LANDMARKS)
        
        
#         if self.cfg.VOICE2POSE.GENERATOR.SEED_POSE:
#             x_torso = torch.cat([x_torso, torso_seed_pose], 2) # B,64,512+9*2
#             # x_hand = torch.cat([x_hand, hand_seed_pose], 2) # B,64,512+42*2

#         torso, _ = self.torso_predictor(x_torso) # B, 64, 400
#         torso = torso[:, :, :self.hidden_size] + torso[:, :, self.hidden_size:]  # B, 64, 200
#         torso = self.torso_encoder(torso.reshape(-1, torso.shape[2])) # B*64, 49*2
#         torso = torso.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_TORSO_LANDMARKS)

#         # hand = self.hand_unet(x_hand.transpose(1,2))  # B, 512, 64
#         # hand = self.hand_decoder(hand)  # B, 49*2, 64
#         # hand = hand.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_HAND_LANDMARKS)

#         # return face, torso, hand
#         return torso

class PoseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=1024, num_layers=1):
        super(PoseEncoder, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)

    def forward(self, x):
        outputs, h_n = self.rnn(x)
        ## TODO include attention
        return outputs

class CorrelationClassifier(nn.Module):
    def __init__(self, in_channel=1024) -> None:
        super().__init__()
        self.correlation_pre_gru = nn.GRU(in_channel, 512, num_layers=1, batch_first=True)
        self.correlation_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, feat1, feat2):
        joint_feat = torch.cat([feat1, feat2], dim=1).permute([0, 2, 1]) # B, Seq, 1024
        classify_feat, _ = self.correlation_pre_gru(joint_feat)
        classify_feat = classify_feat[:,-1,:] #B, 512
        classify_res = self.correlation_classifier(classify_feat) #B ,2

        return classify_res

#3Unet+fushion
class HierarchicalPoseGenerator(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        hand_unet_in_feature = 256
        if cfg.VOICE2POSE.GENERATOR.SEED_POSE:
            hand_unet_in_feature += cfg.DATASET.NUM_HAND_LANDMARKS*2
        if cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
            hand_unet_in_feature += cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE

        face_in_feature = 256
        face_unet_in_feature = 256
        if cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
            face_in_feature += cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE
            face_unet_in_feature += cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE
        face_decoder_in_feature = 512
        if cfg.VOICE2POSE.GENERATOR.SEED_POSE:
            face_decoder_in_feature += cfg.DATASET.NUM_FACE_LANDMARKS*2
            face_unet_in_feature += cfg.DATASET.NUM_FACE_LANDMARKS*2
            

        torso_in_feature = 256
        if cfg.VOICE2POSE.GENERATOR.SEED_POSE:
            torso_in_feature += cfg.DATASET.NUM_TORSO_LANDMARKS*2
        if cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
            torso_in_feature += cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE_SIZE

        self.face_audio_encoder = AudioEncoder(cfg)
        self.body_audio_encoder = AudioEncoder(cfg)
        # self.torso_audio_encoder = AudioEncoder(cfg)
        # self.hand_audio_encoder = AudioEncoder(cfg)

        self.deepspeech_encoder = DeepSpeechScoreEncoder(cfg)

        leaky = cfg.VOICE2POSE.GENERATOR.LEAKY_RELU
        norm = cfg.VOICE2POSE.GENERATOR.NORM

        self.face_predictor = UNet_1D(cfg, face_unet_in_feature)
        self.face_decoder = nn.Sequential(
            ConvNormRelu('1d', 512, 512, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 512, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_FACE_LANDMARKS*2, kernel_size=1, bias=True)
        )

        self.torso_predictor = UNet_1D(cfg, torso_in_feature)
        self.torso_decoder = nn.Sequential(
            ConvNormRelu('1d', 512, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 128, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 128, 64, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 64, 32, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(32, cfg.DATASET.NUM_TORSO_LANDMARKS*2, kernel_size=1, bias=True)
        )

        self.arm_encoder = nn.Sequential(
			nn.Dropout(0.5),
			nn.Conv1d(6*2,256,3,padding=1),
			nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(256),
			# nn.MaxPool1d(kernel_size=2, stride=2),
		)
        self.arm_unet = originalUNet_1D(cfg)
        self.hand_unet = UNet_1D(cfg, hand_unet_in_feature)   
        self.hand_decoder = nn.Sequential(
            ConvNormRelu('1d', 512+256, 512, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 512, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            ConvNormRelu('1d', 256, 256, downsample=False, norm=norm, leaky=leaky),
            nn.Conv1d(256, cfg.DATASET.NUM_HAND_LANDMARKS*2, kernel_size=1, bias=True)
        )

        # pose_gt
        self.torso_encoder = PoseEncoder(input_size=cfg.DATASET.NUM_TORSO_LANDMARKS*2, hidden_size=512)
        # self.hand_encoder = PoseEncoder(input_size=cfg.DATASET.NUM_HAND_LANDMARKS*2, hidden_size=512)
        # self.body_encoder = PoseEncoder(input_size=cfg.DATASET.NUM_HAND_LANDMARKS*2, hidden_size=512)

        # face-body hand-body self-supervised module
        self.face_body_classifier = CorrelationClassifier(in_channel=1024)
        self.hand_body_classifier = CorrelationClassifier(in_channel=1280)
        self.cross_entrophy_F = nn.CrossEntropyLoss()
        # self.hand_torso_fushion_module = nn.Linear(cfg.DATASET.NUM_BODY_LANDMARKS*2,cfg.DATASET.NUM_BODY_LANDMARKS*2)

    def normalize_seed_pose(self, seed_pose):
        face_indices = list(range(9, 79))
        torso_indices = list(range(0,9))
        hand_indices = list(range(79,121))
        face_seed_pose = seed_pose[..., :2, face_indices]
        torso_seed_pose = seed_pose[..., :2, torso_indices]
        hand_seed_pose = seed_pose[..., :2, hand_indices]
        face_seed_pose -= seed_pose[..., :2, 0, None]
        hand_seed_pose[..., :2, 0:21] -= hand_seed_pose[...,:2,0,None]
        hand_seed_pose[..., :2, 21:42] -= hand_seed_pose[...,:2,21,None]

        return face_seed_pose, torso_seed_pose, hand_seed_pose
    
    def forward(self, x, num_frames, deepspeech_feat=None, seed_pose=None, pose_gt=None):
        bs = x.shape[0]
        x_face = self.face_audio_encoder(x, num_frames)
        x_body = self.body_audio_encoder(x, num_frames)  # (B, C, num_frame)
        # x_torso = self.torso_audio_encoder(x, num_frames)
        # x_hand = self.hand_audio_encoder(x, num_frames)  # (B, C, num_frame)

        x_face = x_face.transpose(1,2) # B, 64, 256
        # x_torso = x_torso.transpose(1,2)
        # x_hand = x_hand.transpose(1,2)
        x_torso = x_body.clone().transpose(1,2) # B, 64, 256
        x_hand = x_body.clone().transpose(1,2) # B, 64, 256

        face_seed_pose, torso_seed_pose, hand_seed_pose = self.normalize_seed_pose(seed_pose)
        face_seed_pose = face_seed_pose.unsqueeze(1).repeat(1,64,1,1).reshape(bs,64,-1) #B 64 70*2
        torso_seed_pose = torso_seed_pose.unsqueeze(1).repeat(1,64,1,1).reshape(bs,64,-1) #B 64 9*2
        hand_seed_pose = hand_seed_pose.unsqueeze(1).repeat(1,64,1,1).reshape(bs,64,-1) #B 64 42*2

        deepspeech_feat = self.deepspeech_encoder(deepspeech_feat, num_frames).transpose(1,2) # B, 64, 256
        if self.cfg.VOICE2POSE.GENERATOR.DEEPSPEECH_FEATURE:
            x_face = torch.cat([x_face, deepspeech_feat], 2)
            x_torso = torch.cat([x_torso, deepspeech_feat], 2)
            x_hand = torch.cat([x_hand, deepspeech_feat], 2)
         
        if self.cfg.VOICE2POSE.GENERATOR.SEED_POSE:
            x_torso = torch.cat([x_torso, torso_seed_pose], 2) # B,64,512+9*2
            x_hand = torch.cat([x_hand, hand_seed_pose], 2) # B,64,512+42*2
            x_face = torch.cat([x_face, face_seed_pose], 2)

        face_feat = self.face_predictor(x_face.transpose(1,2))
        face_output = self.face_decoder(face_feat)
        face = face_output.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_FACE_LANDMARKS)

        torso_feat = self.torso_predictor(x_torso.transpose(1,2))
        torso_output = self.torso_decoder(torso_feat)
        torso = torso_output.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_TORSO_LANDMARKS)

        arm_pred = torso[...,1:7]
        arm_pred = arm_pred.reshape(bs,num_frames,-1) # B, 64, 6*2
        arm_feat = self.arm_encoder(arm_pred.transpose(1,2)) #B, 256, 64
        arm_feat = self.arm_unet(arm_feat)
        hand_feat = self.hand_unet(x_hand.transpose(1,2))  # B, 512, 64
        hand_input = torch.cat([hand_feat,arm_feat], 1)
        hand_output = self.hand_decoder(hand_input)  # B, 49*2, 64
        hand = hand_output.permute([0,2,1]).reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_HAND_LANDMARKS)

        # self-supervised correlation
        face_body_correlation_loss = self.calculate_correlation_loss(face_feat,torso_feat,self.face_body_classifier)
        hand_body_correlation_loss = self.calculate_correlation_loss(hand_input,torso_feat,self.hand_body_classifier)


        # # body joint embedding
        # body_indices = list(range(0,9))+list(range(79,121))
        # body_gt = pose_gt[:,:,:,body_indices]
        # body_gt = body_gt.reshape(list(body_gt.shape[:2]) + [-1])
        # pose_body_feat = self.body_encoder(body_gt).permute([0,2,1])


        # torso joint embedding
        torso_indices = list(range(0,9))
        torso_gt = pose_gt[:,:,:,torso_indices]
        torso_gt = torso_gt.reshape(list(torso_gt.shape[:2]) + [-1])  # (B, 64, 18)
        pose_torso_feat = self.torso_encoder(torso_gt).permute([0,2,1])  # (B,512,64)
        pose_torso = self.torso_decoder(pose_torso_feat).permute([0,2,1])  # (B, 64, 18)
        pose_torso = pose_torso.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_TORSO_LANDMARKS)

        # # hand joint embedding
        # hand_indices = list(range(79,121))
        # hand_gt = pose_gt[:,:,:,hand_indices]
        # hand_gt[..., :2, 0:21] -= hand_gt[...,:2,0,None]
        # hand_gt[..., :2, 21:42] -= hand_gt[...,:2,21,None]
        # hand_gt = hand_gt.reshape(list(hand_gt.shape[:2]) + [-1])  # (B, 64, 102)
        # pose_hand_feat = self.hand_encoder(hand_gt).permute([0,2,1])  # (B,512,64)
        # pose_hand = self.hand_decoder(pose_hand_feat).permute([0,2,1])  # (B, 64, 102)
        # pose_hand = pose_hand.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_HAND_LANDMARKS)
        # # hand_gt = hand_gt.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_HAND_LANDMARKS)
        # # pose_hand[..., :2, 0:21] += hand_gt[...,:2,0,None]
        # # pose_hand[..., :2, 21:42] += hand_gt[...,:2,21,None]

        # body_pred = torch.cat([torso , hand], 3).reshape(-1, (self.cfg.DATASET.NUM_HAND_LANDMARKS+self.cfg.DATASET.NUM_TORSO_LANDMARKS)*2)
        # body_pred = self.hand_torso_fushion_module(body_pred)
        # body_pred = body_pred.reshape(-1, num_frames, 2, self.cfg.DATASET.NUM_BODY_LANDMARKS)
        # poses_pred = torch.cat([body_pred[0:9],face,body_pred[9:]], 3)

        # return face, torso, hand, torso_feat, pose_torso_feat, pose_torso, hand_feat, pose_hand_feat, pose_hand, face_body_correlation_loss, hand_body_correlation_loss
        return face, torso, hand, face_body_correlation_loss,hand_body_correlation_loss, torso_feat, pose_torso_feat, pose_torso

    def calculate_correlation_loss(self, feat1, feat2, classifier):
        bs = feat1.shape[0]
        feat1_sample1 = feat1[...,:16]
        feat2_sample1 = feat2[...,:16]
        random_start = random.randint(8, 48)
        feat1_sample2 = feat1[...,random_start:random_start+16]
        feat2_sample2 = feat2[...,random_start:random_start+16]

        positive_classify_res1 = classifier(feat1_sample1, feat2_sample1) # B, 2
        positive_classify_res2 = classifier(feat1_sample2, feat2_sample2)
        negative_classify_res1 = classifier(feat1_sample1, feat2_sample2)
        negative_classify_res2 = classifier(feat1_sample2, feat2_sample1)

        positive_label = torch.ones(bs, dtype=torch.int64).cuda()
        negative_label = torch.zeros(bs, dtype=torch.int64).cuda()

        loss = self.cross_entrophy_F(positive_classify_res1,positive_label)
        loss = loss + self.cross_entrophy_F(positive_classify_res2,positive_label)
        loss = loss + self.cross_entrophy_F(negative_classify_res1,negative_label)
        loss = loss + self.cross_entrophy_F(negative_classify_res2,negative_label)

        return loss


