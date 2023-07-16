from turtle import forward
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import numpy as np
import librosa
import json
import argparse
from configs.default import get_cfg_defaults
from typing import List

from core.networks.building_blocks import ConvNormRelu
from core.deepspeech import DeepSpeech, SpectrogramParser, load_model, TranscribeConfig, load_decoder
from core.networks.keypoints_generation.generator import GenerateDeepspeechScores
from core.datasets import get_dataset

deepspeech_ckpt_path = '/home/wanghexiang/Speech2Gesture/deepspeech.pytorch/ckpt/ted_pretrained_v3.ckpt'

class Audio2GestureWithDeepSpeech(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cfg = TranscribeConfig()
        self.device = torch.device("cuda" if self.cfg.model.cuda else "cpu")
        print(self.cfg.model.model_path)
        self.Deepspeech_Model = load_model(
            device = self.device,
            model_path = deepspeech_ckpt_path
        )
        audio_conf = self.Deepspeech_Model.spect_cfg
        self.window_stride = audio_conf.window_stride
        self.window_size = audio_conf.window_size
        self.sample_rate = audio_conf.sample_rate
        self.window = audio_conf.window.value
        self.normalize = True
        self.precision = self.cfg.model.precision

        leaky = True
        norm = 'IN'
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

        self.specgram_encoder_2d = nn.Sequential(
            down_sample_block_1,
            down_sample_block_2,
            down_sample_block_3
        )


    def forward(self, audio):
        bs = audio.shape[0]
        mel = self.get_audio_feature(audio.numpy()).contiguous()
        print(mel.shape)
        mel = mel.view(bs, 1, mel.size(1), mel.size(2)).to(self.device)
        print(mel.shape)
        input_sizes = torch.IntTensor([mel.size(3)]).repeat(bs).int()
        with autocast(enabled=self.precision == 16):
            out, output_sizes = self.Deepspeech_Model(mel, input_sizes)
        semantic_feat = out.transpose(1,2)
        print(semantic_feat.shape)
        feat = self.specgram_encoder_2d(semantic_feat.unsqueeze(1))
        print(feat.shape)
        x = F.interpolate(feat, (1, 64), mode='bilinear')
        print(x.shape)
        return mel

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

def decode_results(decoded_output: List,
                   decoded_offsets: List,
                   cfg: TranscribeConfig):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "path": cfg.model.model_path
            },
            "language_model": {
                "path": cfg.lm.lm_path
            },
            "decoder": {
                "alpha": cfg.lm.alpha,
                "beta": cfg.lm.beta,
                "type": cfg.lm.decoder_type.value,
            }
        }
    }
    # cfg.offsets = True
    for b in range(len(decoded_output)):
        for pi in range(min(cfg.lm.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if cfg.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="voice2pose main program")
    parser.add_argument("--config_file", default="configs/s2g_deepspeech_feature.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    cfg1 = TranscribeConfig()
    device = torch.device("cuda" if cfg1.model.cuda else "cpu")
    model = GenerateDeepspeechScores(cfg).cuda()
    # deepspeech_model = load_model(
    #     device=device,
    #     model_path=cfg.DEEPSPEECH.CKPT_PATH
    # )
    decoder = load_decoder(
        labels=model.Deepspeech_Model.labels,
        cfg=cfg1.lm
    )
    test_dataset = get_dataset(cfg.DATASET.NAME)(cfg.DATASET.ROOT_DIR, 'oliver', 'val', cfg)
    test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False,num_workers=16,drop_last=False)

    cnt = 0
    test_num = 1

    gru = nn.GRU(1024, 256, 2, batch_first = True)
    gru.cuda()

    for batch in test_dataloader:
        audio = batch['audio'].cuda()

        deepspeech_feature, output_sizes = model(audio)
        # mel = mel.transpose(1,2)
        # print(mel.shape)
        print(deepspeech_feature.shape)
        feat, _ = gru(deepspeech_feature)
        print(feat.shape)
        feat = feat.transpose(1,2)
        feat = F.interpolate(feat, (64,), mode='linear')
        print(feat.shape)
        # decoded_output, decoded_offsets = decoder.decode(mel, output_sizes)

        # results = decode_results(
        #     decoded_output=decoded_output,
        #     decoded_offsets=decoded_offsets,
        #     cfg=cfg1
        # )
        # # print(results.shape)
        # print(json.dumps(results))
        cnt += 1
        if cnt>=test_num:
            break
