import sys
from pathlib import Path

DeepSpeechDir = '../deepspeech.pytorch'

if Path(DeepSpeechDir).exists():
    sys.path.append(DeepSpeechDir)
else:
    # 尝试下探一级路径
    sys.path.append("../../deepspeech.pytorch")

from deepspeech_pytorch.configs.inference_config import *
from deepspeech_pytorch.decoder import *
from deepspeech_pytorch.loader.data_loader import *
from deepspeech_pytorch.model import *
from deepspeech_pytorch.utils import *
