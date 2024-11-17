import torch 
import torchaudio
from torchaudio.io import AudioEffector
from pathlib import Path
import random 
import yaml
from utils import get_audio_paths


class Degraded():
    def __init__(
        self,
        rir_path=None,
        noise_path=None,
        use_rir=True,
        use_noise=True,
        use_codec=True,
        snr_range=[5,30],
        count_rir=100
        ):

        self.use_rir = use_rir
        self.use_noise =use_noise
        self.use_codec = use_codec
        self.snr_range = snr_range

        if use_codec:
             with open('augmentation/codec.yaml', 'r') as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

    def _add_codec(
            self,
            waveform:torch.Tensor,
            sample_rate:int,
            ):
        
        codec_name = random.choice(list(self.yaml['codecs'].keys()))
        config = self.yaml['codecs'][codec_name]
        print(config)
        encoder = AudioEffector(**config)
        
        return encoder.apply(waveform, sample_rate)

    def _add_rir(self, waveform):
        pass

    def _add_noise(self, waveform, sample_rate):
        pass

    def __call__(self):
        pass


wav, sr = torchaudio.load('C:/Users/RedmiBook/Documents/GitHub/audio_augmentation/segment_163.wav',channels_first=False)
degraded = Degraded()
degraded_wav = degraded._add_codec(waveform=wav, sample_rate=sr)
torchaudio.save("output.wav", degraded_wav.T, sr)


    



