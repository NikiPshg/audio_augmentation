import torch 
import torchaudio
from torchaudio.io import AudioEffector, CodecConfig
from pathlib import Path
import random 
import yaml

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
        self.count_rir = count_rir

        if use_codec:
             with open('augmentation/codec.yaml', 'r') as f:
                self.configs = yaml.load(f, Loader=yaml.SafeLoader)

    def _get_audio_paths(self, path:str):
        start_dir = Path(path).resolve()
        return list(start_dir.rglob('*.wav'))
    
    def _add_codec(
            self,
            waveform:torch.Tensor,
            sample_rate:int,
            ):

        codec_name = random.choice(list(self.configs.keys()))
        config = self.configs[codec_name]
        encoder = torchaudio.io.AudioEffector(**config)
        return encoder.apply(waveform, sample_rate)
        
    def _add_rir(self, waveform, sample_rate):
        pass

    def _make_rir(self):
        pass

    def _add_noise(self, waveform, sample_rate):
        pass

    def __call__(self):
        pass


wav, sr = torchaudio.load('C:/Users/RedmiBook/Documents/GitHub/audio_augmentation/segment_163.wav')
degraded = Degraded()
degraded_wav = degraded._add_codec(waveform= wav, sample_rate=sr)
torchaudio.save("",degraded_wav, sr)

    



