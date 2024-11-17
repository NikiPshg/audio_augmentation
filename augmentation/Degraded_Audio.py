import torch 
import torchaudio
from torchaudio.io import AudioEffector
from pathlib import Path
import random 
import yaml
from utils import get_audio_paths


class Degraded():
    def __init__(self,cfg_path:None):
        if not(cfg_path):
             raise RuntimeError
        
        with open('augmentation/config.yaml', 'r') as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

        self.use_rir = self.yaml['use']['use_rir']
        self.use_noise = self.yaml['use']['use_noise']
        self.use_codec = self.yaml['use']['use_codec']

        self.snr_min = self.yaml['snr_range']['min']
        self.snr_max = self.yaml['snr_range']['max']

        if self.use_rir:
            self.rir_paths = get_audio_paths(self.yaml['paths']['rir_path'])
        
        if self.use_noise:
            self.noise_paths = get_audio_paths(self.yaml['paths']['noise_path'])

        if self.use_codec:
            self.codecs = list(self.yaml['codecs'].keys())

    def _add_codec(
            self,
            waveform:torch.Tensor,
            sample_rate:int,
            ):
        codec_name = random.choice(self.codecs)
        config = self.yaml['codecs'][codec_name]
        encoder = AudioEffector(**config)

        return encoder.apply(waveform.T, sample_rate).T

    def _add_rir(self, waveform, sample_rate):

        if len(self.rir_paths) ==0:
            raise RuntimeError
        
        rir_path = random.choice(self.rir_paths)
        rir_wav, rir_sr  = torchaudio.load(rir_path)
        if rir_wav.size()[0] != 1:
            rir_wav = rir_wav.mean(dim=-2, keepdim=True)

        rir = rir_wav[:, int(rir_sr * 1.01) : int(rir_sr * 1.3)]
        rir = rir / torch.linalg.vector_norm(rir, ord=2)

        rir = torchaudio.functional.resample(
            rir_wav, rir_sr, new_freq=sample_rate//2
        )

        return torchaudio.functional.fftconvolve(waveform, rir)

    def _add_noise(self, waveform, sample_rate):
        if len(self.noise_paths) ==0:
            raise RuntimeError
        
        snr_max, snr_min = self.snr_max, self.snr_min
        snr = random.uniform(snr_min, snr_max)
        noise_path = random.choice(self.noise_paths)
        noise, noise_sr = torchaudio.load(noise_path)
        noise /= noise.norm(p=2)

        if noise.size(0) > 1:
            noise = noise[0].unsqueeze(0)
        noise = torchaudio.functional.resample(noise, noise_sr, sample_rate)

        if not noise.size(1) < waveform.size(1):
            start_idx = random.randint(0, noise.size(1) - waveform.size(1))
            end_idx = start_idx + waveform.size(1)
            noise = noise[:, start_idx:end_idx]
        else:
            noise = noise.repeat(1, waveform.size(1) // noise.size(1) + 1)[
                :, : waveform.size(1)
            ]

        augmented = torchaudio.functional.add_noise(
            waveform=waveform, noise=noise, snr=torch.tensor([snr])
        )
        return augmented

    def __call__(self, waveform, sample_rate):

        if random.random() < self.yaml['probs']['rir_prob']:
            waveform = self._add_rir(waveform, sample_rate)

        if random.random() < self.yaml['probs']['noise_prob']:
            waveform = self._add_noise(waveform, sample_rate)

        if random.random() < self.yaml['probs']['codec_prob']:
            waveform = self._add_codec(waveform, sample_rate)

        return waveform
        



wav, sr = torchaudio.load('C:/Users/RedmiBook/Documents/GitHub/audio_augmentation/segment_163.wav')
print(wav.shape)
degraded = Degraded('augmentation/config.yaml')
degraded_wav = degraded(waveform=wav, sample_rate=sr)
print(degraded_wav.shape)
torchaudio.save("output.wav", degraded_wav, sr)


    



