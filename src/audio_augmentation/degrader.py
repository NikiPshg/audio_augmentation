import torch
import torchaudio
from torchaudio.io import AudioEffector
import random
import yaml
import pyroomacoustics as pra
import numpy as np
from tqdm import tqdm
import torchaudio.functional as F
from pathlib import Path
from .utils import get_audio_paths, align_waveform


class Degrader():
    def __init__(self, cfg_path: str, max_audio_length: int = None):
        if not cfg_path:
            raise ValueError("Configuration path must be provided.")
        
        with open(cfg_path, 'r') as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

        self.use_rir = self.yaml['use']['use_rir']
        self.use_noise = self.yaml['use']['use_noise']
        self.use_codec = self.yaml['use']['use_codec']
        self.use_spectr = self.yaml['use']['use_spectr']
        self.use_phone = self.yaml['use']['use_phone']

        self.snr_min = self.yaml['snr_range']['min']
        self.snr_max = self.yaml['snr_range']['max']
        
        self.max_audio_length = max_audio_length

        if self.use_rir:
            self.rirs = []
            self.prepare_rir(self.yaml['room_info']['count'])
        
        if self.use_noise:
            self.noise_paths = get_audio_paths(self.yaml['paths']['noise_path'])
            self.noise_min_ratio = self.yaml['noise_segment']['min_length_ratio']
            self.noise_max_ratio = self.yaml['noise_segment']['max_length_ratio']
            
            print("Pre-loading noise files into memory...")
            self.noises = []

            num_to_load = self.yaml.get('noise_segment', {}).get('max_preloaded', 200)
            paths_to_load = random.sample(self.noise_paths, min(len(self.noise_paths), num_to_load))

            for path in tqdm(paths_to_load, desc="Loading noises"):
                try:
                    noise, sr = torchaudio.load(path)
                    if sr != 16000:
                        noise = F.resample(noise, sr, 16000)
                    if noise.size(0) > 1:
                        noise = noise.mean(dim=0, keepdim=True)
                    noise /= (noise.norm(p=2) + 1e-8)
                    self.noises.append(noise)
                except Exception as e:
                    print(f"\nWarning: Could not load noise file {path}: {e}")
            print(f"Loaded {len(self.noises)} noise files.")

        if self.use_codec:
            self.codecs = list(self.yaml['codecs'].keys())

    def _add_codec(self, waveform: torch.Tensor, sample_rate: int):
        codec_name = random.choice(self.codecs)
        config = self.yaml['codecs'][codec_name]
        encoder = AudioEffector(**config)

        codec_wav = encoder.apply(waveform.T, sample_rate).T

        if waveform.size(1) != codec_wav.size(1):
            _, codec_wav = align_waveform(waveform, codec_wav)

        return codec_wav.float()
    
    def prepare_rir(self, n_rirs):
        print("Preparing Room Impulse Responses...")
        for i in tqdm(range(n_rirs)):
            cfg_room = self.yaml['room_info']
            x_min , x_max = cfg_room['x_min'], cfg_room['x_max']
            z_min, z_max = cfg_room['z_min'], cfg_room['z_max']
            x = random.uniform(x_min, x_max)
            y = random.uniform(x_min, x_max)
            z = random.uniform(z_min, z_max)
            corners = np.array([[0, 0], [0, y], [x, y], [x, 0]]).T
            room = pra.Room.from_corners(corners, fs=16000, max_order=10, absorption=0.2)
            room.extrude(z)

            source_pos = [min(x-0.1, cfg_room['src_pos'][0]), min(y-0.1, cfg_room['src_pos'][1]), min(z-0.1, cfg_room['src_pos'][2])]
            mic_pos = [min(x-0.1, cfg_room['micr_pos'][0]), min(y-0.1, cfg_room['micr_pos'][1]), min(z-0.1, cfg_room['micr_pos'][2])]
            room.add_source(source_pos)
            room.add_microphone(mic_pos)

            room.compute_rir()
            rir = torch.tensor(np.array(room.rir[0]), dtype=torch.float32)
            rir = rir / rir.norm(p=2)
            self.rirs.append(rir)

    def _add_rir(self, waveform, sample_rate):
        if not self.rirs:
            raise RuntimeError("RIRs are not prepared. Check your configuration.")
        rir = random.choice(self.rirs)
        augmented = F.fftconvolve(waveform, rir)
        if augmented.size(1) > waveform.size(1):
            augmented = augmented[:, :waveform.size(1)]
        return augmented.float()

    def _add_noise(self, waveform, sample_rate):
        if not self.noises:
            return waveform 
        noise = random.choice(self.noises)
    
        total_len = waveform.size(1)
        
        noise_ratio = random.uniform(self.noise_min_ratio, self.noise_max_ratio)
        segment_len = int(total_len * noise_ratio)

        if segment_len == 0:
            return waveform.float()

        start_idx = random.randint(0, total_len - segment_len)
        end_idx = start_idx + segment_len
        
        audio_segment = waveform[:, start_idx:end_idx]

    
        noise_len = noise.size(1)
        if noise_len > segment_len:
            noise_start_idx = random.randint(0, noise_len - segment_len)
            noise = noise[:, noise_start_idx:noise_start_idx + segment_len]
        elif noise_len < segment_len:
            repeats = (segment_len // noise_len) + 1
            noise = noise.repeat(1, repeats)
            noise = noise[:, :segment_len]

        snr = random.uniform(self.snr_min, self.snr_max)
        noisy_segment = F.add_noise(waveform=audio_segment, noise=noise, snr=torch.tensor([snr]))
        
        augmented_waveform = waveform.clone()
        augmented_waveform[:, start_idx:end_idx] = noisy_segment
        
        return augmented_waveform.float()

    def _apply_sp_deg(self, waveform, sample_rate):
        waveform = waveform.T
        num_aug = random.randint(0, self.yaml['spectrogramm']['num_aug'])

        for _ in range(num_aug):
            effect = random.choice(self.yaml['spectrogramm']['effects'])
            effector = AudioEffector(effect=effect, pad_end=True)
            waveform = effector.apply(waveform, sample_rate)

        return waveform.T
    
    def _add_phone(self, waveform, sample_rate):
        waveform = waveform.T
        effect = ",".join(
            [
                "lowpass=frequency=4000:poles=1",
                "compand=attacks=0.02:decays=0.05:points=-60/-60|-30/-10|-20/-8|-5/-8|-2/-8:gain=-8:volume=-7:delay=0.05",
            ]
        )
        effector = AudioEffector(effect=effect, format="g722")
        return effector.apply(waveform, sample_rate).T
        
    def __call__(self, waveform, sample_rate):
        if self.max_audio_length and waveform.size(1) > self.max_audio_length:
            max_start = waveform.size(1) - self.max_audio_length
            start_idx = random.randint(0, max_start)
            waveform = waveform[:, start_idx : start_idx + self.max_audio_length]

        degradation_order = []
        if self.use_noise: degradation_order.append('noise')
        if self.use_rir: degradation_order.append('rir')
        if self.use_codec: degradation_order.append('codec')
        if self.use_spectr: degradation_order.append('spectr')
        if self.use_phone: degradation_order.append('phone')
        
        codec_applied = False
        random.shuffle(degradation_order)
        
        for effect in degradation_order:
            if effect == 'rir' and random.random() < self.yaml['probs']['rir_prob']:
                waveform = self._add_rir(waveform, sample_rate)
            
            elif effect == 'noise' and random.random() < self.yaml['probs']['noise_prob']:
                waveform = self._add_noise(waveform, sample_rate)
            
            elif effect == 'codec' and random.random() < self.yaml['probs']['codec_prob']:
                waveform = self._add_codec(waveform, sample_rate)
                codec_applied = True
            
            elif effect == 'spectr' and random.random() < self.yaml['probs']['specrt_prob']:
                waveform = self._apply_sp_deg(waveform, sample_rate)

            elif effect == 'phone' and not codec_applied and random.random() < self.yaml['probs']['phone_prob']:
                waveform = self._add_phone(waveform, sample_rate)

        return waveform