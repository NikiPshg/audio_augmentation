from torchaudio.io import AudioEffector
import random
import yaml

class SpDegrader():
    def __init__(self, cfg_path:str):
        if not(cfg_path):
            raise RuntimeError
        
        with open('augmentation/config.yaml', 'r') as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

    def _apply_aug(self, waveform, sample_rate, effect):
        effector = AudioEffector(effect=effect, pad_end=True)
        return effector.apply(waveform, sample_rate)
    
    def __call__(self, waveform, sample_rate):
        waveform = waveform.T
        num_aug = random.randint(0, self.yaml['spectrogramm']['num_aug'])
        for _ in range(num_aug):
            effect = random.choice( self.yaml['spectrogramm']['effects'])
            waveform = self._apply_aug(waveform, sample_rate, effect)

        return waveform.T
