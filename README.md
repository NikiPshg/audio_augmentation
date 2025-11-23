# audio_augmentation

This repository contains audio augmentation tools for PyTorch 2.5+, including codec degradation, noise addition, RIR simulation, and spectral transformations.

## Installation

Install directly from git:

```bash
pip install git+https://github.com/yourusername/audio_augmentation.git
```

Or if you have a private repository:

```bash
pip install git+ssh://git@github.com/yourusername/audio_augmentation.git
```

For development installation:

```bash
git clone https://github.com/yourusername/audio_augmentation.git
cd audio_augmentation
pip install -e .
```

## Usage

```python
import torchaudio
from audio_augmentation import Degrader

# Initialize degrader with config file
degrader = Degrader(cfg_path="path/to/config.yaml")

# Load audio
waveform, sample_rate = torchaudio.load("input.wav")

# Apply degradation
degraded_waveform = degrader(waveform, sample_rate)

# Save result
torchaudio.save("output.wav", degraded_waveform, sample_rate)
```

See `test/example.py` for a complete example.

## Features

- **Codec Degradation**: Apply various audio codecs (Opus, AMR, MP3, Speex, etc.)
- **Noise Addition**: Add background noise with configurable SNR
- **RIR Simulation**: Simulate room impulse responses
- **Spectral Transformations**: Apply various audio effects
- **Phone Channel Simulation**: Simulate telephone channel characteristics

## Requirements

- Python >= 3.10
- PyTorch >= 2.5.1
- torchaudio >= 2.5.1
- See `requirements.txt` for full list
