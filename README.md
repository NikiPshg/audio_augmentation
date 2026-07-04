# audio_augmentation

Fast, DataLoader-safe audio degradation pipeline for PyTorch: additive noise,
room reverb, lossy codecs, spectral effects and telephone-channel simulation.

Built for training data augmentation: a picklable `Degrader` you create once,
pass into your `Dataset`, and call inside `__getitem__` with any number of
DataLoader workers (`fork` or `spawn`).

## Why v0.2 is a rewrite

| | old (v0.1) | new (v0.2) |
|---|---|---|
| Codecs / effects | `torchaudio.io.AudioEffector` — **removed in torchaudio 2.9+** | in-memory [torchcodec](https://github.com/pytorch/torchcodec) encode/decode round trip |
| Noise | pre-loaded up to 1000 files into RAM (duplicated per worker) | random window of a random file decoded on demand |
| RIR | pyroomacoustics image-source sim, ~seconds per room at startup | synthetic RIR in pure torch, < 1 ms, drawn fresh per call |
| Multiprocessing | heavy `__init__`, unpicklable state, shared RNG | picklable; lazy per-worker state; independent per-worker RNG streams |
| Effects | FFmpeg filter strings | pure-torch biquads / tremolo / clip / gain |
| Deps | pyroomacoustics, pydub, tqdm, numpy | torch, torchaudio, torchcodec, PyYAML |

## Install

```bash
pip install git+https://github.com/NikiPshg/audio_augmentation.git
# or for development
git clone https://github.com/NikiPshg/audio_augmentation.git
cd audio_augmentation && pip install -e .
```

Requires Python >= 3.10, torch/torchaudio >= 2.6, torchcodec (which needs an
FFmpeg shared-library install, versions 4-8).

## Quickstart

```python
import torch
from audio_augmentation import Degrader

degrader = Degrader()                       # library defaults
# degrader = Degrader("configs/config.yaml")  # or a YAML file
# degrader = Degrader({"noise": {"dir": "/data/noises", "prob": 0.7}})  # or a dict

wav = torch.randn(1, 16000 * 4) * 0.1
out = degrader(wav, sample_rate=16000)      # (1, T) float32 @ config sample_rate
```

Input can be `(T,)` or `(C, T)` at any sample rate — it is downmixed to mono
and resampled to `config.sample_rate` (default 16 kHz). Output always has the
same length as the (resampled) input.

### In a DataLoader

```python
from torch.utils.data import DataLoader, Dataset

class MyDataset(Dataset):
    def __init__(self, files, degrader):
        self.files = files
        self.degrader = degrader            # just store it; it pickles cleanly

    def __getitem__(self, idx):
        wav, sr = load_audio(self.files[idx])
        return self.degrader(wav, sr)

    def __len__(self):
        return len(self.files)

ds = MyDataset(files, Degrader({"noise": {"dir": "/data/noises"}}))
dl = DataLoader(ds, batch_size=32, num_workers=8)   # fork and spawn both work
```

No `worker_init_fn` needed: each worker lazily builds its own RNG (seeded from
the pid, or from `config.seed + worker_id` when a seed is set) and its own
decoder state on first call.

## Pipeline

Enabled stages are applied in **random order**, each with its own
probability:

- **noise** — a random window of a random file from `noise.dir` is decoded
  (partial decode, nothing cached in RAM) and mixed over a random segment of
  the utterance at a random SNR. With `dir: null` synthetic pink/white noise
  is used, so the package works out of the box.
- **rir** — a synthetic room impulse response (exponentially decaying
  gaussian tail, sampled RT60 and direct-to-reverberant ratio, pre-delay) is
  convolved with the signal; output is re-leveled to the input RMS.
- **codec** — a random codec from the config is applied as an in-memory
  encode/decode round trip: mp3, vorbis, opus, speex, plus narrowband (8 kHz)
  variants, with a random bitrate from the configured range. Decoded audio is
  re-aligned to the input by FFT cross-correlation when the codec shifts it.
- **effects** — a random stack of pure-torch effects: low/high/band-pass,
  band-reject, EQ, tremolo, tanh clipping, gain (plus `allpass`, `phaser`,
  `flanger` available by name).
- **phone** — 300-3400 Hz band-pass, mild compression, then a narrowband
  codec: G.711 mu-law by default (pure-torch companded 8-bit quantization at
  8 kHz, ~1 ms), or a speex/opus round trip. Skipped when a codec was
  already applied.

## Config

Everything is optional; unspecified keys fall back to
`audio_augmentation.DEFAULT_CONFIG`. See [configs/config.yaml](configs/config.yaml)
for the full annotated reference:

```yaml
sample_rate: 16000
max_length_seconds: null      # random-crop longer inputs
seed: null                    # set for reproducible augmentation

noise:
  prob: 0.5
  dir: /data/noises           # null -> synthetic noise
  snr_db: [2.0, 30.0]
  segment_ratio: [0.2, 1.0]

rir:
  prob: 0.5
  rt60: [0.15, 0.8]
  drr_db: [-2.0, 12.0]

codec:
  prob: 0.5
  codecs:                     # weight = relative pick probability
    mp3:     {format: mp3,  bit_rate: [8000, 64000]}
    opus_8k: {format: opus, bit_rate: [4000, 16000], sample_rate: 8000, weight: 0.5}

effects:
  prob: 0.5
  max_stack: 2
  effects: [lowpass, highpass, bandpass, bandreject, eq, tremolo, clip, gain]

phone:
  prob: 0.5
  codec: g711                 # g711 | spx | opus
```

Set `prob: 0` to disable a stage entirely.

## Performance

4 s mono @ 16 kHz, 50 iterations, single thread
(`torch.set_num_threads(1)`, i.e. a typical DataLoader worker);
torch 2.11 / torchcodec 0.14 / FFmpeg 4.4:

| Stage | Mean (ms) | Median (ms) |
|---|---|---|
| noise | 2.18 | 3.25 |
| rir | 3.14 | 3.14 |
| codec | 11.53 | 8.42 |
| effects | 0.45 | 0.42 |
| phone | 1.44 | 1.44 |
| full pipeline (p=0.5 each) | 9.07 | 6.66 |
| full pipeline (p=1.0 each) | 18.36 | 15.93 |

Per codec (the only FFmpeg-bound stage; everything else is pure torch):

| Codec | Mean (ms) | |
|---|---|---|
| mp3 | 5.9 | |
| vorbis | 7.1 | |
| speex_8k | 9.2 | |
| opus_8k | 16.8 | libopus encode dominates |
| opus | 19.1 | libopus encode dominates |

opus is the slow one and there is no way to lower libopus complexity through
torchcodec, so the default config picks opus/opus_8k with `weight: 0.5`.
Drop them from `codec.codecs` if you want the codec stage under 8 ms, or
raise the weights if realism matters more than speed.

≈ 9 ms per 4 s clip at default probabilities. DataLoader throughput
(batch 32, default config, same machine):

| num_workers | clips/s |
|---|---|
| 0 | 131 |
| 4 | 419 |
| 8 | 681 |

Reproduce with:

```bash
python benchmarks/bench.py --seconds 4 --iters 50 [--noise-dir /data/noises]
```

## Tests

```bash
pytest test/            # or: python test/test_degrader.py
```

Covers the output contract, every stage in isolation, seed reproducibility,
pickling, and DataLoader runs under both `fork` and `spawn`.

## Troubleshooting

- **`Could not load libtorchcodec`** — torchcodec needs FFmpeg shared
  libraries (versions 4-8) at runtime: `conda install ffmpeg` or
  `apt install ffmpeg`. On CUDA builds it may also need the CUDA runtime
  libs bundled with torch on `LD_LIBRARY_PATH`
  (e.g. `.../site-packages/nvidia/cu13/lib`).
- **A codec fails to encode** — the available encoders depend on your FFmpeg
  build (e.g. `amr`/`gsm` are usually absent). Remove that entry from
  `codec.codecs`; the defaults only use widely available encoders.
