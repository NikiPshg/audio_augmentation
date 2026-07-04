"""Telephone-channel simulation.

Bandpass to the classic 300-3400 Hz voice band, mild compression, then a
narrowband (8 kHz) codec round trip. Replaces the old FFmpeg
``lowpass + compand + g722`` AudioEffector chain.

The default codec is G.711 mu-law - the classic PSTN codec - implemented as
pure-torch companded 8-bit quantization at 8 kHz, so it costs ~1 ms and never
touches FFmpeg. Set ``codec: spx`` or ``opus`` for a lossy-codec round trip
via torchcodec instead.
"""

from __future__ import annotations

import random

import torch
import torchaudio.functional as F

from .codec import apply_codec
from .utils import match_length


def _g711_mu_law(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    down = F.resample(waveform, sample_rate, 8000)
    quantized = F.mu_law_encoding(down.clamp(-1.0, 1.0), 256)
    up = F.resample(F.mu_law_decoding(quantized, 256), 8000, sample_rate)
    return match_length(up, waveform.size(-1))


def apply_phone(waveform: torch.Tensor, sample_rate: int, cfg: dict,
                rng: random.Random) -> torch.Tensor:
    out = F.highpass_biquad(waveform, sample_rate, 300.0)
    out = F.lowpass_biquad(out, sample_rate, 3400.0)

    # Mild compand-style compression.
    drive = rng.uniform(1.5, 3.0)
    rms_in = out.pow(2).mean().sqrt()
    out = torch.tanh(out * drive)
    out = out * (rms_in / (out.pow(2).mean().sqrt() + 1e-8))

    codec = cfg.get("codec", "g711")
    if codec == "g711":
        return _g711_mu_law(out, sample_rate)
    spec = {
        "format": codec,
        "bit_rate": cfg.get("bit_rate", [4000, 12000]),
        "sample_rate": 8000,
    }
    return apply_codec(out, sample_rate, spec, rng)
