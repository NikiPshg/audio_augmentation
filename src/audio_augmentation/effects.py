"""Length-preserving spectral/dynamics effects in pure torch.

Replaces the FFmpeg filter strings previously run through
``torchaudio.io.AudioEffector`` (removed in torchaudio 2.9+). Each effect
takes (waveform (1, T), sample_rate, rng) and returns the same shape.
"""

from __future__ import annotations

import math
import random

import torch
import torchaudio.functional as F

from .utils import uniform


def _lowpass(w, sr, rng):
    return F.lowpass_biquad(w, sr, rng.uniform(800.0, sr * 0.45))


def _highpass(w, sr, rng):
    return F.highpass_biquad(w, sr, rng.uniform(80.0, 1500.0))


def _bandpass(w, sr, rng):
    return F.bandpass_biquad(w, sr, rng.uniform(300.0, min(3500.0, sr * 0.4)))


def _bandreject(w, sr, rng):
    return F.bandreject_biquad(w, sr, rng.uniform(300.0, min(3500.0, sr * 0.4)))


def _allpass(w, sr, rng):
    return F.allpass_biquad(w, sr, rng.uniform(300.0, sr * 0.45))


def _eq(w, sr, rng):
    center = rng.uniform(100.0, sr * 0.45)
    gain = rng.uniform(-12.0, 12.0)
    q = rng.uniform(0.5, 2.0)
    return F.equalizer_biquad(w, sr, center, gain, q)


def _tremolo(w, sr, rng):
    freq = rng.uniform(2.0, 12.0)
    depth = rng.uniform(0.3, 0.9)
    t = torch.arange(w.size(-1), dtype=torch.float32) / sr
    mod = (1.0 - depth / 2.0) + (depth / 2.0) * torch.sin(2 * math.pi * freq * t)
    return w * mod


def _clip(w, sr, rng):
    drive = rng.uniform(2.0, 10.0)
    rms_in = w.pow(2).mean().sqrt()
    out = torch.tanh(w * drive)
    rms_out = out.pow(2).mean().sqrt()
    return out * (rms_in / (rms_out + 1e-8))


def _gain(w, sr, rng):
    return w * 10.0 ** (rng.uniform(-12.0, 6.0) / 20.0)


def _phaser(w, sr, rng):
    return F.phaser(w, sr, mod_speed=rng.uniform(0.3, 2.0))


def _flanger(w, sr, rng):
    return F.flanger(w.unsqueeze(0), sr, speed=rng.uniform(0.3, 2.0)).squeeze(0)


EFFECTS = {
    "lowpass": _lowpass,
    "highpass": _highpass,
    "bandpass": _bandpass,
    "bandreject": _bandreject,
    "allpass": _allpass,
    "eq": _eq,
    "tremolo": _tremolo,
    "clip": _clip,
    "gain": _gain,
    "phaser": _phaser,
    "flanger": _flanger,
}


def apply_random_effects(waveform: torch.Tensor, sample_rate: int, names: list[str],
                         max_stack: int, rng: random.Random) -> torch.Tensor:
    unknown = set(names) - set(EFFECTS)
    if unknown:
        raise KeyError(f"Unknown effects {sorted(unknown)}; available: {sorted(EFFECTS)}")
    for name in rng.sample(names, k=min(rng.randint(1, max_stack), len(names))):
        waveform = EFFECTS[name](waveform, sample_rate, rng)
    return waveform
