"""Synthetic room impulse responses in pure torch.

Replaces the pyroomacoustics image-source simulation (seconds per RIR,
computed eagerly at startup) with the standard stochastic model: a direct
impulse followed by exponentially decaying gaussian reverberation with the
decay set by a sampled RT60. Generation takes well under a millisecond, so
RIRs are drawn fresh for every call and nothing is cached per worker.
"""

from __future__ import annotations

import random

import torch

from .utils import fast_fftconvolve, match_length, uniform

MAX_RIR_SECONDS = 1.0


def sample_rir(sample_rate: int, rt60_range, drr_db_range,
               rng: random.Random, generator: torch.Generator) -> torch.Tensor:
    """Draw a random (1, L) impulse response with unit direct path."""
    rt60 = uniform(rng, rt60_range)
    length = int(min(rt60, MAX_RIR_SECONDS) * sample_rate)
    length = max(length, sample_rate // 100)

    t = torch.arange(length, dtype=torch.float32) / sample_rate
    # -60 dB at t = rt60  =>  amplitude decay exp(-6.908 * t / rt60).
    envelope = torch.exp(t * (-6.908 / rt60))
    tail = torch.randn(length, generator=generator) * envelope

    # Pre-delay gap between the direct sound and the reverberant tail.
    predelay = int(sample_rate * rng.uniform(0.002, 0.02))
    tail[: min(predelay, length)] = 0.0

    drr_db = uniform(rng, drr_db_range)
    tail_gain = 10.0 ** (-drr_db / 20.0) / (tail.norm(p=2) + 1e-8)
    rir = tail * tail_gain
    rir[0] = 1.0  # direct path, keeps the output time-aligned with the input
    return rir.unsqueeze(0)


def apply_rir(waveform: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """Convolve and restore the original length and RMS level."""
    rms_in = waveform.pow(2).mean().sqrt()
    out = match_length(fast_fftconvolve(waveform, rir), waveform.size(-1))
    rms_out = out.pow(2).mean().sqrt()
    return out * (rms_in / (rms_out + 1e-8))
