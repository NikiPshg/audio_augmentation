"""Small shared helpers."""

from __future__ import annotations

import random
from pathlib import Path

import torch

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".opus")


def get_audio_paths(root: str | Path, extensions=AUDIO_EXTENSIONS) -> list[Path]:
    root = Path(root)
    exts = {e.lower() for e in extensions}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def to_mono_float(waveform: torch.Tensor) -> torch.Tensor:
    """Any (T,) / (C, T) tensor -> contiguous float32 (1, T)."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() != 2:
        raise ValueError(f"Expected 1D or 2D waveform, got shape {tuple(waveform.shape)}")
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.to(torch.float32).contiguous()


def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def fast_fftconvolve(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Linear convolution via power-of-2 FFTs.

    ~3x faster than torchaudio.functional.fftconvolve, which transforms at
    the exact (usually prime-factor-unfriendly) output length.
    """
    n = x.size(-1) + h.size(-1) - 1
    nfft = next_pow2(n)
    out = torch.fft.irfft(torch.fft.rfft(x, nfft) * torch.fft.rfft(h, nfft), nfft)
    return out[..., :n]


def align_to_reference(reference: torch.Tensor, degraded: torch.Tensor,
                       max_shift: int = 4096, window: int = 32768) -> torch.Tensor:
    """Time-align ``degraded`` to ``reference`` via FFT cross-correlation and
    return it cropped/padded to the reference length.

    Codec delays are tiny (< max_shift samples), so the lag is estimated on a
    prefix window with a power-of-2 FFT instead of transforming the full
    signals - sub-millisecond vs the old O(n * delay) MSE scan.
    """
    ref_len = reference.size(-1)
    ref = reference[..., : min(ref_len, window)]
    deg = degraded[..., : min(degraded.size(-1), window + max_shift)]
    nfft = next_pow2(ref.size(-1) + deg.size(-1))
    # corr[k] = <ref(t), deg(t+k)>: the lag by which the codec delayed us.
    corr = torch.fft.irfft(
        torch.fft.rfft(ref, nfft).conj() * torch.fft.rfft(deg, nfft), nfft
    ).squeeze(0)
    lag = int(torch.argmax(corr[: min(max_shift, deg.size(-1))]).item())
    return match_length(degraded[..., lag:], ref_len)


def match_length(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    """Crop or zero-pad the last dim to ``target_len``."""
    cur = waveform.size(-1)
    if cur > target_len:
        return waveform[..., :target_len]
    if cur < target_len:
        return torch.nn.functional.pad(waveform, (0, target_len - cur))
    return waveform


def uniform(rng: random.Random, lo_hi) -> float:
    lo, hi = lo_hi
    return rng.uniform(float(lo), float(hi))
