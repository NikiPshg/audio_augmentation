"""Additive noise source.

Instead of pre-loading hundreds of files into RAM (which gets duplicated in
every DataLoader worker), we keep only the file list and decode a random
window of a random file on demand with torchcodec. If no noise directory is
configured, synthetic pink/white noise is generated on the fly.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch

from .utils import get_audio_paths, match_length


class NoiseBank:
    def __init__(self, noise_dir: str | Path | None, extensions, sample_rate: int):
        self.sample_rate = sample_rate
        self.paths: list[str] = []
        if noise_dir is not None:
            self.paths = [str(p) for p in get_audio_paths(noise_dir, extensions)]
            if not self.paths:
                raise FileNotFoundError(
                    f"noise.dir='{noise_dir}' contains no audio files "
                    f"(extensions {list(extensions)}). Set noise.dir to null "
                    f"to use synthetic noise instead."
                )

    def sample(self, num_samples: int, rng: random.Random,
               generator: torch.Generator) -> torch.Tensor:
        """Return a (1, num_samples) noise tensor at ``self.sample_rate``."""
        if not self.paths:
            return self._synthetic(num_samples, rng, generator)
        for _ in range(3):
            path = rng.choice(self.paths)
            try:
                return self._read_window(path, num_samples, rng)
            except Exception:
                continue
        # All retries failed (corrupt files?) - degrade gracefully.
        return self._synthetic(num_samples, rng, generator)

    def _read_window(self, path: str, num_samples: int, rng: random.Random) -> torch.Tensor:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(path, sample_rate=self.sample_rate, num_channels=1)
        duration = decoder.metadata.duration_seconds_from_header
        need_sec = num_samples / self.sample_rate
        if duration is not None and duration > need_sec + 0.1:
            start = rng.uniform(0.0, duration - need_sec - 0.05)
            samples = decoder.get_samples_played_in_range(start, start + need_sec + 0.05)
            noise = samples.data
        else:
            noise = decoder.get_all_samples().data
        if noise.size(-1) == 0:
            raise RuntimeError(f"Empty decode: {path}")
        if noise.size(-1) < num_samples:
            repeats = math.ceil(num_samples / noise.size(-1))
            noise = noise.repeat(1, repeats)
        return match_length(noise.to(torch.float32), num_samples)

    @staticmethod
    def _synthetic(num_samples: int, rng: random.Random,
                   generator: torch.Generator) -> torch.Tensor:
        if rng.random() < 0.5:
            return torch.randn(1, num_samples, generator=generator)
        # Pink-ish noise: shape the spectrum by 1/sqrt(f). Generated at a
        # power-of-2 length so both FFTs hit the fast path, then cropped.
        from .utils import next_pow2

        nfft = next_pow2(num_samples)
        white = torch.randn(1, nfft, generator=generator)
        spec = torch.fft.rfft(white)
        freqs = torch.arange(spec.size(-1), dtype=torch.float32).clamp_(min=1.0)
        pink = torch.fft.irfft(spec / freqs.sqrt(), nfft)[..., :num_samples]
        return pink / (pink.std() + 1e-8)
