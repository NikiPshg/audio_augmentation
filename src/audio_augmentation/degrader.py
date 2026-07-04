"""The Degrader: a picklable, DataLoader-friendly audio augmentation pipeline.

Design notes
------------
* ``__init__`` stores only the config and the (small, picklable) noise file
  list. No tensors, no decoders, no RNGs - the instance pickles cleanly into
  spawned DataLoader workers and forks without dragging state along.
* All per-process state (RNGs, resamplers) is created lazily on first call
  and rebuilt automatically if the pid changes, so every worker gets its own
  independent random stream.
* Noise is decoded on demand as a random window of a random file
  (see ``noise.py``) - nothing is preloaded into RAM.
* Every stage preserves length and the output is always
  (1, T) float32 at ``config.sample_rate``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Mapping

import torch
import torchaudio.functional as F

from .codec import apply_codec
from .config import load_config
from .effects import apply_random_effects
from .noise import NoiseBank
from .phone import apply_phone
from .rir import apply_rir, sample_rir
from .utils import to_mono_float, uniform


class Degrader:
    def __init__(self, config: str | Path | Mapping | None = None):
        self.config = load_config(config)
        self.sample_rate = int(self.config["sample_rate"])

        noise_cfg = self.config["noise"]
        self._noise_bank = None
        if noise_cfg["prob"] > 0:
            self._noise_bank = NoiseBank(
                noise_cfg["dir"], noise_cfg["extensions"], self.sample_rate
            )

        self._stages = [
            name for name in ("noise", "rir", "codec", "effects", "phone")
            if self.config[name]["prob"] > 0
        ]

        # Lazy per-process state.
        self._pid: int | None = None
        self._rng: random.Random | None = None
        self._generator: torch.Generator | None = None
        self._resamplers: dict = {}

    # -- per-worker state ---------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_pid=None, _rng=None, _generator=None, _resamplers={})
        return state

    def _ensure_worker_state(self):
        pid = os.getpid()
        if self._pid == pid:
            return
        seed = self.config["seed"]
        if seed is not None:
            info = torch.utils.data.get_worker_info()
            seed = int(seed) + (info.id if info is not None else 0)
        else:
            seed = int.from_bytes(os.urandom(8), "little")
        self._rng = random.Random(seed)
        self._generator = torch.Generator().manual_seed(seed & 0x7FFF_FFFF_FFFF_FFFF)
        self._resamplers = {}
        self._pid = pid

    def _resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr == self.sample_rate:
            return waveform
        key = (orig_sr, self.sample_rate)
        if key not in self._resamplers:
            import torchaudio.transforms as T
            self._resamplers[key] = T.Resample(orig_sr, self.sample_rate)
        return self._resamplers[key](waveform)

    # -- stages ---------------------------------------------------------------

    def _add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        cfg = self.config["noise"]
        rng = self._rng
        total = waveform.size(-1)
        segment = int(total * uniform(rng, cfg["segment_ratio"]))
        if segment == 0:
            return waveform
        start = rng.randint(0, total - segment)

        noise = self._noise_bank.sample(segment, rng, self._generator)
        snr = torch.tensor([uniform(rng, cfg["snr_db"])])
        chunk = waveform[:, start:start + segment]
        if chunk.pow(2).sum() < 1e-10 or noise.pow(2).sum() < 1e-10:
            return waveform  # silent chunk or noise: SNR scaling is undefined
        noisy = F.add_noise(waveform=chunk, noise=noise, snr=snr)

        out = waveform.clone()
        out[:, start:start + segment] = noisy
        return out

    def _add_rir(self, waveform: torch.Tensor) -> torch.Tensor:
        cfg = self.config["rir"]
        rir = sample_rir(self.sample_rate, cfg["rt60"], cfg["drr_db"],
                         self._rng, self._generator)
        return apply_rir(waveform, rir)

    def _add_codec(self, waveform: torch.Tensor) -> torch.Tensor:
        codecs = self.config["codec"]["codecs"]
        names = sorted(codecs)
        weights = [float(codecs[n].get("weight", 1.0)) for n in names]
        name = self._rng.choices(names, weights=weights)[0]
        return apply_codec(waveform, self.sample_rate, codecs[name], self._rng)

    def _add_effects(self, waveform: torch.Tensor) -> torch.Tensor:
        cfg = self.config["effects"]
        return apply_random_effects(waveform, self.sample_rate, cfg["effects"],
                                    int(cfg["max_stack"]), self._rng)

    def _add_phone(self, waveform: torch.Tensor) -> torch.Tensor:
        return apply_phone(waveform, self.sample_rate, self.config["phone"], self._rng)

    # -- pipeline -------------------------------------------------------------

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self._ensure_worker_state()
        rng = self._rng

        waveform = to_mono_float(waveform)
        waveform = self._resample(waveform, sample_rate)

        max_sec = self.config["max_length_seconds"]
        if max_sec is not None:
            max_len = int(max_sec * self.sample_rate)
            if waveform.size(-1) > max_len:
                start = rng.randint(0, waveform.size(-1) - max_len)
                waveform = waveform[:, start:start + max_len]

        order = list(self._stages)
        rng.shuffle(order)
        codec_applied = False
        for stage in order:
            if rng.random() >= self.config[stage]["prob"]:
                continue
            if stage == "noise":
                waveform = self._add_noise(waveform)
            elif stage == "rir":
                waveform = self._add_rir(waveform)
            elif stage == "codec":
                waveform = self._add_codec(waveform)
                codec_applied = True
            elif stage == "effects":
                waveform = self._add_effects(waveform)
            elif stage == "phone" and not codec_applied:
                waveform = self._add_phone(waveform)

        waveform = torch.nan_to_num(waveform)
        peak = waveform.abs().max()
        if peak > 0.99:
            waveform = waveform * (0.99 / peak)
        return waveform
