"""Codec degradation via an in-memory torchcodec encode/decode round trip.

Replaces ``torchaudio.io.AudioEffector`` (removed in torchaudio 2.9+) and the
old pydub/ffmpeg-CLI paths. Everything stays in memory and is safe to run
inside DataLoader workers.
"""

from __future__ import annotations

import random

import torch

from .utils import align_to_reference, uniform


def apply_codec(waveform: torch.Tensor, sample_rate: int, spec: dict,
                rng: random.Random) -> torch.Tensor:
    """Round-trip ``waveform`` (1, T) through the codec described by ``spec``.

    spec keys: format (str), bit_rate ([lo, hi], optional),
    sample_rate (int, optional intermediate rate, e.g. 8000 for narrowband).
    """
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder

    encode_kwargs: dict = {}
    if "bit_rate" in spec:
        encode_kwargs["bit_rate"] = int(uniform(rng, spec["bit_rate"]))
    if "sample_rate" in spec:
        encode_kwargs["sample_rate"] = int(spec["sample_rate"])

    encoded = AudioEncoder(waveform, sample_rate=sample_rate).to_tensor(
        format=spec["format"], **encode_kwargs
    )
    decoded = AudioDecoder(encoded, sample_rate=sample_rate, num_channels=1)
    out = decoded.get_all_samples().data.to(torch.float32)

    if out.size(-1) != waveform.size(-1):
        out = align_to_reference(waveform, out)
    return out
