"""Configuration handling: defaults + deep-merge of a user YAML/dict."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    # The degrader operates at this sample rate. Inputs with a different
    # rate are resampled on the fly; the output is always at this rate.
    "sample_rate": 16000,
    # Random-crop inputs longer than this (seconds). null = no cropping.
    "max_length_seconds": None,
    # Base seed for per-worker RNGs. null = non-deterministic.
    "seed": None,
    "noise": {
        "prob": 0.5,
        # Directory with noise recordings. null = synthetic pink/white noise.
        "dir": None,
        "extensions": [".wav", ".flac", ".mp3", ".ogg", ".opus"],
        "snr_db": [2.0, 30.0],
        # Fraction of the utterance covered by noise.
        "segment_ratio": [0.2, 1.0],
    },
    "rir": {
        "prob": 0.5,
        "rt60": [0.15, 0.8],
        # Direct-to-reverberant energy ratio, dB. Lower = wetter.
        "drr_db": [-2.0, 12.0],
    },
    "codec": {
        "prob": 0.5,
        # name: {format, bit_rate: [lo, hi], sample_rate (opt), weight (opt)}
        # weight sets how often a codec is picked relative to the others -
        # opus is by far the slowest to encode (libopus, ~25 ms per 4 s),
        # so the defaults draw it less often. Bump the weights (or drop the
        # entry) to trade realism vs speed.
        "codecs": {
            "mp3": {"format": "mp3", "bit_rate": [8000, 64000]},
            "vorbis": {"format": "ogg", "bit_rate": [16000, 64000]},
            "speex_8k": {"format": "spx", "bit_rate": [4000, 12000], "sample_rate": 8000},
            "opus": {"format": "opus", "bit_rate": [6000, 32000], "weight": 0.5},
            "opus_8k": {"format": "opus", "bit_rate": [4000, 16000], "sample_rate": 8000,
                        "weight": 0.5},
        },
    },
    "effects": {
        "prob": 0.5,
        # How many randomly chosen effects to stack (uniform in [1, max_stack]).
        "max_stack": 2,
        "effects": [
            "lowpass",
            "highpass",
            "bandpass",
            "bandreject",
            "eq",
            "tremolo",
            "clip",
            "gain",
        ],
    },
    "phone": {
        "prob": 0.5,
        "codec": "g711",  # g711 (pure torch, fastest) | spx | opus
        "bit_rate": [4000, 12000],  # used by spx/opus only
    },
}


def _deep_merge(base: dict, override: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(source: str | Path | Mapping | None = None) -> dict[str, Any]:
    """Build a full config dict from a YAML path, a dict, or defaults."""
    if source is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    if isinstance(source, (str, Path)):
        with open(source, "r") as f:
            user = yaml.safe_load(f) or {}
    elif isinstance(source, Mapping):
        user = source
    else:
        raise TypeError(f"Unsupported config source: {type(source)!r}")
    return _deep_merge(DEFAULT_CONFIG, user)
