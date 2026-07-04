"""Benchmark each augmentation stage and the full pipeline.

Usage:
    python benchmarks/bench.py [--noise-dir PATH] [--seconds 4] [--iters 50]

Prints a markdown table with the mean/median wall time per stage.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import torch

from audio_augmentation import Degrader

SR = 16000


def speech_like(seconds: float) -> torch.Tensor:
    t = torch.arange(int(seconds * SR)) / SR
    tone = sum(torch.sin(2 * torch.pi * f * t) for f in (180.0, 360.0, 720.0, 1440.0))
    envelope = 0.5 + 0.5 * torch.sin(2 * torch.pi * 2.5 * t)
    return (0.2 * tone * envelope + 0.01 * torch.randn_like(t)).unsqueeze(0)


def make_noise_dir(root: Path, n_files: int = 10, seconds: float = 30.0) -> Path:
    from torchcodec.encoders import AudioEncoder

    noise_dir = root / "noises"
    noise_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        wav = torch.randn(1, int(seconds * SR)) * 0.1
        fmt = ["wav", "flac", "mp3"][i % 3]
        AudioEncoder(wav, sample_rate=SR).to_file(noise_dir / f"noise_{i:02d}.{fmt}")
    return noise_dir


def bench(fn, iters: int, warmup: int = 3) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.mean(times), statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-dir", type=Path, default=None,
                        help="real noise folder; default: generated synthetic files")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    torch.set_num_threads(1)  # emulate a DataLoader worker
    wav = speech_like(args.seconds)

    with tempfile.TemporaryDirectory() as tmp:
        noise_dir = args.noise_dir or make_noise_dir(Path(tmp))
        stages = ("noise", "rir", "codec", "effects", "phone")

        rows = []
        for stage in stages:
            cfg = {name: {"prob": 0.0} for name in stages}
            cfg[stage]["prob"] = 1.0
            cfg["noise"]["dir"] = str(noise_dir) if stage == "noise" else None
            deg = Degrader(cfg)
            mean, median = bench(lambda: deg(wav, SR), args.iters)
            rows.append((stage, mean, median))

        from audio_augmentation import DEFAULT_CONFIG
        from audio_augmentation.codec import apply_codec
        import random as _random

        codec_rng = _random.Random(0)
        for cname, spec in DEFAULT_CONFIG["codec"]["codecs"].items():
            mean, median = bench(lambda: apply_codec(wav, SR, spec, codec_rng), args.iters)
            rows.append((f"  codec: {cname}", mean, median))

        full = Degrader({"noise": {"dir": str(noise_dir)}})
        mean, median = bench(lambda: full(wav, SR), args.iters)
        rows.append(("full pipeline (p=0.5 each)", mean, median))

        all_on = Degrader({name: {"prob": 1.0} for name in stages}
                          | {"noise": {"prob": 1.0, "dir": str(noise_dir)}})
        mean, median = bench(lambda: all_on(wav, SR), args.iters)
        rows.append(("full pipeline (p=1.0 each)", mean, median))

    print(f"\nInput: {args.seconds:.0f}s mono @ {SR} Hz, {args.iters} iters, 1 thread\n")
    print("| Stage | Mean (ms) | Median (ms) |")
    print("|---|---|---|")
    for name, mean, median in rows:
        print(f"| {name} | {mean:.2f} | {median:.2f} |")


if __name__ == "__main__":
    main()
