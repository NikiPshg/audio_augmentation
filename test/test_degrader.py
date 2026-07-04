"""Smoke tests: run with `pytest test/` or `python test/test_degrader.py`."""

import pickle

import torch
from torch.utils.data import DataLoader, Dataset

from audio_augmentation import Degrader

SR = 16000


def _speech_like(seconds=2.0, sr=SR):
    t = torch.arange(int(seconds * sr)) / sr
    tone = sum(torch.sin(2 * torch.pi * f * t) for f in (220.0, 440.0, 880.0))
    return (0.2 * tone * (0.5 + 0.5 * torch.sin(2 * torch.pi * 3.0 * t))).unsqueeze(0)


def test_output_contract():
    deg = Degrader()
    wav = _speech_like()
    out = deg(wav, SR)
    assert out.shape == wav.shape
    assert out.dtype == torch.float32
    assert out.abs().max() <= 1.0
    assert torch.isfinite(out).all()


def test_resample_and_stereo_input():
    deg = Degrader()
    wav = torch.rand(2, 8000 * 3) - 0.5  # stereo @ 8 kHz
    out = deg(wav, 8000)
    assert out.size(0) == 1
    assert out.size(1) == 3 * SR  # resampled to config rate


def test_every_stage_runs():
    for stage in ("noise", "rir", "codec", "effects", "phone"):
        cfg = {name: {"prob": 0.0} for name in ("noise", "rir", "codec", "effects", "phone")}
        cfg[stage] = {"prob": 1.0}
        out = Degrader(cfg)(_speech_like(), SR)
        assert torch.isfinite(out).all(), stage


def test_seed_reproducibility():
    wav = _speech_like()
    a = Degrader({"seed": 7})(wav, SR)
    b = Degrader({"seed": 7})(wav, SR)
    assert torch.equal(a, b)


def test_picklable():
    deg = Degrader()
    deg(_speech_like(), SR)  # populate lazy state
    clone = pickle.loads(pickle.dumps(deg))
    out = clone(_speech_like(), SR)
    assert torch.isfinite(out).all()


class _Ds(Dataset):
    def __init__(self, degrader):
        self.degrader = degrader
        self.wav = _speech_like()

    def __len__(self):
        return 16

    def __getitem__(self, idx):
        return self.degrader(self.wav, SR)


def test_dataloader_multiprocessing():
    ds = _Ds(Degrader())
    for ctx in ("fork", "spawn"):
        dl = DataLoader(ds, batch_size=4, num_workers=2, multiprocessing_context=ctx)
        batches = list(dl)
        assert len(batches) == 4
        for b in batches:
            assert torch.isfinite(b).all()


if __name__ == "__main__":
    for fn in sorted(k for k in dir() if k.startswith("test_")):
        print(f"{fn} ...", end=" ", flush=True)
        globals()[fn]()
        print("ok")
