from pathlib import Path


def get_audio_paths(self, path: str):
    start_dir = Path(path).resolve()
    return list(start_dir.rglob('*.wav'))