from pathlib import Path
import torch

def align_waveform(wav1, wav2):
    assert wav2.size(1) >= wav1.size(1)
    diff = wav2.size(1) - wav1.size(1)
    min_mse = float("inf")
    best_i = -1

    for i in range(diff):
        segment = wav2[:, i : i + wav1.size(1)]
        mse = torch.mean((wav1 - segment) ** 2).item()
        if mse < min_mse:
            min_mse = mse
            best_i = i

    return best_i, wav2[:, best_i : best_i + wav1.size(1)]
    
def get_audio_paths(podcast_path: str):
    podcast_path=Path(podcast_path)
    return (
        list(podcast_path.rglob("*.mp3"))  + \
        list(podcast_path.rglob("*.wav"))  + \
        list(podcast_path.rglob("*.flac")) + \
        list(podcast_path.rglob("*.ogg"))  + \
        list(podcast_path.rglob("*.opus"))      
            
    )