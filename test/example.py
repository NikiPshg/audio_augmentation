"""
Example usage of audio_augmentation package
"""
import torchaudio
from pathlib import Path
from audio_augmentation import Degrader

# Get the default config path
def get_default_config_path():
    """Get the path to the default config.yaml file"""
    import audio_augmentation
    from pathlib import Path
    package_dir = Path(audio_augmentation.__file__).parent
    config_path = package_dir / "config.yaml"
    return str(config_path)

# Example usage
if __name__ == "__main__":
    # Initialize degrader with config file
    config_path = get_default_config_path()
    # or you can use your own config path for example: configs/config.yaml
    # config_path = "C:/Users/nicit/Apython/audio_augmentation/configs/config.yaml"
    degrader = Degrader(cfg_path=config_path, max_audio_length=None)
    
    # Load an audio file
    audio_path = "C:/Users/nicit/Downloads/2.mp3"  # Replace with your audio file path
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Apply degradation
    degraded_waveform = degrader(waveform, sample_rate)
    
    # Save the degraded audio
    output_path = "degraded.wav"
    torchaudio.save(output_path, degraded_waveform, sample_rate)
    print(f"Degraded audio saved to {output_path}")
