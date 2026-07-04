from .config import DEFAULT_CONFIG, load_config
from .degrader import Degrader
from .effects import EFFECTS
from .utils import get_audio_paths

__version__ = "0.2.0"
__all__ = ["Degrader", "DEFAULT_CONFIG", "load_config", "EFFECTS", "get_audio_paths"]
