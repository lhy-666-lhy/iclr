"""Robot PC backbones only (geometry / topology). Scene DP3 uses official PointNet."""

from pathlib import Path
import sys

THIRD_PARTY_ROOT = Path(__file__).resolve().parent

def ensure_on_path() -> Path:
    root = str(THIRD_PARTY_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return THIRD_PARTY_ROOT
