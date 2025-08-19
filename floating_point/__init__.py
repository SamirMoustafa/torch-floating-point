import sys
from pathlib import Path

# Try to import version from the current directory first (for Docker builds)
try:
    from version import __version__
except ImportError:
    # If that fails, try to import from the project root (for local development)
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from version import __version__

from .data_types import FloatingPoint
from .floating_point import autograd
from .round import Round

__all__ = ["FloatingPoint", "Round", "__version__", "autograd"]
