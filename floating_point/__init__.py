import sys
from pathlib import Path

# Add the project root to the path to import version
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from version import __version__

from .floating_point import autograd

__all__ = ["__version__", "autograd"]
