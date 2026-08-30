"""Explorer JS kernel must match committed Python/C++ goldens."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fp_codec_goldens():
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "check_fp_codec.py")], cwd=str(ROOT))
