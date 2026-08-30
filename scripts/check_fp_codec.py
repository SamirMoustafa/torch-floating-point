#!/usr/bin/env python3
"""Check explorer JS against the Python/C++ fp-codec goldens.

python scripts/check_fp_codec.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    gen = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "gen_fp_codec_goldens.py"), "--check"],
        cwd=str(ROOT),
        check=False,
    )
    if gen.returncode != 0:
        return gen.returncode
    node = shutil.which("node")
    if node is None:
        print("install Node: node is required to assert docs/javascripts/fp-codec.js", file=sys.stderr)
        return 1
    ass = subprocess.run([node, str(ROOT / "scripts" / "assert_fp_codec.js")], cwd=str(ROOT), check=False)
    return ass.returncode


if __name__ == "__main__":
    sys.exit(main())
