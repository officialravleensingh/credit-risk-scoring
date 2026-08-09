from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_CACHE_DIR = PROJECT_ROOT / ".cache"
MPL_CACHE_DIR = RUNTIME_CACHE_DIR / "matplotlib"
XDG_CACHE_DIR = RUNTIME_CACHE_DIR / "xdg"


def configure_runtime_environment() -> None:
    """Point cache-heavy libraries at writable project-local directories."""
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
    os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
