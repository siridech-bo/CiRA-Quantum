"""Provenance stamp for every QRC result (publication data package).

``provenance()`` returns the code version (git commit + whether the working tree
had uncommitted changes), library versions, hardware and a UTC timestamp, so any
number in the paper can be traced to the exact code and environment that made
it. Import-light: torch / qutip / scipy are reported only if importable.
"""
from __future__ import annotations

import datetime as dt
import platform
import socket
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must never fail a run
        return None


def _version(mod):
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return None


def provenance(**extra) -> dict:
    """Code / environment / time stamp. ``extra`` (e.g. seeds, config, split)
    is merged in under ``run``."""
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no")) if commit else None
    shipped = None
    if not commit:                       # remote hosts get code as a tarball + CODE_VERSION
        f = Path(__file__).resolve().parents[2] / "CODE_VERSION"
        if f.exists():
            shipped = f.read_text(encoding="utf-8").strip()
            commit = shipped.split()[0]
    info = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "code_version_file": shipped,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "qutip": _version("qutip"),
        "torch": _version("torch"),
    }
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    if extra:
        info["run"] = extra
    return info
