"""Symlink-safe creation helpers for spill and cache files.

Ported from Hermes ``tools/spill_safety.py``. Predictable cache names must
never follow a pre-planted symlink when they are refreshed.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import IO

_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def ensure_spill_dir(path: Path, *, private: bool = True) -> Path:
    """Create a real directory, refusing a symlink at the leaf."""
    path = Path(path)
    if private:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)
    state = os.lstat(path)
    if not stat.S_ISDIR(state.st_mode):
        raise OSError(f"spill dir is not a directory (symlink?): {path}")
    if private and stat.S_IMODE(state.st_mode) != 0o700:
        os.chmod(path, 0o700)
    return path


def open_exclusive(
    path: Path,
    *,
    private: bool = True,
    overwrite: bool = False,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> IO[str]:
    """Open a path via exclusive creation and never follow a symlink."""
    path = Path(path)
    if overwrite:
        try:
            state = os.lstat(path)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISDIR(state.st_mode):
                raise OSError(f"refusing to overwrite a directory: {path}")
            os.unlink(path)
    mode = 0o600 if private else 0o666
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW,
        mode,
    )
    try:
        return os.fdopen(
            descriptor,
            "w",
            encoding=encoding,
            errors=errors,
        )
    except Exception:
        os.close(descriptor)
        raise


def write_text_exclusive(
    path: Path,
    text: str,
    *,
    private: bool = True,
    overwrite: bool = False,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> None:
    """``Path.write_text`` equivalent that refuses to follow symlinks."""
    with open_exclusive(
        path,
        private=private,
        overwrite=overwrite,
        encoding=encoding,
        errors=errors,
    ) as stream:
        stream.write(text)


__all__ = ["ensure_spill_dir", "open_exclusive", "write_text_exclusive"]
