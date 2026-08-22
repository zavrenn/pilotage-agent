"""One live runtime per isolated profile.

This is Hermes' gateway runtime-lock mechanism reduced to Pilotage's one-process,
one-profile contract. The operating system owns the advisory lock, so a crash
releases it without trusting a reusable PID or deleting a stale sentinel.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import IO, Optional

try:  # pragma: no cover - selected by platform
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no cover - selected by platform
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]


_WINDOWS_LOCK_OFFSET = 1024 * 1024
_PROCESS_LOCKS: set[str] = set()
_PROCESS_LOCKS_GUARD = threading.Lock()


class RuntimeLockError(RuntimeError):
    """The selected profile's runtime lock cannot be used."""


class RuntimeAlreadyRunning(RuntimeLockError):
    """Another process already owns the selected profile."""


def _try_lock(handle: IO[str]) -> bool:
    try:
        if msvcrt is not None:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("\n")
                handle.flush()
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        elif fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:  # pragma: no cover - supported Python platforms provide one
            return False
        return True
    except (BlockingIOError, OSError):
        return False


def _unlock(handle: IO[str]) -> None:
    try:
        if msvcrt is not None:
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        elif fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


class ProfileRuntimeLock:
    """Hold one profile's cross-process singleton lock until ``release``."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir).expanduser().resolve(strict=False)
        self.path = self.state_dir / ".runtime.lock"
        self._handle: Optional[IO[str]] = None
        self._key = str(self.path)

    def acquire(self) -> None:
        if self._handle is not None:
            return
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeLockError(
                f"Cannot create runtime state directory {self.state_dir}: {exc}"
            ) from exc
        if self.path.is_symlink():
            raise RuntimeLockError(f"Runtime lock cannot be a symbolic link: {self.path}")

        with _PROCESS_LOCKS_GUARD:
            if self._key in _PROCESS_LOCKS:
                raise RuntimeAlreadyRunning(
                    f"Another Pilotage runtime is already using {self.state_dir}."
                )
            try:
                handle = open(self.path, "a+", encoding="utf-8")
            except OSError as exc:
                raise RuntimeLockError(f"Cannot open runtime lock {self.path}: {exc}") from exc
            if not _try_lock(handle):
                handle.close()
                raise RuntimeAlreadyRunning(
                    f"Another Pilotage runtime is already using {self.state_dir}."
                )
            _PROCESS_LOCKS.add(self._key)

        try:
            handle.seek(0)
            handle.truncate()
            json.dump({"pid": os.getpid(), "state_dir": str(self.state_dir)}, handle)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        except BaseException:
            _unlock(handle)
            handle.close()
            with _PROCESS_LOCKS_GUARD:
                _PROCESS_LOCKS.discard(self._key)
            raise
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        _unlock(handle)
        try:
            handle.close()
        except OSError:
            pass
        with _PROCESS_LOCKS_GUARD:
            _PROCESS_LOCKS.discard(self._key)


__all__ = ["ProfileRuntimeLock", "RuntimeAlreadyRunning", "RuntimeLockError"]
