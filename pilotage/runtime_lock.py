"""One live runtime per agent profile.

This is Hermes' gateway runtime-lock mechanism reduced to Pilotage's one-process,
one-profile contract. The operating system owns the advisory lock, so a crash
releases it without trusting a reusable PID or deleting a stale sentinel.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import threading
from pathlib import Path
from typing import IO, Optional

from .deployment import protected_lock

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


def _open_lock(path: Path, *, create: bool) -> IO[str]:
    if not protected_lock(path):
        return open(path, "a+" if create else "r+", encoding="utf-8")
    # The installer owns the sticky parent and provisions this inode. Never
    # create a replacement, follow a link, or chmod away the shared access.
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        from .deployment import load

        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != load().operator_uid or stat.S_IMODE(info.st_mode) != 0o660):
            raise RuntimeLockError("Protected runtime lock ownership or permissions changed.")
        return os.fdopen(fd, "r+", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise


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
    except BlockingIOError:
        return False
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN}:
            return False
        if getattr(exc, "winerror", None) in {32, 33}:
            return False
        raise RuntimeLockError(f"Cannot lock runtime lock {handle.name}: {exc}") from exc


def _unlock(handle: IO[str]) -> None:
    try:
        if msvcrt is not None:
            handle.seek(_WINDOWS_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        elif fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def runtime_lock_is_held(state_dir: Path) -> bool:
    """Probe the operating-system lock without changing its owner record."""

    resolved_state = Path(state_dir).expanduser().resolve(strict=False)
    path = resolved_state / ".runtime.lock"
    key = str(path)
    if path.is_symlink():
        raise RuntimeLockError(f"Runtime lock cannot be a symbolic link: {path}")
    with _PROCESS_LOCKS_GUARD:
        if key in _PROCESS_LOCKS:
            return True
    try:
        handle = _open_lock(path, create=False)
    except FileNotFoundError:
        if protected_lock(path):
            raise RuntimeLockError(f"Missing protected runtime lock: {path}")
        return False
    except OSError as exc:
        raise RuntimeLockError(f"Cannot open runtime lock {path}: {exc}") from exc
    try:
        acquired = _try_lock(handle)
        if acquired:
            _unlock(handle)
        return not acquired
    finally:
        handle.close()


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
                handle = _open_lock(self.path, create=True)
            except OSError as exc:
                raise RuntimeLockError(f"Cannot open runtime lock {self.path}: {exc}") from exc
            try:
                acquired = _try_lock(handle)
            except BaseException:
                try:
                    handle.close()
                except OSError:
                    pass
                raise
            if not acquired:
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
                if not protected_lock(self.path):
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

    def fileno(self) -> int:
        """Descriptor for explicitly passing a held POSIX lock to an installer."""
        if self._handle is None:
            raise RuntimeLockError(f"Runtime lock is not held: {self.path}")
        return self._handle.fileno()

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        # POSIX flock ownership follows the open file description. Closing our
        # copy lets an installer that inherited it keep the lock until it exits;
        # an explicit LOCK_UN would also unlock the installer's copy.
        if msvcrt is not None:
            _unlock(handle)
        try:
            handle.close()
        except OSError:
            pass
        with _PROCESS_LOCKS_GUARD:
            _PROCESS_LOCKS.discard(self._key)


__all__ = [
    "ProfileRuntimeLock",
    "RuntimeAlreadyRunning",
    "RuntimeLockError",
    "runtime_lock_is_held",
]
