"""Operator update of this Git checkout using the existing locked installer."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path

from . import deployment
from .config import state_dir
from .process_tree import terminate_process_tree
from .runtime_lock import RuntimeLock, RuntimeLockError
from .service import SERVICE_TIMEOUT_SECONDS, run_service_command, unit_name


class UpdateError(RuntimeError):
    pass


class UpdateInterrupted(KeyboardInterrupt):
    def __init__(self, signum: int):
        super().__init__()
        self.signum = signum


@contextmanager
def _update_signals():
    """Let installer cleanup finish before unwinding the update locks."""
    previous = {}

    def interrupt(signum, _frame):
        # Repeated signals must not interrupt process-tree cleanup.
        for managed in previous:
            signal.signal(managed, signal.SIG_IGN)
        raise UpdateInterrupted(signum)

    try:
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            signum = getattr(signal, name, None)
            if signum is not None:
                previous[signum] = signal.signal(signum, interrupt)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _install(root: Path, *, lock_fds: tuple[int, ...]) -> int:
    process = subprocess.Popen(
        ["bash", str(root / "scripts" / "install.sh"),
         *(["--dependencies-only"] if deployment.load() else [])],
        cwd=root, start_new_session=True,
        # A killed updater cannot release these locks while installation is
        # still changing shared dependencies. Bash and its children retain them.
        pass_fds=lock_fds,
    )
    try:
        code = process.wait(timeout=1800)
        if code:
            terminate_process_tree(process, pgid=process.pid)
        return code
    except BaseException:
        terminate_process_tree(process, pgid=process.pid)
        raise


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True,
        timeout=120, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    if result.returncode:
        raise UpdateError((result.stderr or result.stdout or "Git command failed").strip())
    return result.stdout.strip()


def _service_running() -> bool:
    result = subprocess.run(
        [*deployment.system_command("systemctl"), "show", unit_name(),
         "--property=LoadState,ActiveState", "--no-pager"],
        capture_output=True, text=True, timeout=SERVICE_TIMEOUT_SECONDS,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if values.get("LoadState") == "not-found":
        return False
    if result.returncode or values.get("LoadState") != "loaded":
        raise UpdateError(result.stderr.strip() or "Cannot determine service state")
    state = values.get("ActiveState")
    if state not in {"active", "inactive", "failed"}:
        raise UpdateError(f"Service is {state or 'unknown'}; wait before updating")
    return state == "active"


def run_update(*, check: bool = False) -> int:
    root = Path(__file__).resolve().parent.parent
    stopped = False
    try:
        managed = deployment.load()
        if managed:
            managed.require_operator()
        if sys.platform != "linux":
            raise UpdateError("Updates run on the Ubuntu deployment; use Git locally")
        for command in (("git",) if check else ("git", "bash", "systemctl")):
            if shutil.which(command) is None:
                raise UpdateError(f"{command} is not installed")
        if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
            raise UpdateError("The runtime must be installed from its own Git checkout")
        # Require an attached branch with an explicit upstream. Never choose a
        # branch, stash edits, reset commits, or elevate permissions implicitly.
        _git(root, "symbolic-ref", "--quiet", "HEAD")
        _git(root, "rev-parse", "--verify", "@{upstream}")
        git_dir = Path(_git(root, "rev-parse", "--absolute-git-dir"))
        with _update_signals(), ExitStack() as update_stack:
            update_lock = RuntimeLock(git_dir / "pilotage-update")
            update_lock.acquire()
            update_stack.callback(update_lock.release)
            if not check and _git(root, "status", "--porcelain", "--untracked-files=normal"):
                raise UpdateError("Local changes found; commit or move them before updating")
            print("Checking the configured Git upstream...", flush=True)
            _git(root, "fetch", "--no-tags")
            target = _git(root, "rev-parse", "--verify", "@{upstream}^{commit}")
            head = _git(root, "rev-parse", "HEAD")
            counts = _git(root, "rev-list", "--left-right", "--count", f"HEAD...{target}").split()
            ahead, behind = map(int, counts)
            if check:
                print(f"Upstream: {behind} new commit(s); local branch: {ahead} ahead.")
                return 0
            if ahead:
                raise UpdateError("Local commits differ from upstream; reconcile the branch before updating")
            agent_state = state_dir()
            running = _service_running()
            if running:
                if run_service_command("stop"):
                    raise UpdateError("Could not stop the service; no code was changed")
                stopped = True
            with ExitStack() as runtime_stack:
                # These same locks gate foreground and service startup.
                install_locks = [update_lock]
                lock = RuntimeLock(agent_state)
                lock.acquire()
                runtime_stack.callback(lock.release)
                install_locks.append(lock)
                if _git(root, "status", "--porcelain", "--untracked-files=normal"):
                    raise UpdateError("Local changes appeared during preflight; update refused")
                if _git(root, "rev-parse", "HEAD") != head:
                    raise UpdateError("Checkout changed during preflight; retry the update")
                if head != target:
                    _git(root, "merge", "--ff-only", target)
                print("Refreshing locked dependencies...", flush=True)
                if _install(root, lock_fds=tuple(lock.fileno() for lock in install_locks)):
                    raise UpdateError("Dependency installation failed; fix the error and rerun pilotage update")
                if managed:
                    deployment.make_runtime_readable(root)
                    verify_command = deployment.agent_python(["-c", "import pilotage.main"])
                else:
                    verify_command = [str(root / ".venv" / "bin" / "python"), "-B", "-c", "import pilotage.main"]
                result = subprocess.run(
                    verify_command,
                    cwd=root, timeout=60,
                )
                if result.returncode:
                    raise UpdateError("Updated runtime failed its import check")
            if running:
                if run_service_command("start"):
                    raise UpdateError("Update installed, but service startup failed; inspect pilotage logs")
                stopped = False
            print("Update complete." + (" Agent restarted." if running else " Agent remains stopped."))
            return 0
    except UpdateInterrupted as exc:
        print("Update interrupted.", file=sys.stderr)
        return 128 + exc.signum
    except KeyboardInterrupt:
        print("Update interrupted.", file=sys.stderr)
        return 130
    except (UpdateError, RuntimeLockError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"Update failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if stopped:
            print("The service was stopped for the update. Resolve the error before restarting it.", file=sys.stderr)
