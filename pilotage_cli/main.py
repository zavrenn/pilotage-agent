#!/usr/bin/env python3
"""
Pilotage CLI - Main entry point.

Usage:
    pilotage                     # Interactive chat (default)
    pilotage chat                # Interactive chat
    pilotage gateway             # Run gateway in foreground
    pilotage gateway start       # Start gateway as service
    pilotage gateway stop        # Stop gateway service
    pilotage gateway status      # Show gateway status
    pilotage gateway install     # Install gateway service
    pilotage gateway uninstall   # Uninstall gateway service
    pilotage setup               # Interactive setup wizard
    pilotage status              # Show status of all components
    pilotage cron                # Manage cron jobs
    pilotage cron list           # List cron jobs
    pilotage cron status         # Check if cron scheduler is running
    pilotage doctor              # Check configuration and dependencies
    pilotage memory status       # Show memory provider config
    pilotage skills list         # List installed skills
    pilotage plugins list        # List plugins
    pilotage sessions browse     # Interactive session picker with search
    pilotage version             # Show version
    pilotage uninstall           # Uninstall Pilotage Agent

Run `pilotage --help` for the full command list.
"""

# IMPORTANT: pilotage_bootstrap must be the very first import — it sets up
# UTF-8 stdio on Windows so print()/subprocess children don't hit
# UnicodeEncodeError with non-ASCII characters.  No-op on POSIX.
#
# Guarded against ModuleNotFoundError because ``pilotage_bootstrap`` is a
# top-level module registered via pyproject.toml's ``py-modules`` list.
# When the user upgrades code via ``git pull`` (or ``pilotage update``
# crashes between ``git reset --hard`` and ``uv pip install -e .``), the
# new code references ``pilotage_bootstrap`` but the editable install's
# ``.pth`` file still points at the old set of top-level modules.  Without
# this guard, pilotage crashes on import and the user can't run
# ``pilotage update`` to recover.  Missing the bootstrap means UTF-8 stdio
# setup is skipped on Windows — degraded, not broken.  POSIX is unaffected.
try:
    import pilotage_bootstrap  # noqa: F401
except ModuleNotFoundError:
    pass

# Windows: neutralize CPython's ``platform._syscmd_ver`` before anything else
# imports — it shells out ``cmd /c ver`` (shell=True, no CREATE_NO_WINDOW), so
# any dependency touching ``platform.uname()`` at import time flashes a
# visible console when this process is windowless (pythonw gateway + every
# worker process).  No-op on POSIX; never raises.
from pilotage_cli._subprocess_compat import suppress_platform_ver_console

suppress_platform_ver_console()

import os
import sys

# ── Startup fast-path bootstrap ─────────────────────────────────────────
# Two lines of inline path math so ``python pilotage_cli/main.py`` (script
# mode — sys.path[0] is pilotage_cli/, not the repo root) can import the
# canonical helpers; everything else lives in pilotage_cli._startup_fast.
_bootstrap_root = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
if _bootstrap_root not in sys.path:
    sys.path.insert(0, _bootstrap_root)
from pilotage_cli import _startup_fast  # noqa: E402

# Early venv self-heal — MUST run before any third-party import below.  When
# a prior ``pilotage update`` left a recovery marker and a core package's import
# files were wiped — failed lazy backend refresh), the module-level
# ``from pilotage_cli.env_loader import ...`` / ``from pilotage_cli.config import
# ...`` imports further down would crash before ``main()`` ever reaches
# ``_recover_from_interrupted_install()``.  ``_early_recovery`` is stdlib-only
# (safe to import on a corrupted venv), repairs just enough for this module to
# finish importing, and leaves the marker lifecycle to the full recovery path.
# The module import itself is unguarded on purpose: it lives in this same
# package directory, so if IT can't import, nothing else in pilotage_cli can
# either. It is also the canonical home of the probe/repair tables reused by
# the full recovery path below.
from pilotage_cli import _early_recovery as _early_recovery_mod

try:
    _early_recovery_mod.recover_if_needed()
except Exception:
    pass


def _exit_after_oneshot(rc: object) -> None:
    """Exit one-shot mode without letting late native finalizers change rc.

    The SIGABRT this guards against (,) fires in a
    native-extension finalizer during CPython's ``Py_FinalizeEx``, *after*
    the response has printed. Flush streams, shut down file logging, then
    ``os._exit`` past interpreter finalization. The ``atexit`` chain is
    deliberately skipped — several handlers re-enter native code that may
    be the abort source. Stateful cleanup is handled in ``_run_agent`` and
    ``_cleanup_oneshot_runtime``.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    try:
        logging.shutdown()
    except Exception:
        pass
    if rc is None:
        exit_code = 0
    elif isinstance(rc, int):
        exit_code = rc
    else:
        exit_code = 1
    os._exit(exit_code)


_oneshot_cleanup_done = False


def _cleanup_oneshot_runtime() -> None:
    """Best-effort process-global cleanup before one-shot hard exit.

    ``run_oneshot`` owns the agent-local cleanup (memory provider, agent.close,
    session_db.close — all in ``_run_agent``'s finally block). This mirrors the
    process-global pieces from ``cli.py:_run_cleanup()`` that would otherwise
    be skipped by ``os._exit``.
    """
    global _oneshot_cleanup_done
    if _oneshot_cleanup_done:
        return
    _oneshot_cleanup_done = True
    try:
        from tools.terminal_tool import cleanup_all_environments
        cleanup_all_environments()
    except Exception:
        pass
    try:
        from tools.async_delegation import interrupt_all
        interrupt_all(reason="oneshot shutdown")
    except Exception:
        pass
    try:
        from agent.auxiliary_client import shutdown_cached_clients
        shutdown_cached_clients()
    except Exception:
        pass


def _run_and_exit_oneshot(
    prompt: str,
    *,
    model: object = None,
    provider: object = None,
    toolsets: object = None,
    usage_file: object = None,
) -> None:
    try:
        from pilotage_cli.oneshot import run_oneshot

        rc = run_oneshot(
            prompt,
            model=model,
            provider=provider,
            toolsets=toolsets,
            usage_file=usage_file,
        )
    except KeyboardInterrupt:
        rc = 130
    except SystemExit as exc:
        if exc.code is not None and not isinstance(exc.code, int):
            print(exc.code, file=sys.stderr)
            rc = 1
        else:
            rc = exc.code
    except BaseException:
        # Defense-in-depth. ``run_oneshot`` already converts agent failures
        # into an int return code and only re-raises KeyboardInterrupt /
        # SystemExit (handled above). Anything still escaping here means
        # ``run_oneshot`` itself malfunctioned — surface it on stderr but never
        # fall through to normal interpreter teardown, which is the exact path
        # that aborts with SIGABRT on AL2023 (the bug this routine fixes).
        import traceback
        try:
            traceback.print_exc()
        except Exception:
            pass
        rc = 1
    try:
        _cleanup_oneshot_runtime()
    finally:
        # The hard exit is the safety boundary for. Even an interrupt
        # during best-effort cleanup must not fall back into interpreter
        # finalization, where the reported native SIGABRT occurs.
        _exit_after_oneshot(rc)


def _project_root_str_fast() -> str:
    return _startup_fast.project_root_str()


def _ensure_project_root_on_path_fast() -> None:
    _startup_fast.ensure_project_root_on_path()


def _set_process_title() -> None:
    """Set the process title to 'pilotage' so tools like 'ps', 'top', and
    'htop' show the app name instead of 'python3.xx'.

    Purely cosmetic — non-fatal on any platform.

    Strategy (try in order):
      1. ``setproctitle`` (opt-in dep — installed via ``pilotage tools`` or
         ``pip install setproctitle``, or bundled in a future release).
      2. ctypes ``prctl(PR_SET_NAME)`` (Linux only, 15-char limit).
      3. ctypes ``pthread_setname_np`` (macOS only, kernel thread name —
         changes lldb/top but not ``ps aux``).
      4. No-op on Windows (the .exe name is already ``pilotage.exe``).
    """
    # Strategy 1: setproctitle (best — works on macOS, Linux, BSD)
    try:
        import setproctitle  # type: ignore[import-untyped]

        setproctitle.setproctitle("pilotage")
        return
    except ImportError:
        pass

    # Strategy 2/3: platform-specific ctypes fallback
    import ctypes
    import platform

    try:
        system = platform.system()
        if system == "Linux":
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            libc.prctl(15, b"pilotage", 0, 0, 0)  # PR_SET_NAME = 15
        elif system == "Darwin":
            libc = ctypes.CDLL("libc.dylib", use_errno=True)
            libc.pthread_setname_np(b"pilotage")
        # Windows: the .exe name is already ``pilotage.exe`` — nothing to do.
    except Exception:
        pass


def _is_termux_startup_environment_fast() -> bool:
    """Tiny Termux check for pre-import startup shortcuts."""
    return _startup_fast.is_termux_env()


def _is_termux_fast_version_argv(argv: list[str]) -> bool:
    return _startup_fast.is_termux_fast_version_argv(argv)


def _is_global_fast_version_argv(argv: list[str]) -> bool:
    return _startup_fast.is_global_fast_version_argv(argv)


def _is_container_startup_environment_fast() -> bool:
    return _startup_fast.is_container_startup_environment()


def _active_profile_may_override_home_fast(pilotage_root: str) -> bool:
    return _startup_fast.active_profile_may_override_home(pilotage_root)


def _read_openai_version_fast() -> str | None:
    """Read OpenAI SDK version without importing ``importlib.metadata``."""
    return _startup_fast.read_openai_version()


def _print_fast_version_info() -> None:
    _startup_fast.print_fast_version_info()


def _try_ultrafast_version() -> bool:
    """Handle ``pilotage --version`` before config/logging imports."""
    return _startup_fast.try_fast_version()


def _try_termux_ultrafast_version() -> bool:
    """Backward-compatible test hook for the Termux startup fast path."""
    if not _is_termux_startup_environment_fast():
        return False
    return _try_ultrafast_version()


_ensure_project_root_on_path_fast()

if _try_ultrafast_version():
    raise SystemExit(0)

import argparse
import hashlib
import json
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Optional


import functools as _functools

from pilotage_cli.subcommands._shared import add_accept_hooks_flag as _add_accept_hooks_flag
from pilotage_cli.subcommands.cron import build_cron_parser
from pilotage_cli.subcommands.gateway import build_gateway_parser
from pilotage_cli.subcommands.profile import build_profile_parser
from pilotage_cli.subcommands.model import build_model_parser
from pilotage_cli.subcommands.setup import build_setup_parser

from pilotage_cli.subcommands.whatsapp import build_whatsapp_parser
from pilotage_cli.subcommands.login import build_login_parser
from pilotage_cli.subcommands.logout import build_logout_parser
from pilotage_cli.subcommands.auth import build_auth_parser
from pilotage_cli.subcommands.status import build_status_parser
from pilotage_cli.subcommands.pause import build_pause_parser
from pilotage_cli.subcommands.webhook import build_webhook_parser
from pilotage_cli.subcommands.hooks import build_hooks_parser
from pilotage_cli.subcommands.doctor import build_doctor_parser
from pilotage_cli.subcommands.security import build_security_parser
from pilotage_cli.subcommands.approvals import build_approvals_parser
from pilotage_cli.subcommands.dump import build_dump_parser
from pilotage_cli.subcommands.debug import build_debug_parser
from pilotage_cli.subcommands.backup import build_backup_parser
from pilotage_cli.subcommands.import_cmd import build_import_cmd_parser
from pilotage_cli.subcommands.config import build_config_parser
from pilotage_cli.subcommands.version import build_version_parser
from pilotage_cli.subcommands.uninstall import build_uninstall_parser
from pilotage_cli.subcommands.logs import build_logs_parser
from pilotage_cli.subcommands.prompt_size import build_prompt_size_parser
from pilotage_cli.subcommands.memory import build_memory_parser
from pilotage_cli.subcommands.tools import build_tools_parser
from pilotage_cli.subcommands.skills import build_skills_parser
from pilotage_cli.subcommands.pairing import build_pairing_parser
from pilotage_cli.subcommands.plugins import build_plugins_parser


def _require_tty(command_name: str) -> None:
    """Exit with a clear error if stdin is not a terminal.

    Interactive TUI commands (pilotage tools, pilotage setup, pilotage model) use
    curses or input() prompts that spin at 100% CPU when stdin is a pipe.
    This guard prevents accidental non-interactive invocation.
    """
    if not sys.stdin.isatty():
        print(
            f"Error: 'pilotage {command_name}' requires an interactive terminal.\n"
            f"It cannot be run through a pipe or non-interactive subprocess.\n"
            f"Run it directly in your terminal instead.",
            file=sys.stderr,
        )
        sys.exit(1)


# Add project root to path
PROJECT_ROOT = Path(_project_root_str_fast())
_ensure_project_root_on_path_fast()


# ---------------------------------------------------------------------------
# Profile override — MUST happen before any pilotage module import.
#
# Many modules cache PILOTAGE_HOME at import time (module-level constants).
# We intercept --profile/-p from sys.argv here and set the env var so that
# every subsequent ``os.getenv("PILOTAGE_HOME", ...)`` resolves correctly.
# The flag is stripped from sys.argv so argparse never sees it.
# Falls back to ~/.pilotage/active_profile for sticky default.
# ---------------------------------------------------------------------------
def _apply_profile_override() -> None:
    """Pre-parse --profile/-p and set PILOTAGE_HOME before imports."""
    argv = sys.argv[1:]
    profile_name = None
    consume = 0
    profile_index = None

    def _inside_mcp_add_args(index: int) -> bool:
        """True once argv reaches `pilotage mcp add ... --args <command argv>`.

        ``mcp add --args`` is command-argv passthrough. Flags after that point
        belong to the child MCP command (for example Docker MCP Toolkit's
        ``--profile``), not to Pilotage' own profile selector.
        """
        try:
            mcp_index = argv.index(0, index)
            argv.index("add", mcp_index + 1, index)
        except ValueError:
            return False
        return True

    def _resolve_sudo_user_profile_env(name: str) -> str | None:
        """Resolve `sudo pilotage -p <name>` against the invoking user's home.

        `_apply_profile_override()` runs before argparse, so `--run-as-user`
        is not available yet. For sudo invocations, the best available signal
        is SUDO_USER: root is only doing the privileged install/start action,
        while the profile store normally belongs to the user who invoked sudo.
        """
        if name == "default":
            return None
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            return None
        sudo_user = os.environ.get("SUDO_USER", "").strip()
        if not sudo_user or sudo_user == "root":
            return None

        try:
            import pwd

            home = Path(pwd.getpwnam(sudo_user).pw_dir)
        except Exception:
            return None

        candidate = home / ".pilotage" / "profiles" / name
        try:
            if candidate.is_dir():
                return str(candidate)
        except OSError:
            return None
        return None

    # 1. Check for explicit -p / --profile flag. Historically this worked even
    # after the subcommand (`pilotage chat -p coder`), so keep scanning broadly.
    # The exception is command-argv passthrough regions such as `mcp add --args`.
    value_flags = {
        "-z", "--oneshot",
        "-m", "--model",
        "--provider",
        "-t", "--toolsets",
        "-r", "--resume",
        "-s", "--skills",
        "--usage-file",
        "--in",
    }
    optional_value_flags = {"-c", "--continue"}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            break
        if arg == "--args" and _inside_mcp_add_args(i):
            break
        if arg in {"--profile", "-p"} and i + 1 < len(argv):
            profile_name = argv[i + 1]
            consume = 2
            profile_index = i
            break
        if arg.startswith("--profile="):
            profile_name = arg.split("=", 1)[1]
            consume = 1
            profile_index = i
            break
        if "=" not in arg and arg in value_flags and i + 1 < len(argv):
            i += 2
        elif (
            "=" not in arg
            and arg in optional_value_flags
            and i + 1 < len(argv)
            and not argv[i + 1].startswith("-")
        ):
            i += 2
        else:
            i += 1

    # 1b. Reject values that can't be valid profile names (e.g. pytest's
    # "-p no:xdist" would be misread as profile "no:xdist" otherwise).
    # Mirrors pilotage_cli.profiles._PROFILE_ID_RE so we never call
    # resolve_profile_env() with a value it must reject + sys.exit on.
    if profile_name is not None and consume == 2:
        import re as _re

        if not _re.match(r"^[a-z0-9][a-z0-9_-]{0,63}$", profile_name):
            profile_name = None
            consume = 0
            profile_index = None

    # 1.5 If PILOTAGE_HOME is already set and no explicit flag was given, trust it
    # only when it already points to a specific profile directory.  The
    # distinguishing heuristic: a profile path has "profiles" as its immediate
    # parent directory name (e.g. ~/.pilotage/profiles/coder or
    # /opt/data/profiles/coder).  If PILOTAGE_HOME points to the pilotage root
    # instead (e.g. systemd hardcodes PILOTAGE_HOME=/root/.pilotage), we must
    # still read active_profile — the user may have switched profiles via
    # `pilotage profile use` and the gateway should honour that choice.
    # See.
    pilotage_home_env = os.environ.get("PILOTAGE_HOME", "")
    if profile_name is None and pilotage_home_env:
        if Path(pilotage_home_env).parent.name == "profiles":
            return

    # 2. If no flag, check active_profile in the pilotage root.
    #
    # EXCEPTION: a supervised s6 gateway child (exported by the container
    # run-script as PILOTAGE_S6_SUPERVISED_CHILD=1) must NOT follow the sticky
    # active_profile. Each supervised slot has a fixed profile identity: named
    # slots pass ``-p <name>`` explicitly (handled in step 1 above), and the
    # reserved ``gateway-default`` slot runs bare ``pilotage gateway run`` to mean
    # "the root PILOTAGE_HOME profile". If the reserved default child read
    # active_profile here, switching the active profile (e.g. via the dashboard)
    # would silently redirect the default gateway into that profile — yielding a
    # duplicate gateway for the active profile and no real default gateway. See
    # the "Docker & Profiles & Dashboard" report.
    if profile_name is None and not os.environ.get("PILOTAGE_S6_SUPERVISED_CHILD"):
        try:
            from pilotage_constants import get_default_pilotage_root

            active_path = get_default_pilotage_root() / "active_profile"
            if active_path.exists():
                name = active_path.read_text(encoding="utf-8").strip()
                if name and name != "default":
                    profile_name = name
                    consume = 0  # don't strip anything from argv
        except (UnicodeDecodeError, OSError):
            pass  # corrupted file, skip

    # 3. If we found a profile, resolve and set PILOTAGE_HOME
    if profile_name is not None:
        try:
            from pilotage_cli.profiles import resolve_profile_env

            pilotage_home = resolve_profile_env(profile_name)
        except FileNotFoundError as exc:
            pilotage_home = _resolve_sudo_user_profile_env(profile_name)
            if not pilotage_home:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            # A bug in profiles.py must NEVER prevent pilotage from starting
            print(
                f"Warning: profile override failed ({exc}), using default",
                file=sys.stderr,
            )
            return
        os.environ["PILOTAGE_HOME"] = pilotage_home
        # Strip the flag from argv so argparse doesn't choke
        if consume > 0 and profile_index is not None:
            start = profile_index + 1  # +1 because argv is sys.argv[1:]
            sys.argv = sys.argv[:start] + sys.argv[start + consume :]


_apply_profile_override()

# Load .env from ~/.pilotage/.env first, then project root as dev fallback.
# User-managed env files should override stale shell exports on restart.
from pilotage_cli.config import get_pilotage_home
from pilotage_cli.env_loader import load_pilotage_dotenv

load_pilotage_dotenv(project_env=PROJECT_ROOT / ".env")

# Bridge security.redact_secrets from config.yaml → PILOTAGE_REDACT_SECRETS env
# var BEFORE pilotage_logging imports agent.redact (which snapshots the flag at
# module-import time). Without this, config.yaml's toggle is ignored because
# the setup_logging() call below imports agent.redact, which reads the env var
# exactly once. Env var in .env still wins — this is config.yaml fallback only.
#
# We also read network.force_ipv4 from the same yaml load to avoid two
# separate config.yaml reads (saves ~17ms on every CLI startup — the second
# `load_config()` was doing a full deep-merge for one boolean lookup).
_FORCE_IPV4_EARLY = False
try:
    # Reuse read_raw_config()'s (mtime, size)-keyed cache instead of a bespoke
    # yaml.load — the SAME parse then serves pilotage_logging's
    # _read_logging_config and any later raw reads in this process, collapsing
    # 3-4 config.yaml parses per invocation into one.
    from pilotage_cli.config import read_raw_config as _read_raw_early

    _cfg_path = get_pilotage_home() / "config.yaml"
    if _cfg_path.exists():
        _early_cfg_raw = _read_raw_early() or {}
        # Managed scope: overlay administrator-pinned values so a managed
        # security.redact_secrets / network.force_ipv4 wins here too. This early
        # bridge reads config.yaml directly (before load_config is usable), so
        # without the overlay a managed redact_secrets toggle would be ignored.
        # Fail-open via the shared helper.
        try:
            from pilotage_cli import managed_scope
            _early_cfg_raw = managed_scope.apply_managed_overlay(_early_cfg_raw)
        except Exception:
            pass
        if "PILOTAGE_REDACT_SECRETS" not in os.environ:
            _early_sec_cfg = _early_cfg_raw.get("security", {})
            if isinstance(_early_sec_cfg, dict):
                _early_redact = _early_sec_cfg.get("redact_secrets")
                if _early_redact is not None:
                    os.environ["PILOTAGE_REDACT_SECRETS"] = str(_early_redact).lower()
        _early_net_cfg = _early_cfg_raw.get("network", {})
        if isinstance(_early_net_cfg, dict) and _early_net_cfg.get("force_ipv4"):
            _FORCE_IPV4_EARLY = True
        del _early_cfg_raw
    del _cfg_path
except Exception:
    pass  # best-effort — redaction stays at default (enabled) on config errors

# Initialize centralized file logging early — all `pilotage` subcommands
# (chat, setup, gateway, config, etc.) write to agent.log + errors.log.
try:
    from pilotage_logging import setup_logging as _setup_logging

    _setup_logging(mode="cli")
except Exception:
    pass  # best-effort — don't crash the CLI if logging setup fails

# Apply IPv4 preference early, before any HTTP clients are created.
# We already determined whether to force IPv4 from the raw yaml read above —
# this just calls the toggle without a redundant load_config() round trip.
if _FORCE_IPV4_EARLY:
    try:
        from pilotage_constants import apply_ipv4_preference as _apply_ipv4

        _apply_ipv4(force=True)
    except Exception:
        pass  # best-effort — don't crash if pilotage_constants not importable yet

import logging
import threading
import time as _time
from datetime import datetime

from pilotage_cli import __version__, __release_date__

# Provider model-selection wizard flows extracted to pilotage_cli/model_setup_flows.py
# (god-file decomposition Phase 2). Re-imported here so select_provider_and_model and
# existing test monkeypatches (pilotage_cli.main._model_flow_*) keep resolving unchanged.
from pilotage_cli.model_setup_flows import (
    _prompt_auth_credentials_choice,
    _model_flow_openai_codex,
    _model_flow_custom,
    _model_flow_named_custom,
    _model_flow_api_key_provider,
)
logger = logging.getLogger(__name__)


def _is_termux_startup_environment(env: dict[str, str] | None = None) -> bool:
    """Import-safe Termux check for cold-start-sensitive CLI paths."""
    check = env or os.environ
    prefix = str(check.get("PREFIX", ""))
    return bool(
        check.get("TERMUX_VERSION")
        or "com.termux/files/usr" in prefix
        or prefix.startswith("/data/data/com.termux/")
    )


def _read_packed_ref(common_dir: Path, ref: str) -> str | None:
    """Look up a ref in .git/packed-refs without spawning git.

    packed-refs lines look like ``<sha> <ref>`` with optional ``^<sha>``
    peel lines and ``#``-prefixed comments / ``# pack-refs with:`` header.
    """
    try:
        text = (common_dir / "packed-refs").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            return parts[0].strip()
    return None


def _read_git_revision_fingerprint(repo_root: Path) -> str | None:
    """Return a cheap checkout fingerprint without spawning git."""
    git_dir = repo_root / ".git"
    try:
        if git_dir.is_file():
            for line in git_dir.read_text(encoding="utf-8", errors="replace").splitlines():
                key, _, value = line.partition(":")
                if key.strip() == "gitdir" and value.strip():
                    git_dir = (repo_root / value.strip()).resolve()
                    break
        # Worktrees point HEAD at a per-worktree gitdir but pack their refs
        # in the main repo's gitdir (referenced via ``commondir``). Resolve
        # that up front so packed-refs lookups hit the right file.
        common_dir = git_dir
        commondir_file = git_dir / "commondir"
        if commondir_file.exists():
            try:
                rel = commondir_file.read_text(encoding="utf-8", errors="replace").strip()
                if rel:
                    common_dir = (git_dir / rel).resolve()
            except OSError:
                pass
        head_file = git_dir / "HEAD"
        head = head_file.read_text(encoding="utf-8", errors="replace").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            # Loose refs may live in the worktree gitdir OR the common dir
            # (branches created via `git worktree add` typically live in the
            # common dir's refs/heads/).
            for candidate in (git_dir, common_dir):
                ref_file = candidate / ref
                if ref_file.exists():
                    return f"git:{ref}:{ref_file.read_text(encoding='utf-8', errors='replace').strip()}"
            packed_sha = _read_packed_ref(common_dir, ref)
            if packed_sha:
                return f"git:{ref}:{packed_sha}"
            # Ref name is known but unresolved — still stable across launches,
            # and the version/release fallback in the caller will invalidate
            # after `pilotage update`.
            return f"git:{ref}:unresolved"
        return f"git:HEAD:{head}"
    except OSError:
        return None


def _termux_should_prefetch_update_check() -> bool:
    if not _is_termux_startup_environment():
        return True
    return os.environ.get("PILOTAGE_TERMUX_PREFETCH_UPDATES") == "1"


def _relative_time(ts) -> str:
    """Format a timestamp as relative time (e.g., '2h ago', 'yesterday').

    Thin wrapper kept for backward compatibility; the implementation lives
    in :mod:`pilotage_cli.timefmt` so lightweight consumers don't have to
    import the whole CLI surface.
    """
    from pilotage_cli.timefmt import relative_time

    return relative_time(ts)


def _has_any_provider_configured() -> bool:
    """Check if at least one inference provider is usable."""
    from pilotage_cli.config import get_env_path, get_pilotage_home, load_config
    from pilotage_cli.auth import get_auth_status

    # Determine whether Pilotage itself has been explicitly configured (model
    # in config that isn't the hardcoded default). Used below to gate external
    # tool credentials (Claude Code, Codex CLI) that shouldn't silently skip
    # the setup wizard on a fresh install.
    from pilotage_cli.config import DEFAULT_CONFIG

    _DEFAULT_MODEL = DEFAULT_CONFIG.get("model", "")
    cfg = load_config()
    model_cfg = cfg.get("model")
    if isinstance(model_cfg, dict):
        _default = model_cfg.get("default")
        if isinstance(_default, dict):
            from pilotage_cli.config import split_model_config_default
            _model_name, _ = split_model_config_default(_default)
        else:
            _model_name = (_default or "")
        _model_name = (str(_model_name) if not isinstance(_model_name, str) else _model_name).strip()
    elif isinstance(model_cfg, str):
        _model_name = model_cfg.strip()
    else:
        _model_name = ""
    _has_pilotage_config = _model_name and _model_name != _DEFAULT_MODEL

    # Check env vars (may be set by .env or shell).
    # OPENAI_BASE_URL alone counts — local models (vLLM, llama.cpp, etc.)
    # often don't require an API key.
    from pilotage_cli.auth import PROVIDER_REGISTRY

    # Collect all provider env vars
    provider_env_vars = {
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "OPENAI_BASE_URL",
    }
    for pconfig in PROVIDER_REGISTRY.values():
        if pconfig.auth_type == "api_key":
            provider_env_vars.update(pconfig.api_key_env_vars)
    if any(os.getenv(v) for v in provider_env_vars):
        return True

    # Check .env file for keys
    env_file = get_env_path()
    if env_file.exists():
        try:
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:]
                key, _, val = line.partition("=")
                val = val.strip().strip("'\"")
                if key.strip() in provider_env_vars and val:
                    return True
        except Exception:
            pass

    # Cheap local checks first: auth.json and config.yaml are on-disk lookups,
    # while the PROVIDER_REGISTRY sweep below spawns subprocesses (gh) and can
    # take 15-20s — long enough that desktop setup.status calls time out.

    # Check for Nous Portal OAuth credentials
    auth_file = get_pilotage_home() / "auth.json"
    if auth_file.exists():
        try:
            import json

            auth = json.loads(auth_file.read_text(encoding="utf-8-sig"))
            active = auth.get("active_provider")
            if active:
                status = get_auth_status(active)
                if status.get("logged_in"):
                    return True
        except Exception:
            pass

    # Check config.yaml — if model is a dict with an explicit provider set,
    # the user has gone through setup (fresh installs have model as a plain
    # string).  Also covers custom endpoints that store api_key/base_url in
    # config rather than .env.
    if isinstance(model_cfg, dict):
        cfg_provider = (model_cfg.get("provider") or "").strip()
        cfg_base_url = (model_cfg.get("base_url") or "").strip()
        cfg_api_key = (model_cfg.get("api_key") or "").strip()
        if cfg_provider or cfg_base_url or cfg_api_key:
            return True

    # Check provider-specific auth fallbacks (for example, Copilot via gh auth).
    try:
        for provider_id, pconfig in PROVIDER_REGISTRY.items():
            if pconfig.auth_type != "api_key":
                continue
            status = get_auth_status(provider_id)
            if status.get("logged_in"):
                return True
    except Exception:
        pass

    return False


def _confirm_startup_expensive_model_override(args) -> None:
    """Guard startup -m/--provider overrides before the first API call."""
    explicit_model = (getattr(args, "model", None) or "").strip()
    explicit_provider = (getattr(args, "provider", None) or "").strip()
    if not explicit_model and not explicit_provider:
        return

    try:
        from pilotage_cli.config import load_config
        from pilotage_cli.model_selection_guards import combined_selection_warning
    except Exception as exc:
        logger.warning("startup model cost guard unavailable: %s", exc)
        return

    try:
        model_cfg = (load_config().get("model") or {})
    except Exception as exc:
        logger.warning("startup model cost guard could not load config: %s", exc)
        model_cfg = {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    model = explicit_model or (model_cfg.get("default") or "").strip()
    if not model:
        return
    provider = (explicit_provider or model_cfg.get("provider") or "").strip()
    try:
        # Unified registry: cost guard + id-keyed guards (e.g. the
        # data-training-tier warning) all fire at startup too.
        warning = combined_selection_warning(
            model,
            provider=provider,
            base_url=(model_cfg.get("base_url") or ""),
            api_key=(model_cfg.get("api_key") or ""),
        )
    except Exception as exc:
        logger.warning("startup model cost guard failed for %s/%s: %s", provider, model, exc)
        return
    if warning is None:
        return

    # Cost and provider-routing confirmation is intentionally independent of
    # --yolo / --accept-hooks: those flags approve local command/tool risk, not
    # paid aggregator spend or a surprising provider route.
    message = warning.message
    if not sys.stdin.isatty():
        sys.stderr.write(message + "\n")
        sys.stderr.write(
            "Refusing this startup model override in non-interactive mode. "
            "Run interactively and confirm if you intend to use it.\n"
        )
        raise SystemExit(1)

    sys.stderr.write(message + "\n")
    try:
        reply = input("Use this model for this invocation? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        reply = ""
    if reply not in {"y", "yes"}:
        sys.stderr.write("Model override cancelled.\n")
        raise SystemExit(1)


def _session_browse_picker(sessions: list) -> Optional[str]:
    """Interactive curses-based session browser with live search filtering.

    Returns the selected session ID, or None if cancelled.
    """
    if not sessions:
        print("No sessions found.")
        return None

    # Try curses-based picker first
    try:
        import curses

        result_holder = [None]

        def _format_row(s, max_x):
            """Format a session row for display."""
            title = (s.get("title") or "").strip()
            preview = (s.get("preview") or "").strip()
            source = s.get("source", "")[:6]
            last_active = _relative_time(s.get("last_active"))
            sid = s["id"][:18]

            # Adaptive column widths based on terminal width
            # Layout: [arrow 3] [title/preview flexible] [active 12] [src 6] [id 18]
            fixed_cols = 3 + 12 + 6 + 18 + 6  # arrow + active + src + id + padding
            name_width = max(20, max_x - fixed_cols)

            if title:
                name = title[:name_width]
            elif preview:
                name = preview[:name_width]
            else:
                name = sid

            return f"{name:<{name_width}}  {last_active:<10}  {source:<5} {sid}"

        def _match(s, query):
            """Check if a session matches the search query (case-insensitive)."""
            q = query.lower()
            return (
                q in (s.get("title") or "").lower()
                or q in (s.get("preview") or "").lower()
                or q in s.get("id", "").lower()
                or q in (s.get("source") or "").lower()
            )

        def _curses_browse(stdscr):
            curses.curs_set(0)
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_GREEN, -1)  # selected
                curses.init_pair(2, curses.COLOR_YELLOW, -1)  # header
                curses.init_pair(3, curses.COLOR_CYAN, -1)  # search
                curses.init_pair(4, 8 if curses.COLORS > 8 else curses.COLOR_WHITE, -1)  # dim

            cursor = 0
            scroll_offset = 0
            search_text = ""
            filtered = list(sessions)

            while True:
                stdscr.clear()
                max_y, max_x = stdscr.getmaxyx()
                if max_y < 5 or max_x < 40:
                    # Terminal too small
                    try:
                        stdscr.addstr(0, 0, "Terminal too small")
                    except curses.error:
                        pass
                    stdscr.refresh()
                    stdscr.getch()
                    return

                # Header line
                if search_text:
                    header = f"  Browse sessions — filter: {search_text}█"
                    header_attr = curses.A_BOLD
                    if curses.has_colors():
                        header_attr |= curses.color_pair(3)
                else:
                    header = "  Browse sessions — ↑↓ navigate  Enter select  Type to filter  Esc quit"
                    header_attr = curses.A_BOLD
                    if curses.has_colors():
                        header_attr |= curses.color_pair(2)
                try:
                    stdscr.addnstr(0, 0, header, max_x - 1, header_attr)
                except curses.error:
                    pass

                # Column header line
                fixed_cols = 3 + 12 + 6 + 18 + 6
                name_width = max(20, max_x - fixed_cols)
                col_header = f"   {'Title / Preview':<{name_width}}  {'Active':<10}  {'Src':<5} {'ID'}"
                try:
                    dim_attr = (
                        curses.color_pair(4) if curses.has_colors() else curses.A_DIM
                    )
                    stdscr.addnstr(1, 0, col_header, max_x - 1, dim_attr)
                except curses.error:
                    pass

                # Compute visible area
                visible_rows = max_y - 4  # header + col header + blank + footer
                visible_rows = max(visible_rows, 1)

                # Clamp cursor and scroll
                if not filtered:
                    try:
                        msg = "  No sessions match the filter."
                        stdscr.addnstr(3, 0, msg, max_x - 1, curses.A_DIM)
                    except curses.error:
                        pass
                else:
                    if cursor >= len(filtered):
                        cursor = len(filtered) - 1
                    cursor = max(cursor, 0)
                    if cursor < scroll_offset:
                        scroll_offset = cursor
                    elif cursor >= scroll_offset + visible_rows:
                        scroll_offset = cursor - visible_rows + 1

                    for draw_i, i in enumerate(
                        range(
                            scroll_offset,
                            min(len(filtered), scroll_offset + visible_rows),
                        )
                    ):
                        y = draw_i + 3
                        if y >= max_y - 1:
                            break
                        s = filtered[i]
                        arrow = " → " if i == cursor else "   "
                        row = arrow + _format_row(s, max_x - 3)
                        attr = curses.A_NORMAL
                        if i == cursor:
                            attr = curses.A_BOLD
                            if curses.has_colors():
                                attr |= curses.color_pair(1)
                        try:
                            stdscr.addnstr(y, 0, row, max_x - 1, attr)
                        except curses.error:
                            pass

                # Footer
                footer_y = max_y - 1
                if filtered:
                    footer = f"  {cursor + 1}/{len(filtered)} sessions"
                    if len(filtered) < len(sessions):
                        footer += f" (filtered from {len(sessions)})"
                else:
                    footer = f"  0/{len(sessions)} sessions"
                try:
                    stdscr.addnstr(
                        footer_y,
                        0,
                        footer,
                        max_x - 1,
                        curses.color_pair(4) if curses.has_colors() else curses.A_DIM,
                    )
                except curses.error:
                    pass

                stdscr.refresh()
                key = stdscr.getch()

                if key in {curses.KEY_UP,}:
                    if filtered:
                        cursor = (cursor - 1) % len(filtered)
                elif key in {curses.KEY_DOWN,}:
                    if filtered:
                        cursor = (cursor + 1) % len(filtered)
                elif key in {curses.KEY_ENTER, 10, 13}:
                    if filtered:
                        result_holder[0] = filtered[cursor]["id"]
                    return
                elif key == 27:  # Esc
                    if search_text:
                        # First Esc clears the search
                        search_text = ""
                        filtered = list(sessions)
                        cursor = 0
                        scroll_offset = 0
                    else:
                        # Second Esc exits
                        return
                elif key in {curses.KEY_BACKSPACE, 127, 8}:
                    if search_text:
                        search_text = search_text[:-1]
                        if search_text:
                            filtered = [s for s in sessions if _match(s, search_text)]
                        else:
                            filtered = list(sessions)
                        cursor = 0
                        scroll_offset = 0
                elif key == ord("q") and not search_text:
                    return
                elif 32 <= key <= 126:
                    # Printable character → add to search filter
                    search_text += chr(key)
                    filtered = [s for s in sessions if _match(s, search_text)]
                    cursor = 0
                    scroll_offset = 0

        curses.wrapper(_curses_browse)
        return result_holder[0]

    except Exception:
        pass

    # Fallback: numbered list (Windows without curses, etc.)
    print("\n  Browse sessions  (enter number to resume, q to cancel)\n")
    for i, s in enumerate(sessions):
        title = (s.get("title") or "").strip()
        preview = (s.get("preview") or "").strip()
        label = title or preview or s["id"]
        if len(label) > 50:
            label = label[:47] + "..."
        last_active = _relative_time(s.get("last_active"))
        src = s.get("source", "")[:6]
        print(f"  {i + 1:>3}. {label:<50}  {last_active:<10}  {src}")

    while True:
        try:
            val = input(f"\n  Select [1-{len(sessions)}]: ").strip()
            if not val or val.lower() in {"q", "quit", "exit"}:
                return None
            idx = int(val) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]["id"]
            print(f"  Invalid selection. Enter 1-{len(sessions)} or q to cancel.")
        except ValueError:
            print("  Invalid input. Enter a number or q to cancel.")
        except (KeyboardInterrupt, EOFError):
            print()
            return None


def _resolve_workspace_key() -> Optional[str]:
    """The current workspace identity for cwd-scoped resume.

    Git repo root when CWD is inside a repo (so all sessions across its
    subdirs/worktrees group together), else the CWD itself. Returns None when
    neither can be determined — callers fall back to the global MRU then.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return os.path.abspath(result.stdout.strip())
    except Exception:
        pass
    try:
        return os.getcwd()
    except Exception:
        return None


def _resolve_last_session(source: str = "cli") -> Optional[str]:
    """Look up the most recently-used session ID for a source.

    Scoped to the current workspace first (git repo root, else cwd) so
    ``pilotage -c`` from repo A continues repo A's last session rather than the
    global MRU. Falls back to the unscoped MRU when no session matches the
    current workspace, preserving the old behaviour for fresh directories.
    """
    db = None
    try:
        from pilotage_state import SessionDB

        db = SessionDB()
        ws_key = _resolve_workspace_key()
        if ws_key:
            sessions = db.search_sessions(source=source, limit=1, workspace_key=ws_key)
            if sessions:
                return sessions[0]["id"]
        # Fallback: global MRU for this source.
        sessions = db.search_sessions(source=source, limit=1)
        return sessions[0]["id"] if sessions else None
    except Exception:
        pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
    return None



def _resolve_session_by_name_or_id(name_or_id: str) -> Optional[str]:
    """Resolve a session name (title) or ID to a session ID.

    - If it looks like a session ID (contains underscore + hex), try direct lookup first.
    - Otherwise, treat it as a title and use resolve_session_by_title (auto-latest).
    - Falls back to the other method if the first doesn't match.
    - If the resolved session is a compression root, follow the chain forward
      to the latest continuation. Users who remember the old root ID (e.g.
      from an exit summary printed before the bug fix, or from notes) get
      resumed at the live tip instead of a stale parent with no messages.
    """
    db = None
    try:
        from pilotage_state import SessionDB

        db = SessionDB()

        # Try as exact session ID first
        session = db.get_session(name_or_id)
        resolved_id: Optional[str] = None
        if session:
            resolved_id = session["id"]
        else:
            # Try as title (with auto-latest for lineage)
            resolved_id = db.resolve_session_by_title(name_or_id)

        if resolved_id:
            # Project forward through compression chain so resumes land on
            # the live tip instead of a dead compressed parent.
            try:
                resolved_id = db.get_compression_tip(resolved_id) or resolved_id
            except Exception:
                pass

        return resolved_id
    except Exception:
        pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
    return None


def _read_tui_active_session_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sid = str(data.get("session_id") or "").strip()
        return sid or None
    except Exception:
        return None


def cmd_chat(args):
    """Run interactive chat CLI."""
    _apply_safe_mode(args)

    # --in DIR: run in DIR. Must happen before any session resolution so the
    # workspace-scoped "latest"/-c lookups key off DIR, and it pins the
    # session there — an explicit --in wins over a resumed session's
    # recorded cwd (so the restore step below is skipped).
    in_dir = getattr(args, "in_dir", None)
    if in_dir:
        # Git Bash / MSYS hands the CLI POSIX-style paths (`--in ~` expands to
        # `/c/Users/x` before Python ever sees it; MSYS2's path conversion is
        # disabled for native executables). Translate the MSYS/Cygwin/WSL
        # drive-root spellings to native Windows form first — no-op elsewhere.
        from tools.environments.local import _msys_to_windows_path

        _target_dir = os.path.abspath(
            os.path.expanduser(_msys_to_windows_path(in_dir))
        )
        if not os.path.isdir(_target_dir):
            print(f"Error: --in directory not found: {in_dir}")
            sys.exit(1)
        try:
            os.chdir(_target_dir)
        except OSError as e:
            print(f"Error: cannot enter --in directory {in_dir}: {e}")
            sys.exit(1)
        args.no_restore_cwd = True

    # --resume latest: keyword for "most recent session" — same resolution
    # as `-c` with no name (workspace-scoped MRU, then global fallback).
    # The keyword wins over a session literally titled "latest"; that
    # session stays reachable via its ID or `-c latest` (title match).
    _resume_raw = getattr(args, "resume", None)
    if isinstance(_resume_raw, str) and _resume_raw.strip().lower() == "latest":
        _last_id = _resolve_last_session(source="cli")
        if _last_id:
            args.resume = _last_id
        else:
            print("No previous CLI session found to resume.")
            print("Use 'pilotage sessions list' to see available sessions.")
            sys.exit(1)

    # Resolve --continue into --resume with the latest session or by name
    continue_val = getattr(args, "continue_last", None)
    if continue_val and not getattr(args, "resume", None):
        if isinstance(continue_val, str):
            # -c "session name" — resolve by title or ID
            resolved = _resolve_session_by_name_or_id(continue_val)
            if resolved:
                args.resume = resolved
            else:
                print(f"No session found matching '{continue_val}'.")
                print("Use 'pilotage sessions list' to see available sessions.")
                sys.exit(1)
        else:
            # -c with no argument — continue the most recent session
            last_id = _resolve_last_session(source="cli")
            if last_id:
                args.resume = last_id
            else:
                print("No previous CLI session found to continue.")
                sys.exit(1)

    # Resolve --resume by title if it's not a direct session ID
    resume_val = getattr(args, "resume", None)
    if resume_val:
        resolved = _resolve_session_by_name_or_id(resume_val)
        if resolved:
            args.resume = resolved
        # If resolution fails, keep the original value — _init_agent will
        # report "Session not found" with the original input

    # Session<->workspace binding: cd back into a resumed session's recorded cwd
    # so it resumes in the repo it belonged to. Opt out with --no-restore-cwd;
    # skipped under --worktree (that path owns its own dir). Best-effort — a
    # missing dir warns and stays put rather than failing the resume.
    if (
        getattr(args, "resume", None)
        and not getattr(args, "no_restore_cwd", False)
        and not getattr(args, "worktree", False)
    ):
        _resume_db = None
        try:
            from pilotage_state import SessionDB

            _resume_db = SessionDB()
            _saved_cwd = ((_resume_db.get_session(args.resume) or {}).get("cwd") or "").strip()
            if _saved_cwd and not os.path.isdir(_saved_cwd):
                print(f"⚠ session's recorded dir is gone ({_saved_cwd}); staying in {os.getcwd()}")
            elif _saved_cwd and os.path.realpath(_saved_cwd) != os.path.realpath(os.getcwd()):
                os.chdir(_saved_cwd)
                print(f"↪ restored workspace dir: {_saved_cwd}")
        except Exception:
            pass  # never let cwd-restore break a resume
        finally:
            if _resume_db is not None:
                try:
                    _resume_db.close()
                except Exception:
                    pass

    # First-run guard: check if any provider is configured before launching
    if not _has_any_provider_configured():
        print()
        print(
            "It looks like Pilotage isn't configured yet -- no API keys or providers found."
        )
        print()
        print("  Run:  pilotage setup")
        print()

        from pilotage_cli.setup import (
            is_interactive_stdin,
            print_noninteractive_setup_guidance,
        )

        if not is_interactive_stdin():
            print_noninteractive_setup_guidance(
                "No interactive TTY detected for the first-run setup prompt."
            )
            sys.exit(1)

        try:
            reply = input("Run setup now? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            reply = "n"
        if reply in {"", "y", "yes"}:
            cmd_setup(args)
            return
        print()
        print("You can run 'pilotage setup' at any time to configure.")
        sys.exit(1)

    # Start update check in background (runs while other init happens).
    # On Termux this imports rich/prompt_toolkit in the foreground and then
    # competes for CPU on single-core devices, so keep it opt-in there.
    if _termux_should_prefetch_update_check():
        try:
            from pilotage_cli.banner import prefetch_banner_data, prefetch_update_check

            prefetch_update_check()
            # Warm git banner state + skills index off-thread too — their
            # subprocess/file-I/O waits overlap the CPU-bound cli import.
            prefetch_banner_data()
        except Exception:
            pass

    # --yolo: bypass all dangerous command approvals.
    # Also set in main() before _prepare_agent_startup() — that is the
    # authoritative site because it runs before tool imports freeze
    # _YOLO_MODE_FROZEN.  This redundant set is a safety net for callers
    # that invoke cmd_chat directly (e.g. subcommand dispatch).
    if getattr(args, "yolo", False):
        os.environ["PILOTAGE_YOLO_MODE"] = "1"

    # --ignore-user-config: make load_cli_config() / load_config() skip the
    # user's ~/.pilotage/config.yaml and return built-in defaults. Set BEFORE
    # importing cli (which runs `CLI_CONFIG = load_cli_config()` at module
    # import time). Credentials in .env are still loaded — this flag only
    # ignores behavioral/config settings.
    if getattr(args, "ignore_user_config", False):
        os.environ["PILOTAGE_IGNORE_USER_CONFIG"] = "1"

    # --ignore-rules: skip auto-injection of AGENTS.md/SOUL.md/.cursorrules
    # (rules), memory entries, and any preloaded skills coming from user config.
    # Maps to AIAgent(skip_context_files=True, skip_memory=True).
    if getattr(args, "ignore_rules", False):
        os.environ["PILOTAGE_IGNORE_RULES"] = "1"

    # --source: tag session source for filtering (e.g. 'tool' for third-party integrations)
    if getattr(args, "source", None):
        os.environ["PILOTAGE_SESSION_SOURCE"] = args.source

    _confirm_startup_expensive_model_override(args)

    # The interactive CLI/TUI chat client was removed — Pilotage runs as a
    # gateway (Telegram / WhatsApp).  The argument handling above is kept so
    # `pilotage chat` still validates flags and reports the change clearly.
    print(
        "Interactive chat was removed from this build.\n"
        "Run the agent as a gateway instead:  pilotage gateway start"
    )
    sys.exit(1)


def cmd_gateway(args):
    """Gateway management commands."""
    from pilotage_cli.gateway import gateway_command

    gateway_command(args)


def cmd_whatsapp(args):
    """Set up WhatsApp: choose mode, configure, install bridge, pair via QR."""
    _require_tty("whatsapp")
    from pilotage_cli.config import get_env_value, save_env_value
    from pilotage_constants import find_node_executable, with_pilotage_node_path

    print()
    print("⚕ WhatsApp Setup")
    print("=" * 50)

    # ── Step 1: Choose mode ──────────────────────────────────────────────
    current_mode = get_env_value("WHATSAPP_MODE") or ""
    if not current_mode:
        print()
        print("How will you use WhatsApp with Pilotage?")
        print()
        print("  1. Separate bot number (recommended)")
        print("     People message the bot's number directly — cleanest experience.")
        print(
            "     Requires a second phone number with WhatsApp installed on a device."
        )
        print()
        print("  2. Personal number (self-chat)")
        print("     You message yourself to talk to the agent.")
        print("     Quick to set up, but the UX is less intuitive.")
        print()
        try:
            choice = input("  Choose [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            return

        if choice == "1":
            save_env_value("WHATSAPP_MODE", "bot")
            wa_mode = "bot"
            print("  ✓ Mode: separate bot number")
            print()
            print("  ┌─────────────────────────────────────────────────┐")
            print("  │  Getting a second number for the bot:           │")
            print("  │                                                 │")
            print("  │  Easiest: Install WhatsApp Business (free app)  │")
            print("  │  on your phone with a second number:            │")
            print("  │    • Dual-SIM: use your 2nd SIM slot            │")
            print("  │    • Google Voice: free US number (voice.google) │")
            print("  │    • Prepaid SIM: $3-10, verify once            │")
            print("  │                                                 │")
            print("  │  WhatsApp Business runs alongside your personal │")
            print("  │  WhatsApp — no second phone needed.             │")
            print("  └─────────────────────────────────────────────────┘")
        else:
            save_env_value("WHATSAPP_MODE", "self-chat")
            wa_mode = "self-chat"
            print("  ✓ Mode: personal number (self-chat)")
    else:
        wa_mode = current_mode
        mode_label = (
            "separate bot number" if wa_mode == "bot" else "personal number (self-chat)"
        )
        print(f"\n✓ Mode: {mode_label}")

    # ── Step 2: Mode is selected, will enable WhatsApp only after pairing ──
    # We intentionally don't write WHATSAPP_ENABLED=true here.  If the user
    # aborts the wizard later (Ctrl+C, failed npm install, missed QR scan),
    # we'd otherwise leave .env claiming WhatsApp is ready when the bridge
    # has no creds.json.  Every subsequent `pilotage gateway` then paid a 30s
    # bridge-bootstrap timeout and queued WhatsApp for indefinite retries.
    # Now: aborted setup leaves WHATSAPP_ENABLED unset → gateway skips it.
    # Re-runs that already have WHATSAPP_ENABLED=true (from a prior
    # successful pairing) stay enabled — we just don't write it pre-emptively.
    print()
    if (get_env_value("WHATSAPP_ENABLED") or "").lower() == "true":
        print("✓ WhatsApp is already enabled")

    # ── Step 3: Allowed users ────────────────────────────────────────────
    current_users = get_env_value("WHATSAPP_ALLOWED_USERS") or ""
    if current_users:
        print(f"✓ Allowed users: {current_users}")
        try:
            response = input("\n  Update allowed users? [y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in {"y", "yes"}:
            if wa_mode == "bot":
                phone = input(
                    "  Phone numbers that can message the bot (comma-separated): "
                ).strip()
            else:
                phone = input("  Your phone number (e.g. 15551234567): ").strip()
            if phone:
                save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
                print(f"  ✓ Updated to: {phone}")
    else:
        print()
        if wa_mode == "bot":
            print("  Who should be allowed to message the bot?")
            phone = input(
                "  Phone numbers (comma-separated, or * for anyone): "
            ).strip()
        else:
            phone = input("  Your phone number (e.g. 15551234567): ").strip()
        if phone:
            save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
            print(f"  ✓ Allowed users set: {phone}")
        else:
            print("  ⚠ No allowlist — the agent will respond to ALL incoming messages")

    # ── Step 4: Install bridge dependencies ──────────────────────────────
    from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
    bridge_dir = resolve_whatsapp_bridge_dir()
    bridge_script = bridge_dir / "bridge.js"

    if not bridge_script.exists():
        print(f"\n✗ Bridge script not found at {bridge_script}")
        return

    if not (bridge_dir / "node_modules").exists():
        print(
            "\n→ Installing WhatsApp bridge dependencies (this can take a few minutes)..."
        )
        npm = find_node_executable("npm")
        if not npm:
            print("  ✗ npm not found on PATH — install Node.js first")
            return
        try:
            result = subprocess.run(
                [npm, "install", "--no-fund", "--no-audit", "--progress=false"],
                cwd=str(bridge_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=with_pilotage_node_path(),
            )
        except KeyboardInterrupt:
            print("\n  ✗ Install cancelled")
            return
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            preview = "\n".join(err.splitlines()[-30:]) if err else "(no output)"
            print("  ✗ npm install failed:")
            print(preview)
            return
        print("  ✓ Dependencies installed")
    else:
        print("✓ Bridge dependencies already installed")

    # ── Step 5: Check for existing session ───────────────────────────────
    session_dir = get_pilotage_home() / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    if (session_dir / "creds.json").exists():
        print("✓ Existing WhatsApp session found")
        try:
            response = input(
                "\n  Re-pair? This will clear the existing session. [y/N] "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in {"y", "yes"}:
            shutil.rmtree(session_dir, ignore_errors=True)
            session_dir.mkdir(parents=True, exist_ok=True)
            print("  ✓ Session cleared")
        else:
            # Existing pairing — ensure WHATSAPP_ENABLED reflects that.
            # (Older installs may have lost the env var; covers re-runs
            # where the user picked "no, keep my session" but the var
            # was never set or got removed.)
            if (get_env_value("WHATSAPP_ENABLED") or "").lower() != "true":
                save_env_value("WHATSAPP_ENABLED", "true")
            print("\n✓ WhatsApp is configured and paired!")
            print("  Start the gateway with: pilotage gateway")
            return

    # ── Step 6: QR code pairing ──────────────────────────────────────────
    print()
    print("─" * 50)
    if wa_mode == "bot":
        print("📱 Open WhatsApp (or WhatsApp Business) on the")
        print("   phone with the BOT's number, then scan:")
    else:
        print("📱 Open WhatsApp on your phone, then scan:")
    print()
    print("   Settings → Linked Devices → Link a Device")
    print("─" * 50)
    print()

    try:
        subprocess.run(
            [
                find_node_executable("node") or "node",
                str(bridge_script),
                "--pair-only",
                "--session",
                str(session_dir),
            ],
            cwd=str(bridge_dir),
            env=with_pilotage_node_path(),
        )
    except KeyboardInterrupt:
        pass

    # ── Step 7: Post-pairing ─────────────────────────────────────────────
    print()
    if (session_dir / "creds.json").exists():
        # Only enable WhatsApp now that pairing actually succeeded.  If the
        # user Ctrl+C'd at any earlier step, WHATSAPP_ENABLED stays unset
        # and `pilotage gateway` skips it cleanly instead of paying a 30s
        # bridge timeout + queueing the platform for indefinite retries.
        save_env_value("WHATSAPP_ENABLED", "true")
        print("✓ WhatsApp paired successfully!")
        print()
        if wa_mode == "bot":
            print("  Next steps:")
            print("    1. Start the gateway:  pilotage gateway")
            print("    2. Send a message to the bot's WhatsApp number")
            print("    3. The agent will reply automatically")
            print()
            print("  Tip: Agent responses are prefixed with '⚕ Pilotage Agent'")
        else:
            print("  Next steps:")
            print("    1. Start the gateway:  pilotage gateway")
            print("    2. Open WhatsApp → Message Yourself")
            print("    3. Type a message — the agent will reply")
            print()
            print("  Tip: Agent responses are prefixed with '⚕ Pilotage Agent'")
            print("  so you can tell them apart from your own messages.")
        print()
        print("  Or install as a service: pilotage gateway install")
    else:
        print("⚠ Pairing may not have completed. Run 'pilotage whatsapp' to try again.")


def cmd_whatsapp_cloud(args):
    """Set up WhatsApp Business Cloud API (official Meta integration).

    Walks the user through the Meta-side credentials (Phone Number ID,
    Access Token, App Secret, optional App/WABA IDs) plus webhook
    configuration. Includes field-shape validators that catch the most
    common setup mistakes (e.g. pasting a phone number into the Phone
    Number ID field).

    Distinct from ``pilotage whatsapp`` (the Baileys bridge wizard) — the
    two adapters are complementary, not alternatives. See
    ``pilotage_cli/setup_whatsapp_cloud.py``.
    """
    _require_tty("whatsapp-cloud")
    from pilotage_cli.setup_whatsapp_cloud import run_whatsapp_cloud_setup

    return run_whatsapp_cloud_setup()


def cmd_setup(args):
    """Interactive setup wizard."""
    from pilotage_cli.setup import run_setup_wizard

    run_setup_wizard(args)


def cmd_model(args):
    """Select default model — starts with provider selection, then model picker."""
    _require_tty("model")
    if getattr(args, "refresh", False):
        try:
            from pilotage_cli.models import clear_provider_models_cache
            clear_provider_models_cache()
            print("  Cleared model picker cache.")
        except Exception:
            pass
    select_provider_and_model(args=args)


def _is_profile_api_key_provider(provider_id: str) -> bool:
    """Return True when provider_id maps to a profile with auth_type='api_key'.

    Used as a catch-all in select_provider_and_model() so that new providers
    declared in plugins/model-providers/<name>/ automatically dispatch to _model_flow_api_key_provider
    without requiring an explicit elif branch here.
    """
    try:
        from providers import get_provider_profile
        _p = get_provider_profile(provider_id)
        return _p is not None and _p.auth_type == "api_key"
    except Exception:
        return False


def select_provider_and_model(args=None):
    """Core provider selection + model picking logic.

    Shared by ``cmd_model`` (``pilotage model``) and the setup wizard
    (``setup_model_provider`` in setup.py).  Handles the full flow:
    provider picker, credential prompting, model selection, and config
    persistence.
    """
    from pilotage_cli.auth import (
        resolve_provider,
        AuthError,
        format_auth_error,
    )
    from pilotage_cli.config import (
        get_compatible_custom_providers,
        load_config,
        get_env_value,
    )
    from pilotage_cli.providers import (
        custom_provider_aliases,
        custom_provider_slug,
        resolve_provider_full,
    )

    config = load_config()
    current_model = config.get("model")
    if isinstance(current_model, dict):
        current_model = current_model.get("default", "")
    current_model = current_model or "(not set)"

    # Read effective provider the same way the CLI does at startup:
    # config.yaml model.provider > env var > auto-detect
    config_provider = None
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        config_provider = model_cfg.get("provider")

    effective_provider = (
        config_provider or os.getenv("PILOTAGE_INFERENCE_PROVIDER") or "auto"
    )
    compatible_custom_providers = get_compatible_custom_providers(config)
    def _named_custom_provider_map(cfg) -> dict[str, dict[str, str]]:
        from pilotage_cli.config import read_raw_config

        # Build lookups of raw (un-expanded) templates keyed by a
        # stable identity. We intentionally bypass
        # ``get_compatible_custom_providers(read_raw_config())`` here because
        # its ``_normalize_custom_provider_entry`` step calls ``urlparse()``
        # on ``base_url`` and drops any entry whose ``base_url`` is itself an
        # env-ref template (e.g. ``${NEURALWATT_API_BASE}``). Dropping those
        # entries is exactly how env-ref preservation fails for the user
        # config that motivated this fix.
        raw_api_key_refs: dict[tuple, str] = {}
        raw_base_url_refs: dict[tuple, str] = {}
        raw_cfg = read_raw_config()

        def _record_raw(
            name: str,
            provider_key: str,
            model: str,
            api_key: str,
            base_url: str,
        ) -> None:
            template = str(api_key or "").strip()
            base_template = str(base_url or "").strip()
            name = str(name or "").strip()
            provider_key = str(provider_key or "").strip()
            model = str(model or "").strip()
            # Index by every plausible identity the loaded (expanded) config
            # might present: (name), (name, model), (provider_key), and
            # (provider_key, model). Case-insensitive on name/provider_key so
            # the loaded entry matches regardless of display casing.
            identities = []
            if name:
                identities.extend(((name.lower(),), (name.lower(), model)))
            if provider_key:
                identities.extend(
                    ((provider_key.lower(),), (provider_key.lower(), model))
                )
            if "${" in template:
                for identity in identities:
                    raw_api_key_refs.setdefault(identity, template)
            if "${" in base_template:
                for identity in identities:
                    raw_base_url_refs.setdefault(identity, base_template)

        raw_list = raw_cfg.get("custom_providers")
        if isinstance(raw_list, list):
            for raw_entry in raw_list:
                if not isinstance(raw_entry, dict):
                    continue
                _record_raw(
                    raw_entry.get("name", ""),
                    "",
                    raw_entry.get("model", "") or raw_entry.get("default_model", ""),
                    raw_entry.get("api_key", ""),
                    raw_entry.get("base_url", "")
                    or raw_entry.get("url", "")
                    or raw_entry.get("api", ""),
                )
        raw_providers = raw_cfg.get("providers")
        if isinstance(raw_providers, dict):
            for raw_key, raw_entry in raw_providers.items():
                if not isinstance(raw_entry, dict):
                    continue
                _record_raw(
                    raw_entry.get("name", "") or raw_key,
                    raw_key,
                    raw_entry.get("model", "") or raw_entry.get("default_model", ""),
                    raw_entry.get("api_key", ""),
                    raw_entry.get("base_url", "")
                    or raw_entry.get("url", "")
                    or raw_entry.get("api", ""),
                )

        def _lookup_ref(
            refs: dict[tuple, str],
            name: str,
            provider_key: str,
            model: str,
        ) -> str:
            name_lc = str(name or "").strip().lower()
            pkey_lc = str(provider_key or "").strip().lower()
            model = str(model or "").strip()
            for identity in (
                (pkey_lc, model),
                (pkey_lc,),
                (name_lc, model),
                (name_lc,),
            ):
                if identity[0] and identity in refs:
                    return refs[identity]
            return ""

        custom_provider_map = {}
        for entry in get_compatible_custom_providers(cfg):
            if not isinstance(entry, dict):
                continue
            name = (entry.get("name") or "").strip()
            base_url = (entry.get("base_url") or "").strip()
            if not name or not base_url:
                continue
            provider_key = (entry.get("provider_key") or "").strip()
            key = custom_provider_slug(name, provider_key)
            custom_provider_map[key] = {
                "name": name,
                "base_url": base_url,
                "api_key": entry.get("api_key", ""),
                "key_env": entry.get("key_env", ""),
                "model": entry.get("model", ""),
                "models": entry.get("models", {}),
                "discover_models": entry.get("discover_models", True),
                "api_mode": entry.get("api_mode", ""),
                "provider_key": provider_key,
                "api_key_ref": _lookup_ref(
                    raw_api_key_refs, name, provider_key, entry.get("model", "")
                ),
                "base_url_ref": _lookup_ref(
                    raw_base_url_refs, name, provider_key, entry.get("model", "")
                ),
            }
        return custom_provider_map

    def _norm_base_url(url: str) -> str:
        return str(url or "").strip().rstrip("/").lower()

    # Add user-defined custom providers from config.yaml
    _custom_provider_map = _named_custom_provider_map(
        config
    )  # key → {name, base_url, api_key}

    def _canonical_named_custom_key(provider_id: str) -> str:
        requested = str(provider_id or "").strip().lower()
        for key, provider_info in _custom_provider_map.items():
            if requested in custom_provider_aliases(
                provider_info.get("name", ""),
                provider_info.get("provider_key", ""),
            ):
                return key
        return provider_id

    def _active_custom_key_from_base_url() -> str:
        if effective_provider != "custom" or not isinstance(model_cfg, dict):
            return ""
        current_base = _norm_base_url(model_cfg.get("base_url", ""))
        if not current_base:
            return ""
        for key, provider_info in _custom_provider_map.items():
            if _norm_base_url(provider_info.get("base_url", "")) == current_base:
                return key
        return ""

    active = _active_custom_key_from_base_url()
    if active is None:
        active = ""
    if not active and effective_provider != "auto":
        active_def = resolve_provider_full(
            effective_provider,
            config.get("providers"),
            compatible_custom_providers,
        )
        if active_def is not None:
            active = active_def.id
            if active_def.source == "user-config":
                active = _canonical_named_custom_key(active)
        else:
            warning = (
                f"Unknown provider '{effective_provider}'. Check 'pilotage model' for "
                "available providers, or run 'pilotage doctor' to diagnose config "
                "issues."
            )
            print(f"Warning: {warning} Falling back to auto provider detection.")
    if not active:
        try:
            active = resolve_provider("auto")
        except AuthError as exc:
            if effective_provider == "auto":
                warning = format_auth_error(exc)
                print(f"Warning: {warning} Falling back to auto provider detection.")
            active = None  # no provider yet; default to first in list

    # Detect custom endpoint
    if active == "openai-api" and get_env_value("OPENAI_BASE_URL"):
        active = "custom"

    from pilotage_cli.models import (
        CANONICAL_PROVIDERS,
        _PROVIDER_LABELS,
        _PROVIDER_ALIASES,
        group_providers,
        provider_group_for_slug,
    )

    provider_labels = dict(_PROVIDER_LABELS)  # derive from canonical list
    if active and active in _custom_provider_map:
        active_label = _custom_provider_map[active]["name"]
    else:
        active_label = provider_labels.get(active, active) if active else "none"

    print()
    print(f"  Current model:    {current_model}")
    print(f"  Active provider:  {active_label}")
    print()

    # Step 1: Provider selection.
    #
    # Canonical providers are folded into top-level groups (display only — see
    # PROVIDER_GROUPS in pilotage_cli/models.py). A multi-member group shows one
    # row; picking it opens a member sub-picker that
    # resolves back to a concrete slug, so the dispatch chain below is
    # unchanged. Custom providers and the trailing actions stay flat.
    canonical_descs = {p.slug: p.tui_desc for p in CANONICAL_PROVIDERS}
    # Honor ``model_catalog.excluded_providers`` so the CLI ``pilotage model``
    # picker hides the same providers the gateway/TUI pickers do. A canonical
    # provider is hidden if its slug OR any of its aliases appears in the
    # exclusion list (case-insensitive), matching list_authenticated_providers'
    # matching against pilotage_id / alias / canonical slug.
    _cli_excluded = {
        str(p).strip().lower()
        for p in (config.get("model_catalog", {}) or {}).get("excluded_providers") or []
        if p
    }
    if _cli_excluded:
        _alias_to_canon = _PROVIDER_ALIASES
        _names_for: dict[str, set[str]] = {}
        for _p in CANONICAL_PROVIDERS:
            _names_for[_p.slug] = {_p.slug.lower()}
        for _alias, _canon in _alias_to_canon.items():
            _names_for.setdefault(_canon, {_canon.lower()}).add(_alias.lower())
        _visible_slugs = [
            p.slug for p in CANONICAL_PROVIDERS
            if not _names_for.get(p.slug, {p.slug.lower()}) & _cli_excluded
        ]
    else:
        _visible_slugs = [p.slug for p in CANONICAL_PROVIDERS]
    grouped_rows = group_providers(_visible_slugs)

    # The group/slug that should be pre-selected: the active provider's group
    # if it's grouped, otherwise the active slug itself.
    active_group = provider_group_for_slug(active) if active else ""

    # ordered entries: (key, label, members)
    #   members == [] → leaf row, key is a provider slug / action
    #   members != [] → group row, key is "group:<gid>"
    ordered: list[tuple[str, str, list[str]]] = []
    default_idx = 0
    for row in grouped_rows:
        if row["kind"] == "group":
            gid = row["group_id"]
            group_desc = row.get("description", "")
            label = f"{row['label']} ▸ ({group_desc})" if group_desc else f"{row['label']} ▸"
            key = f"group:{gid}"
            is_active = bool(active_group) and gid == active_group
            members = row["members"]
        else:
            slug = row["slug"]
            label = canonical_descs.get(slug, provider_labels.get(slug, slug))
            key = slug
            is_active = bool(active) and slug == active
            members = []
        if is_active:
            ordered.append((key, f"{label}  ← currently active", members))
            default_idx = len(ordered) - 1
        else:
            ordered.append((key, label, members))

    for key, provider_info in _custom_provider_map.items():
        name = provider_info["name"]
        base_url = provider_info["base_url"]
        short_url = base_url.replace("https://", "").replace("http://", "").rstrip("/")
        saved_model = provider_info.get("model", "")
        model_hint = f" — {saved_model}" if saved_model else ""
        label = f"{name} ({short_url}){model_hint}"
        if active and key == active:
            ordered.append((key, f"{label}  ← currently active", []))
            default_idx = len(ordered) - 1
        else:
            ordered.append((key, label, []))

    ordered.append(("custom", "Custom endpoint (enter URL manually)", []))
    _has_saved_custom_list = isinstance(config.get("custom_providers"), list) and bool(
        config.get("custom_providers")
    )
    if _has_saved_custom_list:
        ordered.append(("remove-custom", "Remove a saved custom provider", []))
    ordered.append(("aux-config", "Configure auxiliary models...", []))
    ordered.append(("cancel", "Leave unchanged", []))

    provider_idx = _prompt_provider_choice(
        [label for _, label, _ in ordered],
        default=default_idx,
    )
    if provider_idx is None or ordered[provider_idx][0] == "cancel":
        print("No change.")
        return

    selected_key = ordered[provider_idx][0]
    selected_members = ordered[provider_idx][2]

    # Group row → drill into a member sub-picker. Default to the active member
    # if the active provider lives in this group. The descriptive text lives on
    # the group row itself, so member rows show only their short label here.
    if selected_members:
        member_default = 0
        if active in selected_members:
            member_default = selected_members.index(active)
        member_labels = [
            provider_labels.get(m, m) for m in selected_members
        ]
        group_label = ordered[provider_idx][1].split(" ▸", 1)[0]
        member_idx = _prompt_provider_choice(
            member_labels,
            default=member_default,
            title=f"Select {group_label} provider:",
        )
        if member_idx is None:
            print("No change.")
            return
        selected_provider = selected_members[member_idx]
    else:
        selected_provider = selected_key

    if selected_provider == "aux-config":
        _aux_config_menu()
        return

    # Step 2: Provider-specific setup + model selection
    if selected_provider == "openai-codex":
        _model_flow_openai_codex(config, current_model)
    elif selected_provider == "custom":
        _model_flow_custom(config)
    elif (
        selected_provider.startswith("custom:")
        or selected_provider in _custom_provider_map
    ):
        provider_info = _named_custom_provider_map(load_config()).get(selected_provider)
        if provider_info is None:
            print(
                "Warning: the selected saved custom provider is no longer available. "
                "It may have been removed from config.yaml. No change."
            )
            return
        _model_flow_named_custom(config, provider_info)
    elif selected_provider == "remove-custom":
        _remove_custom_provider(config)
    elif selected_provider == "openai-api" or _is_profile_api_key_provider(selected_provider):
        _model_flow_api_key_provider(config, selected_provider, current_model)

    # ── Post-switch cleanup: clear stale OPENAI_BASE_URL ──────────────
    # When the user switches to a named provider (anything except "custom"),
    # a leftover OPENAI_BASE_URL in ~/.pilotage/.env can poison auxiliary
    # clients that use provider:auto. Clear it proactively.
    if selected_provider not in {
        "custom",
        "cancel",
        "remove-custom",
    } and not selected_provider.startswith("custom:"):
        _clear_stale_openai_base_url()


def _clear_stale_openai_base_url():
    """Remove OPENAI_BASE_URL from ~/.pilotage/.env if the active provider is not 'custom'.

    After a provider switch, a leftover OPENAI_BASE_URL causes auxiliary
    clients (compression, vision, delegation) with provider:auto to route
    requests to the old custom endpoint instead of the newly selected
    provider. See.
    """
    from pilotage_cli.config import get_env_value, save_env_value, load_config

    cfg = load_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        provider = (model_cfg.get("provider") or "").strip().lower()
    else:
        provider = ""

    if provider == "custom" or not provider:
        return  # custom provider legitimately uses OPENAI_BASE_URL

    stale_url = get_env_value("OPENAI_BASE_URL")
    if stale_url:
        save_env_value("OPENAI_BASE_URL", "")
        print(
            f"Cleared stale OPENAI_BASE_URL from .env (was: {stale_url[:40]}...)"
            if len(stale_url) > 40
            else f"Cleared stale OPENAI_BASE_URL from .env (was: {stale_url})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary model configuration
#
# Pilotage uses lightweight "auxiliary" models for side tasks (vision analysis,
# context compression, web extraction, session search, etc.). Each task has
# its own provider+model pair in config.yaml under `auxiliary.<task>`.
#
# The UI lives behind "Configure auxiliary models..." at the bottom of the
# `pilotage model` provider picker. It does NOT re-run credential setup — it
# only routes already-authenticated providers to specific aux tasks. Users
# configure new providers through the normal `pilotage model` flow first.
# ─────────────────────────────────────────────────────────────────────────────

# (task_key, display_name, short_description)
_AUX_TASKS: list[tuple[str, str, str]] = [
    ("vision", "Vision", "image/screenshot analysis"),
    ("compression", "Compression", "context summarization"),
    ("web_extract", "Web extract", "web page summarization"),
    ("approval", "Approval", "smart command approval"),
    ("MCP", "MCP", "MCP tool reasoning"),
    ("title_generation", "Title generation", "session titles"),
    ("memory_query_rewrite", "Memory query rewrite", "memory retrieval queries"),
    ("tts_audio_tags", "TTS audio tags", "Gemini TTS tag insertion"),
    ("profile_describer", "Profile describer", "auto profile descriptions"),
]


def _all_aux_tasks() -> list[tuple[str, str, str]]:
    """Return built-in + plugin-registered auxiliary tasks for picker/menu use.

    Built-in tasks come first (preserving order), followed by plugin tasks
    sorted by key. Used by ``_aux_config_menu``, ``_reset_aux_to_auto``, and
    display-name lookups so plugin-registered tasks (registered via
    :meth:`pilotage_cli.plugins.PluginContext.register_auxiliary_task`) appear
    in the same surfaces as built-in ones without core knowing about them.
    """
    tasks = list(_AUX_TASKS)
    try:
        from pilotage_cli.plugins import get_plugin_auxiliary_tasks
        for entry in get_plugin_auxiliary_tasks():
            tasks.append((entry["key"], entry["display_name"], entry["description"]))
    except Exception:
        # Plugin discovery failure must not break the aux config UI.
        # Built-in tasks remain available.
        pass
    return tasks


def _format_aux_current(task_cfg: dict) -> str:
    """Render the current aux config for display in the task menu."""
    if not isinstance(task_cfg, dict):
        return "auto"
    base_url = str(task_cfg.get("base_url") or "").strip()
    provider = str(task_cfg.get("provider") or "auto").strip() or "auto"
    model = str(task_cfg.get("model") or "").strip()
    if base_url:
        short = base_url.replace("https://", "").replace("http://", "").rstrip("/")
        return f"custom ({short})" + (f" · {model}" if model else "")
    if provider == "auto":
        return "auto" + (f" · {model}" if model else "")
    if model:
        return f"{provider} · {model}"
    return provider


def _save_aux_choice(
    task: str,
    *,
    provider: str,
    model: str = "",
    base_url: str = "",
    api_key: str = "",
) -> None:
    """Persist an auxiliary task's provider/model to config.yaml.

    Only writes the four routing fields — timeout, download_timeout, and any
    other task-specific settings are preserved untouched. The main model
    config (``model.default``/``model.provider``) is never modified.
    """
    from pilotage_cli.config import load_config, save_config

    cfg = load_config()
    aux = cfg.setdefault("auxiliary", {})
    if not isinstance(aux, dict):
        aux = {}
        cfg["auxiliary"] = aux
    entry = aux.setdefault(task, {})
    if not isinstance(entry, dict):
        entry = {}
        aux[task] = entry
    entry["provider"] = provider
    entry["model"] = model or ""
    entry["base_url"] = base_url or ""
    entry["api_key"] = api_key or ""
    save_config(cfg)


def _reset_aux_to_auto() -> int:
    """Reset every known aux task back to auto/empty. Returns number reset.

    Includes plugin-registered tasks (via ``_all_aux_tasks``) so a plugin
    that contributed an auxiliary task gets reset alongside built-ins.
    """
    from pilotage_cli.config import load_config, save_config

    cfg = load_config()
    aux = cfg.setdefault("auxiliary", {})
    if not isinstance(aux, dict):
        aux = {}
        cfg["auxiliary"] = aux
    count = 0
    for task, _name, _desc in _all_aux_tasks():
        entry = aux.setdefault(task, {})
        if not isinstance(entry, dict):
            entry = {}
            aux[task] = entry
        changed = False
        if entry.get("provider") not in {None, "", "auto"}:
            entry["provider"] = "auto"
            changed = True
        for field in ("model", "base_url", "api_key"):
            if entry.get(field):
                entry[field] = ""
                changed = True
        # Preserve timeout/download_timeout — those are user-tuned, not routing
        if changed:
            count += 1
    save_config(cfg)
    return count


def _aux_config_menu() -> None:
    """Top-level auxiliary-model picker — choose a task to configure.

    Loops until the user picks "Back" so multiple tasks can be configured
    without returning to the main provider menu.
    """
    from pilotage_cli.config import load_config

    while True:
        cfg = load_config()
        aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}

        print()
        print("  Auxiliary models — side-task routing")
        print()
        print("  Side tasks (vision, compression, web extraction, etc.) default")
        print('  to your main chat model.  "auto" means "use my main model" —')
        print("  Pilotage only falls back to a lightweight backend (OpenRouter,")
        print("  Nous Portal) if the main model is unavailable.  Override a")
        print("  task below if you want it pinned to a specific provider/model.")
        print()

        # Build the task menu with current settings inline
        all_tasks = _all_aux_tasks()
        name_col = max(len(name) for _, name, _ in all_tasks) + 2
        desc_col = max(len(desc) for _, _, desc in all_tasks) + 4
        entries: list[tuple[str, str]] = []
        for task_key, name, desc in all_tasks:
            task_cfg = (
                aux.get(task_key, {}) if isinstance(aux.get(task_key), dict) else {}
            )
            current = _format_aux_current(task_cfg)
            label = (
                f"{name.ljust(name_col)}{('(' + desc + ')').ljust(desc_col)}{current}"
            )
            entries.append((task_key, label))
        entries.append(("__reset__", "Reset all to auto"))
        entries.append(("__back__", "Back"))

        idx = _prompt_provider_choice(
            [label for _, label in entries],
            default=0,
        )
        if idx is None:
            return
        key = entries[idx][0]
        if key == "__back__":
            return
        if key == "__reset__":
            n = _reset_aux_to_auto()
            if n:
                print(f"Reset {n} auxiliary task(s) to auto.")
            else:
                print("All auxiliary tasks were already set to auto.")
            print()
            continue
        # Otherwise configure the specific task
        _aux_select_for_task(key)


def _aux_select_for_task(task: str) -> None:
    """Pick a provider + model for a single auxiliary task and persist it.

    Provider rows come from ``build_aux_picker_rows()`` — the shared aux-picker
    substrate — so this surface shows exactly what every other aux picker
    shows: authenticated built-ins, the user's own ``providers:`` /
    ``custom_providers:`` endpoints, and providers whose credential pool is
    temporarily exhausted. Only already-configured providers appear; users set
    up new ones through the normal ``pilotage model`` flow, then route aux tasks
    to them here.
    """
    from pilotage_cli.config import load_config
    from pilotage_cli.inventory import build_aux_picker_rows, format_aux_picker_entries

    cfg = load_config()
    aux = cfg.get("auxiliary", {}) if isinstance(cfg.get("auxiliary"), dict) else {}
    task_cfg = aux.get(task, {}) if isinstance(aux.get(task), dict) else {}
    current_provider = str(task_cfg.get("provider") or "auto").strip() or "auto"
    current_model = str(task_cfg.get("model") or "").strip()
    current_base_url = str(task_cfg.get("base_url") or "").strip()

    display_name = next((name for key, name, _ in _all_aux_tasks() if key == task), task)

    # Gather authenticated providers (has credentials + curated model list)
    try:
        providers = build_aux_picker_rows(
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
        )
    except Exception as exc:
        print(f"Could not detect authenticated providers: {exc}")
        providers = []

    entries: list[tuple[str, str, list[str]]] = []  # (slug, label, models)
    # "auto" always first
    auto_marker = (
        "  ← current" if current_provider == "auto" and not current_base_url else ""
    )
    entries.append(("__auto__", f"auto (recommended){auto_marker}", []))

    entries.extend(
        format_aux_picker_entries(
            providers,
            current_provider=current_provider,
            current_base_url=current_base_url,
        )
    )

    # Custom endpoint (raw base_url)
    custom_marker = "  ← current" if current_base_url else ""
    entries.append(("__custom__", f"Custom endpoint (direct URL){custom_marker}", []))
    entries.append(("__back__", "Back", []))

    print()
    print(f"  Configure {display_name} — current: {_format_aux_current(task_cfg)}")
    print()

    idx = _prompt_provider_choice([label for _, label, _ in entries], default=0)
    if idx is None:
        return
    slug, _label, models = entries[idx]

    if slug == "__back__":
        return

    if slug == "__auto__":
        _save_aux_choice(task, provider="auto", model="", base_url="", api_key="")
        print(f"{display_name}: reset to auto.")
        return

    if slug == "__custom__":
        _aux_flow_custom_endpoint(task, task_cfg)
        return

    # Regular provider — pick a model from its curated list
    _aux_flow_provider_model(task, slug, models, current_model)


def _aux_flow_provider_model(
    task: str,
    provider_slug: str,
    curated_models: list,
    current_model: str = "",
) -> None:
    """Prompt for a model under an already-authenticated provider, save to aux."""
    from pilotage_cli.auth import _prompt_model_selection
    from pilotage_cli.models import get_pricing_for_provider

    display_name = next((name for key, name, _ in _all_aux_tasks() if key == task), task)

    # Fetch live pricing for this provider (non-blocking)
    pricing: dict = {}
    try:
        pricing = get_pricing_for_provider(provider_slug) or {}
    except Exception:
        pricing = {}

    model_list = list(curated_models)

    # Let the user pick a model. _prompt_model_selection supports "Enter custom
    # model name" and cancel.  When there's no curated list (rare), fall back
    # to a raw input prompt.
    if not model_list:
        print(f"No curated model list for {provider_slug}.")
        print("Enter a model slug manually (blank = use provider default):")
        try:
            val = input("Model: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return
        selected = val or ""
    else:
        selected = _prompt_model_selection(
            model_list,
            current_model=current_model,
            pricing=pricing,
            confirm_provider=provider_slug,
        )
        if selected is None:
            print("No change.")
            return

    _save_aux_choice(
        task, provider=provider_slug, model=selected or "", base_url="", api_key=""
    )
    if selected:
        print(f"{display_name}: {provider_slug} · {selected}")
    else:
        print(f"{display_name}: {provider_slug} (provider default model)")


def _aux_flow_custom_endpoint(task: str, task_cfg: dict) -> None:
    """Prompt for a direct OpenAI-compatible base_url + optional api_key/model."""
    from pilotage_cli.secret_prompt import masked_secret_prompt

    display_name = next((name for key, name, _ in _all_aux_tasks() if key == task), task)
    current_base_url = str(task_cfg.get("base_url") or "").strip()
    current_model = str(task_cfg.get("model") or "").strip()

    print()
    print(f"  Custom endpoint for {display_name}")
    print("  Provide an OpenAI-compatible base URL (e.g. http://localhost:11434/v1)")
    print()
    try:
        url_prompt = (
            f"Base URL [{current_base_url}]: " if current_base_url else "Base URL: "
        )
        url = input(url_prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return
    url = url or current_base_url
    if not url:
        print("No URL provided. No change.")
        return
    try:
        model_prompt = (
            f"Model slug (optional) [{current_model}]: "
            if current_model
            else "Model slug (optional): "
        )
        model = input(model_prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return
    model = model or current_model
    try:
        api_key = masked_secret_prompt(
            "API key (optional, blank = use OPENAI_API_KEY): "
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return

    _save_aux_choice(
        task,
        provider="custom",
        model=model,
        base_url=url,
        api_key=api_key,
    )
    short_url = url.replace("https://", "").replace("http://", "").rstrip("/")
    print(f"{display_name}: custom ({short_url})" + (f" · {model}" if model else ""))


def _prompt_provider_choice(choices, *, default=0, title="Select provider:"):
    """Show provider selection menu with curses arrow-key navigation.

    Falls back to a numbered list when curses is unavailable (e.g. piped
    stdin, non-TTY environments).  Returns the selected index, or None
    if the user cancels.
    """
    try:
        from pilotage_cli.setup import _curses_prompt_choice

        idx = _curses_prompt_choice(title, choices, default)
        if idx >= 0:
            print()
            return idx
    except Exception:
        pass

    # Fallback: numbered list
    print(title)
    for i, c in enumerate(choices, 1):
        marker = "→" if i - 1 == default else " "
        print(f"  {marker} {i}. {c}")
    print()
    while True:
        try:
            val = input(f"Choice [1-{len(choices)}] ({default + 1}): ").strip()
            if not val:
                return default
            idx = int(val) - 1
            if 0 <= idx < len(choices):
                return idx
            print(f"Please enter 1-{len(choices)}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            print()
            return None










def _prompt_custom_api_mode_selection(base_url: str, current_api_mode: str = "") -> Optional[str]:
    """Prompt for a custom provider API mode.

    Returns an explicit mode string, or None to keep auto-detect behavior.
    """
    from pilotage_cli.runtime_provider import _detect_api_mode_for_url

    detected_mode = _detect_api_mode_for_url(base_url)
    normalized_current = str(current_api_mode or "").strip().lower()
    default_mode = normalized_current or detected_mode or ""

    mode_options = [
        (
            "",
            "Auto-detect",
            "Use Pilotage URL heuristics; best for standard OpenAI-compatible endpoints.",
        ),
        (
            "chat_completions",
            "Chat Completions",
            "Use /chat/completions for standard OpenAI-compatible servers.",
        ),
        (
            "codex_responses",
            "Responses / Codex",
            "Use /responses for Codex-compatible tool-calling backends.",
        ),
    ]

    print()
    print("Select API compatibility mode:")
    for idx, (value, label, description) in enumerate(mode_options, 1):
        markers = []
        if value == detected_mode:
            markers.append("detected")
        if value == default_mode:
            markers.append("current")
        suffix = f" [{' / '.join(markers)}]" if markers else ""
        print(f"  {idx}. {label}{suffix}")
        print(f"     {description}")

    try:
        raw = input(
            "Choice [1-3, Enter to keep current/detected]: "
        ).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        raise

    if not raw:
        return default_mode or None

    if raw in {"1", "auto", "detect", "auto-detect"}:
        return None
    if raw in {"2", "chat", "chat_completions", "completions"}:
        return "chat_completions"
    if raw in {"3", "responses", "codex", "codex_responses"}:
        return "codex_responses"

    print(f"Invalid API mode choice: {raw}. Falling back to auto-detect.")
    return None


def _auto_provider_name(base_url: str) -> str:
    """Generate a display name from a custom endpoint URL.

    Returns a human-friendly label like "Local (localhost:11434)" or
    "RunPod (xyz.runpod.io)".  Used as the default when prompting the
    user for a display name during custom endpoint setup.
    """
    import re

    clean = base_url.replace("https://", "").replace("http://", "").rstrip("/")
    clean = re.sub(r"/v1/?$", "", clean)
    name = clean.split("/")[0]
    if "localhost" in name or "127.0.0.1" in name:
        name = f"Local ({name})"
    elif "runpod" in name.lower():
        name = f"RunPod ({name})"
    else:
        name = name.capitalize()
    return name


def _custom_provider_api_key_config_value(provider_info, resolved_api_key=""):
    """Return the value that should be persisted for a custom provider key."""
    api_key_ref = str(provider_info.get("api_key_ref", "") or "").strip()
    if api_key_ref:
        return api_key_ref

    key_env = str(provider_info.get("key_env", "") or "").strip()
    if key_env and not str(provider_info.get("api_key", "") or "").strip():
        return f"${{{key_env}}}"

    return str(resolved_api_key or "").strip()


def _custom_provider_base_url_config_value(provider_info, resolved_base_url=""):
    """Return the value that should be persisted for a custom provider URL."""
    base_url_ref = str(provider_info.get("base_url_ref", "") or "").strip()
    if base_url_ref:
        return base_url_ref
    return str(resolved_base_url or "").strip()


def _save_custom_provider(
    base_url, api_key="", model="", context_length=None, name=None, api_mode=None,
    key_env=""
):
    """Save a custom endpoint to custom_providers in config.yaml.

    Deduplicates by base_url — if the URL already exists, updates the
    model name, context_length, and api_mode but doesn't add a duplicate entry.
    Uses *name* when provided, otherwise auto-generates from the URL.

    When *key_env* is set the caller has already written the key to ``.env``,
    so the entry references it instead of inlining the secret.
    """
    from pilotage_cli.config import load_config, save_config

    cfg = load_config()
    providers = cfg.get("custom_providers") or []
    if not isinstance(providers, list):
        providers = []

    # Check if this URL is already saved — update model/context_length if so
    for entry in providers:
        if isinstance(entry, dict) and entry.get("base_url", "").rstrip(
            "/"
        ) == base_url.rstrip("/"):
            changed = False
            if model and entry.get("model") != model:
                entry["model"] = model
                changed = True
            if model and context_length:
                models_cfg = entry.get("models", {})
                if not isinstance(models_cfg, dict):
                    models_cfg = {}
                models_cfg[model] = {"context_length": context_length}
                entry["models"] = models_cfg
                changed = True
            if api_mode:
                if entry.get("api_mode") != api_mode:
                    entry["api_mode"] = api_mode
                    changed = True
            elif "api_mode" in entry:
                entry.pop("api_mode", None)
                changed = True
            if key_env and (entry.get("key_env") != key_env or entry.get("api_key")):
                entry["key_env"] = key_env
                entry.pop("api_key", None)
                changed = True
            if changed:
                cfg["custom_providers"] = providers
                save_config(cfg)
            return  # already saved, updated if needed

    # Use provided name or auto-generate from URL
    if not name:
        name = _auto_provider_name(base_url)

    entry = {"name": name, "base_url": base_url}
    if key_env:
        entry["key_env"] = key_env
    elif api_key:
        entry["api_key"] = api_key
    if model:
        entry["model"] = model
    if api_mode:
        entry["api_mode"] = api_mode
    if model and context_length:
        entry["models"] = {model: {"context_length": context_length}}

    providers.append(entry)
    cfg["custom_providers"] = providers
    save_config(cfg)
    print(f'  💾 Saved to custom providers as "{name}" (edit in config.yaml)')




def _remove_custom_provider(config):
    """Let the user remove a saved custom provider from config.yaml."""
    from pilotage_cli.config import load_config, save_config

    cfg = load_config()
    providers = cfg.get("custom_providers") or []
    if not isinstance(providers, list) or not providers:
        print("No custom providers configured.")
        return

    print("Remove a custom provider:\n")

    choices = []
    for entry in providers:
        if isinstance(entry, dict):
            name = entry.get("name", "unnamed")
            url = entry.get("base_url", "")
            short_url = url.replace("https://", "").replace("http://", "").rstrip("/")
            choices.append(f"{name} ({short_url})")
        else:
            choices.append(str(entry))
    choices.append("Cancel")

    try:
        from pilotage_cli.curses_ui import curses_radiolist

        idx = curses_radiolist(
            "Select provider to remove:",
            list(choices),
            selected=0,
            cancel_returns=-1,
        )
        print()
        if idx < 0:
            idx = None
    except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
        for i, c in enumerate(choices, 1):
            print(f"  {i}. {c}")
        print()
        try:
            val = input(f"Choice [1-{len(choices)}]: ").strip()
            idx = int(val) - 1 if val else None
        except (ValueError, KeyboardInterrupt, EOFError):
            idx = None

    if idx is None or idx >= len(providers):
        print("No change.")
        return

    removed = providers.pop(idx)
    cfg["custom_providers"] = providers
    save_config(cfg)
    removed_name = (
        removed.get("name", "unnamed") if isinstance(removed, dict) else str(removed)
    )
    print(f'✅ Removed "{removed_name}" from custom providers.')




# Lazy-export the model catalog at module level. Tests and a handful of
# downstream call sites read `pilotage_cli.main._PROVIDER_MODELS` directly,
# so the symbol needs to be reachable as a module attribute. But importing
# the catalog eagerly costs ~55ms on every `pilotage` invocation — including
# fast paths like `pilotage --version` and slash-command dispatch that never
# touch the catalog. PEP 562 module-level __getattr__ defers the import
# until first attribute access, so the cost is only paid by callers that
# actually look up the catalog. Termux already defers via the same
# mechanism (its model-selection handlers do their own function-local
# imports), so the explicit termux branch from before is no longer needed.
_LAZY_MODEL_EXPORTS = ("_PROVIDER_MODELS",)


# The main.py decomposition moved the sessions/update/dashboard command
# implementations into their own modules, but main.py still re-exports their
# surface so argparse wiring and test monkeypatches on pilotage_cli.main.<name>
# keep resolving unchanged. Importing those modules eagerly costs ~50ms on
# every `pilotage` invocation, including fast paths like `pilotage --version`
# that never run a subcommand. Resolve the re-exports through the module
# __getattr__ below instead, so each module is only imported when one of its
# names is actually touched. Monkeypatching keeps working: patch.object sets
# a real module attribute, which shadows __getattr__.
_LAZY_COMMAND_EXPORTS = {
    "pilotage_cli.sessions_cmd": (
        "cmd_sessions",
    ),
}

_LAZY_COMMAND_ATTR_TO_MODULE = {
    attr: module for module, attrs in _LAZY_COMMAND_EXPORTS.items() for attr in attrs
}

# Back-compat alias: some tests and external callers import the old warn-only
# name. The kill behaviour replaced it; resolve to the new name lazily.
_LAZY_COMMAND_ALIASES: dict[str, tuple[str, str]] = {}


def _self():
    """This module, for attribute access at call time.

    Bare-name global lookups inside this module do not go through the PEP 562
    __getattr__ below, so internal callers of the lazily re-exported names use
    _self().<name> instead. That resolves the lazy re-export on first use and
    keeps monkeypatches on pilotage_cli.main.<name> working, exactly like a
    globals lookup did. ``sys`` is imported locally because some tests patch
    this module's ``sys`` attribute.
    """
    import sys as _sys

    return _sys.modules[__name__]


def __getattr__(name):
    """Defer the model-catalog and command-module imports until first read."""
    if name in _LAZY_MODEL_EXPORTS:
        from pilotage_cli.models import _PROVIDER_MODELS
        # Cache on the module so subsequent accesses skip the import machinery.
        globals()[name] = _PROVIDER_MODELS
        return _PROVIDER_MODELS
    module = _LAZY_COMMAND_ATTR_TO_MODULE.get(name)
    if module is not None:
        import importlib

        value = getattr(importlib.import_module(module), name)
        globals()[name] = value
        return value
    alias = _LAZY_COMMAND_ALIASES.get(name)
    if alias is not None:
        import importlib

        module_name, attr = alias
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _current_reasoning_effort(config) -> str:
    agent_cfg = config.get("agent")
    if isinstance(agent_cfg, dict):
        return str(agent_cfg.get("reasoning_effort") or "").strip().lower()
    return ""


def _set_reasoning_effort(config, effort: str) -> None:
    agent_cfg = config.get("agent")
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}
        config["agent"] = agent_cfg
    agent_cfg["reasoning_effort"] = effort


def _prompt_reasoning_effort_selection(efforts, current_effort=""):
    """Prompt for a reasoning effort. Returns effort, 'none', or None to keep current."""
    deduped = list(
        dict.fromkeys(
            str(effort).strip().lower() for effort in efforts if str(effort).strip()
        )
    )
    canonical_order = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")
    ordered = [effort for effort in canonical_order if effort in deduped]
    ordered.extend(effort for effort in deduped if effort not in canonical_order)
    if not ordered:
        return None

    def _label(effort):
        if effort == current_effort:
            return f"{effort}  ← currently in use"
        return effort

    disable_label = "Disable reasoning"
    skip_label = "Skip (keep current)"

    if current_effort == "none":
        default_idx = len(ordered)
    elif current_effort in ordered:
        default_idx = ordered.index(current_effort)
    elif "medium" in ordered:
        default_idx = ordered.index("medium")
    else:
        default_idx = 0

    try:
        from pilotage_cli.curses_ui import curses_radiolist

        choices = [_label(effort) for effort in ordered]
        choices.append(disable_label)
        choices.append(skip_label)
        idx = curses_radiolist(
            "Select reasoning effort:",
            choices,
            selected=default_idx,
            cancel_returns=-1,
        )
        if idx < 0:
            return None
        print()
        if idx < len(ordered):
            return ordered[idx]
        if idx == len(ordered):
            return "none"
        return None
    except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
        pass

    print("Select reasoning effort:")
    for i, effort in enumerate(ordered, 1):
        print(f"  {i}. {_label(effort)}")
    n = len(ordered)
    print(f"  {n + 1}. {disable_label}")
    print(f"  {n + 2}. {skip_label}")
    print()

    while True:
        try:
            choice = input(f"Choice [1-{n + 2}] (default: keep current): ").strip()
            if not choice:
                return None
            idx = int(choice)
            if 1 <= idx <= n:
                return ordered[idx - 1]
            if idx == n + 1:
                return "none"
            if idx == n + 2:
                return None
            print(f"Please enter 1-{n + 2}")
        except ValueError:
            print("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            return None






def _prompt_api_key(
    pconfig,
    existing_key: str,
    provider_id: str = "",
    existing_source: str = "",
) -> tuple:
    """Shared API-key entry point for ``pilotage setup`` / ``pilotage model``.

    Handles both first-time entry and the already-configured case.  When a key
    is already present, offers [K]eep / [R]eplace / [C]lear so the user can
    recover from a malformed paste without editing ``~/.pilotage/.env`` by hand.

    Returns ``(resolved_key, abort)``.  ``abort=True`` means the caller should
    ``return`` immediately — the user cancelled entry, declined to replace, or
    cleared the key and is now unconfigured.
    """
    from pilotage_cli.config import save_env_value
    from pilotage_cli.secret_prompt import masked_secret_prompt

    key_env = pconfig.api_key_env_vars[0] if pconfig.api_key_env_vars else ""

    def _prompt_new_key() -> str:
        prompt = f"{key_env} (or Enter to cancel): "
        try:
            entered = masked_secret_prompt(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return ""
        return entered

    # First-time entry ────────────────────────────────────────────────────
    if not existing_key:
        print(f"No {pconfig.name} API key configured.")
        if not key_env:
            return "", True
        new_key = _prompt_new_key()
        if not new_key:
            print("Cancelled.")
            return "", True
        save_env_value(key_env, new_key)
        print("API key saved.")
        print()
        return new_key, False

    # Already configured — offer K / R / C ────────────────────────────────
    from pilotage_cli.env_loader import format_secret_source_suffix

    source_suffix = format_secret_source_suffix(key_env) if key_env else ""
    print(f"  {pconfig.name} API key: {existing_key[:8]}... ✓{source_suffix}")
    if not key_env:
        # Nothing we can rewrite; just acknowledge and move on.
        print()
        return existing_key, False
    pool_backed = existing_source.startswith("credential_pool:")
    menu = (
        "  [K]eep / [R]eplace (default K): "
        if pool_backed
        else "  [K]eep / [R]eplace / [C]lear (default K): "
    )
    try:
        choice = input(menu).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        choice = "k"

    if choice.startswith("r"):
        new_key = _prompt_new_key()
        if not new_key:
            print("  No change.")
            print()
            return existing_key, False
        save_env_value(key_env, new_key)
        print("  API key updated.")
        print()
        return new_key, False

    if choice.startswith("c") and not pool_backed:
        save_env_value(key_env, "")
        print(
            f"  API key cleared.  Re-run `pilotage setup` to configure {pconfig.name} again."
        )
        return "", True

    # Keep (default, or any other input)
    print()
    return existing_key, False



def cmd_login(args):
    """Authenticate Pilotage CLI with a provider."""
    from pilotage_cli.auth import login_command

    login_command(args)


def cmd_logout(args):
    """Clear provider authentication."""
    from pilotage_cli.auth import logout_command

    logout_command(args)


def cmd_auth(args):
    """Manage pooled credentials."""
    from pilotage_cli.auth_commands import auth_command

    auth_command(args)


def cmd_status(args):
    """Show status of all components."""
    from pilotage_cli.status import show_status

    show_status(args)


def cmd_cron(args):
    """Cron job management."""
    from pilotage_cli.cron import cron_command

    cron_command(args)


def cmd_webhook(args):
    """Webhook subscription management."""
    from pilotage_cli.webhook import webhook_command

    webhook_command(args)


def cmd_hooks(args):
    """Shell-hook inspection and management."""
    from pilotage_cli.hooks import hooks_command

    hooks_command(args)


def cmd_doctor(args):
    """Check configuration and dependencies."""
    from pilotage_cli.doctor import run_doctor

    run_doctor(args)


def cmd_security(args):
    """Dispatch `pilotage security <subcmd>`."""
    sub = getattr(args, "security_command", None)
    if sub in ("audit", None):
        from pilotage_cli.security_audit import cmd_security_audit

        # Default subcommand is `audit` when no subcmd is given.
        code = cmd_security_audit(args)
        sys.exit(int(code or 0))
    print(f"unknown security subcommand: {sub}", file=sys.stderr)
    sys.exit(2)


def cmd_approvals(args):
    """Dispatch `pilotage approvals <subcmd>`."""
    from pilotage_cli.approvals_suggest import approvals_command

    status = approvals_command(args)
    if status:
        sys.exit(status)
    return status


def cmd_dump(args):
    """Dump setup summary for support/debugging."""
    from pilotage_cli.dump import run_dump

    run_dump(args)


def cmd_debug(args):
    """Debug tools (share report, etc.)."""
    from pilotage_cli.debug import run_debug

    run_debug(args)


def cmd_config(args):
    """Configuration management."""
    from pilotage_cli.config import config_command

    config_command(args)


def cmd_backup(args):
    """Back up Pilotage home directory to a zip file."""
    if getattr(args, "quick", False):
        from pilotage_cli.backup import run_quick_backup

        run_quick_backup(args)
    else:
        from pilotage_cli.backup import run_backup

        run_backup(args)


def cmd_import(args):
    """Restore a Pilotage backup from a zip file."""
    from pilotage_cli.backup import run_import

    run_import(args)


def _print_version_info(*, check_updates: bool = True) -> None:
    from pilotage_cli.config import detect_install_method
    from pilotage_cli.slash_exec import CommandContext, execute_command

    # Core version line is registry-owned (shared with the gateway /version);
    # the install/python/SDK detail below is CLI-only decoration.
    print(execute_command("version", CommandContext(surface="cli")).text)
    print(f"Install directory: {PROJECT_ROOT}")
    print(f"Install method: {detect_install_method(PROJECT_ROOT)}")

    # Show Python version
    print(f"Python: {sys.version.split()[0]}")

    # Check for key dependencies.  Use importlib.metadata rather than
    # ``import openai`` — the SDK drags in ~800ms of pydantic-backed type
    # modules just to expose ``__version__``.  Metadata lookup is ~2ms.
    try:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError

        try:
            print(f"OpenAI SDK: {_pkg_version('openai')}")
        except PackageNotFoundError:
            print("OpenAI SDK: Not installed")
    except ImportError:
        print("OpenAI SDK: Not installed")

    if not check_updates:
        return

    # Show update status (synchronous — acceptable since user asked for version info)
    try:
        from pilotage_cli.banner import UPDATE_AVAILABLE_NO_COUNT, check_for_updates
        from pilotage_cli.config import recommended_update_command

        behind = check_for_updates()
        if behind == UPDATE_AVAILABLE_NO_COUNT:
            print(
                f"Update available — run '{recommended_update_command()}'"
            )
        elif behind and behind > 0:
            commits_word = "commit" if behind == 1 else "commits"
            print(
                f"Update available: {behind} {commits_word} behind — "
                f"run '{recommended_update_command()}'"
            )
        elif behind == 0:
            print("Up to date")
    except Exception:
        pass


def cmd_version(args):
    """Show version."""
    _print_version_info(check_updates=True)


def cmd_uninstall(args):
    """Uninstall Pilotage Agent."""
    if not getattr(args, "yes", False):
        _require_tty("uninstall")
    from pilotage_cli.uninstall import run_uninstall

    run_uninstall(args)


def _clear_bytecode_cache(root: Path) -> int:
    """Remove all __pycache__ directories under *root*.

    Stale .pyc files can cause ImportError after code updates when Python
    loads a cached bytecode file that references names that no longer exist
    (or don't yet exist) in the updated source.  Clearing them forces Python
    to recompile from the .py source on next import.

    Returns the number of directories removed.
    """
    removed = 0
    for dirpath, dirnames, _ in os.walk(root):
        # Skip venv / node_modules / .git entirely
        dirnames[:] = [
            d
            for d in dirnames
            if d not in {"venv", ".venv", "node_modules", ".git", ".worktrees"}
        ]
        if os.path.basename(dirpath) == "__pycache__":
            try:
                shutil.rmtree(dirpath)
                removed += 1
            except OSError:
                pass
            dirnames.clear()  # nothing left to recurse into
    return removed


# Update pipeline lives in pilotage_cli/update_cmd.py (main.py decomposition,
# mechanical move). Its names are re-exported lazily through the module-level
# __getattr__ above (see _LAZY_COMMAND_EXPORTS) so argparse wiring and test
# monkeypatches on pilotage_cli.main.<name> keep resolving unchanged without
# paying the update_cmd import cost on every CLI invocation.

# Stamp file recording the checkout fingerprint the bytecode cache was last
# validated against. Lives next to the checkout (NOT in PILOTAGE_HOME) because
# __pycache__ is per-checkout state shared by every profile.
_BYTECODE_FINGERPRINT_FILE = ".bytecode-fingerprint"


def _record_bytecode_fingerprint() -> None:
    """Persist the current checkout fingerprint after a bytecode sweep.

    Never raises. A failed write just means the next launch re-sweeps —
    safe, merely redundant.
    """
    try:
        fingerprint = _read_git_revision_fingerprint(PROJECT_ROOT)
        if not fingerprint:
            return
        stamp_path = PROJECT_ROOT / _BYTECODE_FINGERPRINT_FILE
        tmp_path = stamp_path.with_name(stamp_path.name + ".tmp")
        tmp_path.write_text(fingerprint, encoding="utf-8")
        tmp_path.replace(stamp_path)
    except OSError as exc:
        logger.debug("Could not record bytecode fingerprint: %s", exc)


def _sweep_stale_bytecode_if_checkout_changed() -> None:
    """Clear ``__pycache__`` at launch when the checkout changed underneath us.

    The stale-bytecode bug class (issues,; Dhruv's WhatsApp
    ``cannot import name 'parse_model_flags_detailed'`` report) has one
    shared shape: the checkout's ``.py`` files change (git pull inside
    ``pilotage update``, a manual ``git pull``, a ZIP update, a file-sync
    restore) while ``__pycache__`` retains bytecode from the previous
    revision, and a later process trusts the stale ``.pyc`` instead of the
    fresh source.

    Update-time clears alone can never close this class: ``pilotage update``
    always executes the PRE-pull updater code, so any hardening added to it
    only takes effect one update late, and manual ``git pull`` never runs
    the updater at all. This launch-time guard closes the loop: every
    ``pilotage`` entry point compares the checkout fingerprint (cheap file
    reads, no git subprocess) against the last-validated stamp and sweeps
    the bytecode cache once when they diverge.

    Never raises — a failure here must not block launch.
    """
    try:
        fingerprint = _read_git_revision_fingerprint(PROJECT_ROOT)
        if not fingerprint:
            return  # non-git install — the ZIP update path clears explicitly
        stamp_path = PROJECT_ROOT / _BYTECODE_FINGERPRINT_FILE
        try:
            recorded = stamp_path.read_text(encoding="utf-8").strip()
        except OSError:
            recorded = ""
        if recorded == fingerprint:
            return
        removed = _clear_bytecode_cache(PROJECT_ROOT)
        if removed:
            logger.info(
                "Checkout changed since last launch (%s -> %s): cleared %d stale __pycache__ director%s",
                recorded or "unknown",
                fingerprint,
                removed,
                "y" if removed == 1 else "ies",
            )
        _record_bytecode_fingerprint()
    except Exception as exc:
        logger.debug("Stale-bytecode launch sweep failed: %s", exc)


# Back-compat alias: some tests and any external callers may import the old
# warn-only name.  The new behaviour (kill stale processes) replaces it.
# Resolved lazily via _LAZY_COMMAND_ALIASES near the module __getattr__.


# =========================================================================
# Fork detection and upstream management for `pilotage update`
# =========================================================================


def _load_installable_optional_extras(group: str = "all") -> list[str]:
    """Return optional extras referenced by a dependency group.

    ``group`` is usually ``all`` (desktop/server broad install) or
    ``termux-all`` (Termux-compatible broad install).
    """
    try:
        import tomllib

        with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
    except Exception:
        return []

    optional_deps = project.get("optional-dependencies", {})
    if not isinstance(optional_deps, dict):
        return []

    refs = optional_deps.get(group, [])
    referenced: list[str] = []
    for ref in refs:
        if "[" in ref and "]" in ref:
            name = ref.split("[", 1)[1].split("]", 1)[0]
            if name in optional_deps:
                referenced.append(name)

    return referenced


# Install-scoped breadcrumbs live next to the venv (not under $PILOTAGE_HOME)
# because the venv is shared across profiles.
#
# ``.update-incomplete`` — generic core ``.[all]`` install was interrupted.
# Cleared only after a confirmed full dependency reinstall/recovery.
#
# ``.lazy-refresh-incomplete`` — lazy-backend refresh phase may have corrupted
# packages. Cleared only after import-probe repair confirms healthy (not when
# probes are unavailable/indeterminate). Narrow lazy probes must NEVER clear
# the generic core marker review).
def _update_marker_path() -> Path:
    return PROJECT_ROOT / ".update-incomplete"


def _lazy_refresh_marker_path() -> Path:
    return PROJECT_ROOT / ".lazy-refresh-incomplete"


def _pytest_owns_live_checkout(root: Path) -> bool:
    """True when running under pytest AND ``root`` is this checkout itself.

    Tests that drive update/recovery without sandboxing ``PROJECT_ROOT``
    must neither litter the live repo root with recovery breadcrumbs
    (a leftover ``.lazy-refresh-incomplete`` / ``.update-incomplete``
    false-arms recovery on the developer's next real launch) nor run a real
    reinstall against the executing venv. Sandboxed tests point at a
    tmp_path and are unaffected (same posture as
    ``managed_scope._under_pytest``)."""
    return (
        "PYTEST_CURRENT_TEST" in os.environ
        and root == Path(__file__).resolve().parent.parent
    )


def _clear_marker_file(path: Path, *, label: str) -> None:
    """Remove an update-recovery breadcrumb. Never raises."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Could not clear %s marker: %s", label, exc)


def _clear_update_incomplete_marker() -> None:
    """Remove the interrupted core-install breadcrumb. Never raises."""
    _clear_marker_file(_update_marker_path(), label="update-incomplete")


def _clear_lazy_refresh_incomplete_marker() -> None:
    """Remove the interrupted lazy-refresh breadcrumb. Never raises."""
    _clear_marker_file(_lazy_refresh_marker_path(), label="lazy-refresh-incomplete")


def _recover_from_interrupted_install() -> None:
    """Finish update work left half-done by a prior ``pilotage update``.

    Handles two independent breadcrumbs:

    - ``.update-incomplete`` — core ``.[all]`` install interrupted. Recovers
      via full quarantined reinstall. Never cleared by the narrow lazy-refresh
      import probes alone.
    - ``.lazy-refresh-incomplete`` — lazy-backend refresh may have corrupted
      packages. Recovers via package-only import probes; cleared only when
      probes confirm healthy/repaired (indeterminate keeps the marker).

    Never raises: a recovery failure must not block launch.  If it can't
    self-heal it prints the manual command and leaves the relevant marker so
    the next launch tries again.

    Concurrency: markers live next to the shared venv, so a gateway start
    plus a CLI launch (or two profiles starting at once) can both see them.
    An ``O_EXCL`` lockfile ensures only one process runs recovery; the
    others skip and let the winner clear markers.

    Output: everything — our status lines AND the streamed pip/uv install
    (which inherits fd 1) — is routed to stderr.  Launches whose stdout is a
    protocol stream (``pilotage acp`` speaks JSON-RPC on stdout) must never get
    install noise on stdout.
    """
    if _pytest_owns_live_checkout(PROJECT_ROOT):
        return
    core_marker = _update_marker_path().exists()
    lazy_marker = _lazy_refresh_marker_path().exists()
    if not core_marker and not lazy_marker:
        return

    # Skip in managed/Docker installs and on PyPI installs with no git checkout:
    # those don't run the source-tree update path, so a stray marker is not ours
    # to act on. Just clear it.
    if not (PROJECT_ROOT / "pyproject.toml").is_file():
        _clear_update_incomplete_marker()
        _clear_lazy_refresh_incomplete_marker()
        return

    # Single-flight guard: atomically claim the recovery lock. If another
    # process holds it, skip — it is running the same reinstall into the same
    # shared venv right now. A crashed holder leaves a stale lock; break it
    # after an hour (well past any realistic install) so recovery can't be
    # wedged forever.
    lock_path = PROJECT_ROOT / ".update-incomplete.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
    except FileExistsError:
        try:
            if _time.time() - lock_path.stat().st_mtime > 3600:
                lock_path.unlink()
        except OSError:
            pass
        return
    except OSError as exc:
        # Couldn't create the lock (read-only fs, perms). Proceed unlocked —
        # the install itself will surface the real problem.
        logger.debug("Could not create install-recovery lock: %s", exc)

    saved_stdout_fd = None
    saved_sys_stdout = sys.stdout
    try:
        # Route Python-level prints AND subprocess-inherited fd 1 to stderr
        # for the duration of recovery (see docstring: ACP stdout safety).
        try:
            saved_stdout_fd = os.dup(1)
            os.dup2(2, 1)
        except OSError:
            saved_stdout_fd = None
        sys.stdout = sys.stderr

        if lazy_marker:
            _recover_lazy_refresh_marker_locked()

        if _update_marker_path().exists():
            _recover_core_update_marker_locked()
    finally:
        sys.stdout = saved_sys_stdout
        if saved_stdout_fd is not None:
            try:
                os.dup2(saved_stdout_fd, 1)
                os.close(saved_stdout_fd)
            except OSError:
                pass
        try:
            lock_path.unlink()
        except OSError:
            pass


def _recover_lazy_refresh_marker_locked() -> None:
    """Heal ``.lazy-refresh-incomplete`` via confirmed import-probe repair."""
    print(
        "⚠ A previous lazy-backend refresh may have left the venv unhealthy — "
        "running import-based package repair..."
    )
    install_prefix, install_env = _default_venv_install_target()
    status = _repair_venv_via_import_probes(install_prefix, env=install_env)
    if status in ("healthy", "repaired"):
        _clear_lazy_refresh_incomplete_marker()
        print("✓ Lazy-refresh venv recovery confirmed — install is healthy again.")
        return
    if status == "indeterminate":
        print(
            "  ⚠ Import probes unavailable — cannot confirm venv health. "
            "Leaving `.lazy-refresh-incomplete` for the next launch."
        )
    else:
        print(
            "  ⚠ Lazy-refresh package repair incomplete. "
            "Leaving `.lazy-refresh-incomplete` for the next launch."
        )
        print("  Recover manually with:")
        all_specs = _lazy_refresh_repair_specs(
            sorted(set(_LAZY_REFRESH_REPAIR_PACKAGES.values()))
        )
        print(
            f"    {' '.join(install_prefix)} install --force-reinstall "
            + " ".join(shlex.quote(s) for s in all_specs)
        )


def _recover_core_update_marker_locked() -> None:
    """Heal ``.update-incomplete`` via full ``.[all]`` reinstall only.

    Narrow lazy-refresh import probes are not sufficient proof that a generic
    interrupted core install finished — a missing dep outside that probe set
    would otherwise look healthy and clear the breadcrumb too early.
    """
    print(
        "⚠ A previous `pilotage update` was interrupted mid-install — "
        "finishing dependency installation now..."
    )

    # Windows: a normal ``pilotage.exe`` launch always has the launcher as an
    # ancestor. Full editable reinstall uses quarantine so the live shim can
    # still be replaced. Package-only import repair may help as first aid but
    # must NEVER clear this core marker on its own review).
    self_locked = _windows_running_pilotage_launcher_locked()
    if self_locked:
        install_prefix, install_env = _default_venv_install_target()
        print(
            "  → Running from pilotage.exe; applying package-only first aid, "
            "then quarantined full reinstall (core marker stays until that "
            "succeeds)..."
        )
        _repair_venv_via_import_probes(install_prefix, env=install_env)

    try:
        from pilotage_cli import _install_repair as _ir

        # ensure_uv bootstraps the installer itself when missing (the early
        # pass's stdlib-only lookup cannot); keeping it here means the late
        # path still self-heals a venv whose uv vanished mid-update.
        from pilotage_cli.managed_uv import ensure_uv

        ensure_uv()

        # Delegate the install itself to the shared stdlib executor so both
        # this late path and the pre-import early pass run exactly the same
        # reinstall.  Called inside the same stdout→stderr redirect already
        # established by _recover_from_interrupted_install, so
        # run_core_install's own redirect nests harmlessly.
        _ir.run_core_install(PROJECT_ROOT)

        _clear_update_incomplete_marker()
        print("✓ Dependency installation recovered — your install is healthy again.")
    except Exception as exc:
        # Leave the marker in place so the next launch retries. Give the user
        # the exact manual recovery command in the meantime.
        logger.debug("Interrupted-install recovery failed: %s", exc)
        print("✗ Could not auto-recover the interrupted install.")
        if self_locked:
            print(
                "  Pilotage is still running from the launcher that needs "
                "replacing. Close other Pilotage windows, restart from a "
                "different terminal, then run:"
            )
            print(f'    cd /d "{PROJECT_ROOT}"')
            print(
                f'    "{sys.executable}" -m pip install -e ".[all]"'
            )
        else:
            print("  Recover manually with:")
            print(f"    cd {PROJECT_ROOT}")
            print(f"    {sys.executable} -m ensurepip --upgrade")
            print(f"    {sys.executable} -m pip install -e '.[all]'")


def _windows_running_pilotage_launcher_locked() -> bool:
    """True when a venv ``pilotage*.exe`` shim is this process or an ancestor.

    Best-effort: returns False when psutil is unavailable or inspection fails.
    """
    if not _is_windows():
        return False
    scripts_dir = _venv_scripts_dir()
    if scripts_dir is None:
        return False
    shims = _pilotage_exe_shims(scripts_dir)
    if not shims:
        return False
    shim_set: set[str] = set()
    for shim in shims:
        try:
            shim_set.add(str(shim.resolve()).lower())
        except OSError:
            shim_set.add(str(shim).lower())
    try:
        import psutil

        me = psutil.Process()
        for proc in [me] + list(me.parents()):
            try:
                exe_norm = str(Path(proc.exe()).resolve()).lower()
            except Exception:
                continue
            if exe_norm in shim_set:
                return True
    except Exception:
        return False
    return False


def _default_venv_install_target() -> tuple[list[str], dict[str, str] | None]:
    """Return ``(install_cmd_prefix, env)`` for the project venv when possible."""
    try:
        from pilotage_cli.managed_uv import ensure_uv

        uv_bin = ensure_uv()
    except Exception:
        uv_bin = None
    if uv_bin:
        env = {**os.environ, "VIRTUAL_ENV": str(PROJECT_ROOT / "venv")}
        if _is_termux_env(env):
            env.pop("PYTHONPATH", None)
            env.pop("PYTHONHOME", None)
        return [uv_bin, "pip"], env
    return [sys.executable, "-m", "pip"], None


def _run_install_with_heartbeat(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    heartbeat_interval_seconds: int = 30,
) -> None:
    """Run dependency install command with periodic heartbeat output.

    Some resolvers/build backends (especially when compiling Rust/C extensions)
    can stay quiet for minutes. Emit a simple elapsed-time heartbeat so users
    know ``pilotage update`` is still progressing even if pip/uv itself is silent.
    """
    done = threading.Event()
    start = _time.time()

    def _heartbeat() -> None:
        # Wait first, then print, so short installs don't emit noise.
        while not done.wait(heartbeat_interval_seconds):
            elapsed = int(_time.time() - start)
            print(
                f"  … still installing dependencies ({elapsed}s elapsed)"
                " — compiling Rust/C extensions can take several minutes",
                flush=True,
            )

    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    try:
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            env=env,
        )
    finally:
        done.set()
        t.join(timeout=0.2)


def _is_windows() -> bool:
    return sys.platform == "win32"


def _venv_scripts_dir() -> Path | None:
    """Return the venv Scripts directory if we're running inside the project venv."""
    venv_dir = PROJECT_ROOT / "venv"
    if not venv_dir.is_dir():
        return None
    from pilotage_constants import venv_bin_dir

    scripts = venv_bin_dir(venv_dir, windows=_is_windows())
    return scripts if scripts.is_dir() else None


def _pilotage_exe_shims(scripts_dir: Path) -> list[Path]:
    """Entry-point shims that uv may try to rewrite during ``pip install -e .``.

    On Windows these are .exe launchers generated by setuptools/uv. On POSIX
    they're regular Python scripts which can be replaced atomically — no
    self-replacement hazard exists outside Windows.
    """
    if not _is_windows():
        return []

    names = set(_load_console_script_names()) or {"pilotage", "pilotage-agent"}
    # The gateway shim is not a [project.scripts] entry point, but older
    # update/install paths still rewrite and quarantine it.
    names.add("pilotage-gateway")
    return [scripts_dir / f"{name}.exe" for name in sorted(names)]


def _quarantine_running_pilotage_exe(
    scripts_dir: Path, *, max_attempts: int = 4
) -> list[tuple[Path, Path]]:
    """Pre-empt Windows file lock on the running ``pilotage.exe``.

    Windows allows RENAMING a mapped/running executable (the kernel tracks the
    file by handle, not path), but blocks DELETE/REPLACE while it's loaded. uv
    needs to overwrite the entry-point shims during ``pip install -e .``;
    when ``pilotage update`` runs, ``pilotage.exe`` IS the live process, and uv
    fails with ``Access is denied. (os error 5)``.

    We rename live shims to ``pilotage.exe.old.<unix-ms>`` first. uv then writes
    fresh shims at the original paths. The ``.old`` files are cleaned up on
    the next pilotage invocation by ``_cleanup_quarantined_exes``.

    Rename can still fail when *another* process has opened the .exe without
    ``FILE_SHARE_DELETE`` — typically AV real-time scanners with transient
    handles (recovers in <1s), or the Pilotage Desktop backend child process
    (won't recover until the user closes it). We mitigate:

    1. Retry up to ``max_attempts`` times with exponential backoff
       (100/250/500/1000 ms). Handles the AV-scanner case.
    2. If all retries fail, schedule the .exe for replacement on next
       reboot via ``MoveFileExW(MOVEFILE_DELAY_UNTIL_REBOOT)``. This still
       lets uv create a fresh shim at the original path (Windows will keep
       the old file's content under a new name until the reboot), so the
       update can complete; the user just needs to reboot to fully unload
       the stale image.
    3. Print a clear warning naming the most likely culprit (running
       Pilotage Desktop / gateway / REPL) and pointing to ``--force``.

    Returns the list of (original, quarantined) pairs so the caller can roll
    back if the install itself fails before uv writes a replacement. Pairs
    where we used ``MOVEFILE_DELAY_UNTIL_REBOOT`` are NOT returned — they
    are already deferred and roll-back is meaningless.
    """
    moved: list[tuple[Path, Path]] = []
    if not _is_windows():
        return moved

    import time

    stamp = int(time.time() * 1000)
    # Backoff schedule: first attempt is immediate, subsequent ones sleep.
    # 100ms / 250ms / 500ms covers the typical AV scanner re-scan window.
    backoff_ms = [0, 100, 250, 500, 1000]
    attempts = max(1, min(max_attempts, len(backoff_ms)))

    for shim in _pilotage_exe_shims(scripts_dir):
        if not shim.exists():
            continue
        target = shim.with_suffix(shim.suffix + f".old.{stamp}")

        last_exc: OSError | None = None
        for attempt in range(attempts):
            delay = backoff_ms[attempt] / 1000.0
            if delay:
                time.sleep(delay)
            try:
                shim.rename(target)
                moved.append((shim, target))
                last_exc = None
                break
            except OSError as e:
                last_exc = e
                continue

        if last_exc is None:
            continue

        # All in-process renames failed. Try MoveFileEx with
        # MOVEFILE_DELAY_UNTIL_REBOOT as a last resort. This succeeds in the
        # exact case where the inline rename failed (another process holds
        # the handle without share-delete), at the cost of requiring a
        # reboot to fully reclaim the old .exe.
        scheduled = _schedule_replace_on_reboot(shim, target)
        if scheduled:
            print(
                f"  ⚠ {shim.name} is locked by another process; scheduled "
                f"replacement on next reboot."
            )
            print(
                "    The new shim was written at the same path, but a "
                "reboot is needed to fully unload the old one."
            )
            # Do NOT append to ``moved``: we don't want roll-back to undo a
            # reboot-deferred operation.
            continue

        # Truly couldn't budge the .exe. Print an actionable warning and let
        # uv try its luck — sometimes uv's own retry handling pulls through.
        print(
            f"  ⚠ Could not quarantine {shim.name} ({last_exc.__class__.__name__}: "
            f"another process is holding it open)."
        )
        print(
            "    Close Pilotage Desktop, exit other `pilotage` REPLs, stop the "
            "gateway, or pause AV scanning, then re-run `pilotage update`."
        )

    return moved


def _schedule_replace_on_reboot(shim: Path, quarantine_target: Path) -> bool:
    """Schedule ``shim`` -> ``quarantine_target`` via PendingFileRenameOperations.

    Uses Win32 ``MoveFileExW`` with ``MOVEFILE_REPLACE_EXISTING |
    MOVEFILE_DELAY_UNTIL_REBOOT``. The OS persists the rename in
    ``HKLM\\System\\CurrentControlSet\\Control\\Session Manager\\
    PendingFileRenameOperations`` and applies it before any user-mode code
    runs on next boot — at which point no process can hold the .exe.

    Returns ``True`` if the schedule call succeeded, ``False`` otherwise
    (non-Windows, ctypes failure, lack of privilege, etc.). Never raises.
    """
    if not _is_windows():
        return False
    try:
        import ctypes
        from ctypes import wintypes

        MOVEFILE_REPLACE_EXISTING = 0x1
        MOVEFILE_DELAY_UNTIL_REBOOT = 0x4

        MoveFileExW = ctypes.windll.kernel32.MoveFileExW
        MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        MoveFileExW.restype = wintypes.BOOL

        ok = MoveFileExW(
            str(shim),
            str(quarantine_target),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_DELAY_UNTIL_REBOOT,
        )
        return bool(ok)
    except Exception:
        return False


def _restore_quarantined_exes(moved: list[tuple[Path, Path]]) -> None:
    """Roll back ``_quarantine_running_pilotage_exe`` if uv didn't write replacements."""
    for original, quarantined in moved:
        try:
            if not original.exists() and quarantined.exists():
                quarantined.rename(original)
        except OSError:
            pass


def _run_quarantined_install(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    scripts_dir: Path | None = None,
) -> None:
    """Run an editable install, quarantining the running ``pilotage.exe`` first.

    Any ``pip install -e .`` (or ``--reinstall``) rewrites the entry-point
    shims, and on Windows the live ``pilotage.exe`` is the running process —
    pip can neither delete nor overwrite it, so without quarantine the shim
    is left missing and ``pilotage`` drops off PATH. This wraps
    :func:`_run_install_with_heartbeat` with the same rename-out-of-the-way /
    restore-on-failure dance that the primary install path uses, so EVERY
    install that touches the shims is protected — including the
    verification-repair reinstalls in
    :func:`_verify_core_dependencies_installed`, which previously called
    ``_run_install_with_heartbeat`` directly and bypassed quarantine.

    Off-Windows (``scripts_dir is None``) this is a thin pass-through.
    """
    moved: list[tuple[Path, Path]] = []
    if scripts_dir is not None:
        moved = _quarantine_running_pilotage_exe(scripts_dir)
    try:
        _run_install_with_heartbeat(cmd, env=env)
    except BaseException:
        # Restore shims if pip/uv didn't write replacements (e.g. install
        # failed before the entry-points step). Don't swallow the error.
        if scripts_dir is not None:
            _restore_quarantined_exes(moved)
        raise


def _cleanup_quarantined_exes(scripts_dir: Path | None = None) -> None:
    """Sweep ``pilotage.exe.old.*`` left by prior updates.

    Called early on every pilotage invocation. The .old files are unlocked once
    their owning process exited, so deletion succeeds the next run. Silent
    no-op when nothing's there or on file-locked / permission errors.
    """
    if not _is_windows():
        return
    if scripts_dir is None:
        scripts_dir = _venv_scripts_dir()
    if scripts_dir is None:
        return
    try:
        for stale in scripts_dir.glob("*.exe.old.*"):
            try:
                stale.unlink()
            except OSError:
                pass  # still locked or in use — try again next run
    except OSError:
        pass


# Import probes for venv corruption after a failed lazy ``uv pip install``.
# Metadata can look fine while ``.py`` files were removed mid-install.
# Canonical tables live in the stdlib-only ``_early_recovery`` module (which
# also probes/repairs BEFORE this module's third-party imports can run) so the
# early and full recovery layers can never drift apart.
_LAZY_REFRESH_IMPORT_PROBES: tuple[tuple[str, str], ...] = (
    _early_recovery_mod.LAZY_REFRESH_IMPORT_PROBES
)

_LAZY_REFRESH_REPAIR_PACKAGES: dict[str, str] = (
    _early_recovery_mod.LAZY_REFRESH_REPAIR_PACKAGES
)


def _run_package_only_install(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Run a package-only pip/uv install without quarantining entry-point shims.

    ``pip install --upgrade pip`` and ``--force-reinstall <pkg>`` do not
    rewrite ``pilotage.exe``. The editable-install quarantine path would rename
    shims without uv recreating them on Windows.
    """
    _run_install_with_heartbeat(cmd, env=env)


def _lazy_refresh_repair_specs(packages: list[str]) -> list[str]:
    """Map repair package names to their declared pin specs in pyproject.toml."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover
        return packages

    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return packages

    try:
        with open(pyproject, "rb") as f:
            raw_deps = tomllib.load(f).get("project", {}).get("dependencies", []) or []
    except Exception as exc:
        logger.debug("lazy refresh repair spec lookup failed: %s", exc)
        return packages

    name_to_spec: dict[str, str] = {}
    try:
        from packaging.requirements import Requirement  # type: ignore

        for spec in raw_deps:
            try:
                req = Requirement(spec)
                name_to_spec[req.name.lower()] = spec.split(";", 1)[0].strip()
            except Exception:
                continue
    except Exception:
        for spec in raw_deps:
            head = spec.split(";", 1)[0].strip()
            bare = head
            for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
                if op in bare:
                    bare = bare.split(op, 1)[0]
                    break
            key = bare.strip().split("[", 1)[0].strip().lower()
            if key:
                name_to_spec[key] = head

    return [name_to_spec.get(pkg.lower(), pkg) for pkg in packages]


def _detect_broken_lazy_refresh_imports(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> list[str] | None:
    """Probe lazy-refresh packages via real imports.

    Returns:
      - ``[]`` when probes ran and every package imported cleanly
      - ``[dist, ...]`` when probes ran and some packages failed
      - ``None`` when the probe could not run (missing venv Python, subprocess
        failure, non-zero probe exit) — this is *indeterminate*, not healthy
    """
    venv_python = _resolve_install_target_python(install_cmd_prefix, env)
    if venv_python is None:
        return None

    probe_lines = "\n".join(
        f"    ({mod!r}, {attr!r})," for mod, attr in _LAZY_REFRESH_IMPORT_PROBES
    )
    check_script = (
        "import os\n"
        "import sys\n"
        "probes = [\n"
        f"{probe_lines}\n"
        "]\n"
        "broken = []\n"
        "for mod, attr in probes:\n"
        "    try:\n"
        "        imported = __import__(mod)\n"
        "        if not hasattr(imported, attr):\n"
        "            broken.append(mod)\n"
        "        elif mod == 'certifi':\n"
        "            # The module can import cleanly while cacert.pem is\n"
        "            # missing/corrupt (brew Python upgrade, interrupted venv\n"
        " # rebuild) - every TLS call then fails.\n"
        "            bundle = imported.where()\n"
        "            if not os.path.isfile(bundle) or os.path.getsize(bundle) < 1024:\n"
        "                broken.append(mod)\n"
        "    except Exception:\n"
        "        broken.append(mod)\n"
        "print('\\n'.join(broken))\n"
    )
    try:
        result = subprocess.run(
            [str(venv_python), "-c", check_script],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            check=False,
            env=env,
        )
    except Exception as exc:
        logger.debug("lazy refresh import probe failed: %s", exc)
        return None

    if result.returncode != 0:
        logger.debug(
            "lazy refresh import probe exited %s: %s",
            result.returncode,
            (result.stderr or "")[:200],
        )
        return None

    broken_modules = [
        line.strip() for line in result.stdout.splitlines() if line.strip()
    ]
    packages: list[str] = []
    seen: set[str] = set()
    for mod in broken_modules:
        pkg = _LAZY_REFRESH_REPAIR_PACKAGES.get(mod)
        if pkg and pkg not in seen:
            seen.add(pkg)
            packages.append(pkg)
    return packages


def _repair_broken_lazy_refresh_imports(
    install_cmd_prefix: list[str],
    packages: list[str],
    *,
    env: dict[str, str] | None = None,
) -> bool:
    """Force-reinstall ``packages`` and re-probe imports. Never raises."""
    if not packages:
        return True

    specs = _lazy_refresh_repair_specs(packages)
    try:
        _run_package_only_install(
            install_cmd_prefix + ["install", "--force-reinstall", *specs],
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        logger.warning("lazy refresh venv repair failed: %s", exc)
        return False

    after = _detect_broken_lazy_refresh_imports(install_cmd_prefix, env=env)
    # Indeterminate re-probe is not confirmed success.
    return after == []


def _repair_venv_via_import_probes(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> str:
    """Probe imports and force-reinstall any broken lazy-refresh packages.

    Uses real ``import`` checks (not distribution metadata) so a venv where
    METADATA remains but ``.py`` files were wiped mid-install is still
    detected. Package-only reinstall — never rewrites ``pilotage.exe``.

    Never raises. Returns one of:
      - ``"healthy"`` — probes ran and found nothing broken
      - ``"repaired"`` — probes found breakage and force-reinstall confirmed clean
      - ``"failed"`` — probes found breakage and repair did not confirm clean
      - ``"indeterminate"`` — probes could not run; do NOT treat as healthy
    """
    broken = _detect_broken_lazy_refresh_imports(install_cmd_prefix, env=env)
    if broken is None:
        print(
            "  ⚠ Import probes unavailable — cannot confirm venv package health."
        )
        return "indeterminate"
    if not broken:
        return "healthy"
    print(
        "  → Detected corrupted venv packages via import probes: "
        f"{', '.join(broken)}; repairing..."
    )
    if _repair_broken_lazy_refresh_imports(
        install_cmd_prefix, broken, env=env
    ):
        print("  ✓ Venv repair succeeded")
        return "repaired"
    manual = " ".join(
        shlex.quote(s) for s in _lazy_refresh_repair_specs(broken)
    )
    print("  ⚠ Venv repair incomplete. Run manually, then `pilotage update`:")
    print(
        f"    {' '.join(install_cmd_prefix)} install --force-reinstall {manual}"
    )
    return "failed"


def _install_python_dependencies_with_optional_fallback(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
    group: str = "all",
) -> None:
    """Install base deps plus as many optional extras as the environment supports.

    By default this targets ``.[all]``; Termux callers can pass
    ``group='termux-all'`` to use the curated Android-compatible profile.

    On Windows, pre-renames live ``pilotage.exe`` / ``pilotage-gateway.exe`` shims
    in the venv Scripts dir before each install attempt so uv can write fresh
    copies (Windows blocks REPLACE on a running .exe but allows RENAME). See
    ``_quarantine_running_pilotage_exe`` for the rationale.
    """
    scripts_dir = _venv_scripts_dir() if _is_windows() else None

    def _install(args: list[str]) -> None:
        _run_quarantined_install(
            install_cmd_prefix + args, env=env, scripts_dir=scripts_dir
        )

    try:
        _install(["install", "-e", f".[{group}]"])
        _verify_console_scripts_installed(install_cmd_prefix, env=env)
        return
    except subprocess.CalledProcessError:
        print(
            "  ⚠ Optional extras failed, reinstalling base dependencies and retrying extras individually..."
        )

    _install(["install", "-e", "."])

    failed_extras: list[str] = []
    installed_extras: list[str] = []
    for extra in _load_installable_optional_extras(group=group):
        try:
            _install(["install", "-e", f".[{extra}]"])
            installed_extras.append(extra)
        except subprocess.CalledProcessError:
            failed_extras.append(extra)

    if installed_extras:
        print(
            f"  ✓ Reinstalled optional extras individually: {', '.join(installed_extras)}"
        )
    if failed_extras:
        print(
            f"  ⚠ Skipped optional extras that still failed: {', '.join(failed_extras)}"
        )

    # Belt-and-suspenders: verify every declared core dependency from
    # pyproject.toml's [project.dependencies] is actually importable in the
    # target venv. uv's incremental resolver has — in the wild — produced
    # partial installs where a newly added base dep (e.g. ``pathspec``)
    # silently fails to land on top of a half-stale venv, and the only
    # symptom is a downstream subprocess crashing with ModuleNotFoundError
    # hours later inside ``pilotage update``'s desktop-rebuild or skill-sync
    # stage. Reinstall with --reinstall to force resolution if anything is
    # missing, then re-verify so the failure surfaces here instead of
    # downstream.
    _verify_core_dependencies_installed(install_cmd_prefix, env=env, group=group)
    _verify_console_scripts_installed(install_cmd_prefix, env=env)


def _load_console_script_names() -> list[str]:
    """Return ``[project.scripts]`` entry-point names from pyproject.toml."""
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover
        return []

    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return []

    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        scripts = data.get("project", {}).get("scripts", {}) or {}
        return [str(name) for name in scripts if name]
    except Exception as e:
        logger.debug("console script verification: failed to read pyproject.toml: %s", e)
        return []


def _verify_console_scripts_installed(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Ensure every declared console_script shim exists on disk after install.

    On Windows, ``uv pip install -e .`` can register ``pilotage.exe`` in the
    wheel RECORD while the file never lands on disk — typically when the live
    ``pilotage.exe`` shim is locked during ``pilotage update``, or when uv/distlib
    skips a launcher write. The symptom is ``pilotage-agent.exe`` and
    ``pilotage-acp.exe`` present but ``pilotage.exe`` missing, so ``pilotage`` drops
    off PATH even though the install reported success.

    If any shim is missing we reinstall with ``--reinstall -e .`` under the
    same quarantine dance as the primary install path, then re-check.
    """
    if not _is_windows():
        return

    scripts_dir = _venv_scripts_dir()
    if scripts_dir is None:
        return

    names = _load_console_script_names()
    if not names:
        return

    def _missing() -> list[str]:
        return [
            name
            for name in names
            if not (scripts_dir / f"{name}.exe").is_file()
        ]

    missing = _missing()
    if not missing:
        return

    print(
        f"  ⚠ Verification: {len(missing)} console script(s) missing on disk: "
        f"{', '.join(missing)}"
    )
    print("  → Reinstalling entry points with --reinstall...")

    try:
        _run_quarantined_install(
            install_cmd_prefix + ["install", "--reinstall", "-e", "."],
            env=env,
            scripts_dir=scripts_dir,
        )
    except subprocess.CalledProcessError as e:
        logger.warning("console script verification: repair install failed: %s", e)
        print(
            "  ⚠ Entry point repair failed; try `pilotage update --force` after "
            "closing other pilotage processes."
        )
        return

    still_missing = _missing()
    if still_missing:
        print(
            f"  ⚠ Still missing after repair: {', '.join(still_missing)}. "
            "Workaround: python -m pilotage_cli.main <command>"
        )
    else:
        print("  ✓ All console entry points restored")


def _verify_core_dependencies_installed(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
    group: str = "all",
) -> None:
    """Check that every base dep from pyproject.toml is importable; if not, retry.

    Reads ``pyproject.toml`` directly (so we don't trust the venv's stale
    metadata), filters out deps gated by ``;`` environment markers that don't
    apply to this platform, and runs ``importlib.metadata.version()`` in the
    venv interpreter for each one. If anything is missing we reinstall the
    base group with ``--reinstall`` to force uv to re-resolve, then check
    again. We treat the final state as a warning rather than a hard failure
    so a single broken-on-PyPI dep can't block an otherwise-successful
    update — but the warning makes the partial install visible at the spot
    that caused it, instead of hours later in a downstream subprocess.
    """
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover — Python < 3.11 unsupported but be safe
        return

    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return

    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        raw_deps = data.get("project", {}).get("dependencies", []) or []
    except Exception as e:
        logger.debug("dep verification: failed to read pyproject.toml: %s", e)
        return

    # Parse each "name OP version ; marker" string into (dist_name, marker_obj).
    # We use packaging.requirements when available (it ships with pip/uv envs),
    # falling back to a naive split that's good enough for the canonical
    # ``name==version[; marker]`` style this repo uses.
    deps: list[tuple[str, "object | None"]] = []
    try:
        from packaging.requirements import Requirement  # type: ignore

        for spec in raw_deps:
            try:
                req = Requirement(spec)
                deps.append((req.name, req.marker))
            except Exception:
                continue
    except Exception:
        for spec in raw_deps:
            head = spec.split(";", 1)[0]
            for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
                if op in head:
                    head = head.split(op, 1)[0]
                    break
            name = head.strip().split("[", 1)[0].strip()
            if name:
                deps.append((name, None))

    # Apply environment markers to drop deps that don't apply on this platform
    # (e.g. ``ptyprocess ; sys_platform != 'win32'`` is correctly skipped on
    # Windows). Without markers we'd false-positive every cross-platform exclusion.
    applicable: list[str] = []
    for name, marker in deps:
        if marker is None:
            applicable.append(name)
            continue
        try:
            if marker.evaluate():  # type: ignore[union-attr]
                applicable.append(name)
        except Exception:
            applicable.append(name)

    if not applicable:
        return

    # Run the check inside the venv Python — sys.executable here may be the
    # outer Python that drove ``pilotage update``, not the venv we just wrote
    # to. The uv install_cmd_prefix encodes which environment we targeted
    # (either ``[uv, pip]`` with VIRTUAL_ENV in env, or
    # ``[sys.executable, -m, pip]`` for the in-process Python); resolve the
    # right interpreter for the verification.
    venv_python = _resolve_install_target_python(install_cmd_prefix, env)
    if venv_python is None:
        return

    def _missing_deps() -> list[str]:
        check_script = (
            "import importlib.metadata as md, sys\n"
            "missing=[]\n"
            "for name in sys.argv[1:]:\n"
            "    try: md.version(name)\n"
            "    except md.PackageNotFoundError: missing.append(name)\n"
            "print('\\n'.join(missing))\n"
        )
        try:
            result = subprocess.run(
                [str(venv_python), "-c", check_script, *applicable],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                check=False,
                env=env,
            )
        except Exception as e:
            logger.debug("dep verification: subprocess failed: %s", e)
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    missing = _missing_deps()
    if not missing:
        return

    print(
        f"  ⚠ Verification: {len(missing)} declared dep(s) missing after install: "
        f"{', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}"
    )
    print("  → Reinstalling base group with --reinstall to repair...")

    # Reinstall base group with --reinstall so uv re-resolves from scratch
    # against the current pyproject. We don't pass ``[{group}]`` here on
    # purpose — the missing dep is in *base* deps; rerunning the full all-
    # extras install can cost minutes and trips on whatever optional extra
    # was already broken upstream. Base is fast and is what's actually wrong.
    #
    # Quarantine the running ``pilotage.exe`` first: ``--reinstall -e .``
    # rewrites the entry-point shims, and on Windows pip can't overwrite the
    # live launcher, which would leave ``pilotage`` off PATH.
    scripts_dir = _venv_scripts_dir() if _is_windows() else None
    repair_args = ["install", "--reinstall", "-e", "."]
    try:
        _run_quarantined_install(
            install_cmd_prefix + repair_args, env=env, scripts_dir=scripts_dir
        )
    except subprocess.CalledProcessError as e:
        logger.warning("dep verification: repair install failed: %s", e)
        print("  ⚠ Repair install failed; check `pilotage update` output above.")
        return

    still_missing = _missing_deps()
    if not still_missing:
        print("  ✓ All declared core dependencies now installed")
        return

    # Last-ditch: install each remaining missing dep with its pin directly.
    # Useful when uv's resolver thinks the env is satisfied but the on-disk
    # package metadata says otherwise (rare but observed).
    name_to_spec = {}
    for spec in raw_deps:
        head = spec.split(";", 1)[0].strip()
        bare = head
        for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
            if op in bare:
                bare = bare.split(op, 1)[0]
                break
        name_to_spec[bare.strip().split("[", 1)[0].strip()] = head

    specs = [name_to_spec.get(n, n) for n in still_missing]
    print(
        f"  → Force-installing remaining missing dep(s): {', '.join(specs)}"
    )
    try:
        _run_install_with_heartbeat(
            install_cmd_prefix + ["install", "--reinstall", *specs], env=env
        )
    except subprocess.CalledProcessError as e:
        logger.warning("dep verification: per-package repair failed: %s", e)
        print(
            f"  ⚠ Could not install: {', '.join(still_missing)}. "
            "Run `pilotage update --force` after closing other pilotage processes."
        )
        return

    final_missing = _missing_deps()
    if final_missing:
        print(
            f"  ⚠ Still missing after repair: {', '.join(final_missing)}. "
            "Run `pilotage update --force` after closing other pilotage processes."
        )
    else:
        print("  ✓ All declared core dependencies now installed")


def _resolve_install_target_python(
    install_cmd_prefix: list[str], env: dict[str, str] | None
) -> Path | None:
    """Figure out which Python interpreter the install just targeted.

    ``_install_python_dependencies_with_optional_fallback`` is called with
    either ``[uv, pip]`` (and a ``VIRTUAL_ENV`` env var pointing at the
    target venv) or ``[sys.executable, -m, pip]`` (the in-process Python).
    The verification step needs the *resulting* environment's Python so
    ``importlib.metadata`` queries the right site-packages.
    """
    if env and "VIRTUAL_ENV" in env:
        from pilotage_constants import venv_python_path

        venv_root = Path(env["VIRTUAL_ENV"])
        candidate = venv_python_path(venv_root, windows=_is_windows())
        if candidate.exists():
            return candidate

    # Fallback: assume install_cmd_prefix[0] is the python interpreter (the
    # ``[sys.executable, -m, pip]`` shape). Skip if it looks like ``uv``.
    if install_cmd_prefix:
        first = Path(install_cmd_prefix[0])
        if first.exists() and "uv" not in first.name.lower():
            return first

    return None


def _is_termux_env(env: dict[str, str] | None = None) -> bool:
    return _is_termux_startup_environment(env)


class _UpdateOutputStream:
    """Stream wrapper used during ``pilotage update`` to survive terminal loss.

    Wraps the process's original stdout/stderr so that:

    * Every write is also mirrored to an append-only log file
      (``~/.pilotage/logs/update.log``) that users can inspect after the
      terminal disconnects.
    * Writes to the original stream that fail with ``BrokenPipeError`` /
      ``OSError`` / ``ValueError`` (closed file) no longer cascade into
      process exit — the update keeps going, only the on-screen output
      stops.

    Combined with ``SIGHUP -> SIG_IGN`` installed by
    ``_install_hangup_protection``, this makes ``pilotage update`` safe to
    run in a plain SSH session that might disconnect mid-install.
    """

    def __init__(self, original, log_file):
        self._original = original
        self._log = log_file
        self._original_broken = False

    def write(self, data):
        # Mirror to the log file first — it's the most reliable destination.
        if self._log is not None:
            try:
                self._log.write(data)
            except Exception:
                # Log errors should never abort the update.
                pass

        if self._original_broken:
            return len(data) if isinstance(data, (str, bytes)) else 0

        try:
            return self._original.write(data)
        except (BrokenPipeError, OSError, ValueError):
            # Terminal vanished (SSH disconnect, shell close).  Stop trying
            # to write to it, but keep the update running.
            self._original_broken = True
            return len(data) if isinstance(data, (str, bytes)) else 0

    def flush(self):
        if self._log is not None:
            try:
                self._log.flush()
            except Exception:
                pass
        if self._original_broken:
            return
        try:
            self._original.flush()
        except (BrokenPipeError, OSError, ValueError):
            self._original_broken = True

    def isatty(self):
        if self._original_broken:
            return False
        try:
            return self._original.isatty()
        except Exception:
            return False

    def fileno(self):
        # Some tools probe fileno(); defer to the underlying stream and let
        # callers handle failures (same behaviour as the unwrapped stream).
        return self._original.fileno()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _install_hangup_protection(gateway_mode: bool = False):
    """Protect ``cmd_update`` from SIGHUP and broken terminal pipes.

    Users commonly run ``pilotage update`` in an SSH session or a terminal
    that may close mid-install.  Without protection, ``SIGHUP`` from the
    terminal kills the Python process during ``pip install`` and leaves
    the venv half-installed; the documented workaround ("use screen /
    tmux") shouldn't be required for something as routine as an update.

    Protections installed:

    1. ``SIGHUP`` is set to ``SIG_IGN``.  POSIX preserves ``SIG_IGN``
       across ``exec()``, so pip and git subprocesses also stop dying on
       hangup.
    2. ``sys.stdout`` / ``sys.stderr`` are wrapped to mirror output to
       ``~/.pilotage/logs/update.log`` and to silently absorb
       ``BrokenPipeError`` when the terminal vanishes.

    ``SIGINT`` (Ctrl-C) and ``SIGTERM`` (systemd shutdown) are
    **intentionally left alone** — those are legitimate cancellation
    signals the user or OS sent on purpose.

    In gateway mode (``pilotage update --gateway``) the update is already
    spawned detached from a terminal, so this function is a no-op.

    Returns a dict that ``cmd_update`` can pass to
    ``_finalize_update_output`` on exit.  Returning a dict rather than a
    tuple keeps the call site forward-compatible with future additions.
    """
    state = {
        "prev_stdout": sys.stdout,
        "prev_stderr": sys.stderr,
        "log_file": None,
        "installed": False,
    }

    if gateway_mode:
        return state

    import signal as _signal

    # (1) Ignore SIGHUP for the remainder of this process.
    if hasattr(_signal, "SIGHUP"):
        try:
            _signal.signal(_signal.SIGHUP, _signal.SIG_IGN)
        except (ValueError, OSError):
            # Called from a non-main thread — not fatal.  The update still
            # runs, just without hangup protection.
            pass

    # (2) Mirror output to update.log and wrap stdio for broken-pipe
    # tolerance.  Any failure here is non-fatal; we just skip the wrap.
    try:
        # Late-bound import so tests can monkeypatch
        # pilotage_cli.config.get_pilotage_home to simulate setup failure.
        from pilotage_cli.config import get_pilotage_home as _get_pilotage_home

        logs_dir = _get_pilotage_home() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "update.log"
        log_file = open(log_path, "a", buffering=1, encoding="utf-8")

        import datetime as _dt

        log_file.write(
            f"\n=== pilotage update started "
            f"{_dt.datetime.now().isoformat(timespec='seconds')} ===\n"
        )

        state["log_file"] = log_file
        sys.stdout = _UpdateOutputStream(state["prev_stdout"], log_file)
        sys.stderr = _UpdateOutputStream(state["prev_stderr"], log_file)
        state["installed"] = True
    except Exception:
        # Leave stdio untouched on any setup failure.  Update continues
        # without mirroring.
        state["log_file"] = None

    return state


def _finalize_update_output(state):
    """Restore stdio and close the update.log handle opened by ``_install_hangup_protection``."""
    if not state:
        return
    if state.get("installed"):
        try:
            sys.stdout = state.get("prev_stdout", sys.stdout)
        except Exception:
            pass
        try:
            sys.stderr = state.get("prev_stderr", sys.stderr)
        except Exception:
            pass
    log_file = state.get("log_file")
    if log_file is not None:
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass


def _resolve_update_branch(args) -> str:
    """Normalize ``args.branch`` into a non-empty branch name.

    Centralizes the "default to main, accept --branch override, treat empty
    or whitespace-only values as the default" parsing so every consumer of
    ``--branch`` (check path, git-update path, ZIP-fallback path) agrees on
    the same answer.
    """
    return (getattr(args, "branch", None) or "main").strip() or "main"


def _size_delta_label(saved_mb: float) -> str:
    """Human label for a before/after database size delta, in MB.

    A negative delta means the file GREW — concurrent session writes during a
    long optimize can outweigh what the rebuild freed. Printing
    "reclaimed -163.0 MB" for that reads as data loss, so say "grew by"
    instead.
    """
    if saved_mb >= 0:
        return f"reclaimed {saved_mb:.1f} MB"
    return f"grew by {-saved_mb:.1f} MB"


def _coalesce_session_name_args(argv: list) -> list:
    """Join unquoted multi-word session names after -c/--continue and -r/--resume.

    When a user types ``pilotage -c Pokemon Agent Dev`` without quoting the
    session name, argparse sees three separate tokens.  This function merges
    them into a single argument so argparse receives
    ``['-c', 'Pokemon Agent Dev']`` instead.

    Tokens are collected after the flag until we hit another flag (``-*``)
    or a known top-level subcommand.
    """
    _SUBCOMMANDS = {
        "chat",
        "model",
        "gateway",
        "setup",
        "whatsapp",
        "whatsapp-cloud",
        "login",
        "logout",
        "auth",
        "status",
        "cron",
        "doctor",
        "config",
        "pairing",
        "skills",
        "tools",
        "sessions",
        "version",
        "uninstall",
        "profile",
        "honcho",
        "plugins",
        "security",
        "acp",
        "webhook",
        "memory",
        "dump",
        "debug",
        "backup",
        "import",
        "completion",
        "logs",
    }
    _SESSION_FLAGS = {"-c", "--continue", "-r", "--resume"}

    result = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _SESSION_FLAGS:
            result.append(token)
            i += 1
            # Collect subsequent non-flag, non-subcommand tokens as one name
            parts: list = []
            while (
                i < len(argv)
                and not argv[i].startswith("-")
                and argv[i] not in _SUBCOMMANDS
            ):
                parts.append(argv[i])
                i += 1
            if parts:
                result.append(" ".join(parts))
        else:
            result.append(token)
            i += 1
    return result


def cmd_profile(args):
    """Profile management — create, delete, list, switch, alias."""
    from pilotage_cli.profiles import (
        list_profiles,
        create_profile,
        delete_profile,
        set_active_profile,
        get_active_profile_name,
        check_alias_collision,
        create_wrapper_script,
        remove_wrapper_script,
        _is_wrapper_dir_in_path,
        _get_wrapper_dir,
    )
    from pilotage_constants import display_pilotage_home

    action = getattr(args, "profile_action", None)

    if action is None:
        # Bare `pilotage profile` — show current profile status
        profile_name = get_active_profile_name()
        dhh = display_pilotage_home()
        print(f"\nActive profile: {profile_name}")
        print(f"Path:           {dhh}")

        profiles = list_profiles()
        for p in profiles:
            if p.name == profile_name or (profile_name == "default" and p.is_default):
                if p.model:
                    print(
                        f"Model:          {p.model}"
                        + (f" ({p.provider})" if p.provider else "")
                    )
                print(
                    f"Gateway:        {'running' if p.gateway_running else 'stopped'}"
                )
                print(f"Skills:         {p.skill_count} installed")
                if p.alias_path:
                    alias_display = p.alias_name or p.name
                    print(f"Alias:          {alias_display} → pilotage -p {p.name}")
                break
        print()
        return

    if action == "list":
        profiles = list_profiles()
        active = get_active_profile_name()

        if not profiles:
            print("No profiles found.")
            return

        # Header
        print(
            f"\n {'Profile':<16} {'Model':<28} {'Gateway':<12} {'Alias'}"
        )
        print(
            f" {'─' * 15}    {'─' * 27}    {'─' * 11}    {'─' * 11}"
        )

        for p in profiles:
            marker = (
                " ◆"
                if (p.name == active or (active == "default" and p.is_default))
                else "  "
            )
            name = p.name
            model = (p.model or "—")[:26]
            gw = "running" if p.gateway_running else "stopped"
            alias = (p.alias_name or p.name) if p.alias_path else "—"
            if p.is_default:
                alias = "—"
            print(f"{marker}{name:<15} {model:<28} {gw:<12} {alias}")
        print()

    elif action == "use":
        name = args.profile_name
        try:
            set_active_profile(name)
            if name == "default":
                print("Switched to: default (~/.pilotage)")
            else:
                print(f"Switched to: {name}")
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "create":
        name = args.profile_name
        clone = getattr(args, "clone", False)
        clone_all = getattr(args, "clone_all", False)
        no_alias = getattr(args, "no_alias", False)
        no_skills = getattr(args, "no_skills", False)

        try:
            clone_from = getattr(args, "clone_from", None)
            clone_config = clone or clone_from is not None

            profile_dir = create_profile(
                name=name,
                clone_from=clone_from,
                clone_all=clone_all,
                clone_config=clone_config,
                no_alias=no_alias,
                no_skills=no_skills,
                description=getattr(args, "description", None),
            )
            print(f"\nProfile '{name}' created at {profile_dir}")

            if clone_config or clone_all:
                source_label = (
                    getattr(args, "clone_from", None) or get_active_profile_name()
                )
                if clone_all:
                    print(
                        f"Full copy from {source_label} "
                        "(excluding session history, backups, and snapshots)."
                    )
                else:
                    print(
                        f"Cloned config, .env, SOUL.md, and skills from {source_label}."
                    )

            # Auto-clone Honcho config for the new profile (only with clone operations)
            if clone_config or clone_all:
                try:
                    from plugins.memory.honcho.cli import clone_honcho_for_profile

                    if clone_honcho_for_profile(name):
                        print(f"Honcho config cloned (peer: {name})")
                except Exception:
                    pass  # Honcho plugin not installed or not configured


            # Create wrapper alias
            if not no_alias:
                collision = check_alias_collision(name)
                if collision:
                    print(f"\n⚠ Cannot create alias '{name}' — {collision}")
                    print(
                        f"  Choose a custom alias:  pilotage profile alias {name} --name <custom>"
                    )
                    print(f"  Or access via flag:     pilotage -p {name} chat")
                else:
                    wrapper_path = create_wrapper_script(name)
                    if wrapper_path:
                        print(f"Wrapper created: {wrapper_path}")
                        if not _is_wrapper_dir_in_path():
                            print(f"\n⚠ {_get_wrapper_dir()} is not in your PATH.")
                            print(
                                "  Add to your shell config (~/.bashrc or ~/.zshrc):"
                            )
                            print('    export PATH="$HOME/.local/bin:$PATH"')

            # Profile dir for display
            try:
                profile_dir_display = "~/" + str(profile_dir.relative_to(Path.home()))
            except ValueError:
                profile_dir_display = str(profile_dir)

            # Next steps
            print("\nNext steps:")
            print(f"  {name} setup              Configure API keys and model")
            print(f"  {name} chat               Start chatting")
            print(f"  {name} gateway start      Start the messaging gateway")
            if clone or clone_all:
                print(f"\n  Edit {profile_dir_display}/.env for different API keys")
                print(f"  Edit {profile_dir_display}/SOUL.md for different personality")
            else:
                print(
                    f"\n  ⚠ This profile has no API keys yet. Run '{name} setup' first,"
                )
                print("    or it will inherit keys from your shell environment.")
                print(f"  Edit {profile_dir_display}/SOUL.md to customize personality")
            print()

        except (ValueError, FileExistsError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "delete":
        name = args.profile_name
        yes = getattr(args, "yes", False)
        try:
            delete_profile(name, yes=yes)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "describe":
        # Read or write a profile's description. The description is
        # used to route tasks based on role instead of name alone.
        from pilotage_cli import profiles as _profiles_mod

        all_flag = bool(getattr(args, "all_missing", False))
        auto_flag = bool(getattr(args, "auto", False))
        overwrite_flag = bool(getattr(args, "overwrite", False))
        text_value = getattr(args, "text", None)
        name = getattr(args, "profile_name", None)

        if all_flag and not auto_flag:
            print("profile describe: --all requires --auto", file=sys.stderr)
            sys.exit(2)
        if all_flag and (text_value or name):
            print(
                "profile describe: --all is mutually exclusive with a profile name / --text",
                file=sys.stderr,
            )
            sys.exit(2)
        if not all_flag and not name:
            print("profile describe: profile name is required (or --all --auto)", file=sys.stderr)
            sys.exit(2)
        if text_value and auto_flag:
            print(
                "profile describe: --text is mutually exclusive with --auto",
                file=sys.stderr,
            )
            sys.exit(2)

        # Show current description if no operation requested.
        if name and not text_value and not auto_flag:
            try:
                if _profiles_mod.normalize_profile_name(name) == "default":
                    from pilotage_constants import get_pilotage_home as _hh
                    profile_dir = Path(_hh())
                else:
                    profile_dir = _profiles_mod.get_profile_dir(name)
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            if not profile_dir.is_dir():
                print(f"Error: profile '{name}' not found", file=sys.stderr)
                sys.exit(1)
            meta = _profiles_mod.read_profile_meta(profile_dir)
            desc = meta.get("description") or ""
            if not desc:
                print(f"(no description set for '{name}')")
            else:
                tag = "[auto] " if meta.get("description_auto") else ""
                print(f"{tag}{desc}")
            sys.exit(0)

        # --text path: just write the user-authored description.
        if text_value:
            try:
                if _profiles_mod.normalize_profile_name(name) == "default":
                    from pilotage_constants import get_pilotage_home as _hh
                    profile_dir = Path(_hh())
                else:
                    profile_dir = _profiles_mod.get_profile_dir(name)
                _profiles_mod.write_profile_meta(
                    profile_dir,
                    description=text_value,
                    description_auto=False,
                )
                print(f"Description updated for '{name}'.")
            except Exception as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            sys.exit(0)

        # --auto path: invoke the LLM describer.
        from pilotage_cli import profile_describer as _pd

        if all_flag:
            targets = _pd.list_describable_profiles(missing_only=True)
            if not targets:
                print("All profiles already have descriptions.")
                sys.exit(0)
        else:
            targets = [name]

        ok_count = 0
        fail_count = 0
        for tgt in targets:
            outcome = _pd.describe_profile(tgt, overwrite=overwrite_flag)
            if outcome.ok:
                ok_count += 1
                print(f"Described '{outcome.profile_name}': {outcome.description}")
            else:
                fail_count += 1
                print(
                    f"profile describe {outcome.profile_name}: {outcome.reason}",
                    file=sys.stderr,
                )
        if not all_flag:
            sys.exit(0 if ok_count == 1 else 1)
        sys.exit(0 if ok_count > 0 else 1)

    elif action == "show":
        name = args.profile_name
        from pilotage_cli.profiles import (
            get_profile_dir,
            profile_exists,
            _read_config_model,
            _check_gateway_running,
            _count_skills,
            _get_wrapper_dir,
            find_alias_for_profile,
        )

        if not profile_exists(name):
            print(f"Error: Profile '{name}' does not exist.")
            sys.exit(1)
        profile_dir = get_profile_dir(name)
        model, provider = _read_config_model(profile_dir)
        gw = _check_gateway_running(profile_dir)
        skills = _count_skills(profile_dir)
        alias_name = find_alias_for_profile(name)

        print(f"\nProfile: {name}")
        print(f"Path:    {profile_dir}")
        if model:
            print(f"Model:   {model}" + (f" ({provider})" if provider else ""))
        print(f"Gateway: {'running' if gw else 'stopped'}")
        print(f"Skills:  {skills}")
        print(
            f".env:    {'exists' if (profile_dir / '.env').exists() else 'not configured'}"
        )
        print(
            f"SOUL.md: {'exists' if (profile_dir / 'SOUL.md').exists() else 'not configured'}"
        )
        if alias_name:
            is_windows = sys.platform == "win32"
            wrapper = _get_wrapper_dir() / (f"{alias_name}.bat" if is_windows else alias_name)
            print(f"Alias:   {alias_name} → pilotage -p {name}  ({wrapper})")
        print()

    elif action == "alias":
        name = args.profile_name
        remove = getattr(args, "remove", False)
        custom_name = getattr(args, "alias_name", None)

        from pilotage_cli.profiles import profile_exists, validate_alias_name

        if not profile_exists(name):
            print(f"Error: Profile '{name}' does not exist.")
            sys.exit(1)

        alias_name = custom_name or name

        try:
            validate_alias_name(alias_name)
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

        if remove:
            if remove_wrapper_script(alias_name):
                print(f"✓ Removed alias '{alias_name}'")
            else:
                print(f"No alias '{alias_name}' found to remove.")
        else:
            collision = check_alias_collision(alias_name)
            if collision:
                print(f"Error: {collision}")
                sys.exit(1)
            wrapper_path = create_wrapper_script(
                alias_name, target=name if custom_name else None
            )
            if wrapper_path:
                print(f"✓ Alias created: {wrapper_path}")
                if not _is_wrapper_dir_in_path():
                    print(f"⚠ {_get_wrapper_dir()} is not in your PATH.")

    elif action == "rename":
        from pilotage_cli.profiles import rename_profile

        try:
            new_dir = rename_profile(args.old_name, args.new_name)
            print(f"\nProfile renamed: {args.old_name} → {args.new_name}")
            print(f"Path: {new_dir}\n")
        except (ValueError, FileExistsError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "export":
        from pilotage_cli.profiles import export_profile

        name = args.profile_name
        output = args.output or f"{name}.tar.gz"
        try:
            result_path = export_profile(name, output)
            print(f"✓ Exported '{name}' to {result_path}")
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif action == "import":
        from pilotage_cli.profiles import import_profile

        try:
            profile_dir = import_profile(
                args.archive, name=getattr(args, "import_name", None)
            )
            name = profile_dir.name
            print(f"✓ Imported profile '{name}' at {profile_dir}")

            # Offer to create alias
            collision = check_alias_collision(name)
            if not collision:
                wrapper_path = create_wrapper_script(name)
                if wrapper_path:
                    print(f"  Wrapper created: {wrapper_path}")
            print()
        except (ValueError, FileExistsError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)


def cmd_completion(args, parser=None):
    """Print shell completion script."""
    from pilotage_cli.completion import generate_bash, generate_zsh, generate_fish

    shell = getattr(args, "shell", "bash")
    if shell == "zsh":
        print(generate_zsh(parser))
    elif shell == "fish":
        print(generate_fish(parser))
    else:
        print(generate_bash(parser))


def cmd_prompt_size(args):
    """Show a byte/char breakdown of the system prompt + tool schemas."""
    from pilotage_cli.prompt_size import cmd_prompt_size as _impl

    _impl(args)


def cmd_logs(args):
    """View and filter Pilotage log files."""
    from pilotage_cli.logs import tail_log, list_logs

    log_name = getattr(args, "log_name", "agent") or "agent"

    if log_name == "list":
        list_logs()
        return

    tail_log(
        log_name,
        num_lines=getattr(args, "lines", 50),
        follow=getattr(args, "follow", False),
        level=getattr(args, "level", None),
        session=getattr(args, "session", None),
        since=getattr(args, "since", None),
        component=getattr(args, "component", None),
    )


def _build_provider_choices() -> list[str]:
    """Build the --provider choices list from CANONICAL_PROVIDERS + 'auto'."""
    try:
        from pilotage_cli.models import CANONICAL_PROVIDERS as _cp
        return ["auto"] + [p.slug for p in _cp]
    except Exception:
        # Fallback: static list guarantees the CLI always works
        return [
            "auto", "openai-codex", "openai-api", "custom",
        ]


# Top-level subcommands that argparse knows about WITHOUT running plugin
# discovery.  Used to short-circuit eager plugin imports (which can take
# 500ms+ pulling in google.cloud.pubsub_v1, aiohttp, grpc, etc.) when the
# user's invocation clearly doesn't need any plugin-registered subcommand.
#
# Keep this in sync with the ``subparsers.add_parser("NAME", ...)`` calls
# below in ``main()``. Missing an entry here only costs a one-time
# discovery; extra entries here would let a plugin command silently fail
# to parse.
_BUILTIN_SUBCOMMANDS = frozenset(
    {
        "approvals", "auth", "backup", "bundles", "checkpoints", "completion",
        "config", "cron", "debug", "doctor",
        "dump", "fallback", "gateway", "hooks", "import",
        "login", "logout", "logs", "memory",
        "model", "pairing", "pause", "plugins", "profile",
        "prompt-size",
        "resume",
        "send", "sessions", "setup",
        "skills", "status", "tools", "uninstall",
        "version", "webhook", "whatsapp", "whatsapp-cloud", "chat", "security",
        # Help-ish invocations — plugin commands not being listed in
        # top-level --help is an acceptable trade-off for skipping an
        # expensive eager import of every bundled plugin module.
        "help",
    }
)


# Top-level flags that take a value. Needed by ``_first_positional_argv``
# so that in ``pilotage -m gpt5 chat``, ``gpt5`` is correctly skipped as a
# flag value rather than misclassified as a subcommand. Kept in sync with
# the top-level flags declared in ``pilotage_cli/_parser.py``.
#
# Correctness-safe either way: missing an entry here only makes the
# fast-path bail out too eagerly (we run plugin discovery when we didn't
# need to); extra entries would make us skip a real positional.
_TOP_LEVEL_VALUE_FLAGS = frozenset(
    {
        "-z", "--oneshot",
        "-m", "--model",
        "--provider",
        "-t", "--toolsets",
        "-r", "--resume",
        "-s", "--skills",
        "--usage-file",
        "--in",
        # ``-c / --continue`` is nargs='?' (optional value). Treat it as
        # value-taking: if the next token is a subcommand-looking word
        # the user almost certainly meant it as the session name, and
        # either interpretation keeps us on the safe side.
        "-c", "--continue",
    }
)


def _first_positional_argv() -> str | None:
    """Return the first non-flag, non-flag-value token in ``sys.argv[1:]``.

    Used by ``main()`` to decide whether plugin discovery has to run at
    argparse-setup time. Handles common invocations like
    ``pilotage -m gpt5 --provider openai chat "msg"`` by skipping the
    values attached to known top-level flags.

    Does NOT fully simulate argparse — unknown ``--foo=bar`` / ``--foo
    bar`` flags degrade gracefully (``bar`` may be wrongly classified as
    a positional, which at worst forces a one-time plugin discovery).
    """
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--":
            # Everything after ``--`` is positional.
            if i + 1 < len(argv):
                return argv[i + 1]
            return None
        if tok.startswith("-"):
            # ``--flag=value`` carries its value inline — single token.
            if "=" in tok:
                i += 1
                continue
            if tok in _TOP_LEVEL_VALUE_FLAGS and i + 1 < len(argv):
                i += 2
                continue
            i += 1
            continue
        return tok
    return None


def _plugin_cli_discovery_needed() -> bool:
    """True when the CLI might be invoking a plugin-registered subcommand.

    Returning False lets ``main()`` skip plugin discovery entirely during
    argparse setup, saving ~500-650ms per invocation for users whose
    enabled plugins don't contribute any CLI command.
    """
    first = _first_positional_argv()
    if first is None:
        # Bare ``pilotage`` or only flags → defaults to ``chat``.
        return False
    if first in _BUILTIN_SUBCOMMANDS:
        return False
    # Unknown token — could be a plugin subcommand, OR a chat prompt
    # starting with a non-flag word. Either way we need discovery: if it
    # IS a plugin command, argparse needs the subparser; if it's a chat
    # prompt, argparse will route it via positional handling and the
    # extra discovery cost is amortized over a full agent run anyway.
    return True


def _resolve_deferred_platform_cli_command(command_name: str | None) -> None:
    """Materialize the deferred platform whose top-level CLI command matches.

    Bundled platform plugins are cheap-registered as *deferred* entries to
    avoid importing every gateway SDK during normal startup. A platform that
    registers a top-level ``pilotage <name>`` command (e.g. Photon ->
    ``ctx.register_cli_command(name="photon", ...)``) only runs that side
    effect when its module is imported. On the unknown-top-level-command slow
    path, ``discover_plugins()`` records the deferred loader but does not
    import it, so the CLI registration never happens and ``pilotage photon``
    fails with argparse ``invalid choice``.

    Resolving only the platform whose name matches the first positional token
    keeps normal startup cheap while making the targeted command available.
    """
    if not command_name:
        return
    try:
        from gateway.platform_registry import platform_registry

        platform_registry.get(command_name)
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Deferred platform CLI resolution failed for %s: %s",
            command_name,
            exc,
        )


_AGENT_COMMANDS = {None, "chat", "rl"}
_AGENT_SUBCOMMANDS = {
    "cron": ("cron_command", {"run", "tick"}),
    "gateway": ("gateway_command", {"run"}),
}


def _prepare_agent_startup(args) -> None:
    """Discover plugins/MCP/hooks for commands that can run an agent turn."""
    # --yolo: chokepoint guarantee that PILOTAGE_YOLO_MODE is set before ANY
    # plugin/tool discovery below imports tools.approval, which freezes
    # _YOLO_MODE_FROZEN at import time ( security design). main's
    # dispatch path also sets this earlier, but _prepare_agent_startup() is
    # reachable from other launchers too (e.g. the Termux fast-CLI path),
    # so the guarantee lives here where the import is actually triggered
    #.
    if getattr(args, "yolo", False):
        os.environ["PILOTAGE_YOLO_MODE"] = "1"
    _apply_safe_mode(args)

    _sub_attr, _sub_set = _AGENT_SUBCOMMANDS.get(args.command, (None, None))
    if not (
        args.command in _AGENT_COMMANDS
        or (_sub_attr and getattr(args, _sub_attr, None) in _sub_set)
    ):
        return

    _accept_hooks = bool(getattr(args, "accept_hooks", False))
    try:
        from pilotage_cli.plugins import start_background_plugin_discovery

        # Discovery runs in a daemon thread so its ~150ms of manifest
        # scanning + plugin imports overlaps the rest of startup (cli /
        # prompt_toolkit imports, worktree git calls). Correctness is
        # unchanged: every synchronous reader goes through
        # discover_plugins(), which joins this thread first — including
        # the discover_plugins() call model_tools makes at import time,
        # which happens before any tool list is built.
        start_background_plugin_discovery()
    except Exception:
        logger.warning(
            "plugin discovery failed at CLI startup",
            exc_info=True,
        )
    try:
        from pilotage_cli.config import load_config
        from agent.shell_hooks import register_from_config

        _hooks_cfg = load_config()
        register_from_config(_hooks_cfg, accept_hooks=_accept_hooks)

        from agent.outbound_webhooks import (
            register_from_config as register_outbound_webhooks,
        )

        register_outbound_webhooks(_hooks_cfg)
    except Exception:
        logger.debug(
            "shell-hook registration failed at CLI startup",
            exc_info=True,
        )


def _apply_safe_mode(args) -> None:
    if not getattr(args, "safe_mode", False):
        return
    os.environ["PILOTAGE_SAFE_MODE"] = "1"
    os.environ["PILOTAGE_IGNORE_USER_CONFIG"] = "1"
    os.environ["PILOTAGE_IGNORE_RULES"] = "1"


def _set_chat_arg_defaults(args) -> None:
    for attr, default in [
        ("query", None),
        ("model", None),
        ("provider", None),
        ("toolsets", None),
        ("verbose", False),
        ("resume", None),
        ("continue_last", None),
        ("worktree", False),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, default)


def _try_fast_chat_launch() -> bool:
    """Fast path for unambiguous interactive chat launches (all hosts).

    ``pilotage`` / ``pilotage -w -s foo --yolo`` / ``pilotage chat`` don't need the
    full argparse tree: building all ~40 subcommand parsers costs ~140ms of
    pure-Python argparse setup plus their module imports, none of which the
    chat path uses. Parse the lightweight top-level/chat parser instead and
    dispatch straight to ``cmd_chat``.

    Bails out (returns False) whenever the invocation is not certainly a
    chat launch — a subcommand positional, ``--help``, unknown flags — so
    every other path still goes through the full parser unchanged. Mirrors
    ``_try_termux_fast_cli_launch`` minus the Termux-specific deferred
    startup; kept separate so phone-tuned behavior doesn't leak to desktops.
    """
    if os.environ.get("PILOTAGE_DISABLE_FAST_CHAT_LAUNCH") == "1":
        return False
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        return False
    if _first_positional_argv() not in {None, "chat"}:
        return False

    from pilotage_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)
    try:
        args, unknown = parser.parse_known_args(_coalesce_session_name_args(argv))
    except SystemExit:
        return False
    if unknown:
        # Flags the light parser doesn't know — could belong to a plugin
        # subcommand or a newer full-parser flag. Fall back to full dispatch.
        return False
    if getattr(args, "version", False):
        return False
    if getattr(args, "command", None) not in {None, "chat"}:
        return False

    if getattr(args, "yolo", False):
        os.environ["PILOTAGE_YOLO_MODE"] = "1"
    _prepare_agent_startup(args)

    if getattr(args, "oneshot", None):
        _confirm_startup_expensive_model_override(args)
        _run_and_exit_oneshot(
            args.oneshot,
            model=getattr(args, "model", None),
            provider=getattr(args, "provider", None),
            toolsets=getattr(args, "toolsets", None),
            usage_file=getattr(args, "usage_file", None),
        )

    if (args.resume or args.continue_last) and args.command is None:
        args.command = "chat"

    _set_chat_arg_defaults(args)
    cmd_chat(args)
    return True


def _try_termux_fast_cli_launch() -> bool:
    """Run obvious Termux non-TUI chat/oneshot/version paths on a light parser."""
    if not _is_termux_startup_environment():
        return False
    if os.environ.get("PILOTAGE_TERMUX_DISABLE_FAST_CLI") == "1":
        return False

    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        return False
    if _is_termux_fast_version_argv(argv):
        _print_version_info(check_updates=False)
        return True

    first = _first_positional_argv()
    has_oneshot = any(
        arg == "-z" or arg == "--oneshot" or arg.startswith("--oneshot=")
        for arg in argv
    )
    if not has_oneshot and first not in {None, "chat"}:
        return False

    from pilotage_cli._parser import build_top_level_parser

    parser, _subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)
    args = parser.parse_args(_coalesce_session_name_args(argv))

    if getattr(args, "version", False):
        _print_version_info(check_updates=False)
        return True

    if getattr(args, "oneshot", None):
        _prepare_agent_startup(args)
        _confirm_startup_expensive_model_override(args)
        _run_and_exit_oneshot(
            args.oneshot,
            model=getattr(args, "model", None),
            provider=getattr(args, "provider", None),
            toolsets=getattr(args, "toolsets", None),
            usage_file=getattr(args, "usage_file", None),
        )

    if (args.resume or args.continue_last) and args.command is None:
        args.command = "chat"

    if args.command in {None, "chat"}:
        _set_chat_arg_defaults(args)
        interactive_prompt = not getattr(args, "query", None) and not getattr(args, "image", None)
        if interactive_prompt:
            # Bare Termux CLI should reach the prompt first and do agent-only
            # discovery on the first submitted turn instead of before input.
            setattr(args, "compact", True)
            os.environ["PILOTAGE_DEFER_AGENT_STARTUP"] = "1"
            os.environ["PILOTAGE_FAST_STARTUP_BANNER"] = "1"
            if getattr(args, "accept_hooks", False):
                os.environ["PILOTAGE_ACCEPT_HOOKS"] = "1"
        else:
            _prepare_agent_startup(args)
        cmd_chat(args)
        return True

    return False


def cmd_memory(args):
    sub = getattr(args, "memory_command", None)
    if sub == "off":
        from pilotage_cli.config import load_config, save_config

        config = load_config()
        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = ""
        save_config(config)
        print("\n  ✓ Memory provider: built-in only")
        print("  Saved to config.yaml\n")
    elif sub == "reset":
        from pilotage_constants import get_pilotage_home, display_pilotage_home

        mem_dir = get_pilotage_home() / "memories"
        target = getattr(args, "target", "all")
        files_to_reset = []
        if target in {"all", "memory"}:
            files_to_reset.append(("MEMORY.md", "agent notes"))
        if target in {"all", "user"}:
            files_to_reset.append(("USER.md", "user profile"))

        # Check what exists
        existing = [
            (f, desc) for f, desc in files_to_reset if (mem_dir / f).exists()
        ]
        if not existing:
            print(
                f"\n  Nothing to reset — no memory files found in {display_pilotage_home()}/memories/\n"
            )
            return

        print("\n  This will permanently erase the following memory files:")
        for f, desc in existing:
            path = mem_dir / f
            size = path.stat().st_size
            print(f"    ◆ {f} ({desc}) — {size:,} bytes")

        if not getattr(args, "yes", False):
            try:
                answer = input("\n  Type 'yes' to confirm: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.\n")
                return
            if answer != "yes":
                print("  Cancelled.\n")
                return

        for f, desc in existing:
            (mem_dir / f).unlink()
            print(f"  ✓ Deleted {f} ({desc})")

        print(
            "\n  Memory reset complete. New sessions will start with a blank slate."
        )
        print(f"  Files were in: {display_pilotage_home()}/memories/\n")
    else:
        from pilotage_cli.memory_setup import memory_command

        memory_command(args)


def cmd_tools(args):
    action = getattr(args, "tools_action", None)
    if action in {"list", "disable", "enable"}:
        from pilotage_cli.tools_config import tools_disable_enable_command

        tools_disable_enable_command(args)
    elif action == "post-setup":
        from pilotage_cli.tools_config import run_post_setup_command

        sys.exit(run_post_setup_command(args))
    else:
        _require_tty("tools")
        from pilotage_cli.tools_config import tools_command

        tools_command(args)


def cmd_pairing(args):
    from pilotage_cli.pairing import pairing_command

    pairing_command(args)


def cmd_plugins(args):
    from pilotage_cli.plugins_cmd import plugins_command

    plugins_command(args)


def _advertise_agent_env() -> None:
    """Advertise the agent harness to child processes.

    ``AI_AGENT`` is the emerging cross-agent standard (huggingface_hub's agent
    detection reads it; pi and other agents set it — earendil-works/pi)
    so generic tooling can attribute subprocesses to the harness that spawned
    them. The value must be our id in the public agent-harness registry
    (``pilotage-agent`` in huggingface.js ``agent-harnesses.ts``): standard-var
    matching is exact, so any other value is counted as "unknown".
    ``PILOTAGE_AGENT`` is the Pilotage-specific marker. setdefault: never
    clobber an outer harness (e.g. Pilotage running inside another agent's
    terminal).
    """
    os.environ.setdefault("AI_AGENT", "pilotage-agent")
    os.environ.setdefault("PILOTAGE_AGENT", "true")


def main():
    """Main entry point for pilotage CLI."""
    # Cosmetic: make the process show up as 'pilotage' instead of 'python3.11'
    # in ps/top/htop.  Non-fatal — just a nicer UX.
    _set_process_title()

    # Let child processes (and tools like huggingface_hub) detect they run
    # under an AI agent harness.
    _advertise_agent_env()

    # Force UTF-8 stdio on Windows before anything prints.  No-op elsewhere.
    try:
        from pilotage_cli.stdio import configure_windows_stdio
        configure_windows_stdio()
    except Exception:
        pass

    # Sweep stale ``pilotage.exe.old.*`` quarantine files left by previous
    # ``pilotage update`` runs on Windows. Silent no-op on non-Windows or when
    # there's nothing to clean. See ``_quarantine_running_pilotage_exe``.
    try:
        _cleanup_quarantined_exes()
    except Exception:
        pass

    # If the checkout changed since the last launch (pilotage update, manual
    # git pull, old-updater update that predates newer clears), sweep stale
    # __pycache__ once so no process — this one's lazy imports included —
    # resolves fresh source against old bytecode. Never raises.
    _sweep_stale_bytecode_if_checkout_changed()

    # Self-heal a venv left half-built by an interrupted ``pilotage update``
    # (Ctrl-C, terminal close, WSL OOM mid-install). Skip when the user is
    # *running* update — that flow writes and clears its own marker, and we
    # don't want a recovery install racing the real one. Never raises.
    #
    # The substring match is deliberately loose: argv isn't parsed yet at this
    # point, and the failure modes are asymmetric. Over-matching (e.g.
    # ``pilotage skills install update``) merely defers recovery one launch;
    # under-matching (missing ``pilotage -p work update``) would race a recovery
    # install against the real one. Loose wins.
    try:
        if "update" not in sys.argv[1:]:
            _recover_from_interrupted_install()
    except Exception:
        pass

    if _try_termux_fast_cli_launch():
        return
    if _try_fast_chat_launch():
        return

    from pilotage_cli._parser import build_top_level_parser

    parser, subparsers, chat_parser = build_top_level_parser()
    chat_parser.set_defaults(func=cmd_chat)

    # =========================================================================
    # model command  (parser built in pilotage_cli/subcommands/model.py)
    # =========================================================================
    build_model_parser(subparsers, cmd_model=cmd_model)

    # =========================================================================
    # fallback command — manage the fallback provider chain
    # =========================================================================
    from pilotage_cli.fallback_cmd import cmd_fallback

    fallback_parser = subparsers.add_parser(
        "fallback",
        help="Manage fallback providers (tried when the primary model fails)",
        description=(
            "Manage the fallback provider chain.  Fallback providers are tried "
            "in order when the primary model fails with rate-limit, overload, or "
            "connection errors.  See: "
            ""
        ),
    )
    fallback_subparsers = fallback_parser.add_subparsers(dest="fallback_command")
    fallback_subparsers.add_parser(
        "list",
        aliases=["ls"],
        help="Show the current fallback chain (default when no subcommand)",
    )
    fallback_subparsers.add_parser(
        "add",
        help="Pick a provider + model (same picker as `pilotage model`) and append to the chain",
    )
    fallback_subparsers.add_parser(
        "remove",
        aliases=["rm"],
        help="Pick an entry to delete from the chain",
    )
    fallback_subparsers.add_parser(
        "clear",
        help="Remove all fallback entries",
    )
    fallback_parser.set_defaults(func=cmd_fallback)

    # =========================================================================
    # secrets command — external secret managers (Bitwarden, 1Password)
    # =========================================================================
    # =========================================================================
    # gateway command  (parser built in pilotage_cli/subcommands/gateway.py)
    # =========================================================================
    build_gateway_parser(subparsers, cmd_gateway=cmd_gateway)

    # =========================================================================
    # lsp command
    # =========================================================================

    # =========================================================================
    # setup command  (parser built in pilotage_cli/subcommands/setup.py)
    # =========================================================================
    build_setup_parser(subparsers, cmd_setup=cmd_setup)


    # =========================================================================
    # whatsapp command  (parser built in pilotage_cli/subcommands/whatsapp.py)
    # =========================================================================
    build_whatsapp_parser(subparsers, cmd_whatsapp=cmd_whatsapp)

    # =========================================================================
    # whatsapp-cloud command (official Meta Cloud API; complement to Baileys)
    # =========================================================================
    whatsapp_cloud_parser = subparsers.add_parser(
        "whatsapp-cloud",
        help="Set up WhatsApp Business Cloud API integration",
        description=(
            "Configure the official Meta WhatsApp Business Cloud API "
            "adapter (Business account required, public webhook URL "
            "required). Distinct from `pilotage whatsapp` which sets up "
            "the Baileys bridge for personal accounts."
        ),
    )
    whatsapp_cloud_parser.set_defaults(func=cmd_whatsapp_cloud)

    # =========================================================================
    # send command — pipe shell-script output to any configured platform
    # =========================================================================
    from pilotage_cli.send_cmd import register_send_subparser
    register_send_subparser(subparsers)

    # =========================================================================
    # login command  (parser built in pilotage_cli/subcommands/login.py)
    # =========================================================================
    build_login_parser(subparsers, cmd_login=cmd_login)

    # =========================================================================
    # logout command  (parser built in pilotage_cli/subcommands/logout.py)
    # =========================================================================
    build_logout_parser(subparsers, cmd_logout=cmd_logout)

    # =========================================================================
    # auth command  (parser built in pilotage_cli/subcommands/auth.py)
    # =========================================================================
    build_auth_parser(subparsers, cmd_auth=cmd_auth)

    # =========================================================================
    # status command  (parser built in pilotage_cli/subcommands/status.py)
    # =========================================================================
    build_status_parser(subparsers, cmd_status=cmd_status)

    # =========================================================================
    # pause / resume commands  (parser built in pilotage_cli/subcommands/pause.py)
    # =========================================================================
    build_pause_parser(subparsers)

    # =========================================================================
    # cron command  (parser built in pilotage_cli/subcommands/cron.py)
    # =========================================================================
    build_cron_parser(subparsers, cmd_cron=cmd_cron)

    # =========================================================================
    # webhook command  (parser built in pilotage_cli/subcommands/webhook.py)
    # =========================================================================
    build_webhook_parser(subparsers, cmd_webhook=cmd_webhook)

    # =========================================================================
    # hooks command — shell-hook inspection and management
    # =========================================================================
    # hooks command  (parser built in pilotage_cli/subcommands/hooks.py)
    # =========================================================================
    build_hooks_parser(subparsers, cmd_hooks=cmd_hooks)

    # =========================================================================
    # doctor command  (parser built in pilotage_cli/subcommands/doctor.py)
    # =========================================================================
    build_doctor_parser(subparsers, cmd_doctor=cmd_doctor)

    # =========================================================================
    # verify command  (parser built in pilotage_cli/subcommands/verify.py)
    # =========================================================================

    # =========================================================================
    # security command — on-demand supply-chain audit
    # =========================================================================
    # security command  (parser built in pilotage_cli/subcommands/security.py)
    # =========================================================================
    build_security_parser(subparsers, cmd_security=cmd_security)

    # =========================================================================
    # approvals command  (parser built in pilotage_cli/subcommands/approvals.py)
    # =========================================================================
    build_approvals_parser(subparsers, cmd_approvals=cmd_approvals)

    # =========================================================================
    # dump command  (parser built in pilotage_cli/subcommands/dump.py)
    # =========================================================================
    build_dump_parser(subparsers, cmd_dump=cmd_dump)

    # =========================================================================
    # debug command  (parser built in pilotage_cli/subcommands/debug.py)
    # =========================================================================
    build_debug_parser(subparsers, cmd_debug=cmd_debug)

    # =========================================================================
    # backup command  (parser built in pilotage_cli/subcommands/backup.py)
    # =========================================================================
    build_backup_parser(subparsers, cmd_backup=cmd_backup)

    # =========================================================================
    # checkpoints command
    # =========================================================================
    checkpoints_parser = subparsers.add_parser(
        "checkpoints",
        help="Inspect / prune / clear ~/.pilotage/checkpoints/",
        description="Manage the filesystem checkpoint store — the shadow git "
        "repo pilotage uses to snapshot working directories before "
        "write_file/patch/terminal calls. Lets you see how much "
        "space checkpoints occupy, force a prune, or wipe the base.",
    )
    from pilotage_cli.checkpoints import register_cli as _register_checkpoints_cli
    _register_checkpoints_cli(checkpoints_parser)

    # =========================================================================
    # import command  (parser built in pilotage_cli/subcommands/import_cmd.py)
    # =========================================================================
    build_import_cmd_parser(subparsers, cmd_import=cmd_import)

    # =========================================================================
    # config command  (parser built in pilotage_cli/subcommands/config.py)
    # =========================================================================
    build_config_parser(subparsers, cmd_config=cmd_config)

    # =========================================================================
    # skin command  (parser built in pilotage_cli/subcommands/skin.py)
    # =========================================================================

    # =========================================================================
    # pairing command  (parser built in pilotage_cli/subcommands/pairing.py)
    # =========================================================================
    build_pairing_parser(subparsers, cmd_pairing=cmd_pairing)

    # =========================================================================
    # skills command  (parser built in pilotage_cli/subcommands/skills.py)
    # =========================================================================
    build_skills_parser(subparsers)

    # =========================================================================
    # bundles command — skill bundles (alias /<name> for multiple skills)
    # =========================================================================
    bundles_parser = subparsers.add_parser(
        "bundles",
        help="Create, list, and manage skill bundles (aliases for multiple skills)",
        description=(
            "Skill bundles let you load several skills under one slash "
            "command. `/<bundle>` from the CLI or gateway loads every "
            "referenced skill at once."
        ),
    )
    from pilotage_cli.bundles import register_cli as _bundles_register, bundles_command
    _bundles_register(bundles_parser)
    bundles_parser.set_defaults(func=bundles_command)

    # =========================================================================
    # plugins command  (parser built in pilotage_cli/subcommands/plugins.py)
    # =========================================================================
    build_plugins_parser(subparsers, cmd_plugins=cmd_plugins)

    # =========================================================================
    # Plugin CLI commands — dynamically registered by memory/general plugins.
    # Plugins provide a register_cli(subparser) function that builds their
    # own argparse tree.  No hardcoded plugin commands in main.py.
    #
    # Skipped when the invocation is already targeting a known built-in
    # subcommand — ``pilotage --help``, ``pilotage version``, ``pilotage logs``,
    # etc.  This avoids eagerly importing every bundled plugin module
    # (google.cloud.pubsub_v1, aiohttp, grpc, PIL …) which costs
    # 500-650ms on typical installs.
    # =========================================================================
    if _plugin_cli_discovery_needed():
        try:
            from plugins.memory import discover_plugin_cli_commands
            from pilotage_cli.plugins import discover_plugins, get_plugin_manager

            seen_plugin_commands = set()
            for cmd_info in discover_plugin_cli_commands():
                plugin_parser = subparsers.add_parser(
                    cmd_info["name"],
                    help=cmd_info["help"],
                    description=cmd_info.get("description", ""),
                    formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
                )
                cmd_info["setup_fn"](plugin_parser)
                if cmd_info.get("handler_fn") is not None:
                    plugin_parser.set_defaults(func=cmd_info["handler_fn"])
                seen_plugin_commands.add(cmd_info["name"])

            discover_plugins()
            # A bundled platform whose top-level CLI command is the one being
            # invoked is still only a deferred entry at this point; import it
            # so its register_cli_command side effect runs before we read
            # _cli_commands.
            _resolve_deferred_platform_cli_command(_first_positional_argv())
            for cmd_info in get_plugin_manager()._cli_commands.values():
                if cmd_info["name"] in seen_plugin_commands:
                    continue
                plugin_parser = subparsers.add_parser(
                    cmd_info["name"],
                    help=cmd_info["help"],
                    description=cmd_info.get("description", ""),
                    formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
                )
                cmd_info["setup_fn"](plugin_parser)
                if cmd_info.get("handler_fn") is not None:
                    plugin_parser.set_defaults(func=cmd_info["handler_fn"])
        except Exception as _exc:
            logging.getLogger(__name__).debug("Plugin CLI discovery failed: %s", _exc)

    # =========================================================================
    # memory command  (parser built in pilotage_cli/subcommands/memory.py)
    # =========================================================================
    build_memory_parser(subparsers, cmd_memory=cmd_memory)

    # =========================================================================
    # tools command  (parser built in pilotage_cli/subcommands/tools.py)
    # =========================================================================
    build_tools_parser(subparsers, cmd_tools=cmd_tools)

    # =========================================================================
    # mcp command  (parser built in pilotage_cli/subcommands/mcp.py)
    # =========================================================================

    # =========================================================================
    # sessions command
    # =========================================================================
    sessions_parser = subparsers.add_parser(
        "sessions",
        help="Manage session history (list, rename, export, prune, delete)",
        description="View and manage the SQLite session store",
    )
    sessions_subparsers = sessions_parser.add_subparsers(dest="sessions_action")

    sessions_list = sessions_subparsers.add_parser("list", help="List recent sessions")
    sessions_list.add_argument(
        "--source", help="Filter by source (cli, telegram, discord, etc.)"
    )
    sessions_list.add_argument(
        "--limit", type=int, default=20, help="Max sessions to show"
    )
    sessions_list.add_argument(
        "--workspace",
        metavar="NEEDLE",
        help="Only sessions in one workspace: a git repo root or project dir "
        "(matched by path substring or basename).",
    )

    def _add_session_filter_args(p, default_older_help):
        p.add_argument(
            "--older-than",
            metavar="AGE",
            help=default_older_help,
        )
        p.add_argument(
            "--newer-than",
            metavar="AGE",
            help="Only match sessions active within the last AGE "
            "(e.g. '5h', '2d') or after an ISO timestamp",
        )
        p.add_argument(
            "--before",
            metavar="TIME",
            help="Only match sessions started before TIME "
            "(duration ago like '5h', or ISO timestamp like '2026-07-05 14:30')",
        )
        p.add_argument(
            "--after",
            metavar="TIME",
            help="Only match sessions started at/after TIME "
            "(duration ago like '5h', or ISO timestamp)",
        )
        p.add_argument("--source", help="Only match sessions from this source")
        p.add_argument(
            "--title", help="Only match sessions whose title contains this substring"
        )
        p.add_argument(
            "--end-reason", help="Only match sessions with this end reason"
        )
        p.add_argument(
            "--cwd", help="Only match sessions whose working directory is under this path"
        )
        p.add_argument(
            "--min-messages", type=int, help="Only match sessions with >= N messages"
        )
        p.add_argument(
            "--max-messages", type=int, help="Only match sessions with <= N messages"
        )
        p.add_argument(
            "--model",
            help="Only match sessions whose model name contains this substring "
            "(e.g. 'sonnet', 'gpt-5', 'pilotage')",
        )
        p.add_argument(
            "--provider",
            help="Only match sessions billed through this provider "
            "(e.g. openai-codex, openai-api, custom)",
        )
        p.add_argument(
            "--user", help="Only match sessions from this user ID"
        )
        p.add_argument(
            "--chat-id", help="Only match sessions from this chat/channel ID"
        )
        p.add_argument(
            "--chat-type",
            help="Only match sessions with this chat type (e.g. dm, group)",
        )
        p.add_argument(
            "--branch",
            help="Only match sessions whose git branch contains this substring",
        )
        p.add_argument(
            "--min-tokens", type=int,
            help="Only match sessions with >= N total tokens (input+output)",
        )
        p.add_argument(
            "--max-tokens", type=int,
            help="Only match sessions with <= N total tokens (input+output)",
        )
        p.add_argument(
            "--min-cost", type=float,
            help="Only match sessions costing >= N USD (actual or estimated)",
        )
        p.add_argument(
            "--max-cost", type=float,
            help="Only match sessions costing <= N USD (actual or estimated)",
        )
        p.add_argument(
            "--min-tool-calls", type=int,
            help="Only match sessions with >= N tool calls",
        )
        p.add_argument(
            "--max-tool-calls", type=int,
            help="Only match sessions with <= N tool calls",
        )
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="List matching sessions without changing anything",
        )
        p.add_argument(
            "--yes", "-y", action="store_true", help="Skip confirmation"
        )

    sessions_export = sessions_subparsers.add_parser(
        "export", help="Export sessions to JSONL, Markdown, or QMD"
    )
    sessions_export.add_argument(
        "output",
        nargs="?",
        help=(
            "Output path. JSONL: file path (use - for stdout, required). "
            "md/qmd: output directory (default: <pilotage home>/session-exports)"
        ),
    )
    sessions_export.add_argument(
        "--format",
        choices=["jsonl", "md", "qmd", "html", "trace"],
        default="jsonl",
        help=(
            "Export format (default: jsonl). 'trace' emits Claude Code JSONL "
            "for the Hugging Face Agent Trace Viewer"
        ),
    )
    sessions_export.add_argument(
        "--upload",
        action="store_true",
        help=(
            "trace only: upload to your Hugging Face traces dataset instead "
            "of writing a local file (needs HF_TOKEN)"
        ),
    )
    sessions_export.add_argument(
        "--public",
        action="store_true",
        help="trace --upload only: create/update a public dataset instead of private",
    )
    sessions_export.add_argument(
        "--no-redact",
        action="store_true",
        help=(
            "trace only: skip the forced secret redaction; "
            "only use after manual review"
        ),
    )
    sessions_export.add_argument(
        "--only",
        choices=["user-prompts"],
        help=(
            "Export only a filtered view (user-prompts: one prompt record "
            "per line for jsonl, headed sections for md)"
        ),
    )
    sessions_export.add_argument(
        "--session-id", help="Session ID or unique prefix to export"
    )
    _add_session_filter_args(
        sessions_export,
        "Only export sessions older than AGE (duration like '5h'/'2d', "
        "bare number of days, or an ISO timestamp)",
    )
    sessions_export.add_argument(
        "--redact",
        action="store_true",
        help="Redact secrets (API keys, tokens, credentials) from exported content",
    )
    sessions_export.add_argument(
        "--lineage",
        choices=["single", "logical"],
        default="single",
        help="md/qmd only: export one row or its compression lineage",
    )
    sessions_export.add_argument(
        "--delete-after-verified",
        action="store_true",
        help="md/qmd only: after verified single-session export, delete that session (needs --yes)",
    )
    sessions_export.add_argument(
        "--force",
        action="store_true",
        help="md/qmd only: overwrite an existing export file",
    )

    sessions_delete = sessions_subparsers.add_parser(
        "delete", help="Delete a specific session"
    )
    sessions_delete.add_argument("session_id", help="Session ID to delete")
    sessions_delete.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation"
    )

    sessions_prune = sessions_subparsers.add_parser(
        "prune",
        help="Delete old sessions (filterable by time window, source, title, ...)",
    )
    _add_session_filter_args(
        sessions_prune,
        "Delete sessions older than AGE — days if bare number, or a duration "
        "like '5h'/'2d'/'1w', or an ISO timestamp (bare prune with no filters "
        "defaults to 90 days; any filter matches all ages)",
    )
    sessions_prune.add_argument(
        "--include-archived",
        action="store_true",
        help="Also delete archived sessions (excluded by default)",
    )
    sessions_prune.add_argument(
        "--never-active",
        action="store_true",
        help=(
            "Instead of ended sessions, delete keyed gateway rows that were "
            "opened and never used (no messages, tokens, tool calls or title) "
            "and are older than AGE (default 30 days). Ordinary prune can "
            "never reach these — it only ever selects ended sessions"
        ),
    )

    sessions_archive = sessions_subparsers.add_parser(
        "archive",
        help="Bulk-archive (soft-hide) sessions matching filters — no deletion",
    )
    _add_session_filter_args(
        sessions_archive,
        "Only archive sessions older than AGE (duration like '5h'/'2d', "
        "bare number of days, or ISO timestamp)",
    )

    sessions_subparsers.add_parser(
        "optimize",
        help="Reclaim disk space: merge FTS5 segments + VACUUM (no data change)",
    )

    sessions_clean_markers = sessions_subparsers.add_parser(
        "clean-markers",
        help="Permanently clear stale tool-call marker content left by sessions from before",
        description=(
            "Before the fix, a local tool-call template could persist a "
            "bare bracketed marker (e.g. \"[memory]\") as an assistant turn's "
            "content instead of real text. This is already repaired in memory "
            "on every session load, so running this is optional — it rewrites "
            "the affected rows once, in place, so long-lived sessions stop "
            "re-scanning/re-repairing the same rows on every resume. Only the "
            "content column is touched; tool_calls and every other column on "
            "the row are left untouched."
        ),
    )
    sessions_clean_markers.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Report the affected row count without writing",
    )
    sessions_clean_markers.add_argument(
        "--no-backup",
        action="store_true",
        default=False,
        help="Skip the timestamped state.db backup taken before writing (not recommended)",
    )

    sessions_optimize_storage = sessions_subparsers.add_parser(
        "optimize-storage",
        help="Migrate the search index to the compact v23 layout (reclaims disk on large DBs)",
        description=(
            "Rebuild the full-text search index in the compact v23 "
            "external-content layout. On large databases this reclaims a "
            "large fraction of state.db (the old layout stored duplicate "
            "copies of every message and indexed tool output). Runs "
            "foreground with a progress bar, throttles so a running gateway "
            "stays responsive, and VACUUMs at the end. Safe to interrupt and "
            "re-run — it resumes where it left off. No conversation data is "
            "changed; only the search index is rebuilt."
        ),
    )
    sessions_optimize_storage.add_argument(
        "--no-vacuum",
        action="store_true",
        default=False,
        help="Skip the final VACUUM (index is rebuilt but freed pages aren't returned to the OS until a later VACUUM)",
    )
    sessions_optimize_storage.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip the disk-space confirmation prompt",
    )

    sessions_repair = sessions_subparsers.add_parser(
        "repair",
        help="Repair a malformed state.db schema so hidden sessions reappear",
        description=(
            "Recover a state.db whose schema is malformed (e.g. 'table "
            "messages_fts already exists'), which makes Desktop/Dashboard show "
            "no sessions. A backup is made first; sessions and messages are "
            "preserved and the FTS search index is rebuilt if needed."
        ),
    )
    sessions_repair.add_argument(
        "--check-only",
        action="store_true",
        help="Only report whether the database opens cleanly; do not modify it",
    )
    sessions_repair.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the timestamped backup copy (not recommended)",
    )

    sessions_repair_routing = sessions_subparsers.add_parser(
        "repair-routing",
        help="Re-stamp gateway sessions that lost their routing identity",
        description=(
            "Find gateway conversations stranded in session rows whose "
            "routing identity (session_key/chat_id/origin) was never "
            "written — the damage a corrupt state.db write path leaves "
            "behind. Such a row is invisible to restart recovery, "
            "so the chat resumes an older session instead. Re-stamps each "
            "orphan from the keyed predecessor it continues, and only when "
            "that predecessor is unambiguous. Reports without touching the "
            "database unless --apply is given."
        ),
    )
    sessions_repair_routing.add_argument(
        "--apply",
        action="store_true",
        help="Perform the adoptions (default: report only)",
    )
    sessions_repair_routing.add_argument(
        "--max-gap-seconds",
        type=float,
        default=None,
        help=(
            "Window between a keyed predecessor's last activity and an "
            "orphan's start for them to count as the same conversation "
            "(default: 900)"
        ),
    )

    sessions_recover = sessions_subparsers.add_parser(
        "recover",
        help="Rebuild canonical session data into a separate clean database",
        description=(
            "Offline, non-destructive recovery for a damaged state.db. The "
            "source database and its WAL/SHM/rollback-journal sidecars are "
            "copied before SQLite opens anything. Canonical rows are rebuilt "
            "into a new output database; derived search indexes are recreated "
            "and the active database is never replaced automatically."
        ),
    )
    sessions_recover.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source state.db or preserved backup to inspect/recover",
    )
    sessions_recover.add_argument(
        "--output",
        type=Path,
        help="New recovery database path (required unless --inspect-only)",
    )
    sessions_recover.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only report canonical table readability; do not create an output database",
    )
    sessions_recover.add_argument(
        "--work-dir",
        type=Path,
        help="Existing directory for the disposable source copy (defaults beside the output)",
    )
    sessions_recover.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Rows committed per recovery batch (default: 1000)",
    )
    sessions_recover.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Best-effort salvage across damaged row ranges; the output remains "
            "separate and every skipped range is recorded"
        ),
    )
    sessions_recover.add_argument(
        "--report",
        type=Path,
        help="JSON report path (defaults to <output>.recovery.json)",
    )

    sessions_subparsers.add_parser("stats", help="Show session store statistics")

    sessions_rename = sessions_subparsers.add_parser(
        "rename", help="Set or change a session's title"
    )
    sessions_rename.add_argument("session_id", help="Session ID to rename")
    sessions_rename.add_argument("title", nargs="+", help="New title for the session")

    sessions_retitle = sessions_subparsers.add_parser(
        "retitle-skills",
        help="Re-title sessions whose auto-title came from a /skill's own text",
        description=(
            "Sessions opened with a /skill were auto-titled from the expanded "
            "message, which embeds the whole skill body — so the title "
            "describes the SKILL, not the request. This regenerates those "
            "titles from what the user actually typed. Lists what it would "
            "change unless --apply is passed."
        ),
    )
    sessions_retitle.add_argument(
        "--apply",
        action="store_true",
        help="Write the new titles (default: dry run)",
    )
    sessions_retitle.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum sessions to examine (default: 200)",
    )

    sessions_browse = sessions_subparsers.add_parser(
        "browse",
        help="Interactive session picker — browse, search, and resume sessions",
    )
    sessions_browse.add_argument(
        "--source", help="Filter by source (cli, telegram, discord, etc.)"
    )
    sessions_browse.add_argument(
        "--limit", type=int, default=500, help="Max sessions to load (default: 500)"
    )


    # cmd_sessions lives in pilotage_cli/sessions_cmd.py (main.py decomposition).
    # sessions_parser is threaded in via functools.partial because the
    # fallthrough branch calls sessions_parser.print_help() (formerly a
    # closure capture of this main()-local). The indirection through _self()
    # keeps the sessions_cmd import lazy until the subcommand actually runs
    # and lets monkeypatches on pilotage_cli.main.cmd_sessions keep working.
    def _dispatch_sessions(_args, *, sessions_parser=sessions_parser):
        return _self().cmd_sessions(_args, sessions_parser=sessions_parser)

    sessions_parser.set_defaults(func=_dispatch_sessions)
    # =========================================================================
    # claw command  (parser built in pilotage_cli/subcommands/claw.py)
    # =========================================================================

    # =========================================================================
    # version command  (parser built in pilotage_cli/subcommands/version.py)
    # =========================================================================
    build_version_parser(subparsers, cmd_version=cmd_version)

    # =========================================================================
    # update command  (parser built in pilotage_cli/subcommands/update.py)
    # =========================================================================

    # =========================================================================
    # uninstall command  (parser built in pilotage_cli/subcommands/uninstall.py)
    # =========================================================================
    build_uninstall_parser(subparsers, cmd_uninstall=cmd_uninstall)

    # =========================================================================
    # profile command  (parser built in pilotage_cli/subcommands/profile.py)
    # =========================================================================
    build_profile_parser(subparsers, cmd_profile=cmd_profile)

    # =========================================================================
    # completion command
    # =========================================================================
    completion_parser = subparsers.add_parser(
        "completion",
        help="Print shell completion script (bash, zsh, or fish)",
    )
    completion_parser.add_argument(
        "shell",
        nargs="?",
        default="bash",
        choices=["bash", "zsh", "fish"],
        help="Shell type (default: bash)",
    )
    completion_parser.set_defaults(func=lambda args: cmd_completion(args, parser))


    # =========================================================================
    # logs command  (parser built in pilotage_cli/subcommands/logs.py)
    # =========================================================================
    build_logs_parser(subparsers, cmd_logs=cmd_logs)

    # =========================================================================
    # prompt-size command  (parser built in pilotage_cli/subcommands/prompt_size.py)
    # =========================================================================
    build_prompt_size_parser(subparsers, cmd_prompt_size=cmd_prompt_size)

    # =========================================================================
    # Parse and execute
    # =========================================================================
    # Pre-process argv so unquoted multi-word session names after -c / -r
    # are merged into a single token before argparse sees them.
    # e.g. ``pilotage -c Pokemon Agent Dev`` → ``pilotage -c 'Pokemon Agent Dev'``
    _processed_argv = _coalesce_session_name_args(sys.argv[1:])

    # ── Defensive subparser routing (bpo-9338 workaround) ───────────
    # On some Python versions (notably <3.11), argparse fails to route
    # subcommand tokens when the parent parser has nargs='?' optional
    # arguments (--continue).  The symptom: "unrecognized arguments: model"
    # even though 'model' is a registered subcommand.
    #
    # Fix: when argv contains a token matching a known subcommand, set
    # subparsers.required=True to force deterministic routing.  If that
    # fails (e.g. 'pilotage -c model' where 'model' is consumed as the
    # session name for --continue), fall back to the default behaviour.
    import io as _io

    _known_cmds = (
        set(subparsers.choices.keys()) if hasattr(subparsers, "choices") else set()
    )
    _has_cmd_token = any(
        t in _known_cmds for t in _processed_argv if not t.startswith("-")
    )

    if _has_cmd_token:
        subparsers.required = True
        _saved_stderr = sys.stderr
        try:
            sys.stderr = _io.StringIO()
            args = parser.parse_args(_processed_argv)
            sys.stderr = _saved_stderr
        except SystemExit as exc:
            sys.stderr = _saved_stderr
            # Help/version flags (exit code 0) already printed output —
            # re-raise immediately to avoid a second parse_args printing
            # the same help text again.
            if exc.code == 0:
                raise
            # Subcommand name was consumed as a flag value (e.g. -c model).
            # Fall back to optional subparsers so argparse handles it normally.
            subparsers.required = False
            args = parser.parse_args(_processed_argv)
    else:
        subparsers.required = False
        args = parser.parse_args(_processed_argv)

    # Handle --version flag
    if args.version:
        cmd_version(args)
        return

    # --yolo: set PILOTAGE_YOLO_MODE *before* plugin discovery.  The call to
    # _prepare_agent_startup() below triggers discover_plugins() → tool
    # imports, and tools.approval freezes _YOLO_MODE_FROZEN at module
    # import time (, security hardening against prompt-injection).
    # If the env var is set only later (e.g. inside cmd_chat), the frozen
    # value is already False and --yolo silently does nothing.
    if getattr(args, "yolo", False):
        os.environ["PILOTAGE_YOLO_MODE"] = "1"

    # Discover Python plugins and register shell hooks once, before any
    # command that can fire lifecycle hooks.  Both are idempotent; gated
    # so introspection/management commands (pilotage hooks list, cron
    # list, gateway status, mcp add, ...) don't pay discovery cost or
    # trigger consent prompts for hooks the user is still inspecting.
    _prepare_agent_startup(args)

    # Handle top-level --oneshot / -z: single-shot mode, stdout = final
    # response only, nothing else. Bypasses cli.py entirely.
    if getattr(args, "oneshot", None):
        _confirm_startup_expensive_model_override(args)
        _run_and_exit_oneshot(
            args.oneshot,
            model=getattr(args, "model", None),
            provider=getattr(args, "provider", None),
            toolsets=getattr(args, "toolsets", None),
            usage_file=getattr(args, "usage_file", None),
        )

    # Handle top-level --resume / --continue as shortcut to chat
    if (args.resume or args.continue_last) and args.command is None:
        args.command = "chat"
        for attr, default in [
            ("query", None),
            ("model", None),
            ("provider", None),
            ("toolsets", None),
            ("verbose", None),
            ("worktree", False),
        ]:
            if not hasattr(args, attr):
                setattr(args, attr, default)
        cmd_chat(args)
        return

    # Default to chat if no command specified
    if args.command is None:
        for attr, default in [
            ("query", None),
            ("model", None),
            ("provider", None),
            ("toolsets", None),
            ("verbose", None),
            ("resume", None),
            ("continue_last", None),
            ("worktree", False),
        ]:
            if not hasattr(args, attr):
                setattr(args, attr, default)
        cmd_chat(args)
        return

    # Execute the command.  Propagate the handler's return code as the
    # process exit code so subcommands that signal failure (e.g.
    # ``pilotage egress start`` refusing when credential_source=bitwarden
    # is misconfigured) actually exit non-zero.  Handlers that return
    # None are treated as success (exit 0).
    if hasattr(args, "func"):
        rc = args.func(args)
        if isinstance(rc, int) and rc != 0:
            sys.exit(rc)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
