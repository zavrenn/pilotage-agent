"""Helpers for loading Pilotage .env files consistently across entrypoints."""

from __future__ import annotations

import codecs
import io
import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from utils import atomic_replace, fast_safe_load


# Env var name suffixes that indicate credential values.  These are the
# only env vars whose values we sanitize on load — we must not silently
# alter arbitrary user env vars, but credentials are known to require
# pure ASCII (they become HTTP header values).
_CREDENTIAL_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_KEY")

# Names we've already warned about during this process, so repeated
# load_pilotage_dotenv() calls (user env + project env, gateway hot-reload,
# tests) don't spam the same warning multiple times.
_WARNED_KEYS: set[str] = set()

# Paths we've already emitted a UTF-32 refuse-to-mangle warning for.
# load_pilotage_dotenv can call _sanitize_env_file_if_needed multiple times
# for the same file (user env + project env + hot-reload); once per path
# is enough.
_WARNED_UTF32_PATHS: set[str] = set()

# Map of env-var name → source label ("bitwarden", etc.) for credentials
# that were injected by an external secret source during load_pilotage_dotenv().
# Used by setup / `pilotage model` flows to label detected credentials so
# users understand WHERE a key came from when their .env doesn't contain it
# directly (otherwise the "credentials detected ✓" line looks identical to
# the .env case and they don't know Bitwarden is wired up).
_SECRET_SOURCES: dict[str, str] = {}
# Applied values are immutable per-home snapshots.  ``os.environ`` is shared
# across profiles and may be overwritten by a later home's source apply.
_SECRET_SOURCE_VALUES_BY_HOME: dict[str, dict[str, str]] = {}

# PILOTAGE_HOME paths we've already pulled external secrets for during this
# process.  ``load_pilotage_dotenv()`` is called at module-import time from
# several hot modules (cli.py, pilotage_cli/main.py, run_agent.py,
# trajectory_compressor.py, gateway/run.py, ...), so without this guard the
# Bitwarden status line gets printed 3-5x per startup.  Bitwarden's own
# in-process cache prevents redundant network calls, but the print, the
# config re-parse, and the ASCII sanitization sweep still ran every time.
_APPLIED_HOMES: set[str] = set()
_SECRET_SOURCE_CACHE_LOCK = threading.RLock()


def _known_pilotage_env_keys() -> set[str]:
    """Return the combined set of known Pilotage env-var keys.

    Includes both ``OPTIONAL_ENV_VARS`` (setup-flow vars with metadata) and
    ``_EXTRA_ENV_KEYS`` (provider/platform keys managed outside the setup
    wizard).  Lazy-imported to avoid circular-dependency during early-bootstrap
    ``load_pilotage_dotenv()`` calls.
    """
    from pilotage_cli.config import _EXTRA_ENV_KEYS
    from pilotage_cli.config_defaults import OPTIONAL_ENV_VARS

    return set(OPTIONAL_ENV_VARS.keys()) | set(_EXTRA_ENV_KEYS)


# Behavioral routing keys a parent Pilotage process injects into child env and
# that silently redirect a profile onto the wrong provider path. These — and
# ONLY these — are scrubbed from os.environ at startup when absent from the
# profile's .env. Credential keys (API keys/tokens) are excluded: shell
# exports are a legitimate, documented way to supply them, and read-time
# secret-scope checks (agent/secret_scope.py) own cross-profile credential
# isolation. Empty today: every remaining routing key is read from config,
# not env. The scrub stays wired so a future key only needs listing here.
_PROFILE_MANAGED_ENV_KEYS: frozenset[str] = frozenset()


def _env_keys_defined_in_dotenv(path: Path) -> set[str]:
    """Return KEY names assigned in a dotenv file (including empty ``KEY=``).

    Uses a fast line scanner rather than full dotenv parsing so it works
    during early bootstrap without importing python-dotenv.  Ignores comment
    and blank lines.  Non-ASCII encoding errors fall back to ``latin-1``,
    matching ``_load_dotenv_with_fallback``.
    """
    keys: set[str] = set()
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            text = path.read_text(encoding="latin-1", errors="replace")
        except Exception:
            return keys
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key = line.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def _clear_known_keys_missing_from_dotenv(path: Path) -> None:
    """Remove inherited profile-managed Pilotage keys absent from ``.env``.

    After the profile's ``.env`` has been loaded with ``override=True``,
    scan the file for which profile-managed keys it explicitly defines and
    delete any such key that exists in ``os.environ`` but is *not* present
    in the file.

    Scope is deliberately NARROW: only ``_PROFILE_MANAGED_ENV_KEYS`` —
    behavioral routing keys that a parent Pilotage process injects and that
    silently change *which provider path* a profile uses. Provider API keys
    (OPENAI_API_KEY, …) are
    intentionally excluded: users legitimately export those in their shell
    (``export OPENAI_API_KEY=…`` is a documented flow — see
    ``tests/pilotage_cli/test_dump_env_visibility.py``), and a startup scrub
    cannot distinguish a shell export from parent-process leakage. Clearing
    the full known-key set would delete user-exported credentials on every
    ``pilotage`` invocation.

    Cross-profile *credential* isolation is handled at read time by
    ``agent.secret_scope.get_secret`` (scope authoritative under
    multiplexing), not by mutating ``os.environ`` here.

    Does **not** run when the ``.env`` file does not exist (bare-profile
    case, which follows ```` / ```` semantics).
    """
    if not path.exists():
        return
    defined = _env_keys_defined_in_dotenv(path)
    for key in _PROFILE_MANAGED_ENV_KEYS:
        if key not in defined and key in os.environ:
            del os.environ[key]


def get_secret_source(env_var: str) -> str | None:
    """Return the label of the secret source that supplied ``env_var``, if any.

    Returns ``"bitwarden"`` for keys pulled from Bitwarden Secrets Manager
    during the current process's ``load_pilotage_dotenv()`` call.  Returns
    ``None`` for keys that came from ``.env``, the shell environment, or
    aren't tracked.  The returned label is metadata only: credential-pool
    persistence may store it to explain the origin of a borrowed secret, but
    must never treat it as authorization to persist the raw value.
    """
    return _SECRET_SOURCES.get(env_var)


def get_secret_source_values(
    pilotage_home: str | os.PathLike,
) -> dict[str, str]:
    """Return the external-secret value snapshot for ``pilotage_home``."""
    home_key = str(Path(pilotage_home).resolve())
    return dict(_SECRET_SOURCE_VALUES_BY_HOME.get(home_key, {}))


def hydrate_profile_secret_sources(
    pilotage_home: str | os.PathLike,
) -> dict[str, str]:
    """Resolve one profile's configured sources without mutating ``os.environ``.

    Multiplex gateways can route a first turn to a secondary profile that has
    never run the process-global dotenv startup path.  Resolve that profile's
    sources against a private mapping seeded from its own ``.env`` and record
    the usual per-home snapshot for ``build_profile_secret_scope()``.

    Fail-open and once-per-home semantics intentionally mirror
    ``_apply_external_secret_sources``.  The returned mapping contains only
    values actually contributed by external sources, never the profile's
    plaintext ``.env`` entries.
    """
    with _SECRET_SOURCE_CACHE_LOCK:
        return _hydrate_profile_secret_sources(Path(pilotage_home))


def _hydrate_profile_secret_sources(home: Path) -> dict:
    """External secret sources were removed; nothing to hydrate."""
    return {}



def reset_secret_source_cache() -> None:
    """Forget which PILOTAGE_HOME paths have already had external secrets applied.

    The first call to ``_apply_external_secret_sources(home_path)`` in a
    process pulls from Bitwarden (or other configured backend), records the
    applied keys in ``_SECRET_SOURCES``, and remembers ``home_path`` so
    subsequent calls in the same process are no-ops.  Call this to force the
    next call to re-pull — useful for tests, and for long-running processes
    that want to refresh after a config change.
    """
    _APPLIED_HOMES.clear()
    _SECRET_SOURCES.clear()
    _SECRET_SOURCE_VALUES_BY_HOME.clear()


def format_secret_source_suffix(env_name: str) -> str:
    """External secret sources were removed; no provenance suffix."""
    return ""



def _format_offending_chars(value: str, limit: int = 3) -> str:
    """Return a compact 'U+XXXX ('c'), ...' summary of non-ASCII codepoints."""
    seen: list[str] = []
    for ch in value:
        if ord(ch) > 127:
            label = f"U+{ord(ch):04X}"
            if ch.isprintable():
                label += f" ({ch!r})"
            if label not in seen:
                seen.append(label)
            if len(seen) >= limit:
                break
    return ", ".join(seen)


def _sanitize_loaded_credentials() -> None:
    """Strip non-ASCII characters from credential env vars in os.environ.

    Called after dotenv loads so the rest of the codebase never sees
    non-ASCII API keys.  Only touches env vars whose names end with
    known credential suffixes (``_API_KEY``, ``_TOKEN``, etc.).

    Emits a one-line warning to stderr when characters are stripped.
    Silent stripping would mask copy-paste corruption (Unicode lookalike
    glyphs from PDFs / rich-text editors, ZWSP from web pages) as opaque
    provider-side "invalid API key" errors.
    """
    for key, value in list(os.environ.items()):
        if not any(key.endswith(suffix) for suffix in _CREDENTIAL_SUFFIXES):
            continue
        try:
            value.encode("ascii")
            continue
        except UnicodeEncodeError:
            pass
        cleaned = value.encode("ascii", errors="ignore").decode("ascii")
        os.environ[key] = cleaned
        if key in _WARNED_KEYS:
            continue
        _WARNED_KEYS.add(key)
        stripped = len(value) - len(cleaned)
        detail = _format_offending_chars(value) or "non-printable"
        print(
            f"  Warning: {key} contained {stripped} non-ASCII character"
            f"{'s' if stripped != 1 else ''} ({detail}) — stripped so the "
            f"key can be sent as an HTTP header.",
            file=sys.stderr,
        )
        print(
            "  This usually means the key was copy-pasted from a PDF, "
            "rich-text editor, or web page that substituted lookalike\n"
            "  Unicode glyphs for ASCII letters. If authentication fails "
            "(e.g. \"API key not valid\"), re-copy the key from the\n"
            "  provider's dashboard and run `pilotage setup` (or edit the "
            ".env file in a plain-text editor).",
            file=sys.stderr,
        )


def _load_dotenv_with_fallback(path: Path, *, override: bool) -> None:
    try:
        # utf-8-sig strips a leading UTF-8 BOM if present (PowerShell 5.1
        # Set-Content -Encoding UTF8 / Notepad) and is a no-op for BOM-less
        # UTF-8. Plain "utf-8" would keep U+FEFF on the first key name and
        # silently drop it from os.environ under its canonical name.
        load_dotenv(dotenv_path=path, override=override, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # utf-8-sig can't strip a BOM once we fall back to latin-1 decode.
        raw = path.read_bytes()
        if raw.startswith(codecs.BOM_UTF8):
            raw = raw[len(codecs.BOM_UTF8) :]
        load_dotenv(stream=io.StringIO(raw.decode("latin-1")), override=override)
    # Strip non-ASCII characters from credential env vars that were just
    # loaded.  API keys must be pure ASCII since they're sent as HTTP
    # header values (httpx encodes headers as ASCII).  Non-ASCII chars
    # typically come from copy-pasting keys from PDFs or rich-text editors
    # that substitute Unicode lookalike glyphs (e.g. ʋ U+028B for v).
    _sanitize_loaded_credentials()


def _sanitize_env_file_if_needed(path: Path) -> None:
    """Pre-sanitize a .env file before python-dotenv reads it.

    Strips embedded null bytes which crash ``os.environ[k] = v``
    with ``ValueError: embedded null byte`` — typically introduced by
    copy-pasting API keys from terminals or rich-text editors.

    Encoding: sniffs a leading BOM *before* any text decode. UTF-16
    (Notepad "Unicode") is decoded correctly and rewritten as clean
    UTF-8. UTF-32 is refused (left untouched) so we never fall through
    to the errors=replace corruption path. Order of BOM checks matters:
    UTF-32-LE's BOM starts with UTF-16-LE's FF FE.

    ``pilotage_cli.config._sanitize_env_lines`` normalizes line endings while
    treating content after the first ``=`` as opaque for boundary discovery.
    """
    if not path.exists():
        return
    try:
        from pilotage_cli.config import _sanitize_env_lines
    except ImportError:
        return  # early bootstrap — config module not available yet

    try:
        raw = path.read_bytes()
    except Exception:
        return

    # Sniff leading BOM bytes BEFORE decoding. ORDER MATTERS:
    # codecs.BOM_UTF32_LE is FF FE 00 00, which startswith
    # codecs.BOM_UTF16_LE (FF FE). Checking UTF-16 first would
    # misdetect UTF-32-LE as UTF-16-LE and mangle the file.
    force_utf8_rewrite = False
    if raw.startswith(codecs.BOM_UTF32_LE) or raw.startswith(codecs.BOM_UTF32_BE):
        # Lazy import keeps the module import block identical to's
        # codecs/io additions so the two PRs auto-merge either order.
        path_key = str(path.resolve())
        if path_key not in _WARNED_UTF32_PATHS:
            _WARNED_UTF32_PATHS.add(path_key)
            import logging

            logging.getLogger(__name__).warning(
                "Skipping .env sanitize for %s: UTF-32 BOM detected; "
                "leaving file untouched to avoid corruption",
                path,
            )
        return
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        # "utf-16" uses the BOM to select endianness and strips it.
        # TextIOWrapper + newline=None matches open()'s universal-newlines
        # line splitting (\\n/\\r\\n/\\r only — not splitlines()'s extra
        # Unicode boundaries like U+2028), so sanitize sees the same lines
        # as the UTF-8 path.
        try:
            with io.TextIOWrapper(
                io.BytesIO(raw), encoding="utf-16", newline=None
            ) as f:
                original = f.readlines()
        except UnicodeDecodeError:
            return
        # Source is UTF-16 on disk; always rewrite as clean UTF-8 so
        # the subsequent utf-8 dotenv load sees a canonical file.
        force_utf8_rewrite = True
    else:
        # Default path: utf-8-sig (strips UTF-8 BOM if present) with
        # errors=replace so embedded NULs can be stripped below.
        try:
            with open(path, encoding="utf-8-sig", errors="replace") as f:
                original = f.readlines()
        except Exception:
            return
        # Defense-in-depth: errors=replace turns undecodable leading
        # bytes into U+FFFD. Persisting that glues replacement chars
        # onto the first key name and rewrites the file permanently
        # (the UTF-16-with-BOM corruption path before BOM sniffing).
        # Leave the file untouched rather than write the mangling.
        if original and original[0].startswith("\ufffd"):
            return

    try:
        # Strip null bytes before _sanitize_env_lines so they never
        # reach python-dotenv (which passes them to os.environ and
        # crashes with ValueError). Also intentionally repairs
        # BOM-less UTF-16 (NUL-padded ASCII) into clean UTF-8.
        stripped = [line.replace("\x00", "") for line in original]
        sanitized = _sanitize_env_lines(stripped)
        if sanitized != original or force_utf8_rewrite:
            import tempfile
            fd, tmp = tempfile.mkstemp(
                dir=str(path.parent), suffix=".tmp", prefix=".env_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.writelines(sanitized)
                    f.flush()
                    os.fsync(f.fileno())
                atomic_replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
    except Exception:
        pass  # best-effort — don't block gateway startup


def load_pilotage_dotenv(
    *,
    pilotage_home: str | os.PathLike | None = None,
    project_env: str | os.PathLike | None = None,
) -> list[Path]:
    """Load Pilotage environment files with user config taking precedence.

    Behavior:
    - `~/.pilotage/.env` overrides stale shell-exported values when present.
    - project `.env` acts as a dev fallback and only fills missing values when
      the user env exists.
    - if no user env exists, the project `.env` also overrides stale shell vars.
    """
    loaded: list[Path] = []

    home_path = Path(pilotage_home or os.getenv("PILOTAGE_HOME", Path.home() / ".pilotage"))
    user_env = home_path / ".env"
    project_env_path = Path(project_env) if project_env else None

    # Normalize safe formatting and remove invalid NUL bytes before parsing.
    if user_env.exists():
        _sanitize_env_file_if_needed(user_env)
    if project_env_path and project_env_path.exists():
        _sanitize_env_file_if_needed(project_env_path)

    if user_env.exists():
        _load_dotenv_with_fallback(user_env, override=True)
        loaded.append(user_env)
        # Mirror reload_env() known-key cleanup so inherited Pilotage keys
        # absent from this profile's .env do not leak into the runtime.
        _clear_known_keys_missing_from_dotenv(user_env)

    # Load .op.env AFTER .env so that .env values win, but the bootstrap
    # token (OP_SERVICE_ACCOUNT_TOKEN) becomes available for
    # apply_onepassword_secrets() even in cron / subprocess environments
    # that inherit no shell state (no systemd EnvironmentFile, no op run).
    # .op.env is gitignored — the service-account token never enters the
    # committed .env file.
    # Users on systemd can alternatively use:
    #   EnvironmentFile=-/path/to/.pilotage/.op.env
    # in their gateway unit, which takes precedence (override=False below
    # ensures .op.env never clobbers a token already in the environment).
    op_env = home_path / ".op.env"
    if op_env.exists() and not os.environ.get("OP_SERVICE_ACCOUNT_TOKEN"):
        _load_dotenv_with_fallback(op_env, override=False)

    if project_env_path and project_env_path.exists():
        _load_dotenv_with_fallback(project_env_path, override=not loaded)
        loaded.append(project_env_path)

    _apply_external_secret_sources(home_path)
    _apply_managed_env()

    # config.yaml is the documented source of truth for terminal.* settings,
    # but the dotenv loads above run with override=True — so a stale
    # TERMINAL_ENV=docker left in ~/.pilotage/.env (e.g. written by an older
    # `pilotage setup` before the user switched terminal.backend in config.yaml)
    # silently wins again on every reload. Startup launchers bridge
    # config→env once, but long-lived processes (gateway per-turn reload,
    # cron standalone runs) call load_pilotage_dotenv() repeatedly and used to
    # flip the effective backend back to the stale .env value mid-session
    #. Re-apply config.yaml's explicit terminal keys last so
    # the documented config path always wins. Runs after _apply_managed_env()
    # so the merged config (which already carries the managed overlay) is
    # what lands in the env.
    _reapply_terminal_config_bridge(home_path)

    return loaded


def _reapply_terminal_config_bridge(home_path: Path) -> None:
    """Re-assert config.yaml's explicit ``terminal.*`` keys over reloaded .env.

    Delegates to ``pilotage_cli.config.apply_terminal_config_to_env`` — the
    single shared bridge (same one terminal_tool's fallback and the TUI/
    dashboard launchers use) — so key coverage, explicit-keys-only override
    semantics, cwd placeholder handling, and the managed-scope overlay can't
    drift from the other bridge sites. Only keys the user actually wrote in
    config.yaml's ``terminal`` section override env values; a config.yaml
    without a terminal section leaves .env/shell selections untouched.

    Scoped to the process PILOTAGE_HOME: the shared bridge reads the
    process-global config, so re-applying it for a *different* profile's
    ``load_pilotage_dotenv(pilotage_home=...)`` call would bridge the wrong
    profile's config. Fail-open — a config problem must never break dotenv
    loading (the historical env-driven behavior still applies).
    """
    try:
        if Path(home_path).resolve() != _process_pilotage_home().resolve():
            return
        from pilotage_cli.config import apply_terminal_config_to_env

        apply_terminal_config_to_env(env=None)
    except Exception:  # noqa: BLE001 — early bootstrap / malformed config
        pass


def _apply_managed_env() -> None:
    """Apply the managed-scope .env last, with override, so it beats user/shell.

    Managed scope is machine-global (independent of PILOTAGE_HOME / profile). v1
    enforcement is "applied last with override=True" — at the end of startup load
    ``os.environ`` holds the managed value for every managed key, beating both the
    user ``.env`` and any pre-existing shell export. This deliberately inverts the
    usual env-over-config precedence for the pinned keys (see
    ``docs/design/managed-scope.md`` §4.1).

    This does NOT prevent the agent from later mutating ``os.environ`` in-process
    or ``export``-ing in a subprocess shell; that hard boundary is a documented
    v2 item (design §8.1). v1 relies on filesystem permissions only.

    Fail-open: a missing managed dir or .env is the common case and a no-op; any
    error here is swallowed so managed scope can never block startup.
    """
    try:
        from pilotage_cli import managed_scope

        managed_dir = managed_scope.get_managed_dir()
    except Exception:  # noqa: BLE001 — managed scope must never block startup
        return
    if managed_dir is None:
        return
    managed_env = managed_dir / ".env"
    if not managed_env.exists():
        return
    _sanitize_env_file_if_needed(managed_env)
    _load_dotenv_with_fallback(managed_env, override=True)


def _apply_external_secret_sources(*args, **kwargs) -> None:
    """External secret sources were removed; no-op."""
    return None



def _remediation_hint(*args, **kwargs) -> str:
    """External secret sources were removed; no hints."""
    return ""



def _load_secrets_config(home_path: Path) -> dict:
    """Read just the ``secrets:`` section out of config.yaml.

    Imported lazily and isolated from the main config loader so a
    malformed config can't take down dotenv loading entirely.
    """
    config_path = home_path / "config.yaml"
    if not config_path.exists():
        return {}
    # Prefer the shared (mtime, size)-keyed raw-config cache — this is the
    # first config.yaml read in a normal `pilotage` startup, so populating the
    # shared cache here lets main.py's early bridge and pilotage_logging reuse
    # the same parse (one parse per process instead of 3-4). Falls back to a
    # direct isolated parse if the shared reader is unavailable, preserving
    # the "malformed config can't take down dotenv loading" property (the
    # shared reader also swallows parse errors and returns {}).
    if home_path == _process_pilotage_home():
        try:
            from pilotage_cli.config import read_raw_config

            data = read_raw_config() or {}
            return data.get("secrets") or {}
        except Exception:
            pass
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = fast_safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return {}
    return data.get("secrets") or {}


def _process_pilotage_home() -> Path:
    """The PILOTAGE_HOME the shared config cache is keyed to."""
    try:
        from pilotage_constants import get_pilotage_home

        return get_pilotage_home()
    except Exception:
        return Path.home() / ".pilotage"
