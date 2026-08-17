"""
Status command for pilotage CLI.

Shows the status of all Pilotage Agent components.
"""

import os
import sys
import time
import importlib.util
import subprocess  # noqa: F401 — re-exported for tests that monkeypatch status.subprocess to guard against regressions
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

from pilotage_cli.auth import AuthError, resolve_provider
from pilotage_cli.colors import Colors, color
from pilotage_cli.config import get_env_path, get_env_value, get_pilotage_home, load_config
from pilotage_cli.models import provider_label
from pilotage_cli.runtime_provider import resolve_requested_provider

def check_mark(ok: bool) -> str:
    if ok:
        return color("✓", Colors.GREEN)
    return color("✗", Colors.RED)

def redact_key(key: str) -> str:
    """Redact an API key for display.

    Thin wrapper over :func:`agent.redact.mask_secret`. Preserves the
    "(not set)" placeholder in dim color to match ``pilotage config``'s
    output (previously this variant was missing the DIM color —
    consolidated via PR that also introduced ``mask_secret``).
    """
    from agent.redact import mask_secret
    return mask_secret(key, empty=color("(not set)", Colors.DIM))


def _format_iso_timestamp(value) -> str:
    """Format ISO timestamps for status output, converting to local timezone."""
    if not value or not isinstance(value, str):
        return "(unknown)"
    from datetime import datetime, timezone
    text = value.strip()
    if not text:
        return "(unknown)"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return value
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _format_relative_ts(ts: float) -> str:
    """Format an epoch timestamp as a short relative age for status output."""
    from pilotage_cli.timefmt import relative_time

    return relative_time(ts)


def _configured_model_label(config: dict) -> str:
    """Return the configured default model from config.yaml."""
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        model = (model_cfg.get("default") or model_cfg.get("name") or "").strip()
    elif isinstance(model_cfg, str):
        model = model_cfg.strip()
    else:
        model = ""
    return model or "(not set)"


def _effective_provider_label() -> str:
    """Return the provider label matching current CLI runtime resolution."""
    requested = resolve_requested_provider()
    try:
        effective = resolve_provider(requested)
    except AuthError:
        effective = requested or "auto"

    return provider_label(effective)


from pilotage_constants import is_termux as _is_termux


def _estop_status_line():
    """One-line pause banner for `pilotage status`, or None when not paused.

    Cheap: a single stat on $PILOTAGE_HOME/ESTOP via agent.estop.
    """
    try:
        from agent.estop import get_state
    except ImportError:
        return None
    state = get_state()
    if state is None:
        return None
    reason = state.get("reason")
    suffix = f" — reason: {reason}" if reason else ""
    return f"⏸️  PAUSED (global emergency stop{suffix}; `pilotage resume` to lift)"


def show_status(args):
    """Show status of all Pilotage Agent components."""
    deep = getattr(args, 'deep', False)

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 ⚕ Pilotage Agent Status                  │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))

    _paused_line = _estop_status_line()
    if _paused_line:
        print()
        print(color(_paused_line, Colors.YELLOW, Colors.BOLD))

    # =========================================================================
    # Environment
    # =========================================================================
    print()
    print(color("◆ Environment", Colors.CYAN, Colors.BOLD))
    print(f"  Project:      {PROJECT_ROOT}")
    print(f"  Python:       {sys.version.split()[0]}")

    env_path = get_env_path()
    print(f"  .env file:    {check_mark(env_path.exists())} {'exists' if env_path.exists() else 'not found'}")

    try:
        config = load_config()
    except Exception:
        config = {}

    print(f"  Model:        {_configured_model_label(config)}")
    print(f"  Provider:     {_effective_provider_label()}")

    # =========================================================================
    # API Keys
    # =========================================================================
    print()
    print(color("◆ API Keys", Colors.CYAN, Colors.BOLD))

    # Values may be a single env var name (str) or a tuple of alternates (first found wins).
    keys: dict[str, str | tuple[str, ...]] = {
        "OpenAI": "OPENAI_API_KEY",
        "Firecrawl": "FIRECRAWL_API_KEY",
        "Tavily": "TAVILY_API_KEY",
        "Browser Use": "BROWSER_USE_API_KEY",  # Optional — local browser works without this
        "Browserbase": "BROWSERBASE_API_KEY",  # Optional — direct credentials only
        "FAL": "FAL_KEY",
        "GitHub": "GITHUB_TOKEN",
    }

    def _resolve_env(env_ref) -> str:
        """Return first non-empty env var value from a str or tuple of names."""
        if isinstance(env_ref, tuple):
            for candidate in env_ref:
                v = get_env_value(candidate) or ""
                if v:
                    return v
            return ""
        return get_env_value(env_ref) or ""

    for name, env_ref in keys.items():
        value = _resolve_env(env_ref)
        has_key = bool(value)
        display = redact_key(value)
        print(f"  {name:<12}  {check_mark(has_key)} {display}")

    # =========================================================================
    # Auth Providers (OAuth)
    # =========================================================================
    print()
    print(color("◆ Auth Providers", Colors.CYAN, Colors.BOLD))

    try:
        from pilotage_cli.auth import get_codex_auth_status
        # Read-only display: use the refresh-free snapshot so `pilotage status`
        # never performs an OAuth refresh or burns a single-use refresh token.
        codex_status = get_codex_auth_status()
    except Exception:
        codex_status = {}

    codex_logged_in = bool(codex_status.get("logged_in"))
    print(
        f"  {'OpenAI Codex':<12}  {check_mark(codex_logged_in)} "
        f"{'logged in' if codex_logged_in else 'not logged in (run: pilotage model)'}"
    )
    codex_auth_file = codex_status.get("auth_store")
    if codex_auth_file:
        print(f"    Auth file:  {codex_auth_file}")
    codex_last_refresh = _format_iso_timestamp(codex_status.get("last_refresh"))
    if codex_status.get("last_refresh"):
        print(f"    Refreshed:  {codex_last_refresh}")
    if codex_status.get("error") and not codex_logged_in:
        print(f"    Error:      {codex_status.get('error')}")

    # =========================================================================
    # API-Key Providers
    # =========================================================================
    print()
    print(color("◆ API-Key Providers", Colors.CYAN, Colors.BOLD))

    apikey_providers = {
        "Z.AI / GLM":       ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        "MiniMax":          ("MINIMAX_API_KEY",),
        "MiniMax (China)":  ("MINIMAX_CN_API_KEY",),
        "DeepInfra":        ("DEEPINFRA_API_KEY",),
    }
    for pname, env_vars in apikey_providers.items():
        key_val = ""
        for ev in env_vars:
            key_val = get_env_value(ev) or ""
            if key_val:
                break
        configured = bool(key_val)
        label = "configured" if configured else "not configured (run: pilotage model)"
        print(f"  {pname:<16} {check_mark(configured)} {label}")

    # =========================================================================
    # Terminal Configuration
    # =========================================================================
    print()
    print(color("◆ Terminal Backend", Colors.CYAN, Colors.BOLD))

    terminal_cfg = config.get("terminal", {}) if isinstance(config.get("terminal"), dict) else {}
    terminal_env = os.getenv("TERMINAL_ENV", "")
    if not terminal_env:
        terminal_env = terminal_cfg.get("backend", "local")
    print(f"  Backend:      {terminal_env}")

    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST", "")
        ssh_user = os.getenv("TERMINAL_SSH_USER", "")
        print(f"  SSH Host:     {ssh_host or '(not set)'}")
        print(f"  SSH User:     {ssh_user or '(not set)'}")
    elif terminal_env == "docker":
        docker_image = os.getenv("TERMINAL_DOCKER_IMAGE", "python:3.11-slim")
        print(f"  Docker Image: {docker_image}")
    elif terminal_env == "daytona":
        daytona_image = os.getenv("TERMINAL_DAYTONA_IMAGE", "nikolaik/python-nodejs:python3.11-nodejs20")
        print(f"  Daytona Image: {daytona_image}")
    sudo_password = os.getenv("SUDO_PASSWORD", "")
    print(f"  Sudo:         {check_mark(bool(sudo_password))} {'enabled' if sudo_password else 'disabled'}")

    # =========================================================================
    # Messaging Platforms
    # =========================================================================
    print()
    print(color("◆ Messaging Platforms", Colors.CYAN, Colors.BOLD))

    platforms = {
        "Telegram": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_HOME_CHANNEL"),
        "WhatsApp": ("WHATSAPP_ENABLED", None),
        "WhatsApp Cloud": ("WHATSAPP_CLOUD_ACCESS_TOKEN", None),
    }

    for name, (token_var, home_var) in platforms.items():
        token = os.getenv(token_var, "")
        has_token = bool(token)
        
        home_channel = ""
        if home_var:
            home_channel = os.getenv(home_var, "")
        
        status = "configured" if has_token else "not configured"
        if home_channel:
            status += f" (home: {home_channel})"
        
        print(f"  {name:<12}  {check_mark(has_token)} {status}")

    # Plugin-registered platforms
    try:
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            # Per-entry guard: one raising probe must not abort the listing
            # of every remaining plugin platform (matches the other three
            # check_fn call sites).
            try:
                configured = bool(entry.check_fn())
            except Exception:
                configured = False
            status_str = "configured" if configured else "not configured"
            label = entry.label
            print(f"  {label:<12}  {check_mark(configured)} {status_str} (plugin)")
    except Exception:
        pass

    # =========================================================================
    # Gateway Status
    # =========================================================================
    print()
    print(color("◆ Gateway Service", Colors.CYAN, Colors.BOLD))

    try:
        from pilotage_cli.gateway import get_gateway_runtime_snapshot, _format_gateway_pids

        snapshot = get_gateway_runtime_snapshot()
        is_running = snapshot.running
        print(f"  Status:       {check_mark(is_running)} {'running' if is_running else 'stopped'}")
        print(f"  Manager:      {snapshot.manager}")
        if snapshot.gateway_pids:
            print(f"  PID(s):       {_format_gateway_pids(snapshot.gateway_pids)}")
        if snapshot.has_process_service_mismatch:
            print("  Service:      installed but not managing the current running gateway")
        elif _is_termux() and not snapshot.gateway_pids:
            print("  Start with:   pilotage gateway")
            print("  Note:         Android may stop background jobs when Termux is suspended")
        elif snapshot.service_installed and not snapshot.service_running:
            print("  Service:      installed but stopped")
    except Exception:
        if _is_termux():
            print(f"  Status:       {color('unknown', Colors.DIM)}")
            print("  Manager:      Termux / manual process")
        elif sys.platform.startswith('linux'):
            print(f"  Status:       {color('unknown', Colors.DIM)}")
            print("  Manager:      systemd/manual")
        elif sys.platform == 'darwin':
            print(f"  Status:       {color('unknown', Colors.DIM)}")
            print("  Manager:      launchd")
        else:
            print(f"  Status:       {color('N/A', Colors.DIM)}")
            print("  Manager:      (not supported on this platform)")

    # =========================================================================
    # Cron Jobs
    # =========================================================================
    print()
    print(color("◆ Scheduled Jobs", Colors.CYAN, Colors.BOLD))

    jobs_file = get_pilotage_home() / "cron" / "jobs.json"
    if jobs_file.exists():
        import json
        try:
            # utf-8-sig: same dialect as cron/jobs.load_jobs — Windows editors
            # may leave a UTF-8 BOM that plain utf-8 json.load rejects.
            with open(jobs_file, encoding="utf-8-sig") as f:
                data = json.load(f)
                jobs = data.get("jobs", [])
                enabled_jobs = [j for j in jobs if j.get("enabled", True)]
                print(f"  Jobs:         {len(enabled_jobs)} active, {len(jobs)} total")
        except Exception:
            print("  Jobs:         (error reading jobs file)")
    else:
        print("  Jobs:         0")

    # =========================================================================
    # Sessions
    # =========================================================================
    print()
    print(color("◆ Sessions", Colors.CYAN, Colors.BOLD))

    # Gateway session count: state.db is the source of truth;
    # fall back to sessions.json for pre-migration installs.
    _session_count = None
    _gateway_rows = []
    try:
        from pilotage_state import SessionDB
        _db = SessionDB()
        try:
            _lister = getattr(_db, "list_gateway_sessions", None)
            if callable(_lister):
                _gateway_rows = _lister(active_only=True) or []
                _session_count = len(_gateway_rows)
        finally:
            _db.close()
    except Exception:
        _session_count = None
        _gateway_rows = []

    if _session_count is not None and _session_count > 0:
        print(f"  Active:       {_session_count} session(s)")
        freshest = max(
            (float(r.get("last_active") or 0) for r in _gateway_rows),
            default=0.0,
        )
        if freshest > 0:
            print(f"  Last activity:{_format_relative_ts(freshest):>13}")
    else:
        sessions_file = get_pilotage_home() / "sessions" / "sessions.json"
        if sessions_file.exists():
            import json
            try:
                with open(sessions_file, encoding="utf-8") as f:
                    data = json.load(f)
                    _entries = {
                        k: v for k, v in data.items()
                        if not str(k).startswith("_")
                    } if isinstance(data, dict) else {}
                    print(f"  Active:       {len(_entries)} session(s)")
            except Exception:
                print("  Active:       (error reading sessions file)")
        else:
            print(f"  Active:       {_session_count if _session_count is not None else 0}")

    # Slot usage, only when max_concurrent_sessions is set. The cap is shared
    # across CLI, desktop/TUI and the messaging gateway, so the surface that
    # gets rejected is rarely the one holding the slots — without this the only
    # way to find out is reading runtime/active_sessions.json by hand.
    try:
        from pilotage_cli.active_sessions import (
            active_session_registry_snapshot,
            format_age,
            resolve_max_concurrent_sessions,
        )

        _cap = resolve_max_concurrent_sessions(config)
    except Exception:
        _cap = None
    if _cap:
        try:
            _held = active_session_registry_snapshot()
        except Exception:
            _held = []
        _full = len(_held) >= _cap
        print(
            "  Slots:        "
            + color(
                f"{len(_held)}/{_cap} in use", Colors.YELLOW if _full else Colors.GREEN
            )
        )
        _now = time.time()
        for _entry in sorted(_held, key=lambda e: e.get("started_at") or 0):
            _age = format_age(_now - float(_entry.get("started_at") or _now))
            print(
                f"                {_entry.get('surface') or 'unknown':<17} "
                f"{_entry.get('session_id') or '?':<24} {_age}"
            )

    # =========================================================================
    # Deep checks
    # =========================================================================
    if deep:
        print()
        print(color("◆ Deep Checks", Colors.CYAN, Colors.BOLD))
        
        # Check gateway port
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', 18789))
            sock.close()
            # Port in use = gateway likely running
            port_in_use = result == 0
            # This is informational, not necessarily bad
            print(f"  Port 18789:   {'in use' if port_in_use else 'available'}")
        except OSError:
            pass

    print()
    print(color("─" * 60, Colors.DIM))
    print(color("  Run 'pilotage doctor' for detailed diagnostics", Colors.DIM))
    print(color("  Run 'pilotage setup' to configure", Colors.DIM))
    print()
