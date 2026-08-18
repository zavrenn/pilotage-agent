"""Runtime config loading for the Pilotage process.

Extracted from the deleted interactive ``cli.py``: loads config.yaml, applies
the managed-scope overlay, and bridges terminal/auxiliary/security settings to
the environment variables the tools read.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from pilotage_constants import get_pilotage_home
from utils import fast_safe_load

logger = logging.getLogger(__name__)

_pilotage_home = get_pilotage_home()


def load_cli_config() -> Dict[str, Any]:
    """
    Load CLI configuration from config files.
    
    Config lookup order:
    1. ~/.pilotage/config.yaml (user config - preferred)
    2. ./cli-config.yaml (project config - fallback)
    
    Environment variables take precedence over config file values.
    Returns default values if no config file exists.

    If PILOTAGE_IGNORE_USER_CONFIG=1 is set (via ``pilotage chat --ignore-user-config``),
    the user config at ``~/.pilotage/config.yaml`` is skipped entirely and only the
    built-in defaults plus the project-level ``cli-config.yaml`` (if any) are used.
    Credentials in ``.env`` are still loaded — this flag only suppresses
    behavioral/config settings.
    """
    # Check user config first ({PILOTAGE_HOME}/config.yaml)
    user_config_path = _pilotage_home / 'config.yaml'
    project_config_path = Path(__file__).resolve().parent.parent / 'cli-config.yaml'

    # --ignore-user-config: force-skip the user config.yaml (still honor project
    # config as a fallback so defaults stay sensible).
    ignore_user_config = os.environ.get("PILOTAGE_IGNORE_USER_CONFIG") == "1"

    # Use user config if it exists, otherwise project config
    if user_config_path.exists() and not ignore_user_config:
        config_path = user_config_path
    else:
        config_path = project_config_path

    # Default configuration
    defaults = {
        "model": {
            "default": "",
            "base_url": "",
            "provider": "auto",
        },
        "terminal": {
            "env_type": "local",
            "cwd": ".",  # "." is resolved to os.getcwd() at runtime
            "home_mode": "auto",
            "lifetime_seconds": 300,
        },
        "compression": {
            "enabled": True,      # Auto-compress when approaching context limit
            "threshold": 0.50,    # Compress at 50% of model's context limit
            "min_tail_user_messages": 1,  # Real user messages guaranteed in the tail (1 = existing single anchor)
        },
        "agent": {
            "max_turns": 500,  # Default max tool-calling iterations (shared with subagents)
            "verbose": False,
            "system_prompt": "",
            "prefill_messages_file": "",
            "reasoning_effort": "",
            "service_tier": "",
            # Built-in personalities live in pilotage_cli.personality
            # (BUILTIN_PERSONALITIES) — the single owner. Entries here are
            # user-defined additions/overrides merged on top by name.
            "personalities": {},
        },

        "display": {
            "compact": False,
            "resume_display": "full",
            # Recap tuning for /resume — see pilotage_cli/config.py DEFAULT_CONFIG.
            "resume_exchanges": 10,
            "resume_max_user_chars": 300,
            "resume_max_assistant_chars": 200,
            "resume_max_assistant_lines": 3,
            "resume_skip_tool_only": True,
            # Live reasoning display default ON — keep in sync with
            # pilotage_cli/config.py DEFAULT_CONFIG (display.show_reasoning).
            "show_reasoning": True,
            "reasoning_full": False,
            "streaming": True,
            "busy_input_mode": "interrupt",
            "persistent_output": True,
            "persistent_output_max_lines": 200,
            # Clear terminal scrollback as well as the visible viewport when the
            # classic CLI performs a full redraw/resize recovery. Disabled by
            # default because some users prefer preserving terminal history;
            # enable when a terminal/tmux stack stamps stale prompt chrome into
            # scrollback during fullscreen/restore resizes.
            "cli_rebuild_scrollback_on_redraw": False,
            # Print a one-line summary of resolved modal prompts (approval /
            # clarify) into scrollback so the decision survives the repaint.
            "persist_prompts": True,

            "skin": "default",
        },
        "clarify": {
            "timeout": 120,  # Seconds to wait for a clarify answer before auto-proceeding
        },
        "code_execution": {
            "timeout": 300,    # Max seconds a sandbox script can run before being killed (5 min)
            "max_tool_calls": 50,  # Max RPC tool calls per execution
        },
        "auxiliary": {
            "vision": {
                "provider": "auto",
                "model": "",
                "base_url": "",
                "api_key": "",
            },
            "web_extract": {
                "provider": "auto",
                "model": "",
                "base_url": "",
                "api_key": "",
            },
        },
        "delegation": {
            "max_iterations": 45,  # Max tool-calling turns per child agent
            "model": "",       # Subagent model override (empty = inherit parent model)
            "provider": "",    # Subagent provider override (empty = inherit parent provider)
            "base_url": "",    # Direct OpenAI-compatible endpoint for subagents
            "api_key": "",     # API key for delegation.base_url (falls back to OPENAI_API_KEY)
        },
        "onboarding": {
            # First-touch hint flags (see agent/onboarding.py).  Each hint is
            # shown once per install then latched here.
            "seen": {},
        },
    }
    
    # Track whether the config file explicitly set terminal config.
    # When using defaults (no config file / no terminal section), we should NOT
    # overwrite env vars that were already set by .env -- only a user's config
    # file should be authoritative.
    _file_has_terminal_config = False

    # Load from file if exists
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                from pilotage_cli.config import _normalize_root_model_keys

                file_config = _normalize_root_model_keys(fast_safe_load(f) or {})
            
            _file_has_terminal_config = "terminal" in file_config

            # Handle model config - can be string (new format) or dict (old format)
            if "model" in file_config:
                if isinstance(file_config["model"], str):
                    # New format: model is just a string, convert to dict structure
                    defaults["model"]["default"] = file_config["model"]
                elif isinstance(file_config["model"], dict):
                    # Old format: model is a dict with default/base_url
                    defaults["model"].update(file_config["model"])
                    # If the user config sets model.model but not model.default,
                    # promote model.model to model.default so the user's explicit
                    # choice isn't shadowed by the hardcoded default.  Without this,
                    # profile configs that only set "model:" (not "default:") silently
                    # fall back to claude-opus because the merge preserves the
                    # hardcoded default and PilotageCLI.__init__ checks "default" first.
                    if "model" in file_config["model"] and "default" not in file_config["model"]:
                        defaults["model"]["default"] = file_config["model"]["model"]

            # Deep merge file_config into defaults.
            # First: merge keys that exist in both (deep-merge dicts, overwrite scalars)
            for key in defaults:
                if key == "model":
                    continue  # Already handled above
                if key in file_config:
                    if isinstance(defaults[key], dict) and file_config[key] is None:
                        continue
                    if isinstance(defaults[key], dict) and isinstance(file_config[key], dict):
                        defaults[key].update(file_config[key])
                    else:
                        defaults[key] = file_config[key]
            
            # Second: carry over keys from file_config that aren't in defaults
            # (e.g. platform_toolsets, provider_routing, memory, honcho, etc.)
            for key in file_config:
                if key not in defaults and key != "model":
                    defaults[key] = file_config[key]
            
            # Handle legacy root-level max_turns (backwards compat) - copy to
            # agent.max_turns whenever the nested key is missing.
            agent_file_config = file_config.get("agent")
            if "max_turns" in file_config and not (
                isinstance(agent_file_config, dict)
                and agent_file_config.get("max_turns") is not None
            ):
                defaults["agent"]["max_turns"] = file_config["max_turns"]
        except Exception as e:
            logger.warning("Failed to load cli-config.yaml: %s", e)

    # Expand ${ENV_VAR} references in config values before bridging to env vars.
    from pilotage_cli.config import _expand_env_vars
    defaults = _expand_env_vars(defaults)

    # Managed scope: overlay administrator-pinned values LAST so they win over
    # the user's config here too. cli.py builds its config independently of
    # pilotage_cli.config._load_config_impl (which has its own managed merge), so
    # without this the entire interactive CLI/TUI surface — skin, display prefs,
    # etc. read from CLI_CONFIG — would silently ignore managed scope while
    # `pilotage config`/`doctor`/guards (which use load_config) honor it. The
    # shared helper mirrors _load_config_impl (env-only expansion, root-model
    # normalization, leaf-merge) and is fail-open.
    from pilotage_cli import managed_scope

    defaults = managed_scope.apply_managed_overlay(defaults)

    # Apply terminal config to environment variables (so terminal_tool picks them up)
    terminal_config = defaults.get("terminal", {})
    
    # Normalize config key: the new config system (pilotage_cli/config.py) and all
    # documentation use "backend", the legacy cli-config.yaml uses "env_type".
    # Accept both, with "backend" taking precedence (it's the documented key).
    if "backend" in terminal_config:
        terminal_config["env_type"] = terminal_config["backend"]
    
    # CWD resolution for CLI/TUI. The gateway has its own config bridge in
    # gateway/run.py but may lazily import cli.py (triggering this code).
    # Local backend: always os.getcwd(). Use `cd /dir && pilotage` to control it.
    # Non-local with placeholder: pop so terminal_tool uses its per-backend default.
    # Non-local with explicit path: keep as-is.
    _CWD_PLACEHOLDERS = (".", "auto", "cwd")
    effective_backend = terminal_config.get("env_type", "local")

    if effective_backend == "local":
        terminal_config["cwd"] = os.getcwd()
        defaults["terminal"]["cwd"] = terminal_config["cwd"]
    elif terminal_config.get("cwd") in _CWD_PLACEHOLDERS:
        terminal_config.pop("cwd", None)
    
    env_mappings = {
        "env_type": "TERMINAL_ENV",
        "degraded_mode": "TERMINAL_DEGRADED_MODE",
        "cwd": "TERMINAL_CWD",
        "timeout": "TERMINAL_TIMEOUT",
        "home_mode": "TERMINAL_HOME_MODE",
        "lifetime_seconds": "TERMINAL_LIFETIME_SECONDS",
        "sandbox_dir": "TERMINAL_SANDBOX_DIR",
        "persistent_shell": "TERMINAL_PERSISTENT_SHELL",
        "sudo_password": "SUDO_PASSWORD",
    }
    
    # Bridge config → env vars for terminal_tool. TERMINAL_CWD is force-exported
    # UNLESS we're inside a gateway process (detected by _PILOTAGE_GATEWAY marker)
    # where it was already set correctly by gateway/run.py's config bridge.
    _is_gateway = os.environ.get("_PILOTAGE_GATEWAY") == "1"
    for config_key, env_var in env_mappings.items():
        if config_key in terminal_config:
            if env_var == "TERMINAL_CWD":
                if _is_gateway:
                    continue
                # CLI: always export (overrides stale .env or inherited values)
                os.environ[env_var] = str(terminal_config[config_key])
                continue
            if _file_has_terminal_config or env_var not in os.environ:
                val = terminal_config[config_key]
                if isinstance(val, (list, dict)):
                    os.environ[env_var] = json.dumps(val)
                else:
                    os.environ[env_var] = str(val)
    
    # Apply auxiliary model/direct-endpoint overrides to environment variables.
    # Vision and web_extract each have their own provider/model/base_url/api_key tuple.
    # Compression config is read directly from config.yaml by run_agent.py and
    # auxiliary_client.py — no env var bridging needed.
    # Only set env vars for non-empty / non-default values so auto-detection
    # still works.
    auxiliary_config = defaults.get("auxiliary", {})
    auxiliary_task_env = {
        # config key → env var mapping
        "vision": {
            "provider": "AUXILIARY_VISION_PROVIDER",
            "model": "AUXILIARY_VISION_MODEL",
            "base_url": "AUXILIARY_VISION_BASE_URL",
            "api_key": "AUXILIARY_VISION_API_KEY",
        },
        "web_extract": {
            "provider": "AUXILIARY_WEB_EXTRACT_PROVIDER",
            "model": "AUXILIARY_WEB_EXTRACT_MODEL",
            "base_url": "AUXILIARY_WEB_EXTRACT_BASE_URL",
            "api_key": "AUXILIARY_WEB_EXTRACT_API_KEY",
        },
        "approval": {
            "provider": "AUXILIARY_APPROVAL_PROVIDER",
            "model": "AUXILIARY_APPROVAL_MODEL",
            "base_url": "AUXILIARY_APPROVAL_BASE_URL",
            "api_key": "AUXILIARY_APPROVAL_API_KEY",
        },
    }
    
    for task_key, env_map in auxiliary_task_env.items():
        task_cfg = auxiliary_config.get(task_key, {})
        if not isinstance(task_cfg, dict):
            continue
        prov = str(task_cfg.get("provider", "")).strip()
        model = str(task_cfg.get("model", "")).strip()
        base_url = str(task_cfg.get("base_url", "")).strip()
        api_key = str(task_cfg.get("api_key", "")).strip()
        if prov and prov != "auto":
            os.environ[env_map["provider"]] = prov
        if model:
            os.environ[env_map["model"]] = model
        if base_url:
            os.environ[env_map["base_url"]] = base_url
        if api_key:
            os.environ[env_map["api_key"]] = api_key
    
    # Security settings
    security_config = defaults.get("security", {})
    if isinstance(security_config, dict):
        redact = security_config.get("redact_secrets")
        if redact is not None:
            os.environ["PILOTAGE_REDACT_SECRETS"] = str(redact).lower()

    # Session-search index knobs (pilotage_state reads the env carriers).
    sessions_config = defaults.get("sessions", {})
    if isinstance(sessions_config, dict):
        if "cjk_fts" in sessions_config:
            os.environ["PILOTAGE_CJK_FTS"] = str(sessions_config["cjk_fts"])
        if "search_slow_ms" in sessions_config:
            os.environ["PILOTAGE_SEARCH_SLOW_MS"] = str(
                sessions_config["search_slow_ms"]
            )

    return defaults

# Load configuration at module startup
CLI_CONFIG = load_cli_config()


def save_config_value(key_path: str, value: any) -> bool:
    """
    Save a value to the active config file at the specified key path.
    
    Respects the same lookup order as load_cli_config():
    1. ~/.pilotage/config.yaml (user config - preferred, used if it exists)
    2. ./cli-config.yaml (project config - fallback)
    
    Args:
        key_path: Dot-separated path like "agent.system_prompt"
        value: Value to save
    
    Returns:
        True if successful, False otherwise
    """
    # Runtime persistence ALWAYS targets the user's PILOTAGE_HOME config.yaml,
    # creating it if needed. Resolve PILOTAGE_HOME live (not the import-time
    # _pilotage_home constant) so profile switches and test isolation land right.
    #
    # We deliberately do NOT fall back to the repo's project cli-config.yaml:
    # that file is a shipped default/template, and most config readers
    # (load_config reads get_pilotage_home()/config.yaml) never read it.
    # Writing a user setting there means
    # the reader never sees it. This was the "wake-word ear reverts to disabled
    # after restart" bug — the toggle's persist wrote to cli-config.yaml (which
    # exists in the checkout) while startup read PILOTAGE_HOME/config.yaml, so the
    # setting silently vanished every restart on any install whose
    # PILOTAGE_HOME/config.yaml didn't exist yet.
    config_path = get_pilotage_home() / 'config.yaml'
    
    try:
        # Ensure parent directory exists (for ~/.pilotage/config.yaml on first use)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save back atomically while preserving comments, ordering, quotes, and
        # readable Unicode in user-edited config.yaml.
        from utils import atomic_roundtrip_yaml_update
        atomic_roundtrip_yaml_update(config_path, key_path, value)
        
        # Enforce owner-only permissions on config files (contain API keys)
        try:
            os.chmod(config_path, 0o600)
        except (OSError, NotImplementedError):
            pass

        # Model/provider changes made through /model and the TUI use this
        # persistence path rather than ``pilotage config set``. Surface the same
        # fail-closed cron drift warning for every operator-facing model switch.
        from pilotage_cli.config import (
            warn_unpinned_cron_jobs_after_model_config_change,
        )

        warn_unpinned_cron_jobs_after_model_config_change(key_path, value)
        
        return True
    except Exception as e:
        logger.error("Failed to save config: %s", e)
        return False
