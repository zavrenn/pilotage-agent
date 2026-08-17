"""
Interactive setup wizard for Pilotage Agent.

Modular wizard with independently-runnable sections:
  1. Model & Provider — choose your AI provider and model
  2. Terminal Backend — where your agent runs commands
  3. Agent Settings — iterations, compression, session reset
  4. Messaging Platforms — connect Telegram, WhatsApp
  5. Tools — configure TTS, web search, image generation, etc.

Config files are stored in ~/.pilotage/ for easy access.
"""

import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import copy
from pathlib import Path
from typing import Optional, Dict, Any

from pilotage_constants import get_optional_skills_dir

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

_DOCS_BASE = ""


def _model_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    current_model = config.get("model")
    if isinstance(current_model, dict):
        return dict(current_model)
    if isinstance(current_model, str) and current_model.strip():
        return {"default": current_model.strip()}
    return {}


def _get_credential_pool_strategies(config: Dict[str, Any]) -> Dict[str, str]:
    strategies = config.get("credential_pool_strategies")
    return dict(strategies) if isinstance(strategies, dict) else {}


def _set_credential_pool_strategy(config: Dict[str, Any], provider: str, strategy: str) -> None:
    if not provider:
        return
    strategies = _get_credential_pool_strategies(config)
    strategies[provider] = strategy
    config["credential_pool_strategies"] = strategies


def _supports_same_provider_pool_setup(provider: str) -> bool:
    if not provider or provider == "custom":
        return False
    from pilotage_cli.auth import PROVIDER_REGISTRY

    pconfig = PROVIDER_REGISTRY.get(provider)
    if not pconfig:
        return False
    return pconfig.auth_type in {"api_key", "oauth_device_code"}


# Default model lists per provider — used as fallback when the live
# /models endpoint can't be reached.
_DEFAULT_PROVIDER_MODELS = {
    "openai": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-api": [
        "gpt-5.6-sol",
        "gpt-5.6-sol-pro",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-codex": [
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-5-codex",
    ],
}


def _current_reasoning_effort(config: Dict[str, Any]) -> str:
    agent_cfg = config.get("agent")
    if isinstance(agent_cfg, dict):
        return str(agent_cfg.get("reasoning_effort") or "").strip().lower()
    return ""


def _set_reasoning_effort(config: Dict[str, Any], effort: str) -> None:
    agent_cfg = config.get("agent")
    if not isinstance(agent_cfg, dict):
        agent_cfg = {}
        config["agent"] = agent_cfg
    agent_cfg["reasoning_effort"] = effort




# Import config helpers
from pilotage_cli.config import (
    cfg_get,
    DEFAULT_CONFIG,
    get_pilotage_home,
    get_config_path,
    get_env_path,
    load_config,
    save_config,
    save_env_value,
    remove_env_value,
    get_env_value,
    ensure_pilotage_home,
)
# display_pilotage_home imported lazily at call sites (stale-module safety during pilotage update)

from pilotage_cli.colors import Colors, color


def print_header(title: str):
    """Print a section header."""
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


from pilotage_cli.cli_output import (  # noqa: E402
    print_error,
    print_info,
    print_success,
    print_warning,
)
from pilotage_cli.secret_prompt import masked_secret_prompt  # noqa: E402


def is_interactive_stdin() -> bool:
    """Return True when stdin looks like a usable interactive TTY."""
    stdin = getattr(sys, "stdin", None)
    if stdin is None:
        return False
    try:
        return bool(stdin.isatty())
    except Exception:
        return False


def print_noninteractive_setup_guidance(reason: str | None = None) -> None:
    """Print guidance for headless/non-interactive setup flows."""
    print()
    print(color("⚕ Pilotage Setup — Non-interactive mode", Colors.CYAN, Colors.BOLD))
    print()
    if reason:
        print_info(reason)
    print_info("The interactive wizard cannot be used here.")
    print()
    print_info("Configure Pilotage using environment variables or config commands:")
    print_info("  pilotage config set model.provider custom")
    print_info("  pilotage config set model.base_url http://localhost:8080/v1")
    print_info("  pilotage config set model.default your-model-name")
    print()
    print_info("Or set OPENROUTER_API_KEY / OPENAI_API_KEY in your environment.")
    print_info("Run 'pilotage setup' in an interactive terminal to use the full wizard.")
    print()


def prompt(question: str, default: str = None, password: bool = False) -> str:
    """Prompt for input with optional default."""
    if default:
        display = f"{question} [{default}]: "
    else:
        display = f"{question}: "

    try:
        if password:
            value = masked_secret_prompt(color(display, Colors.YELLOW))
        else:
            value = input(color(display, Colors.YELLOW))

        cleaned = _sanitize_pasted_input(value)
        return cleaned.strip() or default or ""
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(1)


_BRACKETED_PASTE_PATTERN = re.compile(r"\x1b\[\s*200~|\x1b\[\s*201~")


def _sanitize_pasted_input(value: str) -> str:
    """Strip terminal bracketed-paste control markers from pasted text."""
    if not isinstance(value, str) or not value:
        return value
    return _BRACKETED_PASTE_PATTERN.sub("", value)


def _curses_prompt_choice(question: str, choices: list, default: int = 0, description: str | None = None) -> int:
    """Single-select menu using curses. Delegates to curses_radiolist."""
    from pilotage_cli.curses_ui import curses_radiolist
    return curses_radiolist(question, choices, selected=default, cancel_returns=-1, description=description)



def prompt_choice(question: str, choices: list, default: int = 0, description: str | None = None) -> int:
    """Prompt for a choice from a list with arrow key navigation.

    Escape keeps the current default (skips the question).
    Ctrl+C exits the wizard.
    """
    idx = _curses_prompt_choice(question, choices, default, description=description)
    if idx >= 0:
        if idx == default:
            print_info("  Skipped (keeping current)")
            print()
            return default
        print()
        return idx

    print(color(question, Colors.YELLOW))
    for i, choice in enumerate(choices):
        marker = "●" if i == default else "○"
        if i == default:
            print(color(f"  {marker} {choice}", Colors.GREEN))
        else:
            print(f"  {marker} {choice}")

    print_info(f"  Enter for default ({default + 1})  Ctrl+C to exit")

    while True:
        try:
            value = input(
                color(f"  Select [1-{len(choices)}] ({default + 1}): ", Colors.DIM)
            )
            if not value:
                return default
            idx = int(value) - 1
            if 0 <= idx < len(choices):
                return idx
            print_error(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print_error("Please enter a number")
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(1)


def is_noninteractive() -> bool:
    """True when no human is available to answer a prompt.

    The dashboard/desktop spawn CLI actions with ``stdin=DEVNULL`` and
    ``PILOTAGE_NONINTERACTIVE=1`` (see ``pilotage_cli/web_server.py``). In that
    context an ``input()`` raises ``EOFError`` immediately, so a prompt that
    aborts on EOF kills the spawned action — this is what made the desktop
    "restart gateway" fail when the Windows gateway service was not yet
    installed (the start path asks "Install it now?" with no one to answer).
    Honour the explicit env flag here so callers fall back to their default.
    """
    return os.environ.get("PILOTAGE_NONINTERACTIVE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt for yes/no. Ctrl+C exits, empty input returns default.

    Non-interactive callers (``PILOTAGE_NONINTERACTIVE=1`` or a closed/redirected
    stdin) have no one to answer, so fall back to ``default`` instead of
    aborting the whole process.
    """
    if is_noninteractive():
        return default

    default_str = "Y/n" if default else "y/N"

    while True:
        try:
            value = (
                input(color(f"{question} [{default_str}]: ", Colors.YELLOW))
                .strip()
                .lower()
            )
        except KeyboardInterrupt:
            print()
            sys.exit(1)
        except EOFError:
            # No stdin to read (closed/redirected, e.g. a spawned action with
            # stdin=DEVNULL). Accept the default rather than exit so the caller
            # can proceed unattended instead of failing the whole command.
            print()
            return default

        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print_error("Please enter 'y' or 'n'")


def prompt_checklist(title: str, items: list, pre_selected: list = None) -> list:
    """
    Display a multi-select checklist and return the indices of selected items.

    Each item in `items` is a display string. `pre_selected` is a list of
    indices that should be checked by default. A "Continue →" option is
    appended at the end — the user toggles items with Space and confirms
    with Enter on "Continue →".

    Falls back to a numbered toggle interface when curses is
    unavailable.

    Returns:
        List of selected indices (not including the Continue option).
    """
    if pre_selected is None:
        pre_selected = []

    from pilotage_cli.curses_ui import curses_checklist

    chosen = curses_checklist(
        title,
        items,
        set(pre_selected),
        cancel_returns=set(pre_selected),
    )
    return sorted(chosen)


def _prompt_api_key(var: dict):
    """Display a nicely formatted API key input screen for a single env var."""
    tools = var.get("tools", [])
    tools_str = ", ".join(tools[:3])
    if len(tools) > 3:
        tools_str += f", +{len(tools) - 3} more"

    print()
    print(color(f"  ─── {var.get('description', var['name'])} ───", Colors.CYAN))
    print()
    if tools_str:
        print_info(f"  Enables: {tools_str}")
    if var.get("url"):
        print_info(f"  Get your key at: {var['url']}")
    print()

    if var.get("password"):
        value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
    else:
        value = prompt(f"  {var.get('prompt', var['name'])}")

    if value:
        save_env_value(var["name"], value)
        print_success("  ✓ Saved")
    else:
        print_warning("  Skipped (configure later with 'pilotage setup')")


def _print_setup_summary(config: dict, pilotage_home):
    """Print the setup completion summary."""
    # Provider readiness — the one thing setup absolutely must produce.
    # Previously a user could cancel the API-key prompt mid-wizard (Enter →
    # "Cancelled."), watch the wizard continue through Terminal/Gateway/Tools,
    # and exit "successfully" with NO working model — believing they were set
    # up. Say so loudly instead (consumer-onboarding audit finding #7).
    try:
        from pilotage_cli.auth import resolve_provider

        resolve_provider()
        _provider_ready = True
    except Exception:
        _provider_ready = False
    if not _provider_ready:
        print()
        print_warning("No inference provider is configured — Pilotage cannot chat yet.")
        print_info("  Finish this one step with either of:")
        print_info("    pilotage model            (pick any provider/model)")

    # Tool availability summary
    print()
    print_header("Tool Availability Summary")

    tool_status = []

    # Vision — use the same runtime resolver as the actual vision tools
    try:
        from agent.auxiliary_client import get_available_vision_backends

        _vision_backends = get_available_vision_backends()
    except Exception:
        _vision_backends = []

    if _vision_backends:
        tool_status.append(("Vision (image analysis)", True, None))
    else:
        tool_status.append(("Vision (image analysis)", False, "run 'pilotage setup' to configure"))


    # Web tools (Exa, Parallel, Firecrawl, or Tavily)
    _web_keys = [
        ("EXA_API_KEY", "Exa"),
        ("PARALLEL_API_KEY", "Parallel"),
        ("FIRECRAWL_API_KEY", "Firecrawl"),
        ("FIRECRAWL_API_URL", "Firecrawl"),
        ("TAVILY_API_KEY", "Tavily"),
        ("SEARXNG_URL", "SearXNG"),
    ]
    _web_provider = next((n for k, n in _web_keys if get_env_value(k)), None)
    if _web_provider:
        tool_status.append((f"Web Search & Extract ({_web_provider})", True, None))
    else:
        tool_status.append(("Web Search & Extract", False, "EXA_API_KEY, PARALLEL_API_KEY, FIRECRAWL_API_KEY/FIRECRAWL_API_URL, TAVILY_API_KEY, or SEARXNG_URL"))

    # Image generation — FAL, or any plugin-registered provider (OpenAI, etc.)
    if get_env_value("FAL_KEY"):
        tool_status.append(("Image Generation", True, None))
    else:
        # Fall back to probing plugin-registered providers so OpenAI-only
        # setups don't show as "missing FAL_KEY".
        _img_backend = None
        try:
            from agent.image_gen_registry import list_providers
            from pilotage_cli.plugins import _ensure_plugins_discovered

            _ensure_plugins_discovered()
            for _p in list_providers():
                if _p.name == "fal":
                    continue
                try:
                    if _p.is_available():
                        _img_backend = _p.display_name
                        break
                except Exception:
                    continue
        except Exception:
            pass
        if _img_backend:
            tool_status.append((f"Image Generation ({_img_backend})", True, None))
        else:
            tool_status.append(("Image Generation", False, "FAL_KEY or OPENAI_API_KEY"))

    # TTS status (OpenAI is the only backend)
    if get_env_value("VOICE_TOOLS_OPENAI_KEY") or get_env_value("OPENAI_API_KEY"):
        tool_status.append(("Text-to-Speech (OpenAI)", True, None))
    else:
        tool_status.append(
            ("Text-to-Speech", False, "VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY")
        )

    # STT — show configured provider
    stt_provider = cfg_get(config, "stt", "provider", default="openai") or "openai"
    if stt_provider == "openai" and (
        get_env_value("VOICE_TOOLS_OPENAI_KEY") or get_env_value("OPENAI_API_KEY")
    ):
        tool_status.append(("Speech-to-Text (OpenAI)", True, None))
    elif stt_provider != "openai":
        # Plugin-registered provider — resolution happens at call time.
        tool_status.append((f"Speech-to-Text ({stt_provider})", True, None))
    else:
        tool_status.append(
            ("Speech-to-Text (OpenAI — no API key)", False, "set VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY")
        )

    if cfg_get(config, "terminal", "backend") == "modal":
        from tools.tool_backend_helpers import has_direct_modal_credentials

        if has_direct_modal_credentials():
            tool_status.append(("Modal Execution (direct Modal)", True, None))
        else:
            tool_status.append(("Modal Execution", False, "run 'pilotage setup terminal'"))

    # Home Assistant
    if get_env_value("HASS_TOKEN"):
        tool_status.append(("Smart Home (Home Assistant)", True, None))

    # Skills Hub
    if get_env_value("GITHUB_TOKEN"):
        tool_status.append(("Skills Hub (GitHub)", True, None))
    else:
        tool_status.append(("Skills Hub (GitHub)", False, "GITHUB_TOKEN"))

    # Terminal (always available if system deps met)
    tool_status.append(("Terminal/Commands", True, None))

    # Task planning (always available, in-memory)
    tool_status.append(("Task Planning (todo)", True, None))

    # Skills (always available -- bundled skills + user-created skills)
    tool_status.append(("Skills (view, create, edit)", True, None))

    # Print status
    available_count = sum(1 for _, avail, _ in tool_status if avail)
    total_count = len(tool_status)

    print_info(f"{available_count}/{total_count} tool categories available:")
    print()

    for name, available, missing_var in tool_status:
        if available:
            print(f"   {color('✓', Colors.GREEN)} {name}")
        else:
            print(
                f"   {color('✗', Colors.RED)} {name} {color(f'(missing {missing_var})', Colors.DIM)}"
            )

    print()

    disabled_tools = [(name, var) for name, avail, var in tool_status if not avail]
    if disabled_tools:
        print_warning(
            "Some tools are disabled. Run 'pilotage setup tools' to configure them,"
        )
        from pilotage_constants import display_pilotage_home as _dhh
        print_warning(f"or edit {_dhh()}/.env directly to add the missing API keys.")
        print()

    # Done banner
    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────┐", Colors.GREEN
        )
    )
    print(
        color(
            "│              ✓ Setup Complete!                          │", Colors.GREEN
        )
    )
    print(
        color(
            "└─────────────────────────────────────────────────────────┘", Colors.GREEN
        )
    )
    print()

    # Show file locations prominently
    from pilotage_constants import display_pilotage_home as _dhh
    print(color(f"📁 All your files are in {_dhh()}/:", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('Settings:', Colors.YELLOW)}  {get_config_path()}")
    print(f"   {color('API Keys:', Colors.YELLOW)}  {get_env_path()}")
    print(
        f"   {color('Data:', Colors.YELLOW)}      {pilotage_home}/cron/, sessions/, logs/"
    )
    print()

    print(color("─" * 60, Colors.DIM))
    print()
    print(color("📝 To edit your configuration:", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('pilotage setup', Colors.GREEN)}          Re-run the full wizard")
    print(f"   {color('pilotage setup model', Colors.GREEN)}    Change model/provider")
    print(f"   {color('pilotage setup terminal', Colors.GREEN)} Change terminal backend")
    print(f"   {color('pilotage setup gateway', Colors.GREEN)}  Configure messaging")
    print(f"   {color('pilotage setup tools', Colors.GREEN)}    Configure tool providers")
    print()
    print(f"   {color('pilotage config', Colors.GREEN)}         View current settings")
    print(
        f"   {color('pilotage config edit', Colors.GREEN)}    Open config in your editor"
    )
    print(f"   {color('pilotage config set <key> <value>', Colors.GREEN)}")
    print("                          Set a specific value")
    print()
    print("   Or edit the files directly:")
    print(f"   {color(f'nano {get_config_path()}', Colors.DIM)}")
    print(f"   {color(f'nano {get_env_path()}', Colors.DIM)}")
    print()

    print(color("─" * 60, Colors.DIM))
    print()
    print(color("🚀 Ready to go!", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('pilotage', Colors.GREEN)}              Start chatting")
    print(f"   {color('pilotage gateway', Colors.GREEN)}      Start messaging gateway")
    print(f"   {color('pilotage doctor', Colors.GREEN)}       Check for issues")
    print()


def _prompt_container_resources(config: dict):
    """Prompt for container resource settings (Docker, Singularity, Modal, Daytona)."""
    terminal = config.setdefault("terminal", {})

    print()
    print_info("Container Resource Settings:")

    # Persistence
    current_persist = terminal.get("container_persistent", True)
    persist_label = "yes" if current_persist else "no"
    print_info("  Persistent filesystem keeps files between sessions.")
    print_info("  Set to 'no' for ephemeral sandboxes that reset each time.")
    persist_str = prompt(
        "  Persist filesystem across sessions? (yes/no)", persist_label
    )
    terminal["container_persistent"] = persist_str.lower() in {"yes", "true", "y", "1"}

    # CPU
    current_cpu = terminal.get("container_cpu", 1)
    cpu_str = prompt("  CPU cores", str(current_cpu))
    try:
        terminal["container_cpu"] = float(cpu_str)
    except ValueError:
        pass

    # Memory
    current_mem = terminal.get("container_memory", 5120)
    mem_str = prompt("  Memory in MB (5120 = 5GB)", str(current_mem))
    try:
        terminal["container_memory"] = int(mem_str)
    except ValueError:
        pass

    # Disk
    current_disk = terminal.get("container_disk", 51200)
    disk_str = prompt("  Disk in MB (51200 = 50GB)", str(current_disk))
    try:
        terminal["container_disk"] = int(disk_str)
    except ValueError:
        pass


def _prompt_vercel_sandbox_settings(config: dict):
    """Prompt for Vercel Sandbox settings without exposing unsupported disk sizing."""
    terminal = config.setdefault("terminal", {})

    print()
    print_info("Vercel Sandbox settings:")
    print_info("  Filesystem persistence uses Vercel snapshots.")
    print_info("  Snapshots restore files only; live processes do not continue after sandbox recreation.")

    from tools.terminal_tool import _SUPPORTED_VERCEL_RUNTIMES

    current_runtime = terminal.get("vercel_runtime") or "node24"
    supported_label = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
    runtime = prompt(f"  Runtime ({supported_label})", current_runtime).strip() or current_runtime
    if runtime not in _SUPPORTED_VERCEL_RUNTIMES:
        print_warning(f"Unsupported Vercel runtime '{runtime}', keeping {current_runtime}.")
        runtime = current_runtime if current_runtime in _SUPPORTED_VERCEL_RUNTIMES else "node24"
    terminal["vercel_runtime"] = runtime
    save_env_value("TERMINAL_VERCEL_RUNTIME", runtime)

    current_persist = terminal.get("container_persistent", True)
    persist_label = "yes" if current_persist else "no"
    terminal["container_persistent"] = prompt(
        "  Persist filesystem with snapshots? (yes/no)", persist_label
    ).lower() in {"yes", "true", "y", "1"}

    current_cpu = terminal.get("container_cpu", 1)
    cpu_str = prompt("  CPU cores", str(current_cpu))
    try:
        terminal["container_cpu"] = float(cpu_str)
    except ValueError:
        pass

    current_mem = terminal.get("container_memory", 5120)
    mem_str = prompt("  Memory in MB (5120 = 5GB)", str(current_mem))
    try:
        terminal["container_memory"] = int(mem_str)
    except ValueError:
        pass

    if terminal.get("container_disk", 51200) not in {0, 51200}:
        print_warning("Vercel Sandbox does not support custom disk sizing; resetting container_disk to 51200.")
    terminal["container_disk"] = 51200

    print()
    print_info("Vercel authentication:")
    print_info("  Use a long-lived Vercel access token plus project/team IDs.")
    linked_project = _read_nearest_vercel_project()
    if linked_project:
        print_info("  Found defaults in nearest .vercel/project.json.")

    remove_env_value("VERCEL_OIDC_TOKEN")
    token = prompt("    Vercel access token", get_env_value("VERCEL_TOKEN") or "", password=True)
    project = prompt(
        "    Vercel project ID",
        get_env_value("VERCEL_PROJECT_ID") or linked_project.get("projectId", ""),
    )
    team = prompt(
        "    Vercel team ID",
        get_env_value("VERCEL_TEAM_ID") or linked_project.get("orgId", ""),
    )
    if token:
        save_env_value("VERCEL_TOKEN", token)
    if project:
        save_env_value("VERCEL_PROJECT_ID", project)
    if team:
        save_env_value("VERCEL_TEAM_ID", team)


def _read_nearest_vercel_project(start: Path | None = None) -> dict[str, str]:
    """Read project/team defaults from the nearest Vercel link file."""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        project_file = directory / ".vercel" / "project.json"
        if not project_file.exists():
            continue
        try:
            data = json.loads(project_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            key: value
            for key, value in {
                "projectId": data.get("projectId"),
                "orgId": data.get("orgId"),
            }.items()
            if isinstance(value, str) and value.strip()
        }
    return {}


# Tool categories and provider config are now in tools_config.py (shared
# between `pilotage tools` and `pilotage setup tools`).


# =============================================================================
# Section 1: Model & Provider Configuration
# =============================================================================



def setup_model_provider(config: dict, *, quick: bool = False):
    """Configure the inference provider and default model.

    Delegates to ``cmd_model()`` (the same flow used by ``pilotage model``)
    for provider selection, credential prompting, and model picking.
    This ensures a single code path for all provider setup — any new
    provider added to ``pilotage model`` is automatically available here.

    When *quick* is True, skips credential rotation, vision, and TTS
    configuration — used by the streamlined first-time quick setup.
    """
    from pilotage_cli.config import load_config, save_config

    print_header("Inference Provider")
    print_info("Choose how to connect to your main chat model.")
    print_info(f"   Guide: {_DOCS_BASE}/integrations/providers")
    print()

    # Delegate to the shared pilotage model flow — handles provider picker,
    # credential prompting, model selection, and config persistence.
    from pilotage_cli.main import select_provider_and_model
    try:
        select_provider_and_model()
    except (SystemExit, KeyboardInterrupt):
        print()
        print_info("Provider setup skipped.")
    except Exception as exc:
        logger.debug("select_provider_and_model error during setup: %s", exc)
        print_warning(f"Provider setup encountered an error: {exc}")
        print_info("You can try again later with: pilotage model")

    # Re-sync the wizard's config dict from what cmd_model saved to disk.
    # This is critical: cmd_model writes to disk via its own load/save cycle,
    # and the wizard's final save_config(config) must not overwrite those
    # changes with stale values. Refresh the dict in place so callers
    # that keep the same object see every section the shared model picker may
    # have changed (model, custom_providers, auxiliary, provider metadata, etc.).
    _refreshed = load_config()
    config.clear()
    config.update(_refreshed)

    # Credential rotation, vision-backend selection, and TTS provider are no
    # longer prompted here. They have safe defaults (rotation off, vision
    # auto-detected from the main provider, TTS = Edge) and are configurable
    # on demand via `pilotage auth add`, `pilotage setup` vision, and
    # `pilotage setup tts`. This keeps both quick and full setup thin.


    save_config(config)


# =============================================================================
# Section 1b: TTS Provider Configuration
# =============================================================================


def _setup_tts_provider(config: dict):
    """Confirm the OpenAI text-to-speech backend."""
    print()
    print_header("Text-to-Speech (optional)")
    print_info("Provider: OpenAI (the only built-in speech backend).")

    if "tts" not in config:
        config["tts"] = {}
    config["tts"]["provider"] = "openai"
    save_config(config)

    if get_env_value("VOICE_TOOLS_OPENAI_KEY") or get_env_value("OPENAI_API_KEY"):
        print_success("TTS provider set to: OpenAI TTS")
    else:
        print_warning(
            "Set VOICE_TOOLS_OPENAI_KEY or OPENAI_API_KEY to enable speech output."
        )


def setup_tts(config: dict):
    """Standalone TTS setup (for 'pilotage setup tts')."""
    _setup_tts_provider(config)


# =============================================================================
# Section 2: Terminal Backend Configuration
# =============================================================================


def setup_terminal_backend(config: dict):
    """Configure the terminal execution backend."""
    import platform as _platform
    print_header("Terminal Backend")
    print_info("Choose where Pilotage runs shell commands and code.")
    print_info("This affects tool execution, file access, and isolation.")
    print_info(f"   Guide: {_DOCS_BASE}/user-guide/configuration#terminal-backend-configuration")
    print()

    current_backend = cfg_get(config, "terminal", "backend", default="local")
    is_linux = _platform.system() == "Linux"

    # Build backend choices with descriptions
    terminal_choices = [
        "Local - run directly on this machine (default)",
        "Docker - isolated container with configurable resources",
        "Modal - serverless cloud sandbox",
        "SSH - run on a remote machine",
        "Daytona - persistent cloud development environment",
        "Vercel Sandbox - cloud microVM with snapshot filesystem persistence",
    ]
    idx_to_backend = {0: "local", 1: "docker", 2: "modal", 3: "ssh", 4: "daytona", 5: "vercel_sandbox"}
    backend_to_idx = {"local": 0, "docker": 1, "modal": 2, "ssh": 3, "daytona": 4, "vercel_sandbox": 5}

    next_idx = 6
    if is_linux:
        terminal_choices.append("Singularity/Apptainer - HPC-friendly container")
        idx_to_backend[next_idx] = "singularity"
        backend_to_idx["singularity"] = next_idx
        next_idx += 1

    # Add keep current option
    keep_current_idx = next_idx
    terminal_choices.append(f"Keep current ({current_backend})")
    idx_to_backend[keep_current_idx] = current_backend

    terminal_idx = prompt_choice(
        "Select terminal backend:", terminal_choices, keep_current_idx
    )

    selected_backend = idx_to_backend.get(terminal_idx)

    if terminal_idx == keep_current_idx:
        print_info(f"Keeping current backend: {current_backend}")
        return

    config.setdefault("terminal", {})["backend"] = selected_backend

    if selected_backend == "local":
        print_success("Terminal backend: Local")
        print_info("Commands run directly on this machine.")
        # Gateway working directory defaults to home; sudo stays off. Both are
        # configurable later via `pilotage setup terminal` / config.yaml.
        config["terminal"].setdefault("cwd", str(Path.home()))

    elif selected_backend == "docker":
        print_success("Terminal backend: Docker")

        # Check if Docker is available
        docker_bin = shutil.which("docker")
        if not docker_bin:
            print_warning("Docker not found in PATH!")
            print_info("Install Docker: https://docs.docker.com/get-docker/")
        else:
            print_info(f"Docker found: {docker_bin}")

        # Image and resource limits use defaults; tune via `pilotage setup terminal`.
        config["terminal"].setdefault(
            "docker_image", "nikolaik/python-nodejs:python3.11-nodejs20"
        )
        print()
        print_info("Docker sandboxes can be protected with the egress credential firewall.")
        print_info(
            "It routes sandbox traffic through iron-proxy so containers receive "
            "proxy tokens instead of real API keys."
        )
        print_info(
            "   Docker only for now; Modal, SSH, Daytona, and Singularity are not wired yet."
        )
        if prompt_yes_no("  Enable egress firewall for Docker sandboxes?", False):
            proxy_cfg = config.setdefault("proxy", {})
            proxy_cfg["enabled"] = True
            proxy_cfg.setdefault("enforce_on_docker", True)
            print_success("Egress firewall enabled in config")
            print_info(
                "Run `pilotage egress setup` then `pilotage egress start` to mint "
                "tokens and launch the proxy."
            )
        else:
            print_info(
                "Skipping egress firewall. You can enable it later with `pilotage egress setup`."
            )

    elif selected_backend == "singularity":
        print_success("Terminal backend: Singularity/Apptainer")

        # Check if singularity/apptainer is available
        sing_bin = shutil.which("apptainer") or shutil.which("singularity")
        if not sing_bin:
            print_warning("Singularity/Apptainer not found in PATH!")
            print_info(
                "Install: https://apptainer.org/docs/admin/main/installation.html"
            )
        else:
            print_info(f"Found: {sing_bin}")

        # Image and resource limits use defaults; tune via `pilotage setup terminal`.
        config["terminal"].setdefault(
            "singularity_image",
            "docker://nikolaik/python-nodejs:python3.11-nodejs20",
        )

    elif selected_backend == "modal":
        print_success("Terminal backend: Modal")
        print_info("Serverless cloud sandboxes. Each session gets its own container.")
        print_info("Requires a Modal account: https://modal.com")

        # Check if modal SDK is installed
        try:
            __import__("modal")
        except ImportError:
            print_info("Installing modal SDK...")
            from pilotage_cli.tools_config import _pip_install

            result = _pip_install(["modal"])
            if result.returncode == 0:
                print_success("modal SDK installed")
            else:
                print_warning("Install failed — run manually: uv pip install modal")

        # Modal token
        print()
        print_info("Modal authentication:")
        print_info("  Get your token at: https://modal.com/settings")
        existing_token = get_env_value("MODAL_TOKEN_ID")
        if existing_token:
            print_info("  Modal token: already configured")
            if prompt_yes_no("  Update Modal credentials?", False):
                token_id = prompt("    Modal Token ID", password=True)
                token_secret = prompt("    Modal Token Secret", password=True)
                if token_id:
                    save_env_value("MODAL_TOKEN_ID", token_id)
                if token_secret:
                    save_env_value("MODAL_TOKEN_SECRET", token_secret)
        else:
            token_id = prompt("    Modal Token ID", password=True)
            token_secret = prompt("    Modal Token Secret", password=True)
            if token_id:
                save_env_value("MODAL_TOKEN_ID", token_id)
            if token_secret:
                save_env_value("MODAL_TOKEN_SECRET", token_secret)

    elif selected_backend == "daytona":
        print_success("Terminal backend: Daytona")
        print_info("Persistent cloud development environments.")
        print_info("Each session gets a dedicated sandbox with filesystem persistence.")
        print_info("Sign up at: https://daytona.io")

        # Check if daytona SDK is installed
        try:
            __import__("daytona")
        except ImportError:
            print_info("Installing daytona SDK...")
            from pilotage_cli.tools_config import _pip_install

            result = _pip_install(["daytona"])
            if result.returncode == 0:
                print_success("daytona SDK installed")
            else:
                print_warning("Install failed — run manually: uv pip install daytona")
                if result.stderr:
                    print_info(f"  Error: {result.stderr.strip().splitlines()[-1]}")

        # Daytona API key
        print()
        existing_key = get_env_value("DAYTONA_API_KEY")
        if existing_key:
            print_info("  Daytona API key: already configured")
            if prompt_yes_no("  Update API key?", False):
                api_key = prompt("    Daytona API key", password=True)
                if api_key:
                    save_env_value("DAYTONA_API_KEY", api_key)
                    print_success("    Updated")
        else:
            api_key = prompt("    Daytona API key", password=True)
            if api_key:
                save_env_value("DAYTONA_API_KEY", api_key)
                print_success("    Configured")

        # Image and resource limits use defaults; tune via `pilotage setup terminal`.
        config["terminal"].setdefault(
            "daytona_image", "nikolaik/python-nodejs:python3.11-nodejs20"
        )

    elif selected_backend == "vercel_sandbox":
        print_success("Terminal backend: Vercel Sandbox")
        print_info("Cloud microVM sandboxes with snapshot-backed filesystem persistence.")
        print_info("Requires the optional SDK: pip install 'pilotage-agent[vercel]'")

        try:
            __import__("vercel")
        except ImportError:
            print_info("Installing vercel SDK...")
            import subprocess

            # Managed uv first: $PILOTAGE_HOME/bin is never on PATH, so a bare
            # which() misses the uv Pilotage installed. Bootstrapping one is
            # welcome here — this is the interactive setup wizard, already
            # mid-install, and the alternative tier is a pip that a `uv venv`
            # venv may not even have.
            from pilotage_cli.managed_uv import ensure_uv

            uv_bin = ensure_uv()
            if uv_bin:
                result = subprocess.run(
                    [uv_bin, "pip", "install", "--python", sys.executable, "vercel"],
                    capture_output=True,
                    text=True,
                )
            else:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "vercel"],
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                print_success("vercel SDK installed")
            else:
                print_warning("Install failed — run manually: pip install 'pilotage-agent[vercel]'")
                if result.stderr:
                    print_info(f"  Error: {result.stderr.strip().splitlines()[-1]}")

        _prompt_vercel_sandbox_settings(config)

    elif selected_backend == "ssh":
        print_success("Terminal backend: SSH")
        print_info("Run commands on a remote machine via SSH.")

        # SSH host
        current_host = get_env_value("TERMINAL_SSH_HOST") or ""
        host = prompt("  SSH host (hostname or IP)", current_host)
        if host:
            save_env_value("TERMINAL_SSH_HOST", host)

        # SSH user
        current_user = get_env_value("TERMINAL_SSH_USER") or ""
        user = prompt("  SSH user", current_user or os.getenv("USER", ""))
        if user:
            save_env_value("TERMINAL_SSH_USER", user)

        # SSH port
        current_port = get_env_value("TERMINAL_SSH_PORT") or "22"
        port = prompt("  SSH port", current_port)
        if port and port != "22":
            save_env_value("TERMINAL_SSH_PORT", port)

        # SSH key
        current_key = get_env_value("TERMINAL_SSH_KEY") or ""
        default_key = str(Path.home() / ".ssh" / "id_rsa")
        ssh_key = prompt("  SSH private key path", current_key or default_key)
        if ssh_key:
            save_env_value("TERMINAL_SSH_KEY", ssh_key)

        # Test connection
        if host and prompt_yes_no("  Test SSH connection?", True):
            print_info("  Testing connection...")
            import subprocess

            ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
            if ssh_key:
                ssh_cmd.extend(["-i", ssh_key])
            if port and port != "22":
                ssh_cmd.extend(["-p", port])
            ssh_cmd.append(f"{user}@{host}" if user else host)
            ssh_cmd.append("echo ok")
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
            if result.returncode == 0:
                print_success("  SSH connection successful!")
            else:
                print_warning(f"  SSH connection failed: {result.stderr.strip()}")
                print_info("  Check your SSH key and host settings.")

    # Sync terminal backend to .env so terminal_tool picks it up directly.
    # config.yaml is the source of truth, but terminal_tool reads TERMINAL_ENV.
    save_env_value("TERMINAL_ENV", selected_backend)
    if selected_backend == "vercel_sandbox":
        save_env_value("TERMINAL_VERCEL_RUNTIME", config["terminal"].get("vercel_runtime", "node24"))
    save_config(config)
    print()
    print_success(f"Terminal backend set to: {selected_backend}")


# =============================================================================
# Section 3: Agent Settings
# =============================================================================


def _apply_default_agent_settings(config: dict):
    """Apply recommended defaults for all agent settings without prompting."""
    config.setdefault("agent", {})["max_turns"] = 150
    # config.yaml is the authoritative source for max_turns; the gateway
    # bridges it into PILOTAGE_MAX_ITERATIONS at startup. We no longer write
    # to .env to avoid the dual-source inconsistency that caused the
    # 60-vs-500 bug (stale .env entry silently shadowing config.yaml).
    remove_env_value("PILOTAGE_MAX_ITERATIONS")

    config.setdefault("display", {})["tool_progress"] = "all"

    config.setdefault("compression", {})["enabled"] = True
    config["compression"]["threshold"] = 0.50

    # Default: never auto-reset sessions. This matches the gateway's own
    # default (SessionResetPolicy.mode = "none"); we still write it
    # explicitly so the choice is visible/editable in config.yaml.
    config.setdefault("session_reset", {})["mode"] = "none"

    save_config(config)
    print_success("Applied recommended defaults:")
    print_info("  Max iterations: 150")
    print_info("  Tool progress: all")
    print_info("  Compression threshold: 0.50")
    print_info("  Session reset: never (use /reset or compression)")
    print_info("  Run `pilotage setup agent` later to customize.")


def setup_agent_settings(config: dict):
    """Configure agent behavior: iterations, progress display, compression, session reset."""

    print_header("Agent Settings")
    print_info(f"   Guide: {_DOCS_BASE}/user-guide/configuration")
    print()

    # ── Max Iterations ──
    # config.yaml is authoritative; read from there. If a legacy .env
    # entry is still around (from pre- setups), prefer the
    # config value so we don't surface a stale number to the user.
    current_max = str(cfg_get(config, "agent", "max_turns", default=90))
    print_info("Maximum tool-calling iterations per conversation.")
    print_info("Higher = more complex tasks, but costs more tokens.")
    print_info(
        f"Press Enter to keep {current_max}. Use 90 for most tasks or 150+ for open exploration."
    )

    max_iter_str = prompt("Max iterations", current_max)
    try:
        max_iter = int(max_iter_str)
        if max_iter > 0:
            # Write to config.yaml (authoritative) only. Also clean up any
            # stale .env entry from earlier setup runs — the gateway's
            # bridge in gateway/run.py now unconditionally derives
            # PILOTAGE_MAX_ITERATIONS from agent.max_turns at startup.
            config.setdefault("agent", {})["max_turns"] = max_iter
            config.pop("max_turns", None)
            remove_env_value("PILOTAGE_MAX_ITERATIONS")
            print_success(f"Max iterations set to {max_iter}")
    except ValueError:
        print_warning("Invalid number, keeping current value")

    # ── Tool Progress Display ──
    print_info("")
    print_info("Tool Progress Display")
    print_info("Controls how much tool activity is shown (CLI and messaging).")
    print_info("  off     — Silent, just the final response")
    print_info("  new     — Show tool name only when it changes (less noise)")
    print_info("  all     — Show every tool call with a short preview")
    print_info("  verbose — Full args, results, and debug logs")
    print_info("  log     — Silent in chat; write every tool call to ~/.pilotage/logs/tool_calls.log (gateway only)")

    current_mode = cfg_get(config, "display", "tool_progress", default="all")
    mode = prompt("Tool progress mode", current_mode)
    if mode.lower() in {"off", "new", "all", "verbose", "log"}:
        if "display" not in config:
            config["display"] = {}
        config["display"]["tool_progress"] = mode.lower()
        save_config(config)
        print_success(f"Tool progress set to: {mode.lower()}")
    else:
        print_warning(f"Unknown mode '{mode}', keeping '{current_mode}'")

    # ── Context Compression ──
    print_header("Context Compression")
    print_info("Automatically summarizes old messages when context gets too long.")
    print_info(
        "Higher threshold = compress later (use more context). Lower = compress sooner."
    )

    config.setdefault("compression", {})["enabled"] = True

    current_threshold = cfg_get(config, "compression", "threshold", default=0.50)
    threshold_str = prompt("Compression threshold (0.5-0.95)", str(current_threshold))
    try:
        threshold = float(threshold_str)
        if 0.5 <= threshold <= 0.95:
            config["compression"]["threshold"] = threshold
    except ValueError:
        pass

    print_success(
        f"Context compression threshold set to {config['compression'].get('threshold', 0.50)}"
    )

    # ── Session Reset Policy ──
    print_header("Session Reset Policy")
    print_info(
        "Messaging sessions (Telegram, WhatsApp) accumulate context over time."
    )
    print_info(
        "Each message adds to the conversation history, which means growing API costs."
    )
    print_info("")
    print_info(
        "To manage this, sessions can automatically reset after a period of inactivity"
    )
    print_info(
        "or at a fixed time each day. When a reset happens, the agent saves important"
    )
    print_info(
        "things to its persistent memory first — but the conversation context is cleared."
    )
    print_info("")
    print_info("You can also manually reset anytime by typing /reset in chat.")
    print_info("")

    reset_choices = [
        "Inactivity + daily reset (reset whichever comes first)",
        "Inactivity only (reset after N minutes of no messages)",
        "Daily only (reset at a fixed hour each day)",
        "Never auto-reset (recommended - context lives until /reset or context compression)",
        "Keep current settings",
    ]

    current_policy = config.get("session_reset", {})
    current_mode = current_policy.get("mode", "none")
    current_idle = current_policy.get("idle_minutes", 1440)
    current_hour = current_policy.get("at_hour", 4)

    default_reset = {"both": 0, "idle": 1, "daily": 2, "none": 3}.get(current_mode, 3)

    reset_idx = prompt_choice("Session reset mode:", reset_choices, default_reset)

    config.setdefault("session_reset", {})

    if reset_idx == 0:  # Both
        config["session_reset"]["mode"] = "both"
        idle_str = prompt("  Inactivity timeout (minutes)", str(current_idle))
        try:
            idle_val = int(idle_str)
            if idle_val > 0:
                config["session_reset"]["idle_minutes"] = idle_val
        except ValueError:
            pass
        hour_str = prompt("  Daily reset hour (0-23, local time)", str(current_hour))
        try:
            hour_val = int(hour_str)
            if 0 <= hour_val <= 23:
                config["session_reset"]["at_hour"] = hour_val
        except ValueError:
            pass
        print_success(
            f"Sessions reset after {config['session_reset'].get('idle_minutes', 1440)} min idle or daily at {config['session_reset'].get('at_hour', 4)}:00"
        )
    elif reset_idx == 1:  # Idle only
        config["session_reset"]["mode"] = "idle"
        idle_str = prompt("  Inactivity timeout (minutes)", str(current_idle))
        try:
            idle_val = int(idle_str)
            if idle_val > 0:
                config["session_reset"]["idle_minutes"] = idle_val
        except ValueError:
            pass
        print_success(
            f"Sessions reset after {config['session_reset'].get('idle_minutes', 1440)} min of inactivity"
        )
    elif reset_idx == 2:  # Daily only
        config["session_reset"]["mode"] = "daily"
        hour_str = prompt("  Daily reset hour (0-23, local time)", str(current_hour))
        try:
            hour_val = int(hour_str)
            if 0 <= hour_val <= 23:
                config["session_reset"]["at_hour"] = hour_val
        except ValueError:
            pass
        print_success(
            f"Sessions reset daily at {config['session_reset'].get('at_hour', 4)}:00"
        )
    elif reset_idx == 3:  # None
        config["session_reset"]["mode"] = "none"
        print_info(
            "Sessions will never auto-reset. Context is managed only by compression."
        )
        print_warning(
            "Long conversations will grow in cost. Use /reset manually when needed."
        )
    # else: keep current (idx == 4)

    save_config(config)


# =============================================================================
# Section 4: Messaging Platforms (Gateway)
# =============================================================================


_TELEGRAM_BOT_TOKEN_RE = re.compile(r"^\d+:[A-Za-z0-9_-]{30,}$")


def _is_valid_telegram_bot_token(token: str) -> bool:
    return bool(_TELEGRAM_BOT_TOKEN_RE.match(token))


def _setup_telegram_auto_result():
    """Attempt automatic Telegram bot creation via managed QR onboarding."""
    try:
        from pilotage_cli.telegram_managed_bot import auto_setup_telegram_bot_result
    except ImportError:
        return None

    profile_name: str | None = None
    try:
        profile_name = _profile_name_from_pilotage_home(Path(get_pilotage_home()))
    except Exception:
        pass

    return auto_setup_telegram_bot_result(profile_name=profile_name)


def _profile_name_from_pilotage_home(pilotage_home) -> str | None:
    """Return the active profile name when PILOTAGE_HOME is a profile dir."""
    if pilotage_home.parent.name == "profiles":
        return pilotage_home.name
    return None


def _setup_telegram_auto() -> str | None:
    """Attempt automatic Telegram bot creation and return only the token."""
    result = _setup_telegram_auto_result()
    return result.token if result else None


def _prompt_telegram_bot_token() -> str | None:
    print_info("Create a bot via @BotFather on Telegram")
    while True:
        token = prompt("Telegram bot token", password=True)
        if not token:
            return None
        if not _is_valid_telegram_bot_token(token):
            print_error(
                "Invalid token format. Expected: <numeric_id>:<alphanumeric_hash> "
                "(e.g., 123456789:ABCdefGHI-jklMNOpqrSTUvwxYZ)"
            )
            continue
        return token


def _setup_telegram():
    """Configure Telegram bot credentials and allowlist."""
    print_header("Telegram")
    existing = get_env_value("TELEGRAM_BOT_TOKEN")
    if existing:
        print_info("Telegram: already configured")
        if not prompt_yes_no("Reconfigure Telegram?", False):
            # Check missing allowlist on existing config
            if not get_env_value("TELEGRAM_ALLOWED_USERS"):
                print_info("⚠️  Telegram has no user allowlist - anyone can use your bot!")
                if prompt_yes_no("Add allowed users now?", True):
                    print_info("   To find your Telegram user ID: message @userinfobot")
                    allowed_users = prompt("Allowed user IDs (comma-separated)")
                    if allowed_users:
                        save_env_value("TELEGRAM_ALLOWED_USERS", allowed_users.replace(" ", ""))
                        print_success("Telegram allowlist configured")
            return

    print_info("How would you like to create your Telegram bot?")
    print()
    print_info("  [1] Automatic (recommended)")
    print_info("      Scan a QR code → confirm in Telegram → done.")
    print_info("      No token copy-paste needed.")
    print()
    print_info("  [2] Manual")
    print_info("      Create a bot via @BotFather yourself and paste the token.")
    print()

    choice = prompt("Choice [1/2]", default="1")
    token = None
    setup_result = None

    if choice.strip() == "1":
        setup_result = _setup_telegram_auto_result()
        if setup_result:
            token = setup_result.token
            if not _is_valid_telegram_bot_token(token):
                print_error("Automatic setup returned an invalid Telegram bot token.")
                token = None
                setup_result = None
        else:
            token = None
        if not token:
            print()
            print_info("Falling back to manual setup...")
            print()

    if not token:
        token = _prompt_telegram_bot_token()
    if not token:
        return

    save_env_value("TELEGRAM_BOT_TOKEN", token)
    print_success("Telegram token saved")

    print()
    print_info("🔒 Security: Restrict who can use your bot")
    print_info("   To find your Telegram user ID:")
    print_info("   1. Message @userinfobot on Telegram")
    print_info("   2. It will reply with your numeric ID (e.g., 123456789)")
    print()

    detected_user_id = getattr(setup_result, "owner_user_id", None)
    if detected_user_id:
        detected_id = str(detected_user_id)
        print_success(f"Detected your Telegram user ID: {detected_id}")
        if prompt_yes_no("Allow this Telegram account to use the bot?", True):
            extra = prompt("Additional allowed user IDs (comma-separated, optional)")
            ids = [detected_id]
            for uid in extra.replace(" ", "").split(","):
                if uid and uid not in ids:
                    ids.append(uid)
            allowed_users = ",".join(ids)
        else:
            allowed_users = prompt(
                "Allowed user IDs (comma-separated, leave empty for open access)"
            )
    else:
        allowed_users = prompt(
            "Allowed user IDs (comma-separated, leave empty for open access)"
        )

    if allowed_users:
        allowed_users = allowed_users.replace(" ", "")
        save_env_value("TELEGRAM_ALLOWED_USERS", allowed_users)
        print_success("Telegram allowlist configured - only listed users can use the bot")
    else:
        print_info("⚠️  No allowlist set - anyone who finds your bot can use it!")

    print()
    print_info("📬 Home Channel: where Pilotage delivers cron job results,")
    print_info("   cross-platform messages, and notifications.")
    print_info("   For Telegram DMs, this is your user ID (same as above).")

    first_user_id = allowed_users.split(",")[0].strip() if allowed_users else ""
    if first_user_id:
        if prompt_yes_no(f"Use your user ID ({first_user_id}) as the home channel?", True):
            save_env_value("TELEGRAM_HOME_CHANNEL", first_user_id)
            print_success(f"Telegram home channel set to {first_user_id}")
        else:
            home_channel = prompt("Home channel ID (or leave empty to set later with /set-home in Telegram)")
            if home_channel:
                save_env_value("TELEGRAM_HOME_CHANNEL", home_channel)
    else:
        print_info("   You can also set this later by typing /set-home in your Telegram chat.")
        home_channel = prompt("Home channel ID (leave empty to set later)")
        if home_channel:
            save_env_value("TELEGRAM_HOME_CHANNEL", home_channel)


def _setup_webhooks():
    """Configure webhook integration."""
    print_header("Webhooks")
    existing = get_env_value("WEBHOOK_ENABLED")
    if existing:
        print_info("Webhooks: already configured")
        if not prompt_yes_no("Reconfigure webhooks?", False):
            return

    print()
    print_warning("⚠  Webhook platforms require exposing gateway ports to the")
    print_warning("   internet. For security, run the gateway in a sandboxed environment")
    print_warning("   (Docker, VM, etc.) to limit blast radius from prompt injection.")
    print()
    print_info(" Full guide: ")
    print()

    port = prompt("Webhook port (default 8644)")
    if port:
        try:
            save_env_value("WEBHOOK_PORT", str(int(port)))
            print_success(f"Webhook port set to {port}")
        except ValueError:
            print_warning("Invalid port number, using default 8644")

    secret = prompt("Global HMAC secret (shared across all routes)", password=True)
    if secret:
        save_env_value("WEBHOOK_SECRET", secret)
        print_success("Webhook secret saved")
    else:
        print_warning("No secret set — you must configure per-route secrets in config.yaml")

    save_env_value("WEBHOOK_ENABLED", "true")
    print()
    print_success("Webhooks enabled! Next steps:")
    from pilotage_constants import display_pilotage_home as _dhh
    print_info(f"   1. Define webhook routes in {_dhh()}/config.yaml")
    print_info("   2. Point your service (GitHub, GitLab, etc.) at:")
    print_info("      http://your-server:8644/webhooks/<route-name>")
    print()
    print_info("   Route configuration guide:")
    print_info(" #configuring-routes")
    print()
    print_info("   Open config in your editor:  pilotage config edit")
    print_info("   Open config in your editor:  pilotage config edit")


def setup_gateway(config: dict):
    """Configure messaging platform integrations."""
    from pilotage_cli.gateway import _all_platforms, _platform_status, _configure_platform

    print_header("Messaging Platforms")
    print_info("Connect to messaging platforms to chat with Pilotage from anywhere.")
    print_info("Toggle with Space, confirm with Enter.")
    print()

    platforms = _all_platforms()

    # Build checklist, pre-selecting already-configured platforms.
    items = []
    pre_selected = []
    for i, plat in enumerate(platforms):
        status = _platform_status(plat)
        items.append(f"{plat['emoji']} {plat['label']}  ({status})")
        if status == "configured":
            pre_selected.append(i)

    selected = prompt_checklist("Select platforms to configure:", items, pre_selected)

    if not selected:
        print_info("No platforms selected. Run 'pilotage setup gateway' later to configure.")
    else:
        for idx in selected:
            _configure_platform(platforms[idx])

    # ── Gateway Service Setup ──
    # Count any platform (built-in or plugin) the user configured during this
    # setup pass — reuses ``_platform_status`` so plugin platforms like IRC
    # are picked up without another hard-coded env-var list.
    def _is_progress(status: str) -> bool:
        s = status.lower()
        return not (
            s == "not configured"
            or s.startswith("partially")
            or s.startswith("plugin disabled")
        )

    any_messaging = any(
        _is_progress(_platform_status(p)) for p in _all_platforms()
    )
    if any_messaging:
        print()
        print_info("━" * 50)
        print_success("Messaging platforms configured!")

        # Check if any home channels are missing
        missing_home = []
        if get_env_value("TELEGRAM_BOT_TOKEN") and not get_env_value(
            "TELEGRAM_HOME_CHANNEL"
        ):
            missing_home.append("Telegram")

        if missing_home:
            print()
            print_warning(f"No home channel set for: {', '.join(missing_home)}")
            print_info("   Without a home channel, cron jobs and cross-platform")
            print_info("   messages can't be delivered to those platforms.")
            print_info("   Set one later with /set-home in your chat, or:")
            for plat in missing_home:
                print_info(
                    f"     pilotage config set {plat.upper()}_HOME_CHANNEL <channel_id>"
                )

    # ── Gateway Service Setup ──
    # Runs UNCONDITIONALLY — even with zero platforms configured. A gateway
    # without platforms is a supported mode (cron scheduler keeps running,
    # and adapters come up automatically once tokens are added later, e.g.
    # via `pilotage import` or `pilotage setup gateway`). Gating this on
    # messaging config was the bug that left install-then-import machines
    # with registered cron jobs and restored bot tokens but no process to
    # serve them.
    from pilotage_cli.gateway import (
        _is_service_running,
        supports_systemd_services,
        ensure_gateway_service,
        systemd_restart,
        launchd_restart,
        UserSystemdUnavailableError,
        SystemScopeRequiresRootError,
        _system_scope_wizard_would_need_root,
        _print_system_scope_remediation,
    )
    import platform as _platform

    _is_macos = _platform.system() == "Darwin"
    _is_windows = _platform.system() == "Windows"
    supports_systemd = supports_systemd_services()

    print()
    if _is_service_running():
        # Already running: only offer a restart when this setup pass may
        # have changed platform config — a restart interrupts any active
        # session, so it stays behind a prompt.
        if supports_systemd and _system_scope_wizard_would_need_root():
            _print_system_scope_remediation("restart")
        elif any_messaging and prompt_yes_no(
            "  Restart the gateway to pick up changes?", True
        ):
            try:
                if supports_systemd:
                    systemd_restart()
                elif _is_macos:
                    launchd_restart()
                elif _is_windows:
                    from pilotage_cli import gateway_windows
                    gateway_windows.restart()
            except UserSystemdUnavailableError as e:
                print_error("  Restart failed — user systemd not reachable:")
                for line in str(e).splitlines():
                    print(f"  {line}")
            except SystemScopeRequiresRootError as e:
                # Defense in depth: the pre-check above should have
                # caught this, but a race (unit file appearing mid-run)
                # could still land here. Previously this exited the
                # whole wizard via sys.exit(1).
                print_error(f"  Restart failed: {e}")
                _print_system_scope_remediation("restart")
            except Exception as e:
                print_error(f"  Restart failed: {e}")
    else:
        # Not running: install (if needed) and start, no questions asked.
        ensure_gateway_service(context="setup")

    print_info("━" * 50)


# =============================================================================
# Section 5: Tool Configuration (delegates to unified tools_config.py)
# =============================================================================


def setup_tools(config: dict, first_install: bool = False):
    """Configure tools — delegates to the unified tools_command() in tools_config.py.

    Both `pilotage setup tools` and `pilotage tools` use the same flow:
    platform selection → toolset toggles → provider/API key configuration.

    Args:
        first_install: When True, uses the simplified first-install flow
            (no platform menu, prompts for all unconfigured API keys).
    """
    from pilotage_cli.tools_config import tools_command

    tools_command(first_install=first_install, config=config)


# =============================================================================
# Shared Metrics
# =============================================================================


def setup_telemetry(config: dict):
    """Configure the local, privacy-safe shared-metrics subscriber."""
    print_header("Shared Metrics")
    print_info("Shared metrics contain only bounded counters and histograms.")
    print_info("Packages stay under this Pilotage profile and are not uploaded.")

    telemetry = config.get("telemetry")
    if not isinstance(telemetry, dict):
        telemetry = {}
        config["telemetry"] = telemetry
    shared_metrics = telemetry.get("shared_metrics")
    if not isinstance(shared_metrics, dict):
        shared_metrics = {}
        telemetry["shared_metrics"] = shared_metrics

    current = shared_metrics.get("enabled") is True
    shared_metrics["enabled"] = prompt_yes_no(
        "Enable local shared metrics?",
        default=current,
    )
    if shared_metrics["enabled"]:
        print_success("Local shared metrics enabled.")
    else:
        print_info("Local shared metrics disabled.")


# =============================================================================
# Post-Migration Section Skip Logic
# =============================================================================


def _model_section_has_credentials(config: dict) -> bool:
    """Return True when any known inference provider has usable credentials.

    Sources of truth:
      * ``PROVIDER_REGISTRY`` in ``pilotage_cli.auth`` — lists every supported
        provider along with its ``api_key_env_vars``.
      * ``active_provider`` in the auth store — covers OAuth device-code /
        external-OAuth providers (Codex, Qwen, Gemini CLI, ...).
      * The legacy OpenRouter aggregator env vars, which route generic
        ``OPENAI_API_KEY`` / ``OPENROUTER_API_KEY`` values through OpenRouter.
    """
    try:
        from pilotage_cli.auth import get_active_provider
        if get_active_provider():
            return True
    except Exception:
        pass

    try:
        from pilotage_cli.auth import PROVIDER_REGISTRY
    except Exception:
        PROVIDER_REGISTRY = {}  # type: ignore[assignment]

    def _has_key(pconfig) -> bool:
        for env_var in pconfig.api_key_env_vars:
            # CLAUDE_CODE_OAUTH_TOKEN is set by Claude Code itself, not by
            # the user — mirrors is_provider_explicitly_configured in auth.py.
            if env_var == "CLAUDE_CODE_OAUTH_TOKEN":
                continue
            if get_env_value(env_var):
                return True
        return False

    # Prefer the provider declared in config.yaml, avoids false positives
    # from stray env vars (GH_TOKEN, etc.) when the user has already picked
    # a different provider.
    model_cfg = config.get("model") if isinstance(config, dict) else None
    if isinstance(model_cfg, dict):
        provider_id = (model_cfg.get("provider") or "").strip().lower()
        if provider_id in PROVIDER_REGISTRY:
            if _has_key(PROVIDER_REGISTRY[provider_id]):
                return True

    # Aggregator-free fallback (no provider declared in config).
    for env_var in ("OPENAI_API_KEY",):
        if get_env_value(env_var):
            return True

    for pid, pconfig in PROVIDER_REGISTRY.items():
        if _has_key(pconfig):
            return True
    return False


def _gateway_platform_short_label(label: str) -> str:
    """Strip trailing parenthetical qualifiers from a gateway platform label."""
    base = label.split("(", 1)[0].strip()
    return base or label


def _get_section_config_summary(config: dict, section_key: str) -> Optional[str]:
    """Return a short summary if a setup section is already configured, else None.

    Used after OpenClaw migration to detect which sections can be skipped.
    ``get_env_value`` is the module-level import from pilotage_cli.config
    so that test patches on ``setup_mod.get_env_value`` take effect.
    """
    if section_key == "model":
        if not _model_section_has_credentials(config):
            return None
        model = config.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()
        if isinstance(model, dict):
            return str(model.get("default") or model.get("model") or "configured")
        return "configured"

    elif section_key == "terminal":
        backend = cfg_get(config, "terminal", "backend", default="local")
        return f"backend: {backend}"

    elif section_key == "agent":
        max_turns = cfg_get(config, "agent", "max_turns", default=90)
        return f"max turns: {max_turns}"

    elif section_key == "gateway":
        from pilotage_cli.gateway import _all_platforms, _platform_status
        # Count any non-empty status other than the "not configured" sentinel —
        # platforms like WhatsApp ("enabled, not paired") indicate the user
        # has already started setup and we shouldn't force the section to rerun.
        configured = [
            _gateway_platform_short_label(plat["label"])
            for plat in _all_platforms()
            if _platform_status(plat) and _platform_status(plat) != "not configured"
        ]
        if configured:
            return ", ".join(configured)
        return None  # No platforms configured — section must run

    elif section_key == "tools":
        tools = []
        if get_env_value("BROWSERBASE_API_KEY"):
            tools.append("Browser")
        if get_env_value("FIRECRAWL_API_KEY"):
            tools.append("Firecrawl")
        if tools:
            return ", ".join(tools)
        return None

    return None


def _skip_configured_section(
    config: dict, section_key: str, label: str
) -> bool:
    """Show an already-configured section summary and offer to skip.

    Returns True if the user chose to skip, False if the section should run.
    """
    summary = _get_section_config_summary(config, section_key)
    if not summary:
        return False
    print()
    print_success(f"  {label}: {summary}")
    return not prompt_yes_no(f"  Reconfigure {label.lower()}?", default=False)


# =============================================================================
# OpenClaw Migration
# =============================================================================


_OPENCLAW_SCRIPT = (
    get_optional_skills_dir(PROJECT_ROOT / "optional-skills")
    / "migration"
    / "openclaw-migration"
    / "scripts"
    / "openclaw_to_pilotage.py"
)


def _load_openclaw_migration_module():
    """Load the openclaw_to_pilotage migration script as a module.

    Returns the loaded module, or None if the script can't be loaded.
    """
    if not _OPENCLAW_SCRIPT.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "openclaw_to_pilotage", _OPENCLAW_SCRIPT
    )
    if spec is None or spec.loader is None:
        return None

    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass can resolve the module
    # (Python 3.11+ requires this for dynamically loaded modules)
    import sys as _sys
    _sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        _sys.modules.pop(spec.name, None)
        raise
    return mod


# Item kinds that represent high-impact changes warranting explicit warnings.
# Gateway tokens/channels can hijack messaging platforms from the old agent.
# Config values may have different semantics between OpenClaw and Pilotage.
# Instruction/context files (.md) can contain incompatible setup procedures.
_HIGH_IMPACT_KIND_KEYWORDS = {
    "gateway": "⚠ Gateway/messaging — this will configure Pilotage to use your OpenClaw messaging channels",
    "telegram": "⚠ Telegram — this will point Pilotage at your OpenClaw Telegram bot",
    "whatsapp": "⚠ WhatsApp — this will point Pilotage at your OpenClaw WhatsApp connection",
    "config": "⚠ Config values — OpenClaw settings may not map 1:1 to Pilotage equivalents",
    "soul": "⚠ Instruction file — may contain OpenClaw-specific setup/restart procedures",
    "memory": "⚠ Memory/context file — may reference OpenClaw-specific infrastructure",
    "context": "⚠ Context file — may contain OpenClaw-specific instructions",
}


def _print_migration_preview(report: dict):
    """Print a detailed dry-run preview of what migration would do.

    Groups items by category and adds explicit warnings for high-impact
    changes like gateway token takeover and config value differences.
    """
    items = report.get("items", [])
    if not items:
        print_info("Nothing to migrate.")
        return

    migrated_items = [i for i in items if i.get("status") == "migrated"]
    conflict_items = [i for i in items if i.get("status") == "conflict"]
    skipped_items = [i for i in items if i.get("status") == "skipped"]

    warnings_shown = set()

    if migrated_items:
        print(color("  Would import:", Colors.GREEN))
        for item in migrated_items:
            kind = item.get("kind", "unknown")
            dest = item.get("destination", "")
            if dest:
                dest_short = str(dest).replace(str(Path.home()), "~")
                print(f"      {kind:<22s} → {dest_short}")
            else:
                print(f"      {kind}")

            # Check for high-impact items and collect warnings
            kind_lower = kind.lower()
            dest_lower = str(dest).lower()
            for keyword, warning in _HIGH_IMPACT_KIND_KEYWORDS.items():
                if keyword in kind_lower or keyword in dest_lower:
                    warnings_shown.add(warning)
        print()

    if conflict_items:
        print(color("  Would overwrite (conflicts with existing Pilotage config):", Colors.YELLOW))
        for item in conflict_items:
            kind = item.get("kind", "unknown")
            reason = item.get("reason", "already exists")
            print(f"      {kind:<22s}  {reason}")
        print()

    if skipped_items:
        print(color("  Would skip:", Colors.DIM))
        for item in skipped_items:
            kind = item.get("kind", "unknown")
            reason = item.get("reason", "")
            print(f"      {kind:<22s}  {reason}")
        print()

    # Print collected warnings
    if warnings_shown:
        print(color("  ── Warnings ──", Colors.YELLOW))
        for warning in sorted(warnings_shown):
            print(color(f"    {warning}", Colors.YELLOW))
        print()
        print(color("  Note: OpenClaw config values may have different semantics in Pilotage.", Colors.YELLOW))
        print(color("  For example, OpenClaw's tool_call_execution: \"auto\" ≠ Pilotage's yolo mode.", Colors.YELLOW))
        print(color("  Instruction files (.md) from OpenClaw may contain incompatible procedures.", Colors.YELLOW))
        print()


def _offer_openclaw_migration(pilotage_home: Path) -> bool:
    """Detect ~/.openclaw and offer to migrate during first-time setup.

    Runs a dry-run first to show the user exactly what would be imported,
    overwritten, or taken over. Only executes after explicit confirmation.

    Returns True if migration ran successfully, False otherwise.
    """
    openclaw_dir = Path.home() / ".openclaw"
    if not openclaw_dir.is_dir():
        return False

    if not _OPENCLAW_SCRIPT.exists():
        return False

    print()
    print_header("OpenClaw Installation Detected")
    print_info(f"Found OpenClaw data at {openclaw_dir}")
    print_info("Pilotage can preview what would be imported before making any changes.")
    print()

    if not prompt_yes_no("Would you like to see what can be imported?", default=True):
        print_info(
            "Skipping migration. You can run it later with: pilotage claw migrate --dry-run"
        )
        return False

    # Ensure config.yaml exists before migration tries to read it
    config_path = get_config_path()
    if not config_path.exists():
        save_config(load_config())

    # Load the migration module
    try:
        mod = _load_openclaw_migration_module()
        if mod is None:
            print_warning("Could not load migration script.")
            return False
    except Exception as e:
        print_warning(f"Could not load migration script: {e}")
        logger.debug("OpenClaw migration module load error", exc_info=True)
        return False

    # ── Phase 1: Dry-run preview ──
    try:
        selected = mod.resolve_selected_options(None, None, preset="full")
        dry_migrator = mod.Migrator(
            source_root=openclaw_dir.resolve(),
            target_root=pilotage_home.resolve(),
            execute=False,  # dry-run — no files modified
            workspace_target=None,
            overwrite=True,  # show everything including conflicts
            migrate_secrets=True,
            output_dir=None,
            selected_options=selected,
            preset_name="full",
        )
        preview_report = dry_migrator.migrate()
    except Exception as e:
        print_warning(f"Migration preview failed: {e}")
        logger.debug("OpenClaw migration preview error", exc_info=True)
        return False

    # Display the full preview
    preview_summary = preview_report.get("summary", {})
    preview_count = preview_summary.get("migrated", 0)

    if preview_count == 0:
        print()
        print_info("Nothing to import from OpenClaw.")
        return False

    print()
    print_header(f"Migration Preview — {preview_count} item(s) would be imported")
    print_info("No changes have been made yet. Review the list below:")
    print()
    _print_migration_preview(preview_report)

    # ── Phase 2: Confirm and execute ──
    if not prompt_yes_no("Proceed with migration?", default=False):
        print_info(
            "Migration cancelled. You can run it later with: pilotage claw migrate"
        )
        print_info(
            "Use --dry-run to preview again, or --preset minimal for a lighter import."
        )
        return False

    # Execute the migration — overwrite=False so existing Pilotage configs are
    # preserved. The user saw the preview; conflicts are skipped by default.
    try:
        migrator = mod.Migrator(
            source_root=openclaw_dir.resolve(),
            target_root=pilotage_home.resolve(),
            execute=True,
            workspace_target=None,
            overwrite=False,  # preserve existing Pilotage config
            migrate_secrets=True,
            output_dir=None,
            selected_options=selected,
            preset_name="full",
        )
        report = migrator.migrate()
    except Exception as e:
        print_warning(f"Migration failed: {e}")
        logger.debug("OpenClaw migration error", exc_info=True)
        return False

    # Print final summary
    summary = report.get("summary", {})
    migrated = summary.get("migrated", 0)
    skipped = summary.get("skipped", 0)
    conflicts = summary.get("conflict", 0)
    errors = summary.get("error", 0)

    print()
    if migrated:
        print_success(f"Imported {migrated} item(s) from OpenClaw.")
    if conflicts:
        print_info(f"Skipped {conflicts} item(s) that already exist in Pilotage (use pilotage claw migrate --overwrite to force).")
    if skipped:
        print_info(f"Skipped {skipped} item(s) (not found or unchanged).")
    if errors:
        print_warning(f"{errors} item(s) had errors — check the migration report.")

    output_dir = report.get("output_dir")
    if output_dir:
        print_info(f"Full report saved to: {output_dir}")

    print_success("Migration complete! Continuing with setup...")
    return True


# =============================================================================
# Main Wizard Orchestrator
# =============================================================================

SETUP_SECTIONS = [
    ("model", "Model & Provider", setup_model_provider),
    ("tts", "Text-to-Speech", setup_tts),
    ("terminal", "Terminal Backend", setup_terminal_backend),
    ("gateway", "Messaging Platforms (Gateway)", setup_gateway),
    ("tools", "Tools", setup_tools),
    ("telemetry", "Shared Metrics", setup_telemetry),
    ("agent", "Agent Settings", setup_agent_settings),
]


def run_setup_wizard(args):
    """Run the interactive setup wizard.

    Supports full, quick, and section-specific setup:
      pilotage setup           — full or quick (auto-detected)
      pilotage setup model     — just model/provider
      pilotage setup tts       — just text-to-speech
      pilotage setup terminal  — just terminal backend
      pilotage setup gateway   — just messaging platforms
      pilotage setup tools     — just tool configuration
      pilotage setup telemetry — just local shared metrics
      pilotage setup agent     — just agent settings
    """
    from pilotage_cli.config import is_managed, managed_error
    if is_managed():
        managed_error("run setup wizard")
        return
    ensure_pilotage_home()

    reset_requested = bool(getattr(args, "reset", False))
    if reset_requested:
        save_config(copy.deepcopy(DEFAULT_CONFIG))
        print_success("Configuration reset to defaults.")

    reconfigure_requested = bool(getattr(args, "reconfigure", False))
    quick_requested = bool(getattr(args, "quick", False))

    config = load_config()
    pilotage_home = get_pilotage_home()

    # Back up existing config before setup modifies it
    config_path = get_config_path()
    if config_path.exists():
        from datetime import datetime as _dt
        _backup_path = config_path.with_suffix(
            f".yaml.bak.{_dt.now().strftime('%Y%m%d_%H%M%S')}"
        )
        try:
            import shutil
            shutil.copy2(config_path, _backup_path)
        except Exception:
            _backup_path = None
    else:
        _backup_path = None

    # Detect non-interactive environments (headless SSH, Docker, CI/CD)
    non_interactive = getattr(args, 'non_interactive', False)
    if not non_interactive and not is_interactive_stdin():
        non_interactive = True

    if non_interactive:
        print_noninteractive_setup_guidance(
            "Running in a non-interactive environment (no TTY detected)."
        )
        return

    # Check if a specific section was requested
    section = getattr(args, "section", None)
    if section:
        for key, label, func in SETUP_SECTIONS:
            if key == section:
                print()
                print(
                    color(
                        "┌─────────────────────────────────────────────────────────┐",
                        Colors.MAGENTA,
                    )
                )
                print(color(f"│     ⚕ Pilotage Setup — {label:<34s} │", Colors.MAGENTA))
                print(
                    color(
                        "└─────────────────────────────────────────────────────────┘",
                        Colors.MAGENTA,
                    )
                )
                func(config)
                save_config(config)
                print()
                print_success(f"{label} configuration complete!")
                return

        print_error(f"Unknown setup section: {section}")
        print_info(f"Available sections: {', '.join(k for k, _, _ in SETUP_SECTIONS)}")
        return

    # Check if this is an existing installation with a provider configured
    from pilotage_cli.auth import get_active_provider

    active_provider = get_active_provider()
    is_existing = (
        bool(get_env_value("OPENROUTER_API_KEY"))
        or bool(get_env_value("OPENAI_BASE_URL"))
        or active_provider is not None
    )

    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────┐",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "│             ⚕ Pilotage Agent Setup Wizard                │", Colors.MAGENTA
        )
    )
    print(
        color(
            "├─────────────────────────────────────────────────────────┤",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "│  Let's configure your Pilotage Agent installation.       │", Colors.MAGENTA
        )
    )
    print(
        color(
            "│  Press Ctrl+C at any time to exit.                     │", Colors.MAGENTA
        )
    )
    print(
        color(
            "└─────────────────────────────────────────────────────────┘",
            Colors.MAGENTA,
        )
    )

    migration_ran = False

    if is_existing:
        # Existing install — default is the full-wizard reconfigure flow.
        # Every prompt shows the current value as its default, so pressing
        # Enter keeps it.  Opt into `--quick` for the narrow "just fill in
        # missing items" flow (useful after a partial OpenClaw migration
        # or when a required API key got cleared).
        if quick_requested:
            _run_quick_setup(config, pilotage_home)
            return

        print()
        print_header("Reconfigure")
        print_success("You already have Pilotage configured.")
        print_info("Running the full wizard — each prompt shows your current value.")
        print_info("Press Enter to keep it, or type a new value to change it.")
        print_info("")
        print_info("Tip: jump straight to a section with 'pilotage setup model|terminal|")
        print_info("     gateway|tools|agent', or fill only missing items with --quick.")
        # Fall through to the "Full Setup — run all sections" block below.
        # --reconfigure is now the default on existing installs; the flag
        # is preserved for backwards compatibility but is a no-op here.
    else:
        # ── First-Time Setup ──
        print()

        # --reconfigure / --quick on a fresh install are meaningless — fall
        # through to the normal first-time flow.
        if reconfigure_requested or quick_requested:
            print_info("No existing configuration found — running first-time setup.")
            print()

        # Offer OpenClaw migration before configuration begins
        migration_ran = _offer_openclaw_migration(pilotage_home)
        if migration_ran:
            config = load_config()

        setup_mode = prompt_choice(
            "How would you like to set up Pilotage?",
            [
                "Quick Setup (ChatGPT / Codex) — sign in with your ChatGPT account (recommended)",
                "Full setup — configure every provider, tool & option yourself (bring your own keys)",
                "Blank Slate — everything off except the bare minimum; opt in to each capability",
            ],
            0,
        )

        if setup_mode == 0:
            _run_first_time_quick_setup(config, pilotage_home, is_existing)
            return
        if setup_mode == 2:
            _run_blank_slate_setup(config, pilotage_home, is_existing)
            return

    # ── Full Setup — run all sections ──
    print_header("Configuration Location")
    print_info(f"Config file:  {get_config_path()}")
    print_info(f"Secrets file: {get_env_path()}")
    print_info(f"Data folder:  {pilotage_home}")
    print_info(f"Install dir:  {PROJECT_ROOT}")
    print()
    print_info("You can edit these files directly or use 'pilotage config edit'")

    if migration_ran:
        print()
        print_info("Settings were imported from OpenClaw.")
        print_info("Each section below will show what was imported — press Enter to keep,")
        print_info("or choose to reconfigure if needed.")

    # Section 1: Model & Provider
    if not (migration_ran and _skip_configured_section(config, "model", "Model & Provider")):
        setup_model_provider(config)

    # Section 2: Terminal Backend
    if not (migration_ran and _skip_configured_section(config, "terminal", "Terminal Backend")):
        setup_terminal_backend(config)

    # Section 3: Agent Settings — no longer prompted. First installs get the
    # recommended defaults silently; existing installs keep whatever they have.
    # Tune later with `pilotage setup agent`.
    if not is_existing:
        _apply_default_agent_settings(config)

    # Section 4: Messaging Platforms
    if not (migration_ran and _skip_configured_section(config, "gateway", "Messaging Platforms")):
        setup_gateway(config)
    else:
        # Section skipped (migrated config) — still make sure the gateway
        # service exists so cron jobs and migrated platforms actually run.
        from pilotage_cli.gateway import ensure_gateway_service
        ensure_gateway_service(context="setup")

    # Section 5: Tools
    if not (migration_ran and _skip_configured_section(config, "tools", "Tools")):
        setup_tools(config, first_install=not is_existing)

    # Save and show summary
    save_config(config)
    if _backup_path and _backup_path.exists():
        print_info(f"Previous config backed up to: {_backup_path}")
        print_info("If setup changed a value you customized, restore it with:")
        print_info(f"  cp {_backup_path} {config_path}")
    _print_setup_summary(config, pilotage_home)


def _run_first_time_quick_setup(config: dict, pilotage_home, is_existing: bool):
    """Streamlined first-time setup via ChatGPT/Codex: OAuth, model, terminal & messaging.

    Routes straight to the OpenAI Codex provider — runs the browser OAuth
    login, picks a model, then configures the terminal backend and (optionally)
    a messaging platform. Applies sensible defaults for everything else (agent
    settings, tools); the user can customize later via ``pilotage setup <section>``
    or switch providers with ``pilotage model``.
    """
    from pilotage_cli.config import load_config

    # Step 1: ChatGPT / Codex — OAuth login + model selection.
    print()
    print_header("ChatGPT (Codex)")
    print_info("Sign in with your ChatGPT account — no API key needed.")
    print()
    try:
        from pilotage_cli.main import _model_flow_openai_codex
        _model_flow_openai_codex(config)
    except (KeyboardInterrupt, EOFError, SystemExit):
        print()
        print_info("ChatGPT setup cancelled.")
    except Exception as exc:
        logger.debug("_model_flow_openai_codex error during quick setup: %s", exc)
        print_warning(f"ChatGPT setup encountered an error: {exc}")
        print_info("You can try again later with: pilotage model")

    # Re-sync the wizard's config dict from disk — the login/model save writes
    # via its own load/save cycle, and the wizard's later save_config(config)
    # must not clobber those values.
    _refreshed = load_config()
    config.clear()
    config.update(_refreshed)

    # Step 2: Terminal Backend — where commands run is a core decision
    setup_terminal_backend(config)

    # Step 3: Apply defaults for everything else
    _apply_default_agent_settings(config)

    save_config(config)

    # Step 4: Offer messaging gateway setup
    print()
    gateway_choice = prompt_choice(
        "Connect a messaging platform? (Telegram, WhatsApp)",
        [
            "Set up messaging now (recommended)",
            "Skip — set up later with 'pilotage setup gateway'",
        ],
        0,
    )

    if gateway_choice == 0:
        setup_gateway(config)
        save_config(config)
    else:
        # Messaging skipped — still install/start the gateway service so cron
        # jobs run and platforms come alive as soon as tokens are added later
        # (e.g. via `pilotage import` from another machine).
        from pilotage_cli.gateway import ensure_gateway_service
        ensure_gateway_service(context="setup")

    print()
    print_success("Setup complete! You're ready to go.")
    print()
    print_info("  Configure all settings:    pilotage setup")
    if gateway_choice != 0:
        print_info("  Connect Telegram/WhatsApp: pilotage setup gateway")
    print()

    _print_setup_summary(config, pilotage_home)


def _blank_slate_minimal_toolsets(config: dict):
    """Write the minimal toolset state for a Blank Slate install.

    Only ``file`` and ``terminal`` are enabled. Two layers enforce this:

    1. ``platform_toolsets["cli"] = ["file", "terminal"]`` — an explicit list of
       configurable keys, which the resolver treats as authoritative
       (``has_explicit_config``) so default toolsets aren't re-expanded.
    2. ``agent.disabled_toolsets`` — a global hard-suppression list (applied last
       in ``_get_platform_tools``, overriding every other path including the
       non-configurable platform-toolset recovery that would otherwise re-add
       toolsets like ``kanban``). We list every known toolset except the two we
       keep, guaranteeing a true blank slate regardless of platform/recovery
       quirks. The user re-enables any of them later via ``pilotage tools`` (which
       rewrites ``platform_toolsets``) or by editing ``agent.disabled_toolsets``.
    """
    keep = {"file", "terminal"}
    config.setdefault("platform_toolsets", {})["cli"] = sorted(keep)

    try:
        from toolsets import TOOLSETS
        from pilotage_cli.tools_config import CONFIGURABLE_TOOLSETS, _get_plugin_toolset_keys

        all_keys = set()
        all_keys.update(k for k, _, _ in CONFIGURABLE_TOOLSETS)
        all_keys.update(_get_plugin_toolset_keys())
        # Plain (non-composite) TOOLSETS entries — catches recovered toolsets
        # like ``kanban`` that aren't in CONFIGURABLE_TOOLSETS but get re-added.
        for k, tdef in TOOLSETS.items():
            if k.startswith("pilotage-"):
                continue  # platform composites — not user-facing toolsets
            if isinstance(tdef, dict) and tdef.get("includes"):
                continue  # composite groupings, not leaf toolsets
            if isinstance(tdef, dict) and tdef.get("posture"):
                continue  # posture toolsets (e.g. coding) are session-level
                # selections made by agent/coding_context.py — not permanent
                # user-facing disables. Adding them here causes model_tools
                # to subtract their tools (terminal, read_file, …) from the
                # minimal Blank Slate surface.
            all_keys.add(k)

        disabled = sorted(all_keys - keep)
        if disabled:
            config.setdefault("agent", {})["disabled_toolsets"] = disabled
    except Exception as exc:
        logger.debug("blank-slate disabled_toolsets computation skipped: %s", exc)


def _blank_slate_minimize_config(config: dict):
    """Turn OFF the optional config features for a Blank Slate install.

    Everything here is opt-in afterwards via ``pilotage setup agent`` /
    ``pilotage config set``. We keep only what's needed to run.
    """
    config.setdefault("agent", {})["max_turns"] = 90

    # Compression off — minimal footprint; user opts in if they want long sessions.
    config.setdefault("compression", {})["enabled"] = False

    # No automatic memory / user-profile capture.
    mem = config.setdefault("memory", {})
    mem["memory_enabled"] = False
    mem["user_profile_enabled"] = False

    # No filesystem checkpoints, no smart model routing, no auto session reset.
    config.setdefault("checkpoints", {})["enabled"] = False
    config.setdefault("smart_model_routing", {})["enabled"] = False
    config.setdefault("session_reset", {})["mode"] = "none"

    # Quiet, minimal display.
    config.setdefault("display", {})["tool_progress"] = "all"


def _run_blank_slate_setup(config: dict, pilotage_home, is_existing: bool):
    """Blank Slate setup — start with everything off except the bare minimum.

    Forces only the essentials to run an agent (provider + model, the file and
    terminal toolsets) and turns every other tool/skill/plugin/MCP/config
    feature OFF. After applying that minimal baseline, the user chooses one of
    two paths:

      1. Start with everything disabled — finish now with the minimal agent.
      2. Walk through every configuration — opt each capability back in.

    Either way nothing is enabled that the user did not explicitly choose.
    """

    print()
    print_header("Blank Slate Setup")
    print_info("Everything starts OFF. First we force-enable only what's required")
    print_info("to run an agent, then you choose whether to stop there or walk")
    print_info("through enabling more — opting in to exactly what you want.")
    print_info("")
    print_info("Forced on: Provider & Model, File Operations, Terminal.")
    print_info("Everything else (web, browser, code exec, vision, memory,")
    print_info("delegation, cron, skills, plugins, MCP, …) starts disabled.")
    print()

    # ── Step 1: Provider & Model (REQUIRED — the agent cannot run without it) ──
    print_header("Step 1 — Provider & Model (required)")
    setup_model_provider(config)
    save_config(config)

    # ── Step 2: Terminal backend (where commands run — a core decision) ──
    print_header("Step 2 — Terminal Backend")
    setup_terminal_backend(config)

    # ── Step 3: Lock in the minimal toolset + minimized config knobs ──
    _blank_slate_minimal_toolsets(config)
    _blank_slate_minimize_config(config)
    save_config(config)
    print()
    print_success("Minimal baseline applied:")
    print_info("  Toolsets: file, terminal (everything else off)")
    print_info("  Compression, memory, checkpoints, smart routing: off")

    # ── The fork: stop here, or walk through enabling things ──
    print()
    print_header("How far do you want to go?")
    path = prompt_choice(
        "Your minimal agent is ready. What next?",
        [
            "Start with everything disabled — finish now (most minimal)",
            "Walk through all configurations — opt in to tools, skills, plugins, MCP",
        ],
        0,
    )

    if path == 0:
        save_config(config)
        # Blank Slate means no bundled skills; record the opt-out so future
        # `pilotage update` runs don't re-inject them.
        try:
            from tools.skills_sync import set_bundled_skills_opt_out
            set_bundled_skills_opt_out(True)
        except Exception as exc:
            logger.debug("blank-slate skill opt-out error: %s", exc)
        print()
        print_success("Blank Slate setup complete — minimal agent ready.")
        print_info("Enable anything later, on demand:")
        print_info("  Enable tools:        pilotage tools")
        print_info("  Seed skills:         pilotage skills opt-in --sync")
        print_info("  Add MCP servers:     pilotage mcp add")
        print_info("  Enable plugins:      pilotage plugins")
        print_info("  Tune agent settings: pilotage setup agent")
        print()
        _print_setup_summary(config, pilotage_home)
        return

    # ── Walkthrough path — opt in to each capability ──
    _blank_slate_walkthrough(config, pilotage_home)


def _blank_slate_walkthrough(config: dict, pilotage_home):
    """Opt-in walkthrough for Blank Slate: skills, tools, plugins, MCP, gateway."""
    from pilotage_cli.config import load_config

    # ── Bundled skills — default to NONE, offer to seed all ──
    print()
    print_header("Bundled Skills")
    print_info("Blank Slate ships with NO bundled skills by default.")
    seed_skills = prompt_yes_no(
        "Seed the full bundled skill catalog? (No = start with zero skills)",
        default=False,
    )
    try:
        from tools.skills_sync import set_bundled_skills_opt_out, sync_skills
        if seed_skills:
            # Make sure no stale opt-out marker blocks the seed, then sync.
            set_bundled_skills_opt_out(False)
            result = sync_skills(quiet=True)
            copied = len(result.get("copied", [])) if isinstance(result, dict) else 0
            print_success(f"Seeded {copied} bundled skills.")
        else:
            set_bundled_skills_opt_out(True)
            print_info("No skills seeded. A .no-bundled-skills marker keeps future")
            print_info("`pilotage update` runs from re-injecting them. Opt back in any")
            print_info("time with `pilotage skills opt-in --sync`.")
    except Exception as exc:
        logger.debug("blank-slate skill handling error: %s", exc)
        print_warning(f"Skill setup step encountered an error: {exc}")

    # ── Walk through enabling additional tools ──
    print()
    print_header("Tools")
    print_info("Pick exactly which additional toolsets to turn on.")
    print_info("(file and terminal are already on; leave the rest off if you want")
    print_info(" the most minimal agent.)")
    if prompt_yes_no("Open the tool selector to enable more tools?", default=False):
        try:
            from pilotage_cli.tools_config import tools_command
            tools_command(first_install=False, config=config)
            # tools_command saves via its own load/save cycle — re-sync.
            _refreshed = load_config()
            config.clear()
            config.update(_refreshed)
        except Exception as exc:
            logger.debug("blank-slate tools_command error: %s", exc)
            print_warning(f"Tool selector encountered an error: {exc}")
    else:
        print_info("Keeping the minimal toolset. Add tools later with `pilotage tools`.")

    # ── Built-in plugins (off unless chosen) ──
    print()
    print_header("Plugins")
    if prompt_yes_no("Review and enable built-in plugins now?", default=False):
        print_info("Manage plugins with `pilotage plugins list` / `pilotage plugins install`.")
    else:
        print_info("No plugins enabled. Add later with `pilotage plugins`.")

    # ── MCP servers (off unless chosen) ──
    print()
    print_header("MCP Servers")
    if prompt_yes_no("Add an MCP server now?", default=False):
        print_info("Add servers with `pilotage mcp add <name> --url ... | --command ...`.")
    else:
        print_info("No MCP servers configured. Add later with `pilotage mcp add`.")

    # ── Optional messaging gateway ──
    print()
    if prompt_yes_no("Connect a messaging platform (Telegram, WhatsApp)?", default=False):
        setup_gateway(config)

    save_config(config)

    print()
    print_success("Blank Slate setup complete — minimal agent ready.")
    print_info("  Enable more tools:   pilotage tools")
    print_info("  Seed skills:         pilotage skills opt-in --sync")
    print_info("  Add MCP servers:     pilotage mcp add")
    print_info("  Tune agent settings: pilotage setup agent")
    print()

    _print_setup_summary(config, pilotage_home)


def _run_quick_setup(config: dict, pilotage_home):
    """Quick setup — only configure items that are missing."""
    from pilotage_cli.config import (
        get_missing_env_vars,
        get_missing_config_fields,
        check_config_version,
    )

    print()
    print_header("Quick Setup — Missing Items Only")

    # Check what's missing
    missing_required = [
        v for v in get_missing_env_vars(required_only=False) if v.get("is_required")
    ]
    missing_optional = [
        v for v in get_missing_env_vars(required_only=False) if not v.get("is_required")
    ]
    missing_config = get_missing_config_fields()
    current_ver, latest_ver = check_config_version()

    has_anything_missing = (
        missing_required
        or missing_optional
        or missing_config
        or current_ver < latest_ver
    )

    if not has_anything_missing:
        print_success("Everything is configured! Nothing to do.")
        print()
        print_info("Run 'pilotage setup' and choose 'Full Setup' to reconfigure,")
        print_info("or pick a specific section from the menu.")
        return

    # Handle missing required env vars
    if missing_required:
        print()
        print_info(f"{len(missing_required)} required setting(s) missing:")
        for var in missing_required:
            print(f"     • {var['name']}")
        print()

        for var in missing_required:
            print()
            print(color(f"  {var['name']}", Colors.CYAN))
            print_info(f"  {var.get('description', '')}")
            if var.get("url"):
                print_info(f"  Get key at: {var['url']}")

            if var.get("password"):
                value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
            else:
                value = prompt(f"  {var.get('prompt', var['name'])}")

            if value:
                save_env_value(var["name"], value)
                print_success(f"  Saved {var['name']}")
            else:
                print_warning(f"  Skipped {var['name']}")

    # Split missing optional vars by category
    missing_tools = [v for v in missing_optional if v.get("category") == "tool"]
    missing_messaging = [
        v
        for v in missing_optional
        if v.get("category") == "messaging" and not v.get("advanced")
    ]

    # ── Tool API keys (checklist) ──
    if missing_tools:
        print()
        print_header("Tool API Keys")

        checklist_labels = []
        for var in missing_tools:
            tools = var.get("tools", [])
            tools_str = f" → {', '.join(tools[:2])}" if tools else ""
            checklist_labels.append(f"{var.get('description', var['name'])}{tools_str}")

        selected_indices = prompt_checklist(
            "Which tools would you like to configure?",
            checklist_labels,
        )

        for idx in selected_indices:
            var = missing_tools[idx]
            _prompt_api_key(var)

    # ── Messaging platforms (checklist then prompt for selected) ──
    if missing_messaging:
        print()
        print_header("Messaging Platforms")
        print_info("Connect Pilotage to messaging apps to chat from anywhere.")
        print_info("You can configure these later with 'pilotage setup gateway'.")

        # Group by platform (preserving order)
        platform_order = []
        platforms = {}
        for var in missing_messaging:
            name = var["name"]
            if "TELEGRAM" in name:
                plat = "Telegram"
            else:
                continue
            if plat not in platforms:
                platform_order.append(plat)
            platforms.setdefault(plat, []).append(var)

        platform_labels = [
            {"Telegram": "📱 Telegram"}.get(p, p)
            for p in platform_order
        ]

        selected_indices = prompt_checklist(
            "Which platforms would you like to set up?",
            platform_labels,
        )

        for idx in selected_indices:
            plat = platform_order[idx]
            vars_list = platforms[plat]
            emoji = {"Telegram": "📱"}.get(plat, "")
            print()
            print(color(f"  ─── {emoji} {plat} ───", Colors.CYAN))
            print()
            for var in vars_list:
                print_info(f"  {var.get('description', '')}")
                if var.get("url"):
                    print_info(f"  {var['url']}")
                if var.get("password"):
                    value = prompt(f"  {var.get('prompt', var['name'])}", password=True)
                else:
                    value = prompt(f"  {var.get('prompt', var['name'])}")
                if value:
                    save_env_value(var["name"], value)
                    print_success("  ✓ Saved")
                else:
                    print_warning("  Skipped")
                print()

    # Handle missing config fields
    if missing_config:
        print()
        print_info(
            f"Adding {len(missing_config)} new config option(s) with defaults..."
        )
        for field in missing_config:
            print_success(f"  Added {field['key']} = {field['default']}")

        # Update config version
        config["_config_version"] = latest_ver
        save_config(config)

    # Jump to summary
    _print_setup_summary(config, pilotage_home)
