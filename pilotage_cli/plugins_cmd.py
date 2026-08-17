"""``pilotage plugins`` CLI subcommand — list, inspect, enable, and disable plugins.

Plugins are authored by hand as directories under ``~/.pilotage/plugins/``.
There is no installer: a plugin exists because its directory exists.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from pilotage_constants import get_pilotage_home
from pilotage_cli.config import cfg_get

logger = logging.getLogger(__name__)


# Minimum manifest version this installer understands.
# Plugins may declare ``manifest_version: 1`` in plugin.yaml;
# future breaking changes to the manifest schema bump this.
_SUPPORTED_MANIFEST_VERSION = 1


def _plugins_dir() -> Path:
    """Return the user plugins directory, creating it if needed."""
    plugins = get_pilotage_home() / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    return plugins


def _read_manifest(plugin_dir: Path) -> dict:
    """Read a native or portable manifest, preferring native YAML."""
    manifest_file = plugin_dir / "plugin.yaml"
    if not manifest_file.exists():
        manifest_file = plugin_dir / "plugin.yml"
    if not manifest_file.exists():
        portable_file = plugin_dir / "plugin.json"
        if not portable_file.exists() and not portable_file.is_symlink():
            return {}
        try:
            from pilotage_cli.agent_plugins import read_agent_plugin_manifest

            manifest, _ = read_agent_plugin_manifest(plugin_dir)
            return manifest
        except Exception as e:
            logger.warning("Failed to read plugin.json in %s: %s", plugin_dir, e)
            return {}
    try:
        import yaml

        with open(manifest_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to read plugin.yaml in %s: %s", plugin_dir, e)
        return {}


def _get_disabled_set() -> set:
    """Read the disabled plugins set from config.yaml.

    An explicit deny-list. A plugin name here never loads, even if also
    listed in ``plugins.enabled``.
    """
    try:
        from pilotage_cli.config import load_config
        config = load_config()
        disabled = cfg_get(config, "plugins", "disabled", default=[])
        return set(disabled) if isinstance(disabled, list) else set()
    except Exception:
        return set()


def _save_disabled_set(disabled: set) -> None:
    """Write the disabled plugins list to config.yaml."""
    from pilotage_cli.config import load_config, save_config
    config = load_config()
    if "plugins" not in config:
        config["plugins"] = {}
    config["plugins"]["disabled"] = sorted(disabled)
    save_config(config)


_BASIC_AUTH_PLUGIN_KEYS = frozenset({"basic", "dashboard_auth/basic"})


def ensure_basic_auth_plugin_enabled_in_config(cfg: dict) -> bool:
    """Re-enable the bundled basic dashboard-auth plugin in *cfg*.

    ``pilotage setup`` / ``pilotage plugins disable basic`` can park the plugin
    in ``plugins.disabled`` while ``dashboard.basic_auth`` is configured.
    The basic provider is a bundled backend that still respects the
    deny-list, so password auth silently fails until the block is removed.

    Returns True when ``plugins.disabled`` was modified.
    """
    plugins_cfg = cfg.get("plugins")
    if not isinstance(plugins_cfg, dict):
        return False
    disabled = plugins_cfg.get("disabled")
    if not isinstance(disabled, list):
        return False
    if not (set(disabled) & _BASIC_AUTH_PLUGIN_KEYS):
        return False
    plugins_cfg["disabled"] = sorted(
        set(disabled) - _BASIC_AUTH_PLUGIN_KEYS
    )
    return True


def _get_enabled_set() -> set:
    """Read the enabled plugins allow-list from config.yaml.

    Plugins are opt-in: only names here are loaded. Returns ``set()`` if
    the key is missing (same behaviour as "nothing enabled yet").
    """
    try:
        from pilotage_cli.config import load_config
        config = load_config()
        plugins_cfg = config.get("plugins", {})
        if not isinstance(plugins_cfg, dict):
            return set()
        enabled = plugins_cfg.get("enabled", [])
        return set(enabled) if isinstance(enabled, list) else set()
    except Exception:
        return set()


def _save_enabled_set(enabled: set) -> None:
    """Write the enabled plugins list to config.yaml."""
    from pilotage_cli.config import load_config, save_config
    config = load_config()
    if "plugins" not in config:
        config["plugins"] = {}
    config["plugins"]["enabled"] = sorted(enabled)
    save_config(config)


def _resolve_plugin_key(name: str) -> Optional[str]:
    """Resolve a user-supplied plugin identifier to its canonical registry key.

    Accepts either the bare manifest name (``nemo_relay``), the directory
    name, or the full path-derived key (``observability/nemo_relay``) and
    returns the canonical key the loader gates on (``manifest.key`` or, for a
    flat plugin, the bare name). Returns ``None`` when no plugin matches.

    This is the single normalization point so ``pilotage plugins enable`` /
    ``disable`` write the same key that ``PluginManager`` matches against —
    nested category plugins (e.g. ``observability/nemo_relay``) included.
    """
    entries = _discover_all_plugins()
    # 1. Exact match on canonical key or manifest name — always unambiguous.
    for entry in entries:
        # entry = (name, version, description, source, dir_path, key)
        if name == entry[5] or name == entry[0]:
            return entry[5]
    # 2. Fall back to a bare leaf-name match (e.g. "nemo_relay" ->
    #    "observability/nemo_relay"), but only when it resolves to exactly one
    #    plugin so we never silently pick the wrong same-named nested plugin.
    leaf_matches = [entry[5] for entry in entries if name == entry[5].split("/")[-1]]
    if len(leaf_matches) == 1:
        return leaf_matches[0]
    return None


def _resolve_plugin_key_and_source(name: str) -> Optional[tuple]:
    """Resolve *name* to ``(canonical_key, source)`` or ``None`` if no match.

    Mirrors :func:`_resolve_plugin_key`'s normalization but also returns the
    plugin's source (``"bundled"``, ``"user"``, ``"project"``, ...) so the
    enable path can tell whether a built-in-override consent prompt is needed.
    """
    entries = _discover_all_plugins()
    for entry in entries:
        # entry = (name, version, description, source, dir_path, key)
        if name == entry[5] or name == entry[0]:
            return (entry[5], entry[3])
    leaf_matches = [
        (entry[5], entry[3]) for entry in entries
        if name == entry[5].split("/")[-1]
    ]
    if len(leaf_matches) == 1:
        return leaf_matches[0]
    return None


def _set_plugin_entry_flag(plugin_id: str, key: str, value: bool) -> None:
    """Write ``plugins.entries.<plugin_id>.<key> = value`` into config.yaml."""
    from pilotage_cli.config import load_config, save_config
    config = load_config()
    plugins_cfg = config.setdefault("plugins", {})
    if not isinstance(plugins_cfg, dict):
        plugins_cfg = {}
        config["plugins"] = plugins_cfg
    entries = plugins_cfg.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        plugins_cfg["entries"] = entries
    entry = entries.setdefault(plugin_id, {})
    if not isinstance(entry, dict):
        entry = {}
        entries[plugin_id] = entry
    entry[key] = bool(value)
    save_config(config)


def cmd_enable(name: str, allow_tool_override: Optional[bool] = None) -> None:
    """Add a plugin to the enabled allow-list (and remove it from disabled).

    For non-bundled plugins, prompt the operator about granting the
    privileged ``allow_tool_override`` capability (replacing built-in tools
    like ``shell_exec`` / ``write_file``). ``allow_tool_override`` is a
    tri-state: ``True`` grants without prompting, ``False`` declines without
    prompting, ``None`` (default) asks interactively. Bundled plugins are
    trusted and never prompted.
    """
    from rich.console import Console

    console = Console()
    # Discover the plugin — check installed (user) AND bundled, including
    # nested category plugins — and normalize to its canonical registry key.
    resolved = _resolve_plugin_key_and_source(name)
    if resolved is None:
        console.print(f"[red]Plugin '{name}' is not installed or bundled.[/red]")
        sys.exit(1)
    key, source = resolved

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()

    already_enabled = key in enabled and key not in disabled

    if not already_enabled:
        enabled.add(key)
        disabled.discard(key)
        # Drop every alias of this plugin from the disabled list so an
        # explicit disable under a different form can't keep it off. The
        # loader's disable check matches on BOTH the canonical key
        # (``web/firecrawl``) AND the manifest name (``web-firecrawl``);
        # a stale entry under either form makes "explicit disable wins"
        # (plugins.py) silently veto this enable. Discard the key, its
        # bare leaf, and the manifest name. follow-up.)
        bare = key.split("/")[-1]
        if bare != key:
            disabled.discard(bare)
        for entry in _discover_all_plugins():
            # entry = (name, version, description, source, dir_path, key)
            if entry[5] == key:
                disabled.discard(entry[0])
                break
        _save_enabled_set(enabled)
        _save_disabled_set(disabled)
        console.print(
            f"[green]✓[/green] Plugin [bold]{key}[/bold] enabled. "
            "Takes effect on next session."
        )
    else:
        console.print(f"[dim]Plugin '{key}' is already enabled.[/dim]")

    # Built-in tool override is a privileged grant. Bundled plugins ship with
    # Pilotage core and are trusted; every other source needs operator opt-in.
    if source == "bundled":
        return

    # Capability consent: when the manifest declares capabilities,
    # the consent screen is the canonical grant path — it covers
    # tools.override too, so skip the legacy standalone prompt unless the
    # operator explicitly passed --allow-tool-override/--no-allow-tool-override.
    declared_caps = _declared_capabilities_for_key(key)
    if declared_caps:
        _run_capability_consent(console, key, declared_caps, context="enable")
        if allow_tool_override is not None:
            _resolve_tool_override_grant(console, key, allow_tool_override)
        return

    _resolve_tool_override_grant(console, key, allow_tool_override)


# ── Capability consent flow ─────────────────────────────────────────


def _declared_capabilities_from_manifest(manifest: dict, plugin_name: str = "?") -> list:
    """Extract + normalize the ``capabilities:`` declaration from a manifest."""
    from pilotage_cli.plugin_capabilities import parse_declared_capabilities

    return parse_declared_capabilities(
        (manifest or {}).get("capabilities"), plugin_name
    )


def _declared_capabilities_for_key(key: str) -> list:
    """Read the declared capabilities for an installed/bundled plugin by key."""
    for entry in _discover_all_plugins():
        # entry = (name, version, description, source, dir_path, key)
        if entry[5] == key or entry[0] == key:
            if entry[3] == "entrypoint":
                from pilotage_cli.plugins import discover_entrypoint_manifests

                for manifest in discover_entrypoint_manifests():
                    if key in (manifest.key, manifest.name):
                        return list(manifest.capabilities)
                return []
            dir_path = entry[4]
            if not dir_path:
                return []
            manifest = _read_manifest(Path(dir_path))
            return _declared_capabilities_from_manifest(manifest, entry[0])
    return []


def _print_capability_list(console, capabilities: list) -> None:
    """Render the consent screen body: one line per capability."""
    from pilotage_cli.plugin_capabilities import CAPABILITY_REGISTRY

    for cap in capabilities:
        spec = CAPABILITY_REGISTRY.get(cap)
        desc = spec.description if spec else ""
        console.print(f"    [bold]{cap}[/bold] — {desc}")


def _run_capability_consent(
    console,
    plugin_id: str,
    declared: list,
    *,
    context: str = "install",
) -> bool:
    """Show the capability consent screen and record the decision.

    Prints the declared capability list with one-line risk descriptions and
    asks a single Y/n. On consent, the *pending* capabilities are granted
    (recorded under ``plugins.entries.<plugin_id>.granted_capabilities`` with
    a consent hash of the declared set). On decline — or in ANY
    non-interactive context — capabilities stay ungranted (fail closed) and
    the plugin must degrade gracefully via ``ctx.has_capability()``.

    The consent wording deliberately does not imply a code audit: granting a
    capability trusts the plugin author. This is consent + audit, NOT a
    sandbox — an in-process plugin can run arbitrary Python regardless.

    Returns True when consent was granted.
    """
    from pilotage_cli.plugin_capabilities import (
        pending_capabilities,
        record_consent,
    )

    pending = pending_capabilities(plugin_id, declared)
    if not pending:
        # Everything declared is already granted — refresh the consent hash
        # so a later declaration change is detected against the current set.
        if declared:
            record_consent(plugin_id, [], declared)
        return True

    verb = "requests" if context == "install" else "now requests"
    console.print(
        f"\n  [yellow]Plugin [bold]{plugin_id}[/bold] {verb} the following "
        "capabilities:[/yellow]"
    )
    _print_capability_list(console, pending)
    console.print(
        "  [dim]Granting trusts the plugin author with these host surfaces. "
        "This is consent, not a sandbox — plugins run as regular Python "
        "in-process.[/dim]"
    )

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print(
            "  [yellow]Non-interactive session: capabilities NOT granted "
            "(fail closed).[/yellow] Run "
            f"`pilotage plugins capabilities {plugin_id}` to review and "
            f"`pilotage plugins enable {plugin_id}` to grant interactively."
        )
        return False

    try:
        answer = console.input("  Grant these capabilities? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer in {"y", "yes"}:
        record_consent(plugin_id, pending, declared)
        console.print(
            f"  [green]✓[/green] Granted: {', '.join(pending)} "
            f"([dim]plugins.entries.{plugin_id}.granted_capabilities[/dim])"
        )
        return True

    console.print(
        f"  [dim]Declined. {plugin_id} stays enabled with these capabilities "
        "off; it should degrade gracefully (ctx.has_capability()). Re-run "
        f"`pilotage plugins enable {plugin_id}` to grant later.[/dim]"
    )
    return False


def cmd_capabilities(name: Optional[str] = None) -> None:
    """``pilotage plugins capabilities [<id>]`` — declared vs granted."""
    from rich.console import Console

    from pilotage_cli.plugin_capabilities import granted_capabilities

    console = Console()

    rows = []
    for entry in _discover_all_plugins():
        # entry = (name, version, description, source, dir_path, key)
        key = entry[5] or entry[0]
        if name is not None and name not in (key, entry[0]):
            continue
        declared = _declared_capabilities_for_key(key)
        granted = granted_capabilities(key)
        # Legacy grants surface too: report capabilities live via deprecated
        # allow_* keys so `capabilities` shows the true effective state.
        from pilotage_cli.plugin_capabilities import (
            CAPABILITY_REGISTRY,
            plugin_capability_granted,
        )
        effective = {
            cap for cap in CAPABILITY_REGISTRY
            if plugin_capability_granted(key, cap)
        }
        if not declared and not effective and name is None:
            continue
        rows.append((key, entry[3], declared, granted, effective))

    if name is not None and not rows:
        console.print(f"[red]Plugin '{name}' is not installed or bundled.[/red]")
        sys.exit(1)

    if not rows:
        console.print("[dim]No plugins declare or hold capabilities.[/dim]")
        return

    for key, source, declared, granted, effective in sorted(rows):
        console.print(f"[bold]{key}[/bold] [dim]({source})[/dim]")
        if not declared:
            console.print("  declared: [dim](none)[/dim]")
        for cap in declared:
            if cap in effective:
                mark = "[green]granted[/green]"
                if cap not in granted:
                    mark += " [dim](via legacy allow_* key — deprecated)[/dim]"
            else:
                mark = "[yellow]not granted[/yellow]"
            console.print(f"  {cap}: {mark}")
        for cap in sorted(effective - set(declared)):
            console.print(
                f"  {cap}: [green]granted[/green] "
                "[dim](not declared in manifest)[/dim]"
            )


def _resolve_tool_override_grant(
    console, key: str, allow_tool_override: Optional[bool]
) -> None:
    """Resolve and persist the ``allow_tool_override`` grant for a plugin.

    ``allow_tool_override`` tri-state: True grants, False declines, None
    prompts interactively (defaulting to deny on a non-interactive stdin).
    """
    if allow_tool_override is None:
        # Interactive consent. Default to NO so a blind Enter doesn't grant
        # a privileged capability, and a non-interactive stdin denies safely.
        prompt = (
            "[yellow]Allow this plugin to replace built-in tools "
            "(e.g. shell_exec, write_file)?[/yellow]\n"
            "  This is a privileged capability: an override can intercept "
            "everything the agent routes through that tool.\n"
            "  Grant it? [y/N] "
        )
        try:
            answer = console.input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        allow_tool_override = answer in {"y", "yes"}

    plugin_id = key
    _set_plugin_entry_flag(plugin_id, "allow_tool_override", allow_tool_override)
    if allow_tool_override:
        console.print(
            f"[green]✓[/green] Granted [bold]{key}[/bold] permission to "
            "override built-in tools "
            f"([dim]plugins.entries.{plugin_id}.allow_tool_override: true[/dim])."
        )
    else:
        console.print(
            f"[dim]{key} may not override built-in tools. Re-run "
            f"`pilotage plugins enable {key} --allow-tool-override` to grant "
            "this later.[/dim]"
        )


def cmd_disable(name: str) -> None:
    """Remove a plugin from the enabled allow-list (and add to disabled)."""
    from rich.console import Console

    console = Console()
    key = _resolve_plugin_key(name)
    if key is None:
        console.print(f"[red]Plugin '{name}' is not installed or bundled.[/red]")
        sys.exit(1)

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()

    if key not in enabled and key in disabled:
        console.print(f"[dim]Plugin '{key}' is already disabled.[/dim]")
        return

    enabled.discard(key)
    # Drop any legacy bare-name entry from the allow-list too, so a stale
    # bare name can't keep a nested plugin loading after an explicit disable.
    bare = key.split("/")[-1]
    if bare != key:
        enabled.discard(bare)
    disabled.add(key)
    _save_enabled_set(enabled)
    _save_disabled_set(disabled)
    console.print(
        f"[yellow]\u2298[/yellow] Plugin [bold]{key}[/bold] disabled. "
        "Takes effect on next session."
    )


def _plugin_exists(name: str) -> bool:
    """Return True if a plugin with *name* (bare name or key) exists."""
    return _resolve_plugin_key(name) is not None


def _read_manifest_info(d: Path, prefix: str):
    """Read a native or portable manifest and return display metadata.

    Returns None if no manifest file exists.
    """
    manifest_file = d / "plugin.yaml"
    if not manifest_file.exists():
        manifest_file = d / "plugin.yml"
    if not manifest_file.exists():
        portable_file = d / "plugin.json"
        if not portable_file.exists() and not portable_file.is_symlink():
            return None
        try:
            from pilotage_cli.agent_plugins import read_agent_plugin_manifest

            manifest, _ = read_agent_plugin_manifest(d)
            name = manifest["name"]
            key = f"{prefix}/{d.name}" if prefix else name
            return (
                name,
                manifest.get("version", ""),
                manifest.get("description", ""),
                key,
            )
        except Exception:
            return None
    try:
        import yaml
    except ImportError:
        yaml = None
    name = d.name
    version = ""
    description = ""
    if yaml:
        try:
            with open(manifest_file, encoding="utf-8") as f:
                manifest = yaml.safe_load(f) or {}
            name = manifest.get("name", d.name)
            version = manifest.get("version", "")
            description = manifest.get("description", "")
        except Exception:
            pass
    key = f"{prefix}/{d.name}" if prefix else name
    return name, version, description, key


def _is_portable_plugin_dir(dir_path) -> bool:
    """True when *dir_path* is an Agent Plugins v1 package (``plugin.json``
    only — a native ``plugin.yaml`` takes precedence, matching the loader)."""
    try:
        d = Path(dir_path)
        if not d.is_dir():
            return False
        if (d / "plugin.yaml").exists() or (d / "plugin.yml").exists():
            return False
        portable_file = d / "plugin.json"
        return portable_file.exists() or portable_file.is_symlink()
    except OSError:
        return False


# Manifest kinds that are active-by-default when bundled: backends auto-load,
# platforms register lazily but are available out of the box, model providers
# run through providers/ discovery (see PluginManager.discover_and_load).
_BUNDLED_DEFAULT_ON_KINDS = frozenset({"backend", "platform", "model-provider"})


def _bundled_default_on(dir_path) -> bool:
    """True when a bundled plugin at *dir_path* is active without an explicit
    ``plugins.enabled`` entry. Standalone/exclusive kinds stay opt-in, and
    portable packages (``plugin.json``) have no kind at all."""
    manifest_file = Path(dir_path) / "plugin.yaml"
    if not manifest_file.exists():
        manifest_file = Path(dir_path) / "plugin.yml"
    if not manifest_file.exists():
        return False
    try:
        import yaml

        with open(manifest_file, encoding="utf-8") as f:
            manifest = yaml.safe_load(f) or {}
        kind = str(manifest.get("kind", "standalone")).strip().lower()
        return kind in _BUNDLED_DEFAULT_ON_KINDS
    except Exception:
        return False


def _scan_level(
    base: Path,
    source: str,
    skip_names: set,
    prefix: str,
    depth: int,
    seen: dict,
) -> None:
    """Recursive directory scan matching PluginManager._scan_directory_level.

    Populates *seen* with key -> (name, version, description, source, dir, key).
    """
    if not base.is_dir():
        return
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        if depth == 0 and skip_names and d.name in skip_names:
            continue
        info = _read_manifest_info(d, prefix)
        if info is not None:
            name, version, description, key = info
            if key in seen and source == "bundled":
                continue
            src_label = source
            if source == "user" and (d / ".git").exists():
                src_label = "git"
            seen[key] = (name, version, description, src_label, d, key)
            continue
        if depth >= 1:
            continue
        sub_prefix = f"{prefix}/{d.name}" if prefix else d.name
        _scan_level(d, source, set(), sub_prefix, depth + 1, seen)


def _discover_all_plugins() -> list:
    """Return a list of (name, version, description, source, dir_path, key) for
    every plugin the loader can see — user + bundled + project + entry point.

    Matches the ordering/dedup of ``PluginManager.discover_and_load``:
    bundled first, then user, then project, then entry points. Later sources
    override earlier ones on key collision.
    """
    seen: dict = {}  # key -> (name, version, description, source, path, key)

    # Bundled (<repo>/plugins/<name>/), excluding memory/ and context_engine/
    from pilotage_cli.plugins import get_bundled_plugins_dir
    repo_plugins = get_bundled_plugins_dir()
    for base, source, skip in (
        (repo_plugins, "bundled", {"memory", "context_engine"}),
        (_plugins_dir(), "user", set()),
    ):
        _scan_level(base, source, skip, "", 0, seen)

    # Entry-point plugins (installed as Python packages; no plugin directory).
    for name, version, description, path in _discover_entrypoint_plugins():
        seen[name] = (name, version, description, "entrypoint", path, name)
    return list(seen.values())


def _discover_entrypoint_plugins() -> list[tuple[str, str, str, str]]:
    """Return plugin entries advertised through ``pilotage_agent.plugins``.

    Entry-point plugins are installed as Python packages, so they do not have a
    plugin directory under ``~/.pilotage/plugins``. Include package metadata here
    so ``pilotage plugins list`` can show and enable them.
    """
    from pilotage_cli.plugins import ENTRY_POINTS_GROUP

    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            group_eps = eps.select(group=ENTRY_POINTS_GROUP)
        elif isinstance(eps, dict):
            group_eps = eps.get(ENTRY_POINTS_GROUP, [])
        else:
            group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]
    except Exception as exc:
        logger.debug("Entry-point plugin discovery failed: %s", exc)
        return []

    entries: list[tuple[str, str, str, str]] = []
    for ep in group_eps:
        version = ""
        description = ""
        dist = getattr(ep, "dist", None)
        metadata = getattr(dist, "metadata", None)
        if metadata is not None:
            version = str(getattr(dist, "version", "") or "")
            description = str(metadata.get("Summary", "") or "")
        entries.append((ep.name, version, description, ep.value))
    return entries


def _plugin_status(name: str, enabled: set, disabled: set, key: str = "") -> str:
    """Return the user-facing activation state for a plugin name or key."""
    if name in disabled or key in disabled:
        return "disabled"
    if name in enabled or key in enabled:
        return "enabled"
    return "not enabled"


def _filter_plugin_entries(entries: list, args: Any, enabled: set, disabled: set) -> list:
    """Apply ``pilotage plugins list`` CLI filters."""
    filtered = entries
    if getattr(args, "no_bundled", False) or getattr(args, "user", False):
        filtered = [entry for entry in filtered if entry[3] != "bundled"]
    if getattr(args, "enabled", False):
        filtered = [
            entry for entry in filtered
            if _plugin_status(entry[0], enabled, disabled, key=entry[5]) == "enabled"
        ]
    return filtered


def cmd_list(args: Any | None = None) -> None:
    """List all plugins (bundled + user) with enabled/disabled state."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    entries = _discover_all_plugins()
    if not entries:
        console.print("[dim]No plugins found.[/dim]")
        console.print("[dim]Add one as a directory under:[/dim] ~/.pilotage/plugins/")
        return

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    entries = _filter_plugin_entries(entries, args, enabled, disabled)

    if getattr(args, "json", False):
        payload = [
            {
                "name": name,
                "status": _plugin_status(name, enabled, disabled, key=key),
                "version": str(version),
                "description": description,
                "source": source,
            }
            for name, version, description, source, _dir, key in entries
        ]
        print(json.dumps(payload, indent=2))
        return

    if getattr(args, "plain", False):
        for name, version, _description, source, _dir, key in entries:
            status = _plugin_status(name, enabled, disabled, key=key)
            print(f"{status:12} {source:8} {str(version):8} {name}")
        return

    if not entries:
        console.print("[dim]No plugins matched the selected filters.[/dim]")
        return

    table = Table(title="Plugins", show_lines=False)
    table.add_column("Name", style="bold")
    table.add_column("Status")
    table.add_column("Version", style="dim")
    table.add_column("Description")
    table.add_column("Source", style="dim")

    for name, version, description, source, _dir, key in entries:
        status_name = _plugin_status(name, enabled, disabled, key=key)
        if status_name == "disabled":
            status = "[red]disabled[/red]"
        elif status_name == "enabled":
            status = "[green]enabled[/green]"
        else:
            status = "[yellow]not enabled[/yellow]"
        table.add_row(name, status, str(version), description, source)

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Compact view:[/dim] pilotage plugins list --plain --no-bundled")
    console.print("[dim]Interactive toggle:[/dim] pilotage plugins")
    console.print("[dim]Enable/disable:[/dim] pilotage plugins enable/disable <name>")
    console.print("[dim]Plugins are opt-in by default — only 'enabled' plugins load.[/dim]")


# ---------------------------------------------------------------------------
# Provider plugin discovery helpers
# ---------------------------------------------------------------------------


def cmd_show(name: str) -> None:
    """Show details for a single plugin, including declared emits/listens.

    Resolves *name* against every discoverable plugin (bundled + user +
    entrypoint) by either its display name or its registry key, then reads
    its ``plugin.yaml`` to surface the advisory event-bus declarations
    (``emits`` / ``listens``) alongside the basic metadata.
    """
    from rich.console import Console

    console = Console()
    entries = _discover_all_plugins()
    match = None
    for entry in entries:
        # entry = (name, version, description, source, dir_path, key)
        if entry[0] == name or entry[5] == name:
            match = entry
            break

    if match is None:
        console.print(f"[red]Plugin '{name}' not found.[/red]")
        console.print("[dim]List installed plugins:[/dim] pilotage plugins list")
        sys.exit(1)

    pname, version, description, source, dir_path, key = match
    manifest = _read_manifest(Path(dir_path)) if dir_path else {}
    emits = manifest.get("emits") or []
    listens = manifest.get("listens") or []

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    status = _plugin_status(pname, enabled, disabled, key=key)

    console.print()
    console.print(f"[bold]{pname}[/bold]" + (f" [dim]v{version}[/dim]" if version else ""))
    if description:
        console.print(description)
    console.print(f"[dim]Status:[/dim] {status}")
    console.print(f"[dim]Source:[/dim] {source}")
    console.print(f"[dim]Key:[/dim] {key}")
    console.print(
        "[dim]Emits:[/dim] " + (", ".join(emits) if emits else "[dim](none)[/dim]")
    )
    console.print(
        "[dim]Listens:[/dim] " + (", ".join(listens) if listens else "[dim](none)[/dim]")
    )
    console.print()


def cmd_plugin_doctor(target: str = ".", *, ci: bool = False) -> None:
    """Validate one plugin through runtime discovery and registration."""
    from rich.console import Console

    from pilotage_cli.plugin_dev import doctor_plugin

    report = doctor_plugin(target)
    Console().print(report.format_text())
    if ci and not report.ok:
        raise SystemExit(1)


def plugins_command(args) -> None:
    """Dispatch pilotage plugins subcommands."""
    action = getattr(args, "plugins_action", None)

    if action == "enable":
        # Tri-state: --allow-tool-override=True, --no-allow-tool-override=False,
        # neither=None (interactive prompt for non-bundled plugins).
        allow_override = None
        if getattr(args, "allow_tool_override", False):
            allow_override = True
        elif getattr(args, "no_allow_tool_override", False):
            allow_override = False
        cmd_enable(args.name, allow_tool_override=allow_override)
    elif action == "disable":
        cmd_disable(args.name)
    elif action == "capabilities":
        cmd_capabilities(getattr(args, "name", None))
    elif action in {"list", "ls"} or action is None:
        cmd_list(args)
    elif action == "doctor":
        cmd_plugin_doctor(args.target, ci=getattr(args, "ci", False))
    elif action in {"show", "info"}:
        cmd_show(args.name)
    else:
        from rich.console import Console

        Console().print(f"[red]Unknown plugins action: {action}[/red]")
        sys.exit(1)
