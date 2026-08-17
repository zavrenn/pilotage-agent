"""``pilotage plugins`` subcommand parser.

Extracted from ``pilotage_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_plugins_parser(subparsers, *, cmd_plugins: Callable) -> None:
    """Attach the ``plugins`` subcommand to ``subparsers``."""
    plugins_parser = subparsers.add_parser(
        "plugins",
        help="Manage and validate plugins",
        description=(
            "List, inspect, enable, disable, or validate native Pilotage plugins "
            "and portable Agent Plugins v1 packages. Plugins are authored by hand "
            "as directories under ~/.pilotage/plugins/."
        ),
    )
    plugins_subparsers = plugins_parser.add_subparsers(dest="plugins_action")

    plugins_list = plugins_subparsers.add_parser(
        "list", aliases=["ls"], help="List plugins"
    )
    plugins_list.add_argument(
        "--enabled",
        action="store_true",
        help="Show only enabled plugins",
    )
    plugins_list.add_argument(
        "--user",
        action="store_true",
        help="Show only user plugins",
    )
    plugins_list.add_argument(
        "--no-bundled",
        action="store_true",
        help="Hide bundled plugins",
    )
    plugins_list.add_argument(
        "--plain",
        action="store_true",
        help="Print compact plain-text output instead of a Rich table",
    )
    plugins_list.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )

    plugins_enable = plugins_subparsers.add_parser(
        "enable", help="Enable a disabled plugin"
    )
    plugins_enable.add_argument("name", help="Plugin name to enable")
    _enable_override_group = plugins_enable.add_mutually_exclusive_group()
    _enable_override_group.add_argument(
        "--allow-tool-override",
        action="store_true",
        help="Grant this plugin permission to replace built-in tools "
        "(e.g. shell_exec, write_file). Skips the confirmation prompt.",
    )
    _enable_override_group.add_argument(
        "--no-allow-tool-override",
        action="store_true",
        help="Enable without granting built-in tool override (skip prompt).",
    )

    plugins_disable = plugins_subparsers.add_parser(
        "disable", help="Disable a plugin without deleting its directory"
    )
    plugins_disable.add_argument("name", help="Plugin name to disable")

    plugins_capabilities = plugins_subparsers.add_parser(
        "capabilities",
        help="Show declared vs granted capabilities per plugin",
        description=(
            "Show each plugin's declared capabilities (from plugin.yaml) "
            "against what the user has granted. Capabilities are a consent "
            "and audit layer over host API surfaces — NOT a sandbox."
        ),
    )
    plugins_capabilities.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Plugin id to inspect (omit to list all plugins with capabilities)",
    )

    plugins_doctor = plugins_subparsers.add_parser(
        "doctor", help="Validate a plugin with the real runtime contracts"
    )
    plugins_doctor.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Plugin path or installed plugin id (default: current directory)",
    )
    plugins_doctor.add_argument(
        "--ci",
        action="store_true",
        help="Exit non-zero when validation reports an error",
    )

    plugins_show = plugins_subparsers.add_parser(
        "show",
        aliases=["info"],
        help="Show details for a single plugin (including emits/listens)",
    )
    plugins_show.add_argument("name", help="Plugin name or key to show")

    plugins_parser.set_defaults(func=cmd_plugins)
