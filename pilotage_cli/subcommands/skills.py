"""``pilotage skills`` subcommand parser.

Skills are authored by hand under the profile's ``skills/`` directory; there
is no registry, no bundled catalog and no installer. The CLI therefore only
lists what is on disk and toggles what is active.
"""

from __future__ import annotations

import json


def _list_skills(args) -> None:
    """Print the skills found on disk for the active profile."""
    from tools.skills_tool import _find_all_skills

    enabled_only = bool(getattr(args, "enabled_only", False))
    skills = _find_all_skills(skip_disabled=not enabled_only)

    if getattr(args, "json", False):
        print(json.dumps(skills, indent=2))
        return

    if not skills:
        print("No skills found.")
        return

    for skill in sorted(skills, key=lambda s: (s.get("category") or "", s["name"])):
        category = skill.get("category") or "uncategorized"
        description = (skill.get("description") or "").strip().replace("\n", " ")
        if len(description) > 70:
            description = description[:69] + "…"
        print(f"  {skill['name']:<28} {category:<18} {description}")

    print(f"\n{len(skills)} skill(s).")


def _cmd_skills(args) -> None:
    """Dispatch ``pilotage skills`` — list, or the interactive config UI."""
    if getattr(args, "skills_action", None) == "list":
        _list_skills(args)
        return

    from pilotage_cli.skills_config import skills_command

    skills_command(args)


def build_skills_parser(subparsers) -> None:
    """Attach the ``skills`` subcommand to ``subparsers``."""
    skills_parser = subparsers.add_parser(
        "skills",
        help="List and enable/disable your skills",
        description=(
            "List the skills installed in the active profile and enable or "
            "disable them, globally or per messaging platform."
        ),
    )
    skills_subparsers = skills_parser.add_subparsers(dest="skills_action")

    skills_list = skills_subparsers.add_parser("list", help="List installed skills")
    skills_list.add_argument(
        "--enabled-only",
        action="store_true",
        help="Hide disabled skills. Use with -p <profile> to see exactly "
        "which skills will load for that profile.",
    )
    skills_list.add_argument(
        "--json", action="store_true", help="Output the list as JSON"
    )

    skills_subparsers.add_parser(
        "config",
        help="Interactive skill configuration — enable/disable individual skills",
    )

    skills_parser.set_defaults(func=_cmd_skills)
