"""Model-independent operator controls for profile cron jobs."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pilotage.config import Config
from pilotage.tools.cron import execute_cronjob
from pilotage.tools.registry import ToolContext

from .jobs import AmbiguousJobReference, CronError, CronStore


def add_cron_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("cron", help="manage scheduled jobs")
    commands = parser.add_subparsers(dest="cron_command", required=True)

    listing = commands.add_parser("list", help="list jobs")
    listing.add_argument("--all", action="store_true", help="include inactive jobs")

    create = commands.add_parser("create", help="create a local-output job")
    create.add_argument("schedule", help="30m, every 2h, cron expression, or ISO time")
    create.add_argument("--prompt", default="", help="self-contained scheduled task")
    create.add_argument("--name", default="")
    create.add_argument("--repeat", type=int)
    create.add_argument("--skill", action="append", default=[])

    update = commands.add_parser("update", help="update a job")
    update.add_argument("job_id")
    update.add_argument("--prompt")
    update.add_argument("--name")
    update.add_argument("--schedule")
    update.add_argument("--repeat", type=int)
    skills = update.add_mutually_exclusive_group()
    skills.add_argument("--skill", action="append")
    skills.add_argument("--clear-skills", action="store_true")

    pause = commands.add_parser("pause", help="pause a job")
    pause.add_argument("job_id")
    pause.add_argument("--reason", default="")
    for action, help_text in (
        ("resume", "resume a job"),
        ("run", "queue a job now"),
        ("remove", "remove a job"),
        ("output", "print the latest saved output"),
    ):
        command = commands.add_parser(action, help=help_text)
        command.add_argument("job_id")


def _store(config: Config) -> CronStore:
    return CronStore(
        config.state_dir,
        timezone_name=config.cron_timezone,
        claim_ttl_seconds=config.cron_claim_ttl_seconds,
        output_retention=config.cron_output_retention,
    )


def _payload(args: argparse.Namespace) -> dict[str, Any]:
    action = args.cron_command
    if action == "list":
        return {"action": "list", "include_disabled": args.all}
    if action == "create":
        return {
            "action": "create",
            "schedule": args.schedule,
            "prompt": args.prompt,
            "name": args.name,
            "repeat": args.repeat,
            "skills": args.skill,
        }
    if action == "update":
        result: dict[str, Any] = {"action": "update", "job_id": args.job_id}
        for field in ("prompt", "name", "schedule", "repeat"):
            value = getattr(args, field)
            if value is not None:
                result[field] = value
        if args.clear_skills:
            result["skills"] = []
        elif args.skill is not None:
            result["skills"] = args.skill
        return result
    result = {"action": action, "job_id": args.job_id}
    if action == "pause":
        result["reason"] = args.reason
    return result


def _print_job(job: dict[str, Any]) -> None:
    print(
        f"{job.get('job_id')}\t{job.get('state')}\t"
        f"{job.get('schedule')}\t{job.get('next_run_at') or '-'}\t"
        f"{job.get('name')}"
    )


def run_cron_command(args: argparse.Namespace, config: Config) -> int:
    store = _store(config)
    if args.cron_command == "output":
        try:
            job = store.resolve_job(args.job_id)
        except (AmbiguousJobReference, CronError, OSError) as exc:
            print(exc, file=sys.stderr)
            return 1
        if job is None:
            print(f"Cron job {args.job_id!r} was not found.", file=sys.stderr)
            return 1
        output = store.latest_output(job["id"])
        if output is None:
            print("No saved output.", file=sys.stderr)
            return 1
        print(output)
        return 0

    context = ToolContext(chat_id="cli", config=config, cron_store=store)
    raw = execute_cronjob(_payload(args), context)
    try:
        result = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        print("Cron command returned unreadable data.", file=sys.stderr)
        return 1
    if not result.get("success"):
        print(result.get("error") or "Cron command failed.", file=sys.stderr)
        for match in result.get("matches") or ():
            _print_job(match)
        return 1
    if args.cron_command == "list":
        for job in result["jobs"]:
            _print_job(job)
        if not result["jobs"]:
            print("No cron jobs.")
    elif result.get("job"):
        _print_job(result["job"])
    elif result.get("removed_job"):
        removed = result["removed_job"]
        print(f"Removed {removed.get('job_id')}\t{removed.get('name')}")
    return 0


__all__ = ["add_cron_parser", "run_cron_command"]
