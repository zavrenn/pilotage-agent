"""One Hermes-shaped tool for profile-local scheduled work."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List

from pilotage.cron.jobs import AmbiguousJobReference, CronError, CronStore

from ..approvals import approval_error
from .registry import Tool, ToolContext, tool_error

logger = logging.getLogger(__name__)

_ACTIONS = {"create", "list", "update", "pause", "resume", "remove", "run"}
_MUTATING_ACTIONS = _ACTIONS - {"list"}
_ACTION_ARGUMENTS = {
    "create": {
        "action",
        "prompt",
        "schedule",
        "name",
        "repeat",
        "skills",
        "enabled_toolsets",
        "workdir",
        "deliver",
    },
    "list": {"action", "include_disabled"},
    "update": {
        "action",
        "job_id",
        "prompt",
        "schedule",
        "name",
        "repeat",
        "skills",
        "enabled_toolsets",
        "workdir",
        "deliver",
    },
    "pause": {"action", "job_id", "reason"},
    "resume": {"action", "job_id"},
    "remove": {"action", "job_id"},
    "run": {"action", "job_id"},
}
_OUTPUT_PREVIEW_CHARS = 500


def _repeat_display(job: Dict[str, Any]) -> str:
    repeat = job.get("repeat") or {}
    times = repeat.get("times")
    completed = int(repeat.get("completed") or 0)
    if times is None:
        return "forever"
    if int(times) == 1:
        return "once" if completed == 0 else "1/1"
    return f"{completed}/{times}" if completed else f"{times} times"


def _format_job(job: Dict[str, Any], store: CronStore) -> Dict[str, Any]:
    prompt = str(job.get("prompt") or "")
    output = store.latest_output(str(job.get("id") or ""))
    result: Dict[str, Any] = {
        "job_id": job.get("id"),
        "name": job.get("name"),
        "prompt_preview": prompt[:100] + ("..." if len(prompt) > 100 else ""),
        "skills": list(job.get("skills") or []),
        "enabled_toolsets": job.get("enabled_toolsets"),
        "workdir": job.get("workdir"),
        "schedule": job.get("schedule_display"),
        "repeat": _repeat_display(job),
        "deliver": job.get("deliver", "local"),
        "next_run_at": job.get("next_run_at"),
        "last_run_at": job.get("last_run_at"),
        "last_status": job.get("last_status"),
        "last_error": job.get("last_error"),
        "last_delivery_error": job.get("last_delivery_error"),
        "enabled": job.get("enabled", True),
        "state": job.get("state", "scheduled"),
        "paused_reason": job.get("paused_reason"),
    }
    if output is not None:
        result["last_output_preview"] = output[:_OUTPUT_PREVIEW_CHARS] + (
            "..." if len(output) > _OUTPUT_PREVIEW_CHARS else ""
        )
    return result


def _store(context: ToolContext) -> CronStore:
    store = context.cron_store
    if not isinstance(store, CronStore):
        raise CronError("Cron is unavailable in this agent session.")
    return store


def _skills(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, list):
        raise ValueError("skills must be a list of skill names")
    return value


def _enabled_toolsets(value: Any, context: ToolContext):
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("enabled_toolsets must be a list of tool group names")
    names = []
    for raw in value:
        name = str(raw or "").strip()
        if name and name not in names:
            names.append(name)

    from pilotage.tools import build_registry, enabled_groups

    registry = build_registry()
    known = set(registry.groups())
    unknown = sorted(set(names) - known)
    if unknown:
        raise ValueError(
            "Unknown tool groups: " + ", ".join(unknown)
        )
    unavailable = sorted(
        set(names) - set(enabled_groups(context.config.settings, registry))
    )
    if unavailable:
        raise ValueError(
            "Tool groups are disabled for this profile/channel: "
            + ", ".join(unavailable)
        )
    if "cron" in names:
        raise ValueError("The cron tool cannot be enabled inside a scheduled run.")
    return names


def _workdir(value: Any):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("workdir must be an absolute path string")
    return value


def _wake(context: ToolContext) -> None:
    if context.cron_wake is None:
        return
    try:
        context.cron_wake()
    except Exception:
        logger.debug("Could not wake cron scheduler", exc_info=True)


def _missing(reference: str) -> str:
    return tool_error(
        f"Job with ID or name {reference!r} was not found. List jobs first.",
        success=False,
    )


def execute_cronjob(args: Dict[str, Any], context: ToolContext) -> str:
    """Execute a validated cron operation without a messaging approval round-trip.

    The model-facing handler below gates mutations first. The local operator CLI
    calls this directly because invoking that command is already the operator's
    explicit authorization.
    """
    try:
        store = _store(context)
        action = str(args.get("action") or "").strip().lower()
        if action not in _ACTIONS:
            return tool_error(
                f"Unknown cron action {action!r}. Use: {', '.join(sorted(_ACTIONS))}.",
                success=False,
            )
        unexpected = set(args) - _ACTION_ARGUMENTS[action]
        if unexpected:
            return tool_error(
                f"Argument(s) not valid for {action!r}: "
                f"{', '.join(sorted(unexpected))}",
                success=False,
            )

        if action in {"create", "resume", "run"} and not getattr(
            context.config, "cron_enabled", True
        ):
            return tool_error(
                "Cron execution is disabled in config.yaml.", success=False
            )

        if action == "create":
            schedule = str(args.get("schedule") or "").strip()
            if not schedule:
                return tool_error("schedule is required for create", success=False)
            job = store.create_job(
                prompt=str(args.get("prompt") or ""),
                schedule=schedule,
                name=str(args.get("name") or ""),
                repeat=args.get("repeat"),
                skills=_skills(args.get("skills")),
                enabled_toolsets=_enabled_toolsets(
                    args.get("enabled_toolsets"), context
                ),
                workdir=_workdir(args.get("workdir")),
                origin=context.origin,
                deliver=args.get("deliver"),
            )
            _wake(context)
            return json.dumps(
                {
                    "success": True,
                    "job": _format_job(job, store),
                    "message": f"Cron job {job['name']!r} created.",
                },
                ensure_ascii=False,
            )

        if action == "list":
            include_disabled = args.get("include_disabled", False)
            if not isinstance(include_disabled, bool):
                return tool_error("include_disabled must be true or false", success=False)
            jobs = [
                _format_job(job, store)
                for job in store.list_jobs(include_disabled=include_disabled)
            ]
            return json.dumps(
                {"success": True, "count": len(jobs), "jobs": jobs},
                ensure_ascii=False,
            )

        reference = str(args.get("job_id") or "").strip()
        if not reference:
            return tool_error(f"job_id is required for action {action!r}", success=False)
        try:
            job = store.resolve_job(reference)
        except AmbiguousJobReference as exc:
            return json.dumps(
                {
                    "success": False,
                    "error": str(exc),
                    "matches": [
                        {"job_id": match["id"], "name": match.get("name")}
                        for match in exc.matches
                    ],
                },
                ensure_ascii=False,
            )
        if job is None:
            return _missing(reference)
        job_id = str(job["id"])

        if action == "remove":
            if not store.remove_job(job_id):
                return _missing(reference)
            _wake(context)
            return json.dumps(
                {
                    "success": True,
                    "removed_job": {"job_id": job_id, "name": job.get("name")},
                },
                ensure_ascii=False,
            )

        if action == "pause":
            updated = store.pause_job(job_id, reason=str(args.get("reason") or ""))
        elif action == "resume":
            updated = store.resume_job(job_id)
            _wake(context)
        elif action == "run":
            updated = store.trigger_job(job_id)
            _wake(context)
        else:
            updates: Dict[str, Any] = {}
            for field in ("name", "prompt", "schedule", "repeat", "deliver"):
                if field in args:
                    updates[field] = args[field]
            if "skills" in args:
                updates["skills"] = _skills(args["skills"])
            if "enabled_toolsets" in args:
                updates["enabled_toolsets"] = _enabled_toolsets(
                    args["enabled_toolsets"], context
                )
            if "workdir" in args:
                updates["workdir"] = _workdir(args["workdir"])
            if not updates:
                return tool_error("No updates were provided.", success=False)
            updated = store.update_job(job_id, updates)
            _wake(context)

        if updated is None:
            return _missing(reference)
        response: Dict[str, Any] = {
            "success": True,
            "job": _format_job(updated, store),
        }
        if action == "run":
            response["message"] = "The job is queued for the next scheduler tick."
        return json.dumps(response, ensure_ascii=False)
    except (CronError, OSError, TypeError, ValueError) as exc:
        return tool_error(str(exc), success=False)


def _cron_approval_request(
    args: Dict[str, Any], context: ToolContext
) -> tuple[Dict[str, Any], str] | None:
    """Validate and canonicalize enough to approve the exact durable target."""

    action = str(args.get("action") or "").strip().lower()
    if action not in _MUTATING_ACTIONS:
        return None
    if set(args) - _ACTION_ARGUMENTS[action]:
        return None
    if action in {"create", "resume", "run"} and not getattr(
        context.config, "cron_enabled", True
    ):
        return None

    canonical = dict(args)
    canonical["action"] = action
    if action == "create":
        if not str(args.get("schedule") or "").strip():
            return None
        detail = {
            key: args.get(key)
            for key in (
                "name",
                "schedule",
                "repeat",
                "skills",
                "enabled_toolsets",
                "workdir",
                "deliver",
                "prompt",
            )
            if key in args
        }
        rendered = json.dumps(detail, ensure_ascii=False)
        summary = "Create cron job:\n" + rendered
    else:
        reference = str(args.get("job_id") or "").strip()
        if not reference:
            return None
        try:
            job = _store(context).resolve_job(reference)
        except (AmbiguousJobReference, CronError, OSError):
            return None
        if job is None:
            return None
        canonical["job_id"] = str(job["id"])
        if action == "update":
            changes = {
                key: args.get(key)
                for key in (
                    "name",
                    "prompt",
                    "schedule",
                    "repeat",
                    "skills",
                    "enabled_toolsets",
                    "workdir",
                    "deliver",
                )
                if key in args
            }
            if not changes:
                return None
            detail = json.dumps(changes, ensure_ascii=False)
            summary = (
                f"Update cron job {job.get('name')!r} ({job['id']}):\n{detail}"
            )
        elif action == "pause":
            reason = str(args.get("reason") or "").strip()
            summary = f"Pause cron job {job.get('name')!r} ({job['id']})"
            if reason:
                summary += f": {reason}"
        else:
            summary = f"{action.title()} cron job {job.get('name')!r} ({job['id']})"

    if len(summary) > 1800:
        summary = summary[:1800] + "…"
    return canonical, summary


async def handle_cronjob(args: Dict[str, Any], context: ToolContext) -> str:
    prepared = _cron_approval_request(args, context)
    if prepared is None:
        return execute_cronjob(args, context)
    canonical, summary = prepared
    outcome = await context.authorize("cron", summary)
    if not outcome.approved:
        return tool_error(
            approval_error(outcome),
            success=False,
            approval=outcome.status,
        )
    return execute_cronjob(canonical, context)


CRONJOB_SCHEMA = {
    "name": "cronjob",
    "description": (
        "Manage this profile's scheduled AI jobs. Use create, list, update, "
        "pause, resume, remove, or run. Jobs execute in a fresh isolated "
        "conversation and normally deliver back to the messaging chat that "
        "created them. A declared WhatsApp or Telegram home channel may be used "
        "for unattended operator jobs; local jobs save output without sending. "
        "Prompts must be "
        "self-contained. Attached skills are loaded in order at run time. "
        "Only create, change, pause, resume, run, or remove a job when the "
        "current user explicitly requested that change. Always list before "
        "removing a job; never guess an ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(_ACTIONS),
                "description": "The management action.",
            },
            "job_id": {
                "type": "string",
                "description": "Job ID or exact unique name for non-create actions.",
            },
            "prompt": {
                "type": "string",
                "description": "Self-contained task for create or update.",
            },
            "schedule": {
                "type": "string",
                "description": "30m, every 2h, a five-field cron expression, or ISO time.",
            },
            "name": {"type": "string", "description": "Optional readable name."},
            "repeat": {
                "type": "integer",
                "description": "Optional run count; omit for the schedule default.",
            },
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered profile-local skills; [] clears them on update.",
            },
            "enabled_toolsets": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional tool-group allowlist for this job, such as "
                    '["web", "file"]. Omit for the profile defaults; [] clears '
                    "the restriction on update."
                ),
            },
            "workdir": {
                "type": "string",
                "description": (
                    "Optional existing absolute working directory. Its context "
                    "instructions are loaded and terminal/file/code tools start "
                    "there. An empty string clears it on update."
                ),
            },
            "deliver": {
                "type": "string",
                "enum": ["origin", "local", "whatsapp", "telegram"],
                "description": (
                    "Where to send output. origin is the creating chat; whatsapp "
                    "or telegram uses that configured home channel; local only "
                    "saves the output."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Optional operator-visible reason when pausing.",
            },
            "include_disabled": {
                "type": "boolean",
                "description": "Include paused and completed jobs when listing.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    },
}

CRONJOB_TOOL = Tool("cronjob", "cron", CRONJOB_SCHEMA, handle_cronjob, emoji="⏰")


__all__ = [
    "CRONJOB_SCHEMA",
    "CRONJOB_TOOL",
    "execute_cronjob",
    "handle_cronjob",
]
