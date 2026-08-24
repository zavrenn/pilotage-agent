"""Hermes-derived, profile-scoped cron job storage and lifecycle."""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
import re
import shutil
import socket
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

logger = logging.getLogger(__name__)

ONESHOT_GRACE_SECONDS = 120
DEFAULT_CLAIM_TTL_SECONDS = 1800
DEFAULT_OUTPUT_RETENTION = 50
MAX_PROMPT_CHARS = 50_000
MAX_NAME_CHARS = 100
_LOCK_TIMEOUT_SECONDS = 30.0
_SAFE_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_CRON_THREATS = (
    (re.compile(r"cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|id_rsa|id_ed25519)", re.I), "read_secrets"),
    (re.compile(r"authorized_keys", re.I), "ssh_backdoor"),
    (re.compile(r"/etc/sudoers|visudo", re.I), "sudoers_mod"),
    (re.compile(r"rm\s+-rf\s+/", re.I), "destructive_root_rm"),
)
_STORE_LOCKS: dict[str, threading.RLock] = {}
_STORE_LOCKS_GUARD = threading.Lock()


class CronError(RuntimeError):
    pass


class AmbiguousJobReference(LookupError):
    def __init__(self, reference: str, matches: List[Dict[str, Any]]):
        self.reference = reference
        self.matches = matches
        ids = ", ".join(str(job["id"]) for job in matches)
        super().__init__(f"Job name {reference!r} is ambiguous; matches: {ids}. Use an ID.")


def timezone_for_name(name: str = ""):
    written = str(name or "").strip()
    if not written:
        return datetime.now().astimezone().tzinfo or timezone.utc
    try:
        return ZoneInfo(written)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {written}") from exc


def _aware(value: datetime, tz) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=tz)
    return value.astimezone(tz)


def parse_duration(value: str) -> int:
    written = str(value or "").strip().lower()
    match = re.fullmatch(
        r"(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)",
        written,
    )
    if not match:
        raise ValueError(f"Invalid duration {written!r}. Use 30m, 2h, or 1d.")
    amount = int(match.group(1))
    return amount * {"m": 1, "h": 60, "d": 1440}[match.group(2)[0]]


def _croniter(expression: str, base: Optional[datetime] = None):
    try:
        from croniter import croniter
    except ImportError as exc:
        raise ValueError("Cron expressions require Pilotage's croniter dependency.") from exc
    try:
        return croniter(expression, base) if base is not None else croniter(expression)
    except Exception as exc:
        raise ValueError(f"Invalid cron expression {expression!r}: {exc}") from exc


def parse_schedule(
    schedule: str,
    *,
    now: Optional[datetime] = None,
    tz=None,
) -> Dict[str, Any]:
    """Parse Hermes delay, interval, five-field cron, and ISO syntax."""
    written = str(schedule or "").strip()
    if not written:
        raise ValueError("schedule cannot be empty")
    target_tz = tz or (now.tzinfo if now is not None and now.tzinfo else timezone.utc)
    current = _aware(now or datetime.now(target_tz), target_tz)
    lower = written.lower()

    if lower.startswith("every "):
        minutes = parse_duration(written[6:])
        if minutes <= 0:
            raise ValueError("Recurring intervals must be greater than zero.")
        return {"kind": "interval", "minutes": minutes, "display": f"every {minutes}m"}

    parts = written.split()
    if len(parts) == 5 and all(re.fullmatch(r"[\d*\-,/]+", part) for part in parts):
        _croniter(written, current)
        return {"kind": "cron", "expr": written, "display": written}

    if "T" in written or re.match(r"^\d{4}-\d{2}-\d{2}", written):
        try:
            run_at = datetime.fromisoformat(written.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid timestamp {written!r}: {exc}") from exc
        run_at = _aware(run_at, target_tz)
        return {
            "kind": "once",
            "run_at": run_at.isoformat(),
            "display": f"once at {run_at.strftime('%Y-%m-%d %H:%M')}",
        }

    try:
        minutes = parse_duration(written)
    except ValueError:
        minutes = -1
    if minutes >= 0:
        run_at = current + timedelta(minutes=minutes)
        return {"kind": "once", "run_at": run_at.isoformat(), "display": f"once in {written}"}
    raise ValueError(
        f"Invalid schedule {written!r}. Use 30m, every 2h, 0 9 * * *, or an ISO timestamp."
    )


def compute_next_run(
    schedule: Dict[str, Any],
    *,
    now: datetime,
    last_run_at: Optional[str] = None,
) -> Optional[str]:
    if not isinstance(schedule, dict):
        return None
    current = _aware(now, now.tzinfo or timezone.utc)
    kind = schedule.get("kind")

    if kind == "once":
        if last_run_at:
            return None
        try:
            run_at = _aware(datetime.fromisoformat(str(schedule["run_at"])), current.tzinfo)
        except (KeyError, TypeError, ValueError):
            return None
        if run_at < current - timedelta(seconds=ONESHOT_GRACE_SECONDS):
            return None
        return run_at.isoformat()

    if kind == "interval":
        try:
            minutes = int(schedule["minutes"])
        except (KeyError, TypeError, ValueError):
            return None
        if minutes <= 0:
            return None
        base = current
        if last_run_at:
            try:
                base = _aware(datetime.fromisoformat(last_run_at), current.tzinfo)
            except ValueError:
                pass
        return (base + timedelta(minutes=minutes)).isoformat()

    if kind == "cron":
        expression = str(schedule.get("expr") or "")
        if not expression:
            return None
        base = current
        if last_run_at:
            try:
                base = _aware(datetime.fromisoformat(last_run_at), current.tzinfo)
            except ValueError:
                pass
        next_run = _croniter(expression, base).get_next(datetime)
        return _aware(next_run, current.tzinfo).isoformat()
    return None


def validate_prompt(prompt: str, *, current_profile: str = "default") -> str:
    """Hermes' strict persisted-cron-prompt boundary."""
    # Lazy to keep cron storage independent of the tool registry import graph.
    from pilotage.tools.command_guard import find_embedded_self_lifecycle
    from pilotage.tools.threat_patterns import first_threat_message

    written = str(prompt or "").strip()
    if len(written) > MAX_PROMPT_CHARS:
        raise ValueError(f"Cron prompt exceeds {MAX_PROMPT_CHARS} characters.")
    threat = first_threat_message(written, scope="strict")
    if threat:
        raise ValueError(f"Blocked unsafe cron prompt: {threat}")
    lifecycle = find_embedded_self_lifecycle(
        written, current_profile=current_profile
    )
    if lifecycle:
        raise ValueError(f"Blocked unsafe cron prompt: {lifecycle.message}")
    for pattern, pattern_id in _CRON_THREATS:
        if pattern.search(written):
            raise ValueError(f"Blocked unsafe cron prompt: threat pattern {pattern_id!r}.")
    return written


def _normalize_skills(skills: Optional[Iterable[Any]]) -> List[str]:
    result: List[str] = []
    raw_skills = (skills,) if isinstance(skills, str) else (skills or ())
    for raw in raw_skills:
        name = str(raw or "").strip()
        if not name or name in result:
            continue
        if len(name) > 128:
            raise ValueError("Skill references cannot exceed 128 characters.")
        result.append(name)
    return result


def _normalize_enabled_toolsets(
    toolsets: Optional[Iterable[Any]],
) -> Optional[List[str]]:
    if toolsets is None:
        return None
    if isinstance(toolsets, (str, bytes)) or not isinstance(toolsets, Iterable):
        raise ValueError("enabled_toolsets must be a list of tool group names")
    result: List[str] = []
    for raw in toolsets:
        name = str(raw or "").strip()
        if not name or name in result:
            continue
        if len(name) > 128:
            raise ValueError("Tool group names cannot exceed 128 characters.")
        result.append(name)
    return result or None


def _normalize_workdir(workdir: Optional[str]) -> Optional[str]:
    """Normalize current Hermes' absolute, existing cron workdir contract."""

    if workdir is None:
        return None
    raw = str(workdir).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"Cron workdir must be an absolute path (got {raw!r})."
        )
    resolved = path.resolve(strict=False)
    if not resolved.exists():
        raise ValueError(f"Cron workdir does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Cron workdir is not a directory: {resolved}")
    return str(resolved)


def _normalize_origin(origin: Optional[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not isinstance(origin, dict):
        return None
    channel = str(origin.get("channel") or "").strip().lower()
    chat_id = str(origin.get("chat_id") or "").strip()
    if channel == "telegram" and chat_id:
        normalized = {"channel": channel, "chat_id": chat_id}
        thread_id = str(origin.get("thread_id") or "").strip()
        if thread_id:
            normalized["thread_id"] = thread_id
        return normalized
    return {"channel": channel, "chat_id": chat_id} if channel and chat_id else None


def _normalize_delivery(value: Any, origin: Optional[Dict[str, str]]) -> str:
    """Normalize the current-Hermes delivery targets Pilotage actually uses."""

    if value is None or not str(value).strip():
        return "origin" if origin else "local"
    written = str(value).strip().lower()
    if written not in {"origin", "local", "whatsapp", "telegram"}:
        raise ValueError(
            "Cron delivery must be origin, local, whatsapp, or telegram."
        )
    return written


def _normalize_stored_delivery(
    value: Any,
    origin: Optional[Dict[str, str]],
    *,
    job_id: str,
) -> str:
    """Read old or hand-edited delivery data without breaking startup."""

    if isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value if str(part).strip()]
        value = parts[0] if len(parts) == 1 else value
    try:
        return _normalize_delivery(value, origin)
    except (TypeError, ValueError):
        # Delivery corruption must fail safe: keep the job readable but do not
        # route its output to any external chat until an operator fixes it.
        logger.warning(
            "Cron job %s has invalid delivery data; using local delivery",
            job_id,
        )
        return "local"


def _normalize_repeat_count(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("repeat must be a positive whole number")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("repeat must be a positive whole number")
    try:
        count = int(value)
    except (TypeError, ValueError):
        raise ValueError("repeat must be a positive whole number") from None
    if count <= 0:
        raise ValueError("repeat must be a positive whole number")
    return count


def _atomic_write(path: Path, text: str) -> None:
    """Hermes' tempfile, fsync, replace pattern."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


class CronStore:
    """One profile's durable job database."""

    def __init__(
        self,
        state_dir: Path,
        *,
        timezone_name: str = "",
        now: Optional[Callable[[], datetime]] = None,
        claim_ttl_seconds: float = DEFAULT_CLAIM_TTL_SECONDS,
        output_retention: int = DEFAULT_OUTPUT_RETENTION,
    ):
        self.state_dir = Path(state_dir).expanduser().resolve(strict=False)
        self.cron_dir = self.state_dir / "cron"
        self.jobs_path = self.cron_dir / "jobs.json"
        self.output_dir = self.cron_dir / "output"
        self.lock_path = self.cron_dir / ".jobs.lock"
        self.timezone = timezone_for_name(timezone_name)
        self._clock = now
        self.claim_ttl_seconds = max(1.0, float(claim_ttl_seconds))
        self.output_retention = int(output_retention)
        with _STORE_LOCKS_GUARD:
            self._thread_lock = _STORE_LOCKS.setdefault(
                str(self.cron_dir), threading.RLock()
            )

    def now(self) -> datetime:
        current = self._clock() if self._clock is not None else datetime.now(self.timezone)
        return _aware(current, self.timezone)

    def _assert_local_path(self, path: Path, label: str) -> None:
        if path.is_symlink():
            raise CronError(f"Cron {label} cannot be a symbolic link: {path}")
        try:
            path.resolve(strict=False).relative_to(self.state_dir)
        except ValueError as exc:
            raise CronError(
                f"Cron {label} escaped its profile directory: {path}"
            ) from exc

    def ensure_dirs(self) -> None:
        for directory in (self.cron_dir, self.output_dir):
            self._assert_local_path(directory, "directory")
            directory.mkdir(parents=True, exist_ok=True)
            self._assert_local_path(directory, "directory")
            try:
                os.chmod(directory, 0o700)
            except OSError:
                pass
        for path, label in (
            (self.jobs_path, "database"),
            (self.lock_path, "lock file"),
        ):
            self._assert_local_path(path, label)

    @contextlib.contextmanager
    def _locked(self):
        """Bounded Hermes advisory locking, but fail closed instead of racing."""
        self.ensure_dirs()
        with self._thread_lock:
            lock_file = open(self.lock_path, "a+b")
            acquired = False
            try:
                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
                while not acquired:
                    try:
                        lock_file.seek(0)
                        if fcntl is not None:
                            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        elif msvcrt is not None:
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                        acquired = True
                    except (OSError, IOError) as exc:
                        if time.monotonic() >= deadline:
                            raise CronError(
                                f"Timed out waiting for cron store lock {self.lock_path}"
                            ) from exc
                        time.sleep(0.05)
                yield
            finally:
                if acquired:
                    try:
                        lock_file.seek(0)
                        if fcntl is not None:
                            fcntl.flock(lock_file, fcntl.LOCK_UN)
                        elif msvcrt is not None:
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                    except (OSError, IOError):
                        logger.warning("Could not release cron lock %s", self.lock_path)
                lock_file.close()

    def _load_unlocked(self) -> List[Dict[str, Any]]:
        if not self.jobs_path.exists():
            return []
        try:
            raw = self.jobs_path.read_text(encoding="utf-8-sig")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = json.loads(raw, strict=False)
        except (OSError, ValueError) as exc:
            raise CronError(f"Cron database is unreadable: {exc}") from exc
        jobs = payload.get("jobs") if isinstance(payload, dict) else payload
        if not isinstance(jobs, list):
            raise CronError("Cron database must contain a jobs list.")
        result = []
        seen_ids: set[str] = set()
        for index, job in enumerate(jobs):
            job_id = str(job.get("id") or "") if isinstance(job, dict) else ""
            if not _SAFE_JOB_ID.fullmatch(job_id):
                raise CronError(f"Cron database job {index} has an invalid ID.")
            if job_id in seen_ids:
                raise CronError(f"Cron database contains duplicate job ID {job_id}.")
            seen_ids.add(job_id)
            normalized = copy.deepcopy(job)
            if "skills" not in normalized and normalized.get("skill"):
                normalized["skills"] = [normalized["skill"]]
            normalized["skills"] = _normalize_skills(normalized.get("skills"))
            normalized["skill"] = (
                normalized["skills"][0] if normalized["skills"] else None
            )
            normalized["enabled_toolsets"] = _normalize_enabled_toolsets(
                normalized.get("enabled_toolsets")
            )
            raw_workdir = str(normalized.get("workdir") or "").strip()
            normalized["workdir"] = raw_workdir or None
            normalized["origin"] = _normalize_origin(normalized.get("origin"))
            normalized["deliver"] = _normalize_stored_delivery(
                normalized.get("deliver"),
                normalized["origin"],
                job_id=job_id,
            )
            result.append(normalized)
        return result

    def _save_unlocked(self, jobs: List[Dict[str, Any]]) -> None:
        payload = {"jobs": jobs, "updated_at": self.now().isoformat()}
        _atomic_write(
            self.jobs_path,
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        )

    def load_jobs(self) -> List[Dict[str, Any]]:
        with self._locked():
            return copy.deepcopy(self._load_unlocked())

    @staticmethod
    def _resolve_index(jobs: List[Dict[str, Any]], reference: str) -> Optional[int]:
        written = str(reference or "").strip()
        if not written:
            return None
        for index, job in enumerate(jobs):
            if job["id"] == written:
                return index
        matches = [
            index
            for index, job in enumerate(jobs)
            if str(job.get("name") or "").casefold() == written.casefold()
        ]
        if len(matches) > 1:
            raise AmbiguousJobReference(written, [jobs[index] for index in matches])
        return matches[0] if matches else None

    def resolve_job(self, reference: str) -> Optional[Dict[str, Any]]:
        with self._locked():
            jobs = self._load_unlocked()
            index = self._resolve_index(jobs, reference)
            return copy.deepcopy(jobs[index]) if index is not None else None

    def list_jobs(self, *, include_disabled: bool = False) -> List[Dict[str, Any]]:
        jobs = self.load_jobs()
        return jobs if include_disabled else [
            job for job in jobs if job.get("enabled", True)
        ]

    def create_job(
        self,
        *,
        prompt: str,
        schedule: str,
        name: str = "",
        repeat: Optional[int] = None,
        skills: Optional[Iterable[Any]] = None,
        enabled_toolsets: Optional[Iterable[Any]] = None,
        workdir: Optional[str] = None,
        origin: Optional[Dict[str, Any]] = None,
        deliver: Optional[str] = None,
    ) -> Dict[str, Any]:
        from pilotage.tools.command_guard import profile_name_for_state_dir

        prompt_text = validate_prompt(
            prompt,
            current_profile=profile_name_for_state_dir(self.state_dir),
        )
        skill_names = _normalize_skills(skills)
        normalized_toolsets = _normalize_enabled_toolsets(enabled_toolsets)
        normalized_workdir = _normalize_workdir(workdir)
        if not prompt_text and not skill_names:
            raise ValueError("A cron job requires a prompt or at least one skill.")
        parsed = parse_schedule(schedule, now=self.now(), tz=self.timezone)
        repeat = _normalize_repeat_count(repeat)
        if parsed["kind"] == "once":
            repeat = 1
        next_run = compute_next_run(parsed, now=self.now())
        if parsed["kind"] == "once" and next_run is None:
            raise ValueError(
                f"One-shot time is more than {ONESHOT_GRACE_SECONDS}s in the past."
            )
        normalized_origin = _normalize_origin(origin)
        normalized_delivery = _normalize_delivery(deliver, normalized_origin)
        label = str(name or "").strip() or prompt_text[:50].strip()
        if not label and skill_names:
            label = skill_names[0]
        job = {
            "id": uuid.uuid4().hex[:12],
            "name": label[:MAX_NAME_CHARS] or "cron job",
            "prompt": prompt_text,
            "skills": skill_names,
            "skill": skill_names[0] if skill_names else None,
            "enabled_toolsets": normalized_toolsets,
            "workdir": normalized_workdir,
            "schedule": parsed,
            "schedule_display": parsed["display"],
            "repeat": {"times": repeat, "completed": 0},
            "enabled": True,
            "state": "scheduled",
            "paused_at": None,
            "paused_reason": None,
            "created_at": self.now().isoformat(),
            "next_run_at": next_run,
            "last_run_at": None,
            "last_status": None,
            "last_error": None,
            "last_delivery_error": None,
            "claim": None,
            "deliver": normalized_delivery,
            "origin": normalized_origin,
        }
        with self._locked():
            jobs = self._load_unlocked()
            existing_ids = {existing["id"] for existing in jobs}
            while job["id"] in existing_ids:
                job["id"] = uuid.uuid4().hex[:12]
            jobs.append(job)
            self._save_unlocked(jobs)
        return copy.deepcopy(job)

    def update_job(
        self, reference: str, updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        allowed = {
            "name",
            "prompt",
            "schedule",
            "skills",
            "repeat",
            "enabled_toolsets",
            "workdir",
            "deliver",
        }
        unknown = set(updates) - allowed
        if unknown:
            raise ValueError(
                f"Cron fields cannot be updated: {', '.join(sorted(unknown))}"
            )
        with self._locked():
            jobs = self._load_unlocked()
            index = self._resolve_index(jobs, reference)
            if index is None:
                return None
            job = jobs[index]
            if job.get("claim") and {"schedule", "repeat"}.intersection(updates):
                raise CronError(
                    "A running job's schedule or repeat count cannot be changed."
                )
            if "name" in updates:
                name = str(updates["name"] or "").strip()
                if not name:
                    raise ValueError("name cannot be empty")
                job["name"] = name[:MAX_NAME_CHARS]
            if "prompt" in updates:
                from pilotage.tools.command_guard import profile_name_for_state_dir

                job["prompt"] = validate_prompt(
                    updates["prompt"],
                    current_profile=profile_name_for_state_dir(self.state_dir),
                )
            if "skills" in updates:
                job["skills"] = _normalize_skills(updates["skills"])
                job["skill"] = job["skills"][0] if job["skills"] else None
            if "enabled_toolsets" in updates:
                job["enabled_toolsets"] = _normalize_enabled_toolsets(
                    updates["enabled_toolsets"]
                )
            if "workdir" in updates:
                job["workdir"] = _normalize_workdir(updates["workdir"])
            if "deliver" in updates:
                job["deliver"] = _normalize_delivery(
                    updates["deliver"], _normalize_origin(job.get("origin"))
                )
            if not job.get("prompt") and not job.get("skills"):
                raise ValueError("A cron job requires a prompt or at least one skill.")
            if "repeat" in updates:
                repeat = _normalize_repeat_count(updates["repeat"])
                if job.get("schedule", {}).get("kind") == "once":
                    repeat = 1
                job.setdefault("repeat", {})["times"] = repeat
            if "schedule" in updates:
                previous_kind = job.get("schedule", {}).get("kind")
                was_terminal = job.get("state") in {"completed", "error"}
                parsed = parse_schedule(
                    str(updates["schedule"]), now=self.now(), tz=self.timezone
                )
                job["schedule"] = parsed
                job["schedule_display"] = parsed["display"]
                repeat_state = job.setdefault(
                    "repeat", {"times": None, "completed": 0}
                )
                if parsed["kind"] == "once":
                    repeat_state["times"] = 1
                elif previous_kind == "once" and "repeat" not in updates:
                    repeat_state["times"] = None
                if parsed["kind"] != previous_kind or was_terminal:
                    repeat_state["completed"] = 0
                if job.get("state") != "paused":
                    next_run = compute_next_run(parsed, now=self.now())
                    if parsed["kind"] == "once" and next_run is None:
                        raise ValueError(
                            f"One-shot time is more than {ONESHOT_GRACE_SECONDS}s in the past."
                        )
                    job["next_run_at"] = next_run
                    job["enabled"] = True
                    job["state"] = "scheduled"
            self._save_unlocked(jobs)
            return copy.deepcopy(job)

    def pause_job(self, reference: str, reason: str = "") -> Optional[Dict[str, Any]]:
        with self._locked():
            jobs = self._load_unlocked()
            index = self._resolve_index(jobs, reference)
            if index is None:
                return None
            job = jobs[index]
            job["enabled"] = False
            job["state"] = "paused"
            job["paused_at"] = self.now().isoformat()
            job["paused_reason"] = str(reason or "").strip() or None
            self._save_unlocked(jobs)
            return copy.deepcopy(job)

    def resume_job(self, reference: str) -> Optional[Dict[str, Any]]:
        with self._locked():
            jobs = self._load_unlocked()
            index = self._resolve_index(jobs, reference)
            if index is None:
                return None
            job = jobs[index]
            if job.get("state") != "paused":
                raise CronError("Only a paused job can be resumed.")
            next_run = compute_next_run(job["schedule"], now=self.now())
            if job.get("schedule", {}).get("kind") == "once" and next_run is None:
                raise ValueError("This one-shot time has passed; create a new job.")
            job["enabled"] = True
            job["paused_at"] = None
            job["paused_reason"] = None
            if job.get("claim"):
                job["state"] = "running"
            else:
                job["state"] = "scheduled"
                job["next_run_at"] = next_run
            self._save_unlocked(jobs)
            return copy.deepcopy(job)

    def trigger_job(self, reference: str) -> Optional[Dict[str, Any]]:
        with self._locked():
            jobs = self._load_unlocked()
            index = self._resolve_index(jobs, reference)
            if index is None:
                return None
            job = jobs[index]
            if not job.get("enabled", True) or job.get("state") == "paused":
                raise CronError(
                    "The job is paused or disabled; resume it before running."
                )
            if job.get("claim") or job.get("state") == "running":
                raise CronError("The job is already running.")
            if job.get("state") == "completed":
                raise CronError("A completed one-shot cannot run again; recreate it.")
            job["state"] = "scheduled"
            job["next_run_at"] = self.now().isoformat()
            self._save_unlocked(jobs)
            return copy.deepcopy(job)

    def _job_output_dir(self, job_id: str, *, create: bool = False) -> Path:
        if not _SAFE_JOB_ID.fullmatch(str(job_id or "")):
            raise CronError("Invalid cron job ID for output path.")
        self.ensure_dirs()
        directory = self.output_dir / job_id
        self._assert_local_path(directory, "job output directory")
        if create:
            directory.mkdir(parents=True, exist_ok=True)
            self._assert_local_path(directory, "job output directory")
        return directory

    def remove_job(self, reference: str) -> bool:
        with self._locked():
            jobs = self._load_unlocked()
            index = self._resolve_index(jobs, reference)
            if index is None:
                return False
            job_id = jobs[index]["id"]
            output = self._job_output_dir(job_id)
            jobs.pop(index)
            self._save_unlocked(jobs)
        if output.exists():
            shutil.rmtree(output)
        return True

    @staticmethod
    def _claim_age(job: Dict[str, Any], now: datetime) -> Optional[float]:
        claim = job.get("claim")
        if not isinstance(claim, dict):
            return None
        try:
            claimed = _aware(datetime.fromisoformat(str(claim["at"])), now.tzinfo)
        except (KeyError, TypeError, ValueError):
            return float("inf")
        age = (now - claimed).total_seconds()
        return age if age >= 0 else float("inf")

    def claim_due_jobs(self, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Claim due jobs before any model or delivery side effect."""
        maximum = None if limit is None else max(0, int(limit))
        with self._locked():
            jobs = self._load_unlocked()
            now = self.now()
            due: List[Dict[str, Any]] = []
            changed = False
            owner_base = f"{socket.gethostname()}:{os.getpid()}"

            for job in jobs:
                if maximum is not None and len(due) >= maximum:
                    break
                if job.get("state") == "running" or job.get("claim"):
                    age = self._claim_age(job, now)
                    if age is None or age < self.claim_ttl_seconds:
                        continue
                    if job.get("schedule", {}).get("kind") == "once":
                        job["enabled"] = False
                        job["state"] = "error"
                        job["next_run_at"] = None
                        job["last_status"] = "error"
                        job["last_error"] = (
                            "Previous claimed run ended without completion."
                        )
                        job["claim"] = None
                        changed = True
                        continue
                    job["state"] = "scheduled"
                    job["claim"] = None
                    changed = True

                if not job.get("enabled", True) or job.get("state") != "scheduled":
                    continue
                if job.get("paused_at") or job.get("paused_reason"):
                    job["enabled"] = False
                    job["state"] = "paused"
                    changed = True
                    continue

                repeat = job.get("repeat") or {}
                times = repeat.get("times")
                completed = int(repeat.get("completed") or 0)
                if times is not None and completed >= int(times):
                    job["enabled"] = False
                    job["state"] = "completed"
                    job["next_run_at"] = None
                    changed = True
                    continue

                next_run_text = job.get("next_run_at")
                if not next_run_text:
                    next_run_text = compute_next_run(
                        job.get("schedule", {}),
                        now=now,
                        last_run_at=job.get("last_run_at"),
                    )
                    if not next_run_text:
                        continue
                    job["next_run_at"] = next_run_text
                    changed = True
                try:
                    next_run = _aware(
                        datetime.fromisoformat(str(next_run_text)), now.tzinfo
                    )
                except ValueError:
                    job["enabled"] = False
                    job["state"] = "error"
                    job["last_error"] = "Invalid next_run_at timestamp."
                    changed = True
                    continue
                if next_run > now:
                    continue

                kind = job.get("schedule", {}).get("kind")
                if kind in {"interval", "cron"}:
                    job["next_run_at"] = compute_next_run(
                        job["schedule"],
                        now=now,
                        last_run_at=now.isoformat(),
                    )
                elif kind == "once":
                    repeat["completed"] = completed + 1
                    job["repeat"] = repeat
                else:
                    job["enabled"] = False
                    job["state"] = "error"
                    job["last_error"] = "Invalid schedule kind."
                    changed = True
                    continue

                owner = f"{owner_base}:{uuid.uuid4().hex}"
                job["claim"] = {"at": now.isoformat(), "by": owner}
                job["state"] = "running"
                due.append(copy.deepcopy(job))
                changed = True

            if changed:
                self._save_unlocked(jobs)
            return due

    def renew_claim(self, job_id: str, *, owner: str) -> bool:
        """Keep a live run from being reclaimed while its model is working."""
        with self._locked():
            jobs = self._load_unlocked()
            index = self._resolve_index(jobs, job_id)
            if index is None:
                return False
            job = jobs[index]
            claim = job.get("claim")
            if not isinstance(claim, dict) or claim.get("by") != owner:
                return False
            claim["at"] = self.now().isoformat()
            job["claim"] = claim
            self._save_unlocked(jobs)
            return True

    def finish_job(
        self,
        job_id: str,
        *,
        owner: str,
        success: bool,
        error: str = "",
        delivery_error: str = "",
    ) -> bool:
        """Fence one run's completion against its durable claim."""
        with self._locked():
            jobs = self._load_unlocked()
            index = self._resolve_index(jobs, job_id)
            if index is None:
                return False
            job = jobs[index]
            claim = job.get("claim")
            if not isinstance(claim, dict) or claim.get("by") != owner:
                logger.warning("Discarding stale completion for cron job %s", job_id)
                return False

            now = self.now()
            job["last_run_at"] = now.isoformat()
            job["last_status"] = "ok" if success else "error"
            job["last_error"] = None if success else str(error or "Cron run failed.")
            job["last_delivery_error"] = str(delivery_error or "") or None
            job["claim"] = None
            kind = job.get("schedule", {}).get("kind")

            if kind == "once":
                job["enabled"] = False
                job["state"] = "completed" if success else "error"
                job["next_run_at"] = None
            else:
                repeat = job.setdefault("repeat", {"times": None, "completed": 0})
                repeat["completed"] = int(repeat.get("completed") or 0) + 1
                times = repeat.get("times")
                if times is not None and repeat["completed"] >= int(times):
                    job["enabled"] = False
                    job["state"] = "completed"
                    job["next_run_at"] = None
                elif not job.get("enabled", True) or job.get("state") == "paused":
                    job["enabled"] = False
                    job["state"] = "paused"
                else:
                    job["state"] = "scheduled"
                    job["next_run_at"] = compute_next_run(
                        job["schedule"],
                        now=now,
                        last_run_at=now.isoformat(),
                    )
            self._save_unlocked(jobs)
            return True

    def _save_output_file(self, job_id: str, output: str) -> Path:
        directory = self._job_output_dir(job_id, create=True)
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        path = directory / f"{self.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}.md"
        _atomic_write(path, str(output))
        if self.output_retention > 0:
            files = sorted(directory.glob("*.md"), reverse=True)
            for stale in files[self.output_retention :]:
                stale.unlink(missing_ok=True)
        return path

    def save_output(
        self,
        job_id: str,
        output: str,
        *,
        owner: Optional[str] = None,
    ) -> Optional[Path]:
        """Save output, optionally fenced to the caller's live durable claim."""
        if owner is None:
            return self._save_output_file(job_id, output)
        with self._locked():
            jobs = self._load_unlocked()
            index = self._resolve_index(jobs, job_id)
            if index is None:
                return None
            claim = jobs[index].get("claim")
            if not isinstance(claim, dict) or claim.get("by") != owner:
                logger.warning("Discarding stale output for cron job %s", job_id)
                return None
            return self._save_output_file(job_id, output)

    def latest_output(self, job_id: str) -> Optional[str]:
        if not _SAFE_JOB_ID.fullmatch(str(job_id or "")):
            return None
        directory = self._job_output_dir(job_id)
        files = sorted(
            path for path in directory.glob("*.md") if not path.is_symlink()
        )
        if not files:
            return None
        try:
            return files[-1].read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
