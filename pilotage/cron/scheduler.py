"""Small in-process scheduler over the durable Hermes-derived job store."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

from pilotage.agent import Agent
from pilotage.config import Config
from pilotage.history import ConversationStore

from .jobs import (
    CronStore,
    _register_active_claim_owner,
    _unregister_active_claim_owner,
    validate_prompt,
)

logger = logging.getLogger(__name__)

MAX_ASSEMBLED_PROMPT_CHARS = 200_000
MAX_SAVED_OUTPUT_CHARS = 200_000
MAX_ERROR_CHARS = 2_000
SILENT_RESPONSE = "[SILENT]"

_ASSEMBLED_THREATS = (
    (re.compile(r"ignore\s+(?:\w+\s+){0,8}(?:previous|all|above|prior)\s+(?:\w+\s+){0,8}instructions", re.I), "prompt_injection"),
    (re.compile(r"do\s+not\s+tell\s+the\s+user", re.I), "deception_hide"),
    (re.compile(r"system\s+prompt\s+override", re.I), "sys_prompt_override"),
    (re.compile(r"disregard\s+(?:your|all|any)\s+(?:instructions|rules|guidelines)", re.I), "disregard_rules"),
)
_EMOJI_RANGES = (
    (0x1F000, 0x1FFFF),
    (0x2600, 0x27BF),
    (0x2300, 0x23FF),
    (0x1F1E6, 0x1F1FF),
    (0x20E3, 0x20E3),
)

Delivery = Callable[[Dict[str, str], str, str], Awaitable[None]]
AgentFactory = Callable[[Config], Agent]


class CronExecutionError(RuntimeError):
    pass


def _is_emoji(codepoint: int) -> bool:
    return any(low <= codepoint <= high for low, high in _EMOJI_RANGES)


def _emoji_joiner(text: str, index: int) -> bool:
    left = index - 1
    while left >= 0 and ord(text[left]) == 0xFE0F:
        left -= 1
    right = index + 1
    while right < len(text) and ord(text[right]) == 0xFE0F:
        right += 1
    return (
        left >= 0
        and right < len(text)
        and _is_emoji(ord(text[left]))
        and _is_emoji(ord(text[right]))
    )


def _scan_assembled_skill_prompt(text: str) -> str:
    """Hermes' looser tripwire for already trusted skill markdown."""
    from pilotage.tools.threat_patterns import INVISIBLE_CHARS

    removed: set[str] = set()
    cleaned = []
    for index, char in enumerate(text):
        if char in INVISIBLE_CHARS:
            if char == "\u200d" and _emoji_joiner(text, index):
                cleaned.append(char)
            else:
                removed.add(f"U+{ord(char):04X}")
            continue
        cleaned.append(char)
    result = "".join(cleaned)
    if removed:
        logger.warning(
            "Cron skill prompt: stripped invisible unicode %s",
            ", ".join(sorted(removed)),
        )
    normalized = unicodedata.normalize("NFKC", result)
    for pattern, pattern_id in _ASSEMBLED_THREATS:
        if pattern.search(normalized):
            raise CronExecutionError(
                f"Attached skill content matched threat pattern {pattern_id!r}."
            )
    return result


def build_job_prompt(config: Config, job: Dict[str, Any], session_id: str) -> str:
    """Load attached skills in order, then append the autonomous task."""
    prompt = validate_prompt(str(job.get("prompt") or ""))
    skills = [str(name).strip() for name in job.get("skills") or () if str(name).strip()]
    parts = []
    if skills:
        from pilotage.tools.skills import skill_view

        for skill_name in skills:
            try:
                loaded = json.loads(
                    skill_view(config, skill_name, session_id=session_id)
                )
            except (json.JSONDecodeError, TypeError) as exc:
                raise CronExecutionError(
                    f"Attached skill {skill_name!r} returned invalid data."
                ) from exc
            if not loaded.get("success"):
                reason = loaded.get("error") or "skill is unavailable"
                raise CronExecutionError(
                    f"Attached skill {skill_name!r} could not be loaded: {reason}"
                )
            parts.extend(
                [
                    f'[IMPORTANT: Follow the attached "{skill_name}" skill. Its full content follows.]',
                    "",
                    str(loaded.get("content") or "").strip(),
                    "",
                ]
            )

    parts.extend(
        [
            "[IMPORTANT: You are running as a scheduled cron job. Your final "
            "response is delivered automatically; do not try to send it yourself. "
            "If there is genuinely nothing to report, respond with exactly "
            '"[SILENT]" and nothing else.]',
            "",
            "## Scheduled task",
            prompt,
        ]
    )
    assembled = "\n".join(parts).strip()
    if len(assembled) > MAX_ASSEMBLED_PROMPT_CHARS:
        raise CronExecutionError(
            f"Assembled cron prompt exceeds {MAX_ASSEMBLED_PROMPT_CHARS} characters."
        )
    return _scan_assembled_skill_prompt(assembled) if skills else assembled


def _bounded(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "\n\n[... truncated ...]"


class CronScheduler:
    """Claim, execute, save, and deliver one profile's jobs."""

    def __init__(
        self,
        config: Config,
        store: CronStore,
        *,
        deliver: Optional[Delivery] = None,
        agent_factory: Optional[AgentFactory] = None,
        channel_configs: Optional[Mapping[str, Config]] = None,
    ):
        self.config = config
        self.store = store
        self._deliver = deliver
        self._agent_factory = agent_factory
        self._channel_configs = {
            str(channel).lower(): channel_config
            for channel, channel_config in (channel_configs or {}).items()
        }
        self._wake = asyncio.Event()
        self._loop_task: Optional[asyncio.Task] = None
        self._active: set[asyncio.Task] = set()
        self._running = False
        self.stopped = asyncio.Event()
        self.failure: Optional[str] = None

    def _fresh_agent(self, config: Config, job: Dict[str, Any]) -> Agent:
        raw_workdir = str(job.get("workdir") or "").strip()
        workdir = Path(raw_workdir) if raw_workdir else None
        if workdir is not None and not workdir.is_dir():
            raise CronExecutionError(
                f"Configured cron workdir no longer exists: {workdir}"
            )
        return Agent(
            config,
            ConversationStore(path=None),
            disabled_tool_groups=("cron",),
            enabled_tool_groups=job.get("enabled_toolsets"),
            enabled_skills=job.get("skills") or (),
            working_directory=workdir,
            allow_persistence_writes=False,
            scheduled_run=True,
        )

    def _agent_for_job(self, config: Config, job: Dict[str, Any]) -> Agent:
        if self._agent_factory is not None:
            agent = self._agent_factory(config)
            if bool(getattr(agent, "_allow_persistence_writes", False)):
                raise CronExecutionError(
                    "Scheduled agent factory returned a foreground-write-capable Agent."
                )
            if getattr(agent, "_scheduled_run", None) is not True:
                raise CronExecutionError(
                    "Scheduled agent factory returned an Agent without the "
                    "scheduled read-only instruction boundary."
                )
            return agent
        return self._fresh_agent(config, job)

    @staticmethod
    async def _close_agent(agent: Any) -> None:
        close = getattr(agent, "close", None)
        if not callable(close):
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("Closing a cron Agent failed", exc_info=True)

    def _config_for_job(self, job: Dict[str, Any]) -> Config:
        origin = self._delivery_origin(job)
        if isinstance(origin, dict):
            channel = str(origin.get("channel") or "").lower()
            if channel in self._channel_configs:
                return self._channel_configs[channel]
        return self.config

    def _home_origin(self, channel: str) -> Optional[Dict[str, str]]:
        config = self._channel_configs.get(channel)
        if config is None and str(getattr(self.config, "channel", "whatsapp")) == channel:
            config = self.config
        origin = getattr(config, "home_origin", None) if config is not None else None
        return dict(origin) if isinstance(origin, dict) else None

    def _delivery_origin(self, job: Dict[str, Any]) -> Optional[Dict[str, str]]:
        deliver = str(job.get("deliver") or "local").strip().lower()
        origin = job.get("origin")
        if deliver == "local":
            return None
        if deliver == "origin" and isinstance(origin, dict):
            channel = str(origin.get("channel") or "").strip().lower()
            chat_id = str(origin.get("chat_id") or "").strip()
            if channel in {"whatsapp", "telegram"} and chat_id:
                resolved = {"channel": channel, "chat_id": chat_id}
                thread_id = str(origin.get("thread_id") or "").strip()
                if channel == "telegram" and thread_id:
                    resolved["thread_id"] = thread_id
                return resolved
        if deliver in {"whatsapp", "telegram"}:
            return self._home_origin(deliver)
        if deliver == "origin":
            # Current Hermes falls back to the first configured home when a
            # CLI/API-created job has no live messaging origin.
            for channel in ("whatsapp", "telegram"):
                if home := self._home_origin(channel):
                    return home
        return None

    def wake(self) -> None:
        self._wake.set()

    async def start(self) -> None:
        if self._running:
            return
        await asyncio.to_thread(self.store.load_jobs)
        self.failure = None
        self.stopped.clear()
        self._running = True
        self._loop_task = asyncio.create_task(
            self._run_loop(), name="pilotage-cron-scheduler"
        )
        logger.info("Cron scheduler ready")

    async def stop(self, *, drain_timeout_seconds: float = 0.0) -> None:
        """Stop claiming jobs, then drain already claimed work for a bound."""

        self._running = False
        self._wake.set()
        if self._loop_task is not None:
            await asyncio.gather(self._loop_task, return_exceptions=True)
            self._loop_task = None
        active = list(self._active)
        live = {task for task in active if not task.done()}
        pending = live
        timeout = max(0.0, float(drain_timeout_seconds))
        if live and timeout > 0:
            _, pending = await asyncio.wait(live, timeout=timeout)
        if pending:
            logger.warning(
                "Cron shutdown drain expired with %d in-flight job(s)",
                len(pending),
            )
        for task in pending:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        self._active.clear()
        self.stopped.set()

    async def _run_loop(self) -> None:
        try:
            while self._running:
                self._wake.clear()
                capacity = max(0, self.config.cron_max_concurrent - len(self._active))
                due = []
                if capacity:
                    due = await asyncio.to_thread(
                        self.store.claim_due_jobs, limit=capacity
                    )
                for job in due:
                    owner = str((job.get("claim") or {}).get("by") or "")
                    _register_active_claim_owner(owner)
                    try:
                        task = asyncio.create_task(
                            self._run_claimed(job),
                            name=f"pilotage-cron-{job['id']}",
                        )
                    except BaseException:
                        _unregister_active_claim_owner(owner)
                        raise
                    self._active.add(task)
                    task.add_done_callback(self._job_done)
                    task.add_done_callback(
                        lambda _task, claim_owner=owner: (
                            _unregister_active_claim_owner(claim_owner)
                        )
                    )
                if due:
                    continue
                try:
                    await asyncio.wait_for(
                        self._wake.wait(), timeout=self.config.cron_tick_seconds
                    )
                except asyncio.TimeoutError:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.failure = f"Cron scheduler failed: {type(exc).__name__}: {exc}"
            self._running = False
            logger.exception(self.failure)
        finally:
            self.stopped.set()

    def _job_done(self, task: asyncio.Task) -> None:
        self._active.discard(task)
        self._wake.set()
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("Cron worker ended unexpectedly: %s", error)

    async def _heartbeat(
        self,
        job_id: str,
        owner: str,
        parent: asyncio.Task,
        lost_claim: asyncio.Event,
    ) -> None:
        interval = max(0.1, min(self.store.claim_ttl_seconds / 3.0, 60.0))
        while True:
            await asyncio.sleep(interval)
            try:
                renewed = await asyncio.to_thread(
                    self.store.renew_claim, job_id, owner=owner
                )
            except Exception:
                logger.exception("Cron claim heartbeat failed for %s", job_id)
                renewed = False
            if not renewed:
                lost_claim.set()
                parent.cancel()
                return

    async def _deliver_text(self, job: Dict[str, Any], text: str) -> str:
        deliver = str(job.get("deliver") or "local").strip().lower()
        if deliver == "local" or text.strip() == SILENT_RESPONSE:
            return ""
        origin = self._delivery_origin(job)
        if not isinstance(origin, dict):
            if deliver == "origin":
                # Match Hermes: output remains saved when no origin or home has
                # been configured, instead of producing a false run failure.
                return ""
            return f"No configured {deliver.title()} home channel."
        channel = str(origin.get("channel") or "").lower()
        if channel not in {"whatsapp", "telegram"}:
            return "Cron delivery channel is unsupported by this runtime."
        if self._deliver is None:
            return "Messaging delivery adapter is unavailable."
        try:
            claim = job.get("claim") or {}
            owner = str(claim.get("by") or "") if isinstance(claim, dict) else ""
            if not owner:
                return "Cron delivery has no durable run reference."
            await self._deliver(origin, text, f"{job.get('id') or ''}:{owner}")
        except Exception as exc:
            logger.exception("Cron delivery failed for %s", job.get("id"))
            return _bounded(f"{type(exc).__name__}: {exc}", MAX_ERROR_CHARS)
        return ""

    async def _save_failure(
        self,
        job: Dict[str, Any],
        error: str,
        owner: str,
    ) -> Optional[bool]:
        artifact = f"# Cron run failed\n\n{error}\n"
        try:
            saved = await asyncio.to_thread(
                self.store.save_output,
                str(job["id"]),
                _bounded(artifact, MAX_SAVED_OUTPUT_CHARS),
                owner=owner,
            )
            return True if saved is not None else None
        except Exception:
            logger.exception("Could not save failure output for cron job %s", job.get("id"))
            return False

    async def _run_claimed(self, job: Dict[str, Any]) -> None:
        claim = job.get("claim") or {}
        owner = str(claim.get("by") or "")
        job_id = str(job.get("id") or "")
        if not owner:
            logger.error("Claimed cron job %s has no owner", job_id)
            return

        parent = asyncio.current_task()
        assert parent is not None
        lost_claim = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat(job_id, owner, parent, lost_claim),
            name=f"pilotage-cron-heartbeat-{job_id}",
        )

        async def retain_claim() -> bool:
            if lost_claim.is_set():
                return False
            try:
                renewed = await asyncio.to_thread(
                    self.store.renew_claim,
                    job_id,
                    owner=owner,
                )
            except Exception:
                logger.exception("Cron claim check failed for %s", job_id)
                renewed = False
            if not renewed:
                lost_claim.set()
            return renewed

        success = False
        cancelled = False
        error = ""
        delivery_error = ""
        try:
            session_id = f"cron:{job_id}:{owner.rsplit(':', 1)[-1]}"
            job_config = self._config_for_job(job)
            prompt = await asyncio.to_thread(
                build_job_prompt, job_config, job, session_id
            )
            agent = self._agent_for_job(job_config, job)
            try:
                respond_result = getattr(agent, "respond_result", None)
                if not callable(respond_result):
                    raise CronExecutionError(
                        "The scheduled agent cannot publish terminal completion proof."
                    )
                turn = await respond_result(session_id, prompt)
            finally:
                await self._close_agent(agent)
            if getattr(turn, "terminal_completed", None) is not True:
                raise CronExecutionError(
                    "The scheduled agent ended without positive terminal completion proof."
                )
            answer = str(getattr(turn, "text", "") or "")
            if not str(answer or "").strip():
                raise CronExecutionError("The scheduled agent returned an empty response.")
            if not await retain_claim():
                cancelled = True
                raise asyncio.CancelledError
            answer = _bounded(answer, MAX_SAVED_OUTPUT_CHARS)
            saved = await asyncio.to_thread(
                self.store.save_output,
                job_id,
                answer,
                owner=owner,
            )
            if saved is None or not await retain_claim():
                lost_claim.set()
                cancelled = True
                raise asyncio.CancelledError
            try:
                persisted = saved.read_text(encoding="utf-8")
            except OSError as exc:
                raise CronExecutionError(
                    "The scheduled output artifact could not be verified."
                ) from exc
            if persisted != answer:
                raise CronExecutionError(
                    "The scheduled output artifact failed verification."
                )
            success = True
            delivery_error = await self._deliver_text(job, answer)
        except asyncio.CancelledError:
            cancelled = True
            success = False
            error = (
                "Cron claim ownership was lost."
                if lost_claim.is_set()
                else "Cron run was cancelled during shutdown."
            )
            if not lost_claim.is_set():
                await self._save_failure(job, error, owner)
        except Exception as exc:
            error = _bounded(f"{type(exc).__name__}: {exc}", MAX_ERROR_CHARS)
            logger.exception("Cron job %s failed", job_id)
            failure_saved = await self._save_failure(job, error, owner)
            if failure_saved is None or not await retain_claim():
                lost_claim.set()
                cancelled = True
            else:
                public = (
                    f"Scheduled job {job.get('name') or job_id!r} failed. "
                    "Check the agent logs."
                )
                delivery_error = await self._deliver_text(job, public)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            try:
                completed = await asyncio.shield(
                    asyncio.to_thread(
                        self.store.finish_job,
                        job_id,
                        owner=owner,
                        success=success,
                        error=error,
                        delivery_error=delivery_error,
                    )
                )
                if not completed:
                    logger.warning("Cron completion lost its claim for %s", job_id)
            except Exception:
                logger.exception("Could not finalize cron job %s", job_id)
        if cancelled:
            raise asyncio.CancelledError


__all__ = [
    "CronExecutionError",
    "CronScheduler",
    "SILENT_RESPONSE",
    "build_job_prompt",
]
