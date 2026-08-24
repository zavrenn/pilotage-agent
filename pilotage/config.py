"""What this agent is and how it behaves.

Behaviour settings live in a configuration file the operator can diff against
another machine; profile identity lives in its small ``SOUL.md`` file. The
environment carries secrets, channel identities, and the few things that are
properties of the box rather than of the agent. Each concern has one canonical
home.

Missing files leave defaults in place, so an unconfigured agent still runs.
Broken configuration or an unreadable/unsafe identity stops startup instead of
silently running with a different contract.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from .approvals import DEFAULT_APPROVAL_TIMEOUT_SECONDS
from .codex.compaction import DEFAULT_COMPACT_THRESHOLD
from .i18n import DEFAULT_PROFILE_LANGUAGE, normalize_language, t
from .profiles import default_state_root
from .settings import ConfigError, Settings, config_path

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"

# Hermes's conservative context-file floor; profile identities should normally
# remain far smaller than this.
SOUL_FILENAME = "SOUL.md"
SOUL_MAX_CHARS = 20_000

DEFAULT_INSTRUCTIONS = (
    "You are a Pilotage agent. You answer over WhatsApp, so keep replies short "
    "and readable on a phone: a few sentences, not a document. Say what you "
    "know, say plainly when you do not know, and never invent facts."
)

TELEGRAM_DEFAULT_INSTRUCTIONS = (
    "You are a Pilotage agent. You answer over Telegram, so keep replies short "
    "and readable on a phone: a few sentences, not a document. Say what you "
    "know, say plainly when you do not know, and never invent facts."
)

# Added to whatever instructions the operator writes, always. The model emits
# markdown on its own and we translate it on the way out, so it has to know
# which marks survive the trip — including when the operator replaces the
# instructions above entirely and never thinks about formatting.
FORMATTING_NOTE = (
    "Write in ordinary markdown. It is converted to WhatsApp formatting before "
    "the message is sent: **bold**, *italic*, ~~strikethrough~~ and `code` all "
    "arrive as WhatsApp expects them. A heading becomes a bold line, so use "
    "headings sparingly, and bullet lists ('- item') freely. Tables do not "
    "render at all — use short lines or bullets instead."
)

WHATSAPP_MEDIA_NOTE = (
    "You can send generated files natively on WhatsApp. To deliver a local "
    "file, include MEDIA:/absolute/path/to/file on its own line in your "
    "response. Images (.jpg, .png, .webp) appear as photos; PDFs, spreadsheets "
    "and other files arrive as downloadable documents. The file must be inside "
    "this profile's workspace or an operator-declared delivery directory. "
    "Use MEDIA: for local files, never a markdown "
    "link or sandbox: URL."
)

TELEGRAM_FORMATTING_NOTE = (
    "Write in ordinary markdown. It is converted to Telegram MarkdownV2 before "
    "delivery, including headings, emphasis, links, code, block quotes and "
    "lists. Tables are converted into readable row groups for mobile."
)

TELEGRAM_MEDIA_NOTE = (
    "You can send generated files natively on Telegram. To deliver a local "
    "file, include MEDIA:/absolute/path/to/file on its own line in your "
    "response. The file must be inside this profile's workspace or an "
    "operator-declared delivery directory."
)

# How many times the model may call tools and look at the results before it has
# to answer with what it has. Hermes' number. It is a runaway guard, not a
# budget: real work finishes far inside it, and a loop that reaches it was
# going nowhere.
DEFAULT_MAX_TOOL_ITERATIONS = 90

# What one tool result may carry, and what one round of them may carry between
# them. Every result is replayed on every later request of the conversation, so
# these are the difference between an expensive turn and an expensive chat.
DEFAULT_MAX_RESULT_CHARS = 100_000
DEFAULT_MAX_STEP_CHARS = 200_000

# Hermes' bounded curated-memory defaults.
DEFAULT_MEMORY_CHAR_LIMIT = 2200
DEFAULT_USER_CHAR_LIMIT = 1375

# Hermes-derived scheduler guardrails. The tick is only a wake-up latency;
# durable claims, not timing luck, prevent duplicate execution.
DEFAULT_CRON_TICK_SECONDS = 1.0
DEFAULT_CRON_CLAIM_TTL_SECONDS = 1800.0
DEFAULT_CRON_MAX_CONCURRENT = 2
DEFAULT_CRON_OUTPUT_RETENTION = 50

# Production's maximum total quiet-window batch age.
DEFAULT_BATCH_HARD_CAP_SECONDS = 20.0
DEFAULT_SESSION_RETENTION_DAYS = 90
DEFAULT_WORKING_NOTICE_INTERVAL_SECONDS = 180.0

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def state_dir() -> Path:
    """Where credentials, the WhatsApp session and logs live."""
    override = os.environ.get("PILOTAGE_HOME", "").strip()
    root = Path(override).expanduser() if override else Path.home() / ".pilotage-agent"
    return root


def _load_soul(home: Path) -> str:
    """Load exactly this profile's optional Hermes-compatible identity file."""
    path = home / SOUL_FILENAME
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except (OSError, UnicodeError) as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc

    content = content.removeprefix("\ufeff").strip()
    if not content:
        return ""
    if len(content) > SOUL_MAX_CHARS:
        raise ConfigError(
            f"{path} exceeds the {SOUL_MAX_CHARS:,}-character identity limit"
        )

    from .tools.threat_patterns import scan_for_threats

    findings = scan_for_threats(content, scope="context")
    if findings:
        raise ConfigError(
            f"{path} contains potentially unsafe instructions: {', '.join(findings)}"
        )
    return content


def _instructions(settings: Settings, home: Path, channel: str = "") -> str:
    """Assemble profile identity, optional operator overlay, and channel rules."""
    written = settings.text("agent.instructions", "")
    soul = _load_soul(home)
    if channel == "telegram":
        parts = [soul or written or TELEGRAM_DEFAULT_INSTRUCTIONS]
        if soul and written:
            parts.append(written)
        parts.extend((TELEGRAM_FORMATTING_NOTE, TELEGRAM_MEDIA_NOTE))
        return "\n\n".join(parts)
    parts = [soul or written or DEFAULT_INSTRUCTIONS]
    if soul and written:
        parts.append(written)
    parts.append(FORMATTING_NOTE)
    if channel == "whatsapp":
        parts.append(WHATSAPP_MEDIA_NOTE)
    return "\n\n".join(parts)


def _count_in_range(
    name: str, value: int, *, minimum: int, maximum: int | None = None
) -> int:
    if value < minimum or (maximum is not None and value > maximum):
        expected = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise ConfigError(f"{name} must be {expected}, not {value!r}")
    return value


def _number_in_range(name: str, value: float, *, minimum: float, inclusive: bool) -> float:
    valid = value >= minimum if inclusive else value > minimum
    if not math.isfinite(value) or not valid:
        relation = "at least" if inclusive else "greater than"
        raise ConfigError(f"{name} must be {relation} {minimum:g}, not {value!r}")
    return value


def _path_is_within(path: Path, root: Path) -> bool:
    """Whether a resolved path is inside one resolved trusted root."""

    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


@dataclass(frozen=True)
class Config:
    model: str
    reasoning_effort: str
    instructions: str
    # Runtime-owned static messages.  The model's language and register remain
    # part of the profile's SOUL.md identity.
    language: str
    # One profile timezone, inherited by cron and automatic daily resets.
    timezone: str
    # Who the agent answers. Empty means nobody: an agent wired to a real phone
    # number must never answer whoever finds it. Entries are phone numbers in
    # digits, or full jids.
    allowed_senders: frozenset[str]
    # Additional location gate for allowed senders. "*" means any group,
    # never any sender.
    group_policy: str
    group_allow_from: frozenset[str]
    require_mention: bool
    bridge_port: int
    bridge_dir: Path
    state_dir: Path
    # Quiet period before a burst of inbound messages is treated as one turn.
    text_batch_delay_seconds: float
    text_batch_split_delay_seconds: float
    # Maximum total age of a pending inbound batch.
    text_batch_hard_cap_seconds: float
    # Turns of history kept per chat, in memory and on disk.
    history_turns: int
    # Current Hermes' messaging reset policy, profile-wide.
    session_reset_mode: str
    session_reset_idle_minutes: int
    session_reset_at_hour: int
    session_reset_notify: bool
    # Optional ended-session history cleanup. Active sessions are never pruned.
    session_auto_prune: bool
    session_retention_days: int
    # Route each conversation generation into its own restricted workspace.
    session_isolated_workspaces: bool
    # Hermes-native server compaction on the fixed ChatGPT Codex route.
    codex_native_compaction: bool
    codex_compact_threshold: int
    # Profile-wide curated notes injected as a frozen per-session snapshot.
    memory_char_limit: int
    user_memory_char_limit: int
    # Hermes-shaped human gates, reduced to the persistent writes production
    # needs. Each switch is per profile (and may be overridden per channel).
    approval_memory: bool
    approval_skills: bool
    approval_cron: bool
    approval_timeout_seconds: float
    # Profile-local durable scheduled work.
    cron_enabled: bool
    cron_timezone: str
    cron_tick_seconds: float
    cron_claim_ttl_seconds: float
    cron_max_concurrent: int
    cron_output_retention: int
    request_timeout_seconds: float
    # How long a model connection may stay silent before we drop it and
    # reconnect: the first wait is for the very first sign of life, the second
    # is for a stream that started and then stopped. Zero waits forever.
    codex_first_event_timeout_seconds: float
    codex_quiet_stream_timeout_seconds: float
    # A bounded, non-technical heartbeat for long messaging turns. Zero disables.
    working_notice_interval_seconds: float
    working_notice_text: str
    # Blue ticks. Off unless the operator asks: an agent that watches a chat
    # should not silently mark everything as read.
    send_read_receipts: bool
    # The tool loop.
    max_tool_iterations: int
    max_tool_result_chars: int
    max_tool_step_chars: int
    # The file itself, so a tool added in a later slice can read its own
    # settings without this dataclass growing a field for every one of them.
    settings: Settings = field(default_factory=Settings, compare=False, repr=False)
    # Current Hermes' home-channel contract, narrowed to Pilotage's two
    # platforms. Channel identities stay in the profile .env, not YAML.
    channel: str = "whatsapp"
    home_chat_id: str = ""
    home_thread_id: str = ""

    @property
    def session_dir(self) -> Path:
        return self.state_dir / "whatsapp"

    @property
    def media_dir(self) -> Path:
        """Short-lived channel cache; active files are copied into session inputs."""
        return self.state_dir / "media"

    @property
    def media_roots(self) -> tuple[Path, ...]:
        """The only directories a bridge-reported file path may live under."""
        return (self.media_dir,)

    @property
    def answer_groups(self) -> bool:
        """Whether the bridge should surface any group traffic."""
        return self.group_policy != "disabled"

    @property
    def conversations_path(self) -> Path:
        """Every turn of every chat. Survives re-pairing; only login state does not."""
        return self.state_dir / "conversations.db"

    @property
    def memory_dir(self) -> Path:
        return self.state_dir / "memories"

    @property
    def workspace_dir(self) -> Path:
        """The profile-local default root for terminal and file tools."""
        return self.state_dir / "workspace"

    @property
    def outbound_media_roots(self) -> tuple[Path, ...]:
        """Directories explicitly trusted for native outbound file delivery."""

        roots = [self.workspace_dir.resolve(strict=False)]
        for written in self.settings.names("gateway.media_delivery_allow_dirs"):
            resolved = Path(written).expanduser().resolve(strict=False)
            if resolved not in roots:
                roots.append(resolved)
        return tuple(roots)

    @property
    def home_origin(self) -> dict[str, str] | None:
        """The declared destination for unattended output on this channel."""

        if self.channel not in {"whatsapp", "telegram"} or not self.home_chat_id:
            return None
        origin = {"channel": self.channel, "chat_id": self.home_chat_id}
        if self.channel == "telegram" and self.home_thread_id:
            origin["thread_id"] = self.home_thread_id
        return origin

    @property
    def credentials_path(self) -> Path:
        return self.state_dir / "codex-auth.json"

    @property
    def main_credentials_path(self) -> Path:
        """The only state a named profile may borrow from the main agent."""
        return default_state_root() / "codex-auth.json"

    @property
    def bridge_script(self) -> Path:
        return self.bridge_dir / "bridge.js"

    def for_channel(self, channel: str) -> "Config":
        """The same agent as seen from one channel.

        A channel may run with fewer tools or a different model than the
        agent's common settings — a group chat on WhatsApp is not the console.
        """
        if not channel:
            return self
        return Config.load(channel=channel)

    @classmethod
    def load(cls, channel: str = "") -> "Config":
        """Read the configuration. Raises ConfigError if the file is unusable."""
        home = state_dir()
        settings = Settings.load(config_path(home))
        if channel:
            settings = settings.for_channel(channel)
        active_channel = (channel or "whatsapp").strip().lower()
        home_chat_id = ""
        home_thread_id = ""
        if active_channel == "whatsapp":
            home_chat_id = _env_str("WHATSAPP_HOME_CHANNEL", "")
        elif active_channel == "telegram":
            home_chat_id = _env_str("TELEGRAM_HOME_CHANNEL", "")
            home_thread_id = _env_str("TELEGRAM_HOME_CHANNEL_THREAD_ID", "")
            if home_thread_id and not home_thread_id.isdigit():
                raise ConfigError(
                    "TELEGRAM_HOME_CHANNEL_THREAD_ID must be a positive numeric "
                    "Telegram topic ID"
                )
            if home_thread_id and int(home_thread_id) <= 0:
                raise ConfigError(
                    "TELEGRAM_HOME_CHANNEL_THREAD_ID must be a positive numeric "
                    "Telegram topic ID"
                )

        if settings.get("whatsapp.allowed_senders") is not None:
            raise ConfigError(
                "whatsapp.allowed_senders contains sensitive identities; "
                "set PILOTAGE_ALLOWED_SENDERS in ~/.pilotage-agent/.env instead"
            )
        env_senders = _env_str("PILOTAGE_ALLOWED_SENDERS", "")
        senders = [
            part.strip()
            for part in env_senders.replace(";", ",").split(",")
            if part.strip()
        ]
        if "*" in senders:
            raise ConfigError(
                "PILOTAGE_ALLOWED_SENDERS must name explicit senders; '*' is not allowed"
            )

        # Validate tool-owned settings at startup as well. Waiting until a
        # model first calls the tool would turn an operator typo into a partial
        # production failure rather than a clear startup error.
        settings.names("tools.enabled")
        settings.names("tools.disabled")
        from .tools import build_registry, enabled_groups

        enabled_groups(settings, build_registry())
        from .tools.image import validate_image_settings

        validate_image_settings(settings)
        from .tools.web import validate_web_settings

        validate_web_settings(settings)
        from .tools.code_execution import validate_settings as validate_code_execution

        validate_code_execution(settings)
        from .transcription import validate_settings as validate_stt_settings

        validate_stt_settings(settings)
        from .channels.telegram import validate_settings as validate_telegram_settings

        validate_telegram_settings(settings)

        settings.names("skills.disabled")
        settings.flag("skills.template_vars", True)
        settings.flag("skills.inline_shell", False)
        for written in settings.names("gateway.media_delivery_allow_dirs"):
            root = Path(written).expanduser()
            if not root.is_absolute():
                raise ConfigError(
                    "gateway.media_delivery_allow_dirs must contain absolute "
                    f"paths, not {written!r}"
                )
            if not root.is_dir():
                raise ConfigError(
                    "gateway.media_delivery_allow_dirs path is not an existing "
                    f"directory: {root}"
                )
        _count_in_range(
            "skills.inline_shell_timeout",
            settings.count("skills.inline_shell_timeout", 10),
            minimum=1,
        )
        terminal_cwd = settings.text("terminal.cwd", "")
        if terminal_cwd:
            expanded_cwd = Path(terminal_cwd).expanduser()
            if not expanded_cwd.is_dir() or not os.access(expanded_cwd, os.X_OK):
                raise ConfigError(
                    "terminal.cwd must be an existing accessible directory, "
                    f"not {terminal_cwd!r}"
                )
        isolated_workspaces = settings.flag(
            "sessions.isolated_workspaces", False
        )
        if isolated_workspaces and terminal_cwd:
            if not expanded_cwd.is_absolute():
                raise ConfigError(
                    "terminal.cwd must be absolute when "
                    "sessions.isolated_workspaces is enabled"
                )
            resolved_cwd = expanded_cwd.resolve(strict=False)
            trusted_roots = [
                (home / "workspace").resolve(strict=False),
                *[
                    Path(written).expanduser().resolve(strict=False)
                    for written in settings.names(
                        "gateway.media_delivery_allow_dirs"
                    )
                ],
            ]
            if not any(
                _path_is_within(resolved_cwd, root) for root in trusted_roots
            ):
                raise ConfigError(
                    "terminal.cwd must be inside the profile workspace or a "
                    "gateway.media_delivery_allow_dirs path when "
                    "sessions.isolated_workspaces is enabled"
                )
        legacy_answer_groups = settings.get("whatsapp.answer_groups")
        if legacy_answer_groups is not None and settings.flag(
            "whatsapp.answer_groups", False
        ):
            raise ConfigError(
                "whatsapp.answer_groups has been replaced by "
                "whatsapp.group_policy and whatsapp.group_allow_from"
            )
        group_policy = settings.text(
            "whatsapp.group_policy", "disabled"
        ).lower()
        if group_policy not in {"disabled", "allowlist"}:
            raise ConfigError(
                "whatsapp.group_policy must be 'disabled' or 'allowlist', "
                f"not {group_policy!r}"
            )
        group_allow_from = frozenset(
            settings.names("whatsapp.group_allow_from")
        )
        require_mention = settings.flag("whatsapp.require_mention", False)
        session_reset_mode = settings.text(
            "session_reset.mode", "none"
        ).strip().lower()
        if session_reset_mode not in {"none", "idle", "daily", "both"}:
            raise ConfigError(
                "session_reset.mode must be 'none', 'idle', 'daily', or 'both', "
                f"not {session_reset_mode!r}"
            )
        language_written = settings.text(
            "display.language", DEFAULT_PROFILE_LANGUAGE
        )
        try:
            language = normalize_language(language_written)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        working_notice_text = settings.text(
            "agent.working_notice_text",
            t("runtime.working", language),
        )
        if len(working_notice_text) > 280:
            raise ConfigError(
                "agent.working_notice_text must be at most 280 characters"
            )
        timezone_name = settings.text("timezone", "")
        cron_timezone = settings.text("cron.timezone", timezone_name)
        try:
            from .cron.jobs import timezone_for_name

            timezone_for_name(timezone_name)
            timezone_for_name(cron_timezone)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        _count_in_range(
            "terminal.timeout",
            settings.count("terminal.timeout", 120),
            minimum=1,
        )

        return cls(
            model=settings.text("agent.model", DEFAULT_MODEL),
            reasoning_effort=settings.text("agent.reasoning_effort", DEFAULT_REASONING_EFFORT),
            instructions=_instructions(settings, home, channel),
            language=language,
            timezone=timezone_name,
            allowed_senders=frozenset(senders),
            group_policy=group_policy,
            group_allow_from=group_allow_from,
            require_mention=require_mention,
            bridge_port=_count_in_range(
                "whatsapp.bridge_port",
                settings.count("whatsapp.bridge_port", 8765),
                minimum=1,
                maximum=65535,
            ),
            bridge_dir=Path(_env_str("PILOTAGE_BRIDGE_DIR", str(REPO_ROOT / "bridge"))),
            state_dir=home,
            text_batch_delay_seconds=_number_in_range(
                "whatsapp.batch_delay",
                settings.number("whatsapp.batch_delay", 5.0),
                minimum=0,
                inclusive=True,
            ),
            text_batch_split_delay_seconds=_number_in_range(
                "whatsapp.batch_split_delay",
                settings.number("whatsapp.batch_split_delay", 10.0),
                minimum=0,
                inclusive=True,
            ),
            text_batch_hard_cap_seconds=_number_in_range(
                "whatsapp.batch_hard_cap",
                settings.number(
                    "whatsapp.batch_hard_cap", DEFAULT_BATCH_HARD_CAP_SECONDS
                ),
                minimum=0,
                inclusive=False,
            ),
            history_turns=_count_in_range(
                "agent.history_turns",
                settings.count("agent.history_turns", 20),
                minimum=1,
            ),
            session_reset_mode=session_reset_mode,
            session_reset_idle_minutes=_count_in_range(
                "session_reset.idle_minutes",
                settings.count("session_reset.idle_minutes", 1440),
                minimum=1,
            ),
            session_reset_at_hour=_count_in_range(
                "session_reset.at_hour",
                settings.count("session_reset.at_hour", 4),
                minimum=0,
                maximum=23,
            ),
            session_reset_notify=settings.flag("session_reset.notify", True),
            session_auto_prune=settings.flag("sessions.auto_prune", False),
            session_retention_days=_count_in_range(
                "sessions.retention_days",
                settings.count(
                    "sessions.retention_days", DEFAULT_SESSION_RETENTION_DAYS
                ),
                minimum=1,
            ),
            session_isolated_workspaces=isolated_workspaces,
            codex_native_compaction=settings.flag(
                "compression.codex_responses_native", True
            ),
            codex_compact_threshold=_count_in_range(
                "compression.codex_responses_compact_threshold",
                settings.count(
                    "compression.codex_responses_compact_threshold",
                    DEFAULT_COMPACT_THRESHOLD,
                ),
                minimum=1024,
            ),
            memory_char_limit=_count_in_range(
                "memory.memory_char_limit",
                settings.count("memory.memory_char_limit", DEFAULT_MEMORY_CHAR_LIMIT),
                minimum=1,
            ),
            user_memory_char_limit=_count_in_range(
                "memory.user_char_limit",
                settings.count("memory.user_char_limit", DEFAULT_USER_CHAR_LIMIT),
                minimum=1,
            ),
            approval_memory=settings.flag("approvals.memory", True),
            approval_skills=settings.flag("approvals.skills", True),
            approval_cron=settings.flag("approvals.cron", True),
            approval_timeout_seconds=_number_in_range(
                "approvals.timeout",
                settings.number(
                    "approvals.timeout", DEFAULT_APPROVAL_TIMEOUT_SECONDS
                ),
                minimum=0,
                inclusive=False,
            ),
            cron_enabled=settings.flag("cron.enabled", True),
            cron_timezone=cron_timezone,
            cron_tick_seconds=_number_in_range(
                "cron.tick_seconds",
                settings.number("cron.tick_seconds", DEFAULT_CRON_TICK_SECONDS),
                minimum=0,
                inclusive=False,
            ),
            cron_claim_ttl_seconds=_number_in_range(
                "cron.claim_ttl_seconds",
                settings.number(
                    "cron.claim_ttl_seconds", DEFAULT_CRON_CLAIM_TTL_SECONDS
                ),
                minimum=1,
                inclusive=True,
            ),
            cron_max_concurrent=_count_in_range(
                "cron.max_concurrent",
                settings.count("cron.max_concurrent", DEFAULT_CRON_MAX_CONCURRENT),
                minimum=1,
            ),
            cron_output_retention=_count_in_range(
                "cron.output_retention",
                settings.count(
                    "cron.output_retention", DEFAULT_CRON_OUTPUT_RETENTION
                ),
                minimum=1,
            ),
            request_timeout_seconds=_number_in_range(
                "agent.request_timeout",
                settings.number("agent.request_timeout", 300.0),
                minimum=0,
                inclusive=False,
            ),
            codex_first_event_timeout_seconds=_number_in_range(
                "agent.first_event_timeout",
                settings.number(
                    "agent.first_event_timeout",
                    120.0,
                ),
                minimum=0,
                inclusive=True,
            ),
            codex_quiet_stream_timeout_seconds=_number_in_range(
                "agent.quiet_stream_timeout",
                settings.number(
                    "agent.quiet_stream_timeout",
                    12.0,
                ),
                minimum=0,
                inclusive=True,
            ),
            working_notice_interval_seconds=_number_in_range(
                "agent.working_notice_interval",
                settings.number(
                    "agent.working_notice_interval",
                    DEFAULT_WORKING_NOTICE_INTERVAL_SECONDS,
                ),
                minimum=0,
                inclusive=True,
            ),
            working_notice_text=working_notice_text,
            send_read_receipts=settings.flag("whatsapp.read_receipts", False),
            max_tool_iterations=_count_in_range(
                "tools.max_iterations",
                settings.count("tools.max_iterations", DEFAULT_MAX_TOOL_ITERATIONS),
                minimum=1,
            ),
            max_tool_result_chars=_count_in_range(
                "tools.max_result_chars",
                settings.count("tools.max_result_chars", DEFAULT_MAX_RESULT_CHARS),
                minimum=1,
            ),
            max_tool_step_chars=_count_in_range(
                "tools.max_step_chars",
                settings.count("tools.max_step_chars", DEFAULT_MAX_STEP_CHARS),
                minimum=1,
            ),
            settings=settings,
            channel=active_channel,
            home_chat_id=home_chat_id,
            home_thread_id=home_thread_id,
        )


__all__ = ["Config", "ConfigError", "state_dir"]
