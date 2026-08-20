"""What this agent is and how it behaves.

Behaviour is written in a configuration file the operator edits and can diff
against another machine; the environment carries sensitive identities,
secrets, and the few things that are properties of the box rather than of the
agent. Each setting has one canonical home, so two files cannot disagree about
which behaviour is running.

A missing file leaves every default in place, so an agent that has never been
configured still runs. A broken file stops the agent instead of falling back,
because a default can silently re-enable something the operator switched off.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from .settings import ConfigError, Settings, config_path

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"

DEFAULT_INSTRUCTIONS = (
    "You are a Pilotage agent. You answer over WhatsApp, so keep replies short "
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

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def state_dir() -> Path:
    """Where credentials, the WhatsApp session and logs live."""
    override = os.environ.get("PILOTAGE_HOME", "").strip()
    root = Path(override).expanduser() if override else Path.home() / ".pilotage-agent"
    return root


def _instructions(settings: Settings) -> str:
    """The operator's instructions, plus the formatting note they cannot drop."""
    # A blank setting means the default, so there is always something here.
    written = settings.text("agent.instructions", "")
    return f"{written or DEFAULT_INSTRUCTIONS}\n\n{FORMATTING_NOTE}"


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


@dataclass(frozen=True)
class Config:
    model: str
    reasoning_effort: str
    instructions: str
    # Who the agent answers. Empty means nobody: an agent wired to a real phone
    # number must never answer whoever finds it. Entries are phone numbers in
    # digits, or full jids.
    allowed_senders: frozenset[str]
    answer_groups: bool
    bridge_port: int
    bridge_dir: Path
    state_dir: Path
    # Quiet period before a burst of inbound messages is treated as one turn.
    text_batch_delay_seconds: float
    text_batch_split_delay_seconds: float
    # Turns of history kept per chat, in memory and on disk.
    history_turns: int
    request_timeout_seconds: float
    # How long a model connection may stay silent before we drop it and
    # reconnect: the first wait is for the very first sign of life, the second
    # is for a stream that started and then stopped. Zero waits forever.
    codex_first_event_timeout_seconds: float
    codex_quiet_stream_timeout_seconds: float
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

    @property
    def session_dir(self) -> Path:
        return self.state_dir / "whatsapp"

    @property
    def media_dir(self) -> Path:
        """Inbound media, kept outside the session directory.

        Re-pairing deletes the session; a cached voice note should not depend
        on that. Nothing prunes this directory yet — see docs/skeleton-drops.md.
        """
        return self.state_dir / "media"

    @property
    def media_roots(self) -> tuple[Path, ...]:
        """The only directories a bridge-reported file path may live under."""
        return (self.media_dir,)

    @property
    def conversations_path(self) -> Path:
        """Every turn of every chat. Survives re-pairing; only login state does not."""
        return self.state_dir / "conversations.db"

    @property
    def credentials_path(self) -> Path:
        return self.state_dir / "codex-auth.json"

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

        # Validate tool-owned settings at startup as well. Waiting until a
        # model first calls the tool would turn an operator typo into a partial
        # production failure rather than a clear startup error.
        settings.names("tools.enabled")
        settings.names("tools.disabled")
        settings.names("skills.disabled")
        settings.text("terminal.cwd", "")
        _count_in_range(
            "terminal.timeout",
            settings.count("terminal.timeout", 120),
            minimum=1,
        )

        return cls(
            model=settings.text("agent.model", DEFAULT_MODEL),
            reasoning_effort=settings.text("agent.reasoning_effort", DEFAULT_REASONING_EFFORT),
            instructions=_instructions(settings),
            allowed_senders=frozenset(senders),
            answer_groups=settings.flag("whatsapp.answer_groups", False),
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
                settings.number(
                    "whatsapp.batch_split_delay",
                    10.0,
                ),
                minimum=0,
                inclusive=True,
            ),
            history_turns=_count_in_range(
                "agent.history_turns",
                settings.count("agent.history_turns", 20),
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
        )


__all__ = ["Config", "ConfigError", "state_dir"]
