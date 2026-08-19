"""Runtime configuration for the walking skeleton.

Environment variables only. A real configuration file arrives with the operator
commands, once the skeleton walks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"

DEFAULT_INSTRUCTIONS = (
    "You are a Pilotage agent. You answer over WhatsApp, so keep replies short "
    "and readable on a phone: plain sentences, no headings, no tables, no code "
    "blocks unless the answer is code. Say what you know, say plainly when you "
    "do not know, and never invent facts."
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def state_dir() -> Path:
    """Where credentials, the WhatsApp session and logs live."""
    override = os.environ.get("PILOTAGE_HOME", "").strip()
    root = Path(override).expanduser() if override else Path.home() / ".pilotage-agent"
    return root


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
    # Turns of history kept per chat, in memory only.
    history_turns: int
    request_timeout_seconds: float

    @property
    def session_dir(self) -> Path:
        return self.state_dir / "whatsapp"

    @property
    def credentials_path(self) -> Path:
        return self.state_dir / "codex-auth.json"

    @property
    def bridge_script(self) -> Path:
        return self.bridge_dir / "bridge.js"

    @classmethod
    def load(cls) -> "Config":
        raw_senders = _env_str("PILOTAGE_ALLOWED_SENDERS", "")
        senders = frozenset(
            part.strip() for part in raw_senders.replace(";", ",").split(",") if part.strip()
        )
        return cls(
            model=_env_str("PILOTAGE_MODEL", DEFAULT_MODEL),
            reasoning_effort=_env_str("PILOTAGE_REASONING_EFFORT", DEFAULT_REASONING_EFFORT),
            instructions=_env_str("PILOTAGE_INSTRUCTIONS", DEFAULT_INSTRUCTIONS),
            allowed_senders=senders,
            answer_groups=_env_str("PILOTAGE_ANSWER_GROUPS", "0") in {"1", "true", "yes"},
            bridge_port=_env_int("PILOTAGE_BRIDGE_PORT", 8765),
            bridge_dir=Path(_env_str("PILOTAGE_BRIDGE_DIR", str(REPO_ROOT / "bridge"))),
            state_dir=state_dir(),
            text_batch_delay_seconds=_env_float("PILOTAGE_TEXT_BATCH_DELAY", 5.0),
            text_batch_split_delay_seconds=_env_float("PILOTAGE_TEXT_BATCH_SPLIT_DELAY", 10.0),
            history_turns=_env_int("PILOTAGE_HISTORY_TURNS", 20),
            request_timeout_seconds=_env_float("PILOTAGE_REQUEST_TIMEOUT", 300.0),
        )
