"""Hermes-derived secret redaction for logs and terminal output.

The runtime keeps one policy for every log handler and for text returned by
the model-controlled terminal. Known credential shapes are masked while
ordinary source code and web-navigation URLs remain usable.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import secrets
import shlex
from pathlib import Path
from typing import Optional


_PREFIX_PATTERNS = (
    r"sk-[A-Za-z0-9_-]{10,}",
    r"ghp_[A-Za-z0-9]{10,}",
    r"github_pat_[A-Za-z0-9_]{10,}",
    r"gh[ours]_[A-Za-z0-9]{10,}",
    r"xapp-\d+-[A-Za-z0-9-]{10,}",
    r"xox[baprs]-[A-Za-z0-9-]{10,}",
    r"AIza[A-Za-z0-9_-]{30,}",
    r"pplx-[A-Za-z0-9]{10,}",
    r"fal_[A-Za-z0-9_-]{10,}",
    r"fc-[A-Za-z0-9]{10,}",
    r"gAAAA[A-Za-z0-9_=-]{20,}",
    r"AKIA[A-Z0-9]{16}",
    r"sk_(?:live_|test_)?[A-Za-z0-9_]{10,}",
    r"rk_live_[A-Za-z0-9]{10,}",
    r"SG\.[A-Za-z0-9_-]{10,}",
    r"hf_[A-Za-z0-9]{10,}",
    r"r8_[A-Za-z0-9]{10,}",
    r"npm_[A-Za-z0-9]{10,}",
    r"pypi-[A-Za-z0-9_-]{10,}",
    r"do[po]_v1_[A-Za-z0-9]{10,}",
    r"tvly-[A-Za-z0-9]{10,}",
    r"exa_[A-Za-z0-9]{10,}",
    r"gsk_[A-Za-z0-9]{10,}",
    r"ntn_[A-Za-z0-9]{10,}",
    r"xai-[A-Za-z0-9]{30,}",
    r"gl(?:pat|oas|dt|rt|rtr|cbt|ptt|ft|imt|agent|soat|ffct|wt)-[A-Za-z0-9_.-]{10,}",
)
_PREFIX_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(" + "|".join(_PREFIX_PATTERNS) + r")(?![A-Za-z0-9_-])"
)

_SECRET_ENV_NAMES = (
    r"(?:API_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASS|PW|CREDENTIAL|AUTH)"
)
_ENV_ASSIGN_RE = re.compile(
    rf"([A-Z0-9_]{{0,50}}{_SECRET_ENV_NAMES}[A-Z0-9_]{{0,50}})"
    rf"\s*=\s*(['\"]?)(\S+)\2"
)
_ENV_ASSIGN_LOWER_RE = re.compile(
    r"(?<![a-z0-9_])"
    r"([a-z0-9_]+_(?:key|pass|pw|token|secret|password|passwd|credential|auth))"
    r"\s*=\s*(['\"]?)(\S+)\2",
    re.IGNORECASE,
)
_ENV_LOOKUP_VALUE_RE = re.compile(
    r"^(?:os\.(?:getenv|environ)|process\.env|\$ENV\{)"
)

_KEY_KEYWORD_RE = re.compile(
    r"(?:api|auth|access|refresh|session|secret)[ _.\-]?(?:key|token)"
    r"|token|secret|passwd|password|pass|pw|credential|auth|key",
    re.IGNORECASE,
)
_JSON_FIELD_RE = re.compile(
    r'("(?:api_?[Kk]ey|token|secret|password|access_token|refresh_token|'
    r'auth_token|bearer|secret_value|raw_secret|secret_input|key_material)")'
    r'\s*:\s*"([^"]+)"',
    re.IGNORECASE,
)
_YAML_ASSIGN_RE = re.compile(
    r"(^[ \t]*[A-Za-z0-9_.\-]*(?:api[ _.\-]?key|token|secret|passwd|"
    r"password|credential)[A-Za-z0-9_.\-]*)(:[ \t]*)(?!['\"])([^\s&]+)",
    re.IGNORECASE | re.MULTILINE,
)

_AUTH_HEADER_RE = re.compile(
    r"((?:Proxy-)?Authorization:\s*)([A-Za-z][\w.+-]*\s+)?([^\s\"']+)",
    re.IGNORECASE,
)
_SECRET_HEADER_RE = re.compile(
    r"((?:x-api-key|x-goog-api-key|api-key|apikey|x-api-token|"
    r"x-auth-token|x-access-token)\s*:\s*)(\S+)",
    re.IGNORECASE,
)
_TELEGRAM_RE = re.compile(r"(bot)?(\d{8,}):([-A-Za-z0-9_]{30,})")
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?"
    r"-----END[A-Z ]*PRIVATE KEY-----"
)
_DB_CONNSTR_RE = re.compile(
    r"((?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp)://"
    r"[^:\s]+:)([^@\s]+)(@)",
    re.IGNORECASE,
)
_URL_BARE_TOKEN_RE = re.compile(
    r"((?:https?|wss?|git|ssh|ftp|ftps|sftp)://)"
    r"([^\s:@/]{8,})(@[^\s]+)",
    re.IGNORECASE,
)
_JWT_RE = re.compile(
    r"eyJ[A-Za-z0-9_-]{10,}(?:\.[A-Za-z0-9_=-]{4,}){0,2}"
)
_PHONE_RE = re.compile(r"(\+[1-9]\d{6,14})(?![A-Za-z0-9])")
_WHATSAPP_JID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d[\d:+-]{4,})@(?:s\.whatsapp\.net|g\.us|lid)"
    r"(?![A-Za-z0-9.])",
    re.IGNORECASE,
)
_CHANNEL_ID_FIELD_RE = re.compile(
    r"(?P<prefix>\b(?:chat|user|sender|thread)_id\b\s*['\"]?\s*[:=]\s*['\"]?)"
    r"(?P<identity>-?\d{5,})",
    re.IGNORECASE,
)
_TELEGRAM_CHAT_RE = re.compile(
    r"(?P<prefix>\bTelegram\s+(?:chat|user)\s+)"
    r"(?P<identity>-?\d{5,})",
    re.IGNORECASE,
)
_FORM_BODY_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*"
    r"(?:&[A-Za-z_][A-Za-z0-9_.-]*=[^&\s]*)+$"
)
_SENSITIVE_QUERY_PARAMS = frozenset(
    {
        "access_token",
        "refresh_token",
        "id_token",
        "token",
        "api_key",
        "apikey",
        "client_secret",
        "password",
        "auth",
        "jwt",
        "session",
        "secret",
        "key",
        "code",
        "signature",
        "x-amz-signature",
    }
)

_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x1f\x7f\u200b-\u200f\u2028-\u202f\u2060\ufeff]"
)
_DISPLAY_CONTROL_RE = re.compile(
    r"[\x00-\x1f\x7f\x80-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2064]"
)
_TOKEN_BODY_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-."
)

_ENV_DUMP_COMMANDS = frozenset({"env", "printenv", "set", "export", "declare"})
_FILE_READ_COMMANDS = frozenset(
    {"cat", "head", "tail", "type", "bat", "less", "more", "nl", "zcat", "tac", "view", "batcat"}
)

_IDENTITY_KEY_BYTES = 32
_identity_key = secrets.token_bytes(_IDENTITY_KEY_BYTES)


def identity_key_path(state_dir: Path) -> Path:
    return Path(state_dir) / "log-identity.key"


def configure_identity_pseudonyms(state_dir: Path) -> Path:
    """Load or create the profile-local key used for stable log aliases."""

    global _identity_key
    target = identity_key_path(state_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        key = target.read_bytes()
    except FileNotFoundError:
        key = secrets.token_bytes(_IDENTITY_KEY_BYTES)
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp"
        )
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(key)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # A hard-link is an atomic create-if-absent.  Unlike replace,
                # two guarded startup paths can never rotate an already-used
                # pseudonym key underneath each other.
                os.link(temporary, target)
            except FileExistsError:
                pass
            try:
                directory = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except OSError:
                pass
        finally:
            try:
                temporary.unlink()
            except OSError:
                pass
        # Another guarded process may have won a creation race.
        key = target.read_bytes()
    if len(key) != _IDENTITY_KEY_BYTES:
        raise ValueError(f"Identity-log key at {target} is corrupt.")
    try:
        os.chmod(target, 0o600)
    except OSError:
        pass
    _identity_key = key
    return target


def identity_pseudonym(value: object, namespace: str = "id") -> str:
    """Return one stable, non-reversible alias for a channel identity."""

    written = str(value or "")
    safe_namespace = re.sub(r"[^a-z0-9_-]", "-", namespace.lower()) or "id"
    digest = hmac.new(
        _identity_key,
        f"{safe_namespace}\0{written}".encode("utf-8", errors="replace"),
        hashlib.sha256,
    ).hexdigest()[:12]
    return f"[{safe_namespace}:{digest}]"


def redact_channel_identities(text: object) -> str:
    """Pseudonymize recognizable channel identities in rendered log text."""

    written = str(text or "")
    written = _WHATSAPP_JID_RE.sub(
        lambda match: identity_pseudonym(match.group(0), "wa"), written
    )
    for pattern in (_CHANNEL_ID_FIELD_RE, _TELEGRAM_CHAT_RE):
        written = pattern.sub(
            lambda match: (
                match.group("prefix")
                + identity_pseudonym(match.group("identity"), "tg")
            ),
            written,
        )
    return written


def mask_secret(
    value: str,
    *,
    head: int = 4,
    tail: int = 4,
    floor: int = 12,
    placeholder: str = "***",
    empty: str = "",
) -> str:
    """Mask a secret while retaining a small diagnostic prefix and suffix."""

    if not value:
        return empty
    value = _DISPLAY_CONTROL_RE.sub("", str(value))
    if not value:
        return empty
    if len(value) < floor:
        return placeholder
    return f"{value[:head]}...{value[-tail:]}"


def _mask_token(token: str) -> str:
    return mask_secret(token, head=6, tail=4, floor=18, empty="***")


def _is_word_start(value: str, index: int) -> bool:
    if index == 0:
        return True
    previous, current = value[index - 1], value[index]
    if not previous.isalpha():
        return True
    if current.isupper() and previous.islower():
        return True
    return bool(
        current.isupper()
        and previous.isupper()
        and index + 1 < len(value)
        and value[index + 1].islower()
    )


def _is_word_end(value: str, index: int, *, allow_plural: bool = True) -> bool:
    if index >= len(value):
        return True
    current = value[index]
    if not current.isalpha():
        return True
    if current.isupper() and value[index - 1].islower():
        return True
    if allow_plural and current in "sS":
        return _is_word_end(value, index + 1, allow_plural=False)
    return False


def _key_has_secret_keyword(key: str) -> bool:
    for match in _KEY_KEYWORD_RE.finditer(key):
        if _is_word_start(key, match.start()) and _is_word_end(key, match.end()):
            return True
    return False


def _mask_control_split_tokens(text: str) -> str:
    stripped = _CONTROL_CHARS_RE.sub("", text)
    if stripped == text:
        return text
    original_indices = [
        index for index, char in enumerate(text) if not _CONTROL_CHARS_RE.match(char)
    ]
    output = list(text)
    matches = []
    for match in _PREFIX_RE.finditer(stripped):
        token = match.group(1)
        start = original_indices[match.start(1)]
        end = original_indices[match.end(1) - 1] + 1
        span = text[start:end]
        if ("\n" in span or "\r" in span) and _PREFIX_RE.search(span):
            continue
        if all(
            char in _TOKEN_BODY_CHARS or _CONTROL_CHARS_RE.match(char)
            for char in span
        ) and (end >= len(text) or text[end] != "="):
            matches.append((start, end, _mask_token(token)))
    for start, end, replacement in reversed(matches):
        output[start:end] = list(replacement)
    return "".join(output)


def _redact_assignment(match: re.Match[str]) -> str:
    name, quote, value = match.group(1), match.group(2), match.group(3)
    if _ENV_LOOKUP_VALUE_RE.match(value) or not _key_has_secret_keyword(name):
        return match.group(0)
    return f"{name}={quote}{_mask_token(value)}{quote}"


def _redact_query_string(text: str) -> str:
    parts = []
    for pair in text.split("&"):
        if "=" not in pair:
            parts.append(pair)
            continue
        key, _, value = pair.partition("=")
        parts.append(f"{key}=***" if key.lower() in _SENSITIVE_QUERY_PARAMS else pair)
    return "&".join(parts)


def redact_sensitive_text(text: object, *, code_file: bool = False) -> str:
    """Mask current Hermes credential forms in arbitrary text."""

    if text is None:
        return ""
    written = str(text)
    if not written:
        return written

    written = _mask_control_split_tokens(written)
    written = _PREFIX_RE.sub(lambda match: _mask_token(match.group(1)), written)

    if not code_file:
        if "=" in written:
            written = _ENV_ASSIGN_RE.sub(_redact_assignment, written)
            if "://" not in written:
                written = _ENV_ASSIGN_LOWER_RE.sub(_redact_assignment, written)
        if ":" in written and '"' in written:
            written = _JSON_FIELD_RE.sub(
                lambda match: f'{match.group(1)}: "{_mask_token(match.group(2))}"',
                written,
            )
        if ":" in written and "://" not in written:
            def redact_yaml(match: re.Match[str]) -> str:
                key, separator, value = match.group(1), match.group(2), match.group(3)
                if not _key_has_secret_keyword(key):
                    return match.group(0)
                return f"{key}{separator}{_mask_token(value)}"

            written = _YAML_ASSIGN_RE.sub(redact_yaml, written)

    if "uthorization" in written or "UTHORIZATION" in written:
        written = _AUTH_HEADER_RE.sub(
            lambda match: (
                match.group(1)
                + (match.group(2) or "")
                + _mask_token(match.group(3))
            ),
            written,
        )
    if ":" in written:
        written = _SECRET_HEADER_RE.sub(
            lambda match: match.group(1) + _mask_token(match.group(2)),
            written,
        )
        written = _TELEGRAM_RE.sub(
            lambda match: f"{match.group(1) or ''}{match.group(2)}:***",
            written,
        )
    if "BEGIN" in written and "PRIVATE KEY" in written:
        written = _PRIVATE_KEY_RE.sub("[REDACTED PRIVATE KEY]", written)
    if "://" in written:
        written = _DB_CONNSTR_RE.sub(
            lambda match: f"{match.group(1)}***{match.group(3)}",
            written,
        )
        written = _URL_BARE_TOKEN_RE.sub(
            lambda match: (
                f"{match.group(1)}{_mask_token(match.group(2))}{match.group(3)}"
            ),
            written,
        )
    if "eyJ" in written:
        written = _JWT_RE.sub(lambda match: _mask_token(match.group(0)), written)
    if "&" in written and "=" in written and _FORM_BODY_RE.fullmatch(written.strip()):
        written = _redact_query_string(written.strip())
    if "+" in written:
        written = _PHONE_RE.sub(
            lambda match: (
                match.group(1)[:4] + "****" + match.group(1)[-4:]
                if len(match.group(1)) > 8
                else match.group(1)[:2] + "****" + match.group(1)[-2:]
            ),
            written,
        )
    return written


def is_env_dump_command(command: Optional[str]) -> bool:
    """Return whether a command directly prints its process environment."""

    if not command or not isinstance(command, str):
        return False
    for segment in re.split(r"[|;&]+", command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        if tokens and tokens[0] in _ENV_DUMP_COMMANDS:
            return True
    return False


def _command_reads_env_file(command: Optional[str]) -> bool:
    if not command:
        return False
    # Late import avoids making the top-level redactor initialize the complete
    # tool registry. The file guard remains the single owner of this list.
    from .tools.file_safety import BLOCKED_ENV_BASENAMES

    for segment in re.split(r"[|;&]+", command):
        tokens = segment.strip().split()
        if not tokens or tokens[0] not in _FILE_READ_COMMANDS:
            continue
        for argument in tokens[1:]:
            if argument.startswith("-"):
                continue
            argument = argument.strip("\"'")
            basename = argument.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            if basename.lower() in BLOCKED_ENV_BASENAMES:
                return True
    return False


def redact_terminal_output(output: object, command: Optional[str] = None) -> str:
    """Apply Hermes's command-aware redaction to terminal output."""

    if output is None:
        return ""
    written = str(output)
    code_file = not (
        is_env_dump_command(command) or _command_reads_env_file(command)
    )
    return redact_sensitive_text(written, code_file=code_file)


class RedactingFormatter(logging.Formatter):
    """Logging formatter that masks secrets after exceptions are rendered."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_channel_identities(
            redact_sensitive_text(super().format(record))
        )


__all__ = [
    "RedactingFormatter",
    "configure_identity_pseudonyms",
    "identity_key_path",
    "identity_pseudonym",
    "is_env_dump_command",
    "mask_secret",
    "redact_channel_identities",
    "redact_sensitive_text",
    "redact_terminal_output",
]
