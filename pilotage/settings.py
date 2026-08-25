"""The configuration file.

Behaviour is written down in a file the operator edits; secrets stay in the
environment. That split is the one Hermes uses and it has earned it: a model
name, a tool list and a timeout are things you want to read back six months
later and diff against another machine, while a token in a YAML file is a
token in a backup.

A file that cannot be parsed stops the agent. It is tempting to log the
problem and carry on with defaults, but the defaults enable tools the operator
may have deliberately switched off — a typo would quietly hand a chat back the
terminal. Failing at startup is the only reading of a broken file that cannot
surprise anyone.

Every key can be overridden per channel. ``channels.whatsapp.tools.disabled``
wins over ``tools.disabled`` for the WhatsApp channel and is invisible to the
others, so one file describes an agent that answers everywhere without
describing it four times.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_FILENAME = "config.yaml"
_CHANNEL_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


class ConfigError(RuntimeError):
    """The configuration file exists but cannot be used."""


def config_path(state_dir: Path) -> Path:
    """Where the configuration file is read from."""
    override = os.environ.get("PILOTAGE_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    return state_dir / CONFIG_FILENAME


def set_channel_enabled(path: Path, channel: str, enabled: bool = True) -> None:
    """Safely update one top-level channel flag without rewriting operator YAML."""

    path = Path(path)
    if not _CHANNEL_NAME_RE.fullmatch(channel):
        raise ConfigError(f"Invalid channel name: {channel!r}")
    if os.path.lexists(path) and path.is_symlink():
        raise ConfigError(f"Refusing to replace symbolic-link configuration: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
    except OSError as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc

    try:
        import yaml
        from yaml.nodes import MappingNode, ScalarNode
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ConfigError("PyYAML is required to update the configuration file.") from exc

    try:
        root = yaml.compose(text) if text.strip() else None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

    written = "true" if enabled else "false"
    block = f"{channel}:\n  enabled: {written}\n"
    if root is None:
        separator = "" if not text or text.endswith("\n") else "\n"
        updated = text + separator + block
    elif not isinstance(root, MappingNode):
        raise ConfigError(f"{path} must describe settings by name")
    else:
        sections = [
            value
            for key, value in root.value
            if isinstance(key, ScalarNode) and key.value == channel
        ]
        if len(sections) > 1:
            raise ConfigError(f"{path} contains duplicate {channel!r} sections")
        if not sections:
            if root.flow_style:
                if root.value:
                    raise ConfigError(
                        f"Cannot safely add {channel}.enabled to flow-style YAML in {path}"
                    )
                updated = (
                    text[: root.start_mark.index]
                    + block.rstrip("\n")
                    + text[root.end_mark.index :]
                )
            else:
                separator = "" if text.endswith("\n") else "\n"
                updated = text + separator + block
        else:
            section = sections[0]
            if isinstance(section, MappingNode):
                flags = [
                    value
                    for key, value in section.value
                    if isinstance(key, ScalarNode) and key.value == "enabled"
                ]
                if len(flags) > 1:
                    raise ConfigError(
                        f"{path} contains duplicate {channel}.enabled values"
                    )
                if flags:
                    flag = flags[0]
                    if not isinstance(flag, ScalarNode):
                        raise ConfigError(f"{channel}.enabled must be a scalar value")
                    updated = (
                        text[: flag.start_mark.index]
                        + written
                        + text[flag.end_mark.index :]
                    )
                elif section.flow_style:
                    raise ConfigError(
                        f"Cannot safely add {channel}.enabled to flow-style YAML in {path}"
                    )
                elif section.value:
                    first_key = section.value[0][0]
                    indentation = max(2, first_key.start_mark.column)
                    insertion = " " * indentation + f"enabled: {written}\n"
                    line_start = first_key.start_mark.index - first_key.start_mark.column
                    updated = (
                        text[:line_start]
                        + insertion
                        + text[line_start:]
                    )
                else:  # pragma: no cover - empty block mappings compose as null
                    raise ConfigError(f"Cannot safely update empty {channel} section")
            elif isinstance(section, ScalarNode) and section.value in {"", None}:
                updated = (
                    text[: section.start_mark.index]
                    + f"\n  enabled: {written}"
                    + text[section.end_mark.index :]
                )
            else:
                raise ConfigError(f"{channel} must be a settings block in {path}")

    try:
        parsed = yaml.safe_load(updated)
    except yaml.YAMLError as exc:  # pragma: no cover - guarded construction
        raise ConfigError(f"Could not safely update {path}: {exc}") from exc
    if (
        not isinstance(parsed, dict)
        or not isinstance(parsed.get(channel), dict)
        or parsed[channel].get("enabled") is not enabled
    ):
        raise ConfigError(f"Could not prove {channel}.enabled was updated in {path}")
    if updated == text:
        return

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


class Settings:
    """One configuration file, read once, looked up by dotted key.

    ``for_channel`` returns a view of the same file in which a key under
    ``channels.<name>`` shadows the one at the top level.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, channel: str = ""):
        self._data: Dict[str, Any] = data if isinstance(data, dict) else {}
        self._channel = channel

    @property
    def channel(self) -> str:
        """The channel this settings view belongs to, if any."""
        return self._channel

    # -- loading ------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Read the file. A missing file is an empty one; a broken file raises."""
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return cls({})
        except OSError as exc:
            raise ConfigError(f"Could not read {path}: {exc}") from exc

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ConfigError("PyYAML is required to read the configuration file.") from exc

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

        if data is None:
            return cls({})
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must describe settings by name, not a {type(data).__name__}.")
        return cls(data)

    # -- lookup -------------------------------------------------------------

    def for_channel(self, channel: str) -> "Settings":
        return Settings(self._data, channel=channel)

    def get(self, dotted: str, default: Any = None) -> Any:
        """The value at ``a.b.c``, from this channel's section if it has one."""
        if self._channel:
            found = _dig(self._data, f"channels.{self._channel}.{dotted}")
            if found is not _MISSING:
                return found
        found = _dig(self._data, dotted)
        return default if found is _MISSING else found

    def section(self, dotted: str) -> Dict[str, Any]:
        """A block of settings as a dict — the channel's, merged over the common one.

        Only the top level is merged: a channel that sets one key in a block
        keeps the rest of the block rather than replacing it wholesale, which
        is what an override is expected to do.
        """
        merged: Dict[str, Any] = {}
        common = _dig(self._data, dotted)
        if isinstance(common, dict):
            merged.update(common)
        if self._channel:
            override = _dig(self._data, f"channels.{self._channel}.{dotted}")
            if isinstance(override, dict):
                merged.update(override)
        return merged

    def text(self, dotted: str, default: str = "") -> str:
        value = self.get(dotted)
        if value is None:
            return default
        if not isinstance(value, str):
            raise ConfigError(f"{dotted} must be text, not {value!r}")
        text = value.strip()
        return text or default

    def flag(self, dotted: str, default: bool = False) -> bool:
        value = self.get(dotted)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            written = value.strip().lower()
            if written in {"1", "true", "yes", "on"}:
                return True
            if written in {"0", "false", "no", "off"}:
                return False
        raise ConfigError(f"{dotted} must be true or false, not {value!r}")

    def number(self, dotted: str, default: float) -> float:
        value = self.get(dotted)
        if value is None:
            return default
        if isinstance(value, bool):
            raise ConfigError(f"{dotted} must be a number, not {value!r}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{dotted} must be a number, not {value!r}") from None
        if not math.isfinite(number):
            raise ConfigError(f"{dotted} must be a finite number, not {value!r}")
        return number

    def count(self, dotted: str, default: int) -> int:
        value = self.get(dotted)
        if value is None:
            return default
        if isinstance(value, bool):
            raise ConfigError(f"{dotted} must be a whole number, not {value!r}")
        if isinstance(value, float) and not value.is_integer():
            raise ConfigError(f"{dotted} must be a whole number, not {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ConfigError(f"{dotted} must be a whole number, not {value!r}") from None

    def names(self, dotted: str, default: Optional[List[str]] = None) -> List[str]:
        """A list of names, written as a YAML list or as one comma-separated line."""
        value = self.get(dotted)
        if value is None:
            return list(default or [])
        if isinstance(value, str):
            parts = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            if not all(isinstance(part, str) for part in value):
                raise ConfigError(f"{dotted} must contain names, not {value!r}")
            parts = list(value)
        else:
            raise ConfigError(f"{dotted} must be a list of names, not {value!r}")
        return [part.strip() for part in parts if str(part).strip()]


_MISSING = object()


def _dig(data: Any, dotted: str) -> Any:
    current = data
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current
