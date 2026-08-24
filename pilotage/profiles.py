"""Isolated agent profiles.

This is the small profile boundary extracted from Hermes' profile manager.
The default agent lives at ~/.pilotage-agent; named agents live below its
profiles directory. Selecting one only changes PILOTAGE_HOME. Every existing
state path therefore follows the profile without a second router.

Profiles never inherit another profile's configuration or state. ChatGPT
authentication is the single exception, implemented explicitly in
pilotage.codex.auth rather than here.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .i18n import DEFAULT_PROFILE_LANGUAGE
from .runtime_lock import ProfileRuntimeLock, RuntimeLockError

_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_RESERVED_NAMES = frozenset({"pilotage", "default", "test", "tmp", "root", "sudo"})
_DEFAULT_BRIDGE_PORT = 8765
_FIRST_NAMED_BRIDGE_PORT = 8766

# Only state the current runtime actually owns. Keeping these roots explicit
# makes the isolation contract visible and gives tools a stable workspace.
_PROFILE_DIRS = (
    "memories",
    "sessions",
    "skills",
    "logs",
    "workspace",
    "cron",
    "whatsapp",
    "media",
    "home",
)


@dataclass(frozen=True)
class ProfileInfo:
    name: str
    path: Path
    is_default: bool
    is_active: bool


def _platform_default_root() -> Path:
    return Path.home() / ".pilotage-agent"


def default_state_root() -> Path:
    """Return the main profile root even while a named profile is active.

    This follows Hermes' custom-deployment rule: an arbitrary PILOTAGE_HOME is
    itself the root, while <root>/profiles/<name> is recognized as a selected
    named profile and resolves back to <root>.
    """
    native_root = _platform_default_root()
    written = os.environ.get("PILOTAGE_HOME", "").strip()
    if not written:
        return native_root

    env_path = Path(written).expanduser()
    try:
        env_path.resolve().relative_to(native_root.resolve())
    except ValueError:
        if env_path.parent.name == "profiles":
            return env_path.parent.parent
        return env_path
    return native_root


def profiles_root() -> Path:
    return default_state_root() / "profiles"


def normalize_profile_name(name: str) -> str:
    written = str(name).strip()
    if not written:
        raise ValueError("profile name cannot be empty")
    if written.casefold() == "default":
        return "default"
    return written.lower()


def validate_profile_name(name: str) -> None:
    if name == "default":
        return
    if not _PROFILE_ID_RE.fullmatch(name):
        raise ValueError(
            f"Invalid profile name {name!r}. Must match "
            "[a-z0-9][a-z0-9_-]{0,63}"
        )
    if name in _RESERVED_NAMES:
        raise ValueError(f"Profile name {name!r} is reserved. Pick a different name.")


def get_profile_dir(name: str) -> Path:
    canon = normalize_profile_name(name)
    validate_profile_name(canon)
    if canon == "default":
        return default_state_root()
    return profiles_root() / canon


def _is_local_profile_dir(path: Path) -> bool:
    """Reject aliases, links, and junctions that would share another state tree."""
    if not path.is_dir() or path.is_symlink():
        return False
    try:
        root = profiles_root().resolve()
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root / path.name


def profile_exists(name: str) -> bool:
    canon = normalize_profile_name(name)
    if canon == "default":
        return True
    return _is_local_profile_dir(get_profile_dir(canon))


def _active_profile_path() -> Path:
    return default_state_root() / "active_profile"


def get_active_profile() -> str:
    """Read the sticky profile, failing closed to default on corrupt state."""
    try:
        written = _active_profile_path().read_text(encoding="utf-8").strip()
    except (FileNotFoundError, UnicodeDecodeError, OSError):
        return "default"
    if not written:
        return "default"
    try:
        canon = normalize_profile_name(written)
        validate_profile_name(canon)
    except ValueError:
        return "default"
    return canon if profile_exists(canon) else "default"


def set_active_profile(name: str) -> None:
    canon = normalize_profile_name(name)
    validate_profile_name(canon)
    if canon != "default" and not profile_exists(canon):
        raise FileNotFoundError(
            f"Profile {canon!r} does not exist. Create it with: "
            f"pilotage profile create {canon}"
        )

    path = _active_profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if canon == "default":
        path.unlink(missing_ok=True)
        return
    temp = path.with_suffix(".tmp")
    temp.write_text(canon + "\n", encoding="utf-8")
    os.replace(temp, path)


def activate_for_process(name: str | None = None) -> tuple[str, Path]:
    """Select a profile before environment and configuration are loaded."""
    canon = normalize_profile_name(name) if name is not None else get_active_profile()
    validate_profile_name(canon)
    path = get_profile_dir(canon)
    if canon != "default" and not profile_exists(canon):
        raise FileNotFoundError(
            f"Profile {canon!r} does not exist. Create it with: "
            f"pilotage profile create {canon}"
        )
    os.environ["PILOTAGE_HOME"] = str(path)
    return canon, path


def list_profiles() -> list[ProfileInfo]:
    active = get_active_profile()
    result = [
        ProfileInfo(
            name="default",
            path=default_state_root(),
            is_default=True,
            is_active=active == "default",
        )
    ]
    root = profiles_root()
    if root.is_dir():
        for entry in sorted(root.iterdir(), key=lambda item: item.name):
            if not _is_local_profile_dir(entry) or not _PROFILE_ID_RE.fullmatch(entry.name):
                continue
            try:
                validate_profile_name(entry.name)
            except ValueError:
                continue
            result.append(
                ProfileInfo(
                    name=entry.name,
                    path=entry,
                    is_default=False,
                    is_active=active == entry.name,
                )
            )
    return result


def _configured_bridge_port(profile_dir: Path) -> int:
    """Read the port exactly as the WhatsApp runtime will read it."""
    from .settings import ConfigError, Settings

    try:
        port = (
            Settings.load(profile_dir / "config.yaml")
            .for_channel("whatsapp")
            .count("whatsapp.bridge_port", _DEFAULT_BRIDGE_PORT)
        )
    except ConfigError as exc:
        raise ValueError(f"Cannot allocate a profile port: {exc}") from exc
    if not 1 <= port <= 65535:
        raise ValueError(
            f"Cannot allocate a profile port: {profile_dir / 'config.yaml'} "
            f"sets whatsapp.bridge_port to {port!r}"
        )
    return port


def _allocate_bridge_port() -> int:
    """Choose an unused loopback port for a new profile."""
    used = {_configured_bridge_port(default_state_root())}
    for info in list_profiles():
        if not info.is_default:
            used.add(_configured_bridge_port(info.path))
    for port in range(_FIRST_NAMED_BRIDGE_PORT, 65536):
        if port not in used:
            return port
    raise ValueError("No free WhatsApp bridge port remains for another profile.")


def create_profile(name: str) -> Path:
    """Create one fresh profile without copying another agent's state."""
    canon = normalize_profile_name(name)
    validate_profile_name(canon)
    if canon == "default":
        raise ValueError("Cannot create the built-in default profile.")

    profile_dir = get_profile_dir(canon)
    if os.path.lexists(profile_dir):
        raise FileExistsError(f"Profile {canon!r} already exists at {profile_dir}")

    bridge_port = _allocate_bridge_port()
    created = False
    try:
        profile_dir.mkdir(parents=True)
        created = True
        for subdir in _PROFILE_DIRS:
            (profile_dir / subdir).mkdir()

        # A blank file is intentional: it prevents a named profile from silently
        # loading the repository .env. Real process environment still wins, as it
        # does for the default profile.
        env_path = profile_dir / ".env"
        env_path.write_text(
            "# Secrets and channel identities for this Pilotage profile.\n"
            "# Behavioral settings belong in config.yaml.\n",
            encoding="utf-8",
        )
        config_path = profile_dir / "config.yaml"
        config_path.write_text(
            "# Profile-local settings. This port must stay unique on the host.\n"
            "# Put this profile's identity in SOUL.md beside this file.\n"
            "display:\n"
            f"  language: {DEFAULT_PROFILE_LANGUAGE}\n"
            "timezone: \"\"\n"
            "whatsapp:\n"
            f"  bridge_port: {bridge_port}\n",
            encoding="utf-8",
        )
        for protected in (env_path, config_path):
            try:
                os.chmod(protected, 0o600)
            except OSError:
                pass
    except BaseException:
        if created:
            shutil.rmtree(profile_dir, ignore_errors=True)
        raise
    return profile_dir


def delete_profile(name: str) -> Path:
    """Delete exactly one validated named profile."""
    canon = normalize_profile_name(name)
    validate_profile_name(canon)
    if canon == "default":
        raise ValueError("Cannot delete the built-in default profile.")
    profile_dir = get_profile_dir(canon)
    if not os.path.lexists(profile_dir):
        raise FileNotFoundError(f"Profile {canon!r} does not exist.")
    if not _is_local_profile_dir(profile_dir):
        raise ValueError(f"Refusing non-local or linked profile: {profile_dir}")

    # A running service owns this state tree. Refuse destructive cleanup until
    # its OS-backed lock is released; a stale lock file itself is harmless.
    runtime_lock = ProfileRuntimeLock(profile_dir)
    try:
        runtime_lock.acquire()
    except RuntimeLockError as exc:
        raise ValueError(
            f"Profile {canon!r} is running or its runtime lock is unavailable. "
            "Stop the profile before deleting it."
        ) from exc
    finally:
        runtime_lock.release()

    was_active = get_active_profile() == canon
    shutil.rmtree(profile_dir)
    if was_active:
        set_active_profile("default")
    return profile_dir


__all__ = [
    "ProfileInfo",
    "activate_for_process",
    "create_profile",
    "default_state_root",
    "delete_profile",
    "get_active_profile",
    "get_profile_dir",
    "list_profiles",
    "normalize_profile_name",
    "profile_exists",
    "profiles_root",
    "set_active_profile",
    "validate_profile_name",
]
