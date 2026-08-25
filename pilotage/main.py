"""Command line entry point.

    pilotage login          authenticate against ChatGPT (device code)
    pilotage ask "..."      one question straight to the model, no messaging
    pilotage whatsapp       configure, pair, and enable WhatsApp
    pilotage telegram       configure, verify, and enable Telegram
    pilotage run            answer enabled messaging channels until stopped
    pilotage doctor         prove deployment readiness
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import inspect
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import httpx

from . import media, profiles, transcription
from .agent import Agent
from .commands import CommandInvocation, execute_command, status_text
from .channels.whatsapp import (
    ChannelError,
    InboundMessage,
    WhatsAppSessionError,
    WhatsAppChannel,
    build_bridge_environment,
    normalize_whatsapp_chat_id,
    validate_whatsapp_session,
)
from .channels.telegram import (
    ChannelError as TelegramChannelError,
    InboundMessage as TelegramInboundMessage,
    TelegramChannel,
    normalize_telegram_allowed_users,
    normalize_telegram_bot_token,
    normalize_telegram_home_chat_id,
    normalize_telegram_topic_id,
)
from .codex import auth
from .config import Config, ConfigError
from .cron.cli import add_cron_parser, run_cron_command
from .cron.jobs import CronError, CronStore
from .cron.scheduler import CronScheduler
from .delivery import (
    DeliveryStore,
    claim_deliveries,
    deliver_final,
    redeliver_claimed_deliveries,
)
from .env import candidate_env_files, load_env_files, update_env_values
from .history import ConversationError, ConversationStore
from .i18n import t
from .redact import RedactingFormatter
from .runtime_lock import ProfileRuntimeLock, RuntimeLockError
from .service import run_service_command
from .settings import config_path, set_channel_enabled

logger = logging.getLogger("pilotage")

SESSION_MAINTENANCE_INTERVAL_SECONDS = 3600
# Hermes bounds both startup owed-delivery drain and shutdown in-flight drain
# to 30 seconds.
STARTUP_RECOVERY_DRAIN_SECONDS = 30.0
SHUTDOWN_DRAIN_SECONDS = 30.0
WHATSAPP_SETUP_ENV_KEYS = frozenset(
    {"PILOTAGE_ALLOWED_SENDERS", "WHATSAPP_HOME_CHANNEL"}
)
TELEGRAM_SETUP_ENV_KEYS = frozenset(
    {
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_ALLOWED_USERS",
        "TELEGRAM_HOME_CHANNEL",
        "TELEGRAM_HOME_CHANNEL_THREAD_ID",
    }
)
CHANNEL_SETUP_ENV_KEYS = WHATSAPP_SETUP_ENV_KEYS | TELEGRAM_SETUP_ENV_KEYS


async def _deliver_scheduled(
    channel: WhatsAppChannel,
    origin: dict[str, str],
    text: str,
) -> None:
    chat_id = str(origin.get("chat_id") or "")
    if not chat_id:
        raise ChannelError("Cron delivery origin has no WhatsApp chat ID.")
    if not await channel.send(chat_id, text):
        raise ChannelError("WhatsApp rejected the scheduled delivery.")


def _configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        RedactingFormatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=[handler],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def command_login(config: Config) -> int:
    print("Signing in to ChatGPT.")
    try:
        credentials = auth.device_code_login()
    except auth.AuthError as exc:
        print(f"Login failed: {exc}", file=sys.stderr)
        return 1
    # Signing in replaces the tokens another agent may be refreshing right now.
    with auth.credentials_lock(config.credentials_path):
        auth.write_credentials(config.credentials_path, credentials)
    print(f"Signed in. Credentials stored at {config.credentials_path}.")
    return 0


def command_status(config: Config, profile_name: str) -> int:
    """Report configuration and verify the selected authentication source."""
    print(status_text(config, profile_name))
    try:
        auth.read_credentials(
            config.credentials_path,
            fallback_path=config.main_credentials_path,
        )
    except auth.AuthError as exc:
        print(f"Health check failed: {exc}", file=sys.stderr)
        return 1
    return 0


async def command_ask(config: Config, question: str) -> int:
    # Nowhere to write. This is the one-shot you run to find out whether the
    # login and the model still work, so it has to answer the same way today as
    # it did yesterday, and it must not add to a running agent's conversations.
    agent = Agent(config, ConversationStore(path=None))

    async def notice(text: str) -> None:
        print(text, file=sys.stderr)

    try:
        answer = await agent.respond("cli", question, on_notice=notice)
    except auth.AuthError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    finally:
        await agent.close()
    print(answer)
    return 0


async def command_run(config: Config, profile_name: str = "default") -> int:
    runtime_lock = ProfileRuntimeLock(config.state_dir)
    try:
        runtime_lock.acquire()
    except RuntimeLockError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        return await _command_run_locked(config, profile_name)
    finally:
        runtime_lock.release()


def _save_channel_enabled(path: Path, channel: str, enabled: bool) -> bool:
    try:
        set_channel_enabled(path, channel, enabled)
    except (ConfigError, OSError) as exc:
        print(f"Could not update {channel} enablement: {exc}", file=sys.stderr)
        return False
    return True


class _SetupCancelled(Exception):
    pass


def command_whatsapp_pair(
    config: Config,
    *,
    env_path: Path | None = None,
    settings_path: Path | None = None,
    external_env: frozenset[str] = frozenset(),
) -> int:
    """Configure and pair WhatsApp without starting the agent runtime."""

    if shutil.which("node") is None:
        print("Node.js is not on PATH.", file=sys.stderr)
        return 1
    if not config.bridge_script.is_file():
        print(f"The bridge script is missing at {config.bridge_script}.", file=sys.stderr)
        return 1
    if not (config.bridge_dir / "node_modules").is_dir():
        print("WhatsApp bridge dependencies are not installed.", file=sys.stderr)
        return 1

    lock = ProfileRuntimeLock(config.state_dir)
    try:
        lock.acquire()
    except RuntimeLockError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        selected_settings_path = settings_path or config_path(config.state_dir)
        try:
            allowed_senders, updates = _prompt_whatsapp_configuration(config)
        except _SetupCancelled:
            print("WhatsApp setup cancelled.", file=sys.stderr)
            return 1

        if updates:
            externally_managed = sorted(updates.keys() & external_env)
            if externally_managed:
                print(
                    "Cannot update externally supplied environment value(s): "
                    + ", ".join(externally_managed)
                    + ". Change them at their deployment source.",
                    file=sys.stderr,
                )
                return 1
            try:
                update_env_values(env_path or config.state_dir / ".env", updates)
            except (OSError, ValueError) as exc:
                print(
                    f"Could not save WhatsApp configuration: {exc}",
                    file=sys.stderr,
                )
                return 1

        config.session_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        credentials_path = config.session_dir / "creds.json"
        if credentials_path.is_file():
            try:
                validate_whatsapp_session(config.session_dir)
            except WhatsAppSessionError as exc:
                print(f"Existing WhatsApp session is invalid: {exc}", file=sys.stderr)
                paired = False
            else:
                print("WhatsApp is already paired for this profile.")
                paired = True
            try:
                repair = _prompt_yes_no(
                    "Re-pair? This will clear the existing WhatsApp session. [y/N] "
                )
            except _SetupCancelled:
                print("WhatsApp setup cancelled.", file=sys.stderr)
                return 1
            if not repair:
                if paired:
                    if not _save_channel_enabled(
                        selected_settings_path, "whatsapp", True
                    ):
                        return 1
                    print("WhatsApp configuration is saved; existing pairing kept.")
                    return 0
                if not _save_channel_enabled(
                    selected_settings_path, "whatsapp", False
                ):
                    return 1
                print("WhatsApp remains unpaired.", file=sys.stderr)
                return 1

        # A failed or interrupted pairing must not leave an unusable channel
        # selected for the resident runtime.
        if not _save_channel_enabled(selected_settings_path, "whatsapp", False):
            return 1
        if credentials_path.is_file():
            try:
                shutil.rmtree(config.session_dir)
                config.session_dir.mkdir(mode=0o700, parents=True)
            except OSError as exc:
                print(
                    f"Could not clear the existing WhatsApp session: {exc}",
                    file=sys.stderr,
                )
                return 1

        try:
            result = subprocess.run(
                [
                    "node",
                    str(config.bridge_script),
                    "--pair-only",
                    "--session",
                    str(config.session_dir),
                ],
                cwd=str(config.bridge_dir),
                env=build_bridge_environment(
                    allowed_senders=list(allowed_senders),
                    allowed_groups=list(
                        getattr(config, "group_allow_from", ())
                    ),
                ),
                check=False,
            )
        except (OSError, KeyboardInterrupt) as exc:
            if isinstance(exc, KeyboardInterrupt):
                print("Pairing cancelled.", file=sys.stderr)
            else:
                print(f"Could not start WhatsApp pairing: {exc}", file=sys.stderr)
            return 1
        if result.returncode != 0:
            print("WhatsApp pairing did not complete.", file=sys.stderr)
            return 1
        try:
            validate_whatsapp_session(config.session_dir)
        except WhatsAppSessionError as exc:
            print(
                f"WhatsApp pairing did not save usable linked-device credentials: {exc}",
                file=sys.stderr,
            )
            return 1
        if not _save_channel_enabled(selected_settings_path, "whatsapp", True):
            return 1
        print("WhatsApp is connected and its profile credentials are saved.")
        return 0
    finally:
        lock.release()


def _read_telegram_setup_input(prompt: str, *, secret: bool = False) -> str:
    try:
        reader = getpass.getpass if secret else input
        return reader(prompt).strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise _SetupCancelled from exc


def _prompt_telegram_token(updates: dict[str, str]) -> str:
    current_raw = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    current = ""
    if current_raw:
        try:
            current = normalize_telegram_bot_token(current_raw)
        except ValueError:
            print("Existing Telegram bot token is invalid.", file=sys.stderr)
    if current:
        print("Telegram bot token is configured.")
        if not _prompt_yes_no("Update the bot token? [y/N] "):
            return current

    while True:
        prompt = (
            "New Telegram bot token (hidden; blank keeps existing): "
            if current
            else "Telegram bot token from @BotFather (hidden): "
        )
        written = _read_telegram_setup_input(prompt, secret=True)
        if not written and current:
            return current
        try:
            token = normalize_telegram_bot_token(written)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            continue
        if token != current:
            updates["TELEGRAM_BOT_TOKEN"] = token
        return token


def _prompt_telegram_allowed_users(updates: dict[str, str]) -> tuple[str, ...]:
    current_raw = os.environ.get("TELEGRAM_ALLOWED_USERS", "").strip()
    current: tuple[str, ...] = ()
    if current_raw:
        try:
            current = normalize_telegram_allowed_users(current_raw)
        except ValueError as exc:
            print(f"Existing Telegram allowlist is invalid: {exc}", file=sys.stderr)
    if current:
        print("Allowed Telegram user IDs: " + ",".join(current))
        if not _prompt_yes_no("Update allowed user IDs? [y/N] "):
            return current
    else:
        print("Find your numeric Telegram user ID by messaging @userinfobot.")

    while True:
        prompt = (
            "Allowed user IDs, comma-separated (blank keeps existing): "
            if current
            else "Allowed Telegram user IDs, comma-separated: "
        )
        written = _read_telegram_setup_input(prompt)
        if not written and current:
            return current
        try:
            users = normalize_telegram_allowed_users(written)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            continue
        if users != current:
            updates["TELEGRAM_ALLOWED_USERS"] = ",".join(users)
        return users


def _prompt_telegram_home(
    allowed_users: tuple[str, ...], updates: dict[str, str]
) -> str:
    current_raw = os.environ.get("TELEGRAM_HOME_CHANNEL", "").strip()
    current = ""
    if current_raw:
        try:
            current = normalize_telegram_home_chat_id(current_raw)
        except ValueError as exc:
            print(f"Existing Telegram home chat is invalid: {exc}", file=sys.stderr)
    if current:
        print(f"Telegram home chat: {current}")
        if not _prompt_yes_no("Update the home chat? [y/N] "):
            return current

    while True:
        default = allowed_users[0]
        prompt = (
            "Home chat ID (blank keeps existing): "
            if current
            else f"Home Telegram chat ID [{default}]: "
        )
        written = _read_telegram_setup_input(prompt)
        if not written:
            if current:
                return current
            written = default
        try:
            home = normalize_telegram_home_chat_id(written)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            continue
        if home != current:
            updates["TELEGRAM_HOME_CHANNEL"] = home
        return home


def _prompt_telegram_topic(home: str, updates: dict[str, str]) -> str:
    current_raw = os.environ.get("TELEGRAM_HOME_CHANNEL_THREAD_ID", "").strip()
    if not home.startswith("-"):
        if current_raw:
            print("Clearing the Telegram home topic for a direct-message home.")
            updates["TELEGRAM_HOME_CHANNEL_THREAD_ID"] = ""
        return ""
    current = ""
    if current_raw:
        try:
            current = normalize_telegram_topic_id(current_raw)
        except ValueError as exc:
            print(f"Existing Telegram topic is invalid: {exc}", file=sys.stderr)
            while True:
                written = _read_telegram_setup_input(
                    "Home topic ID (blank clears it): "
                )
                try:
                    topic = normalize_telegram_topic_id(written)
                except ValueError as topic_exc:
                    print(topic_exc, file=sys.stderr)
                    continue
                updates["TELEGRAM_HOME_CHANNEL_THREAD_ID"] = topic
                return topic
    if current:
        print(f"Telegram home topic: {current}")
        if not _prompt_yes_no("Update or clear the home topic? [y/N] "):
            return current
        while True:
            written = _read_telegram_setup_input(
                "Home topic ID (blank clears it): "
            )
            try:
                topic = normalize_telegram_topic_id(written)
            except ValueError as exc:
                print(exc, file=sys.stderr)
                continue
            if topic != current:
                updates["TELEGRAM_HOME_CHANNEL_THREAD_ID"] = topic
            return topic
    if home.startswith("-"):
        while True:
            written = _read_telegram_setup_input(
                "Home topic ID (blank for no topic): "
            )
            try:
                topic = normalize_telegram_topic_id(written)
            except ValueError as exc:
                print(exc, file=sys.stderr)
                continue
            if topic:
                updates["TELEGRAM_HOME_CHANNEL_THREAD_ID"] = topic
            return topic
    return ""


def _prompt_telegram_configuration() -> tuple[str, dict[str, str]]:
    updates: dict[str, str] = {}
    token = _prompt_telegram_token(updates)
    allowed_users = _prompt_telegram_allowed_users(updates)
    home = _prompt_telegram_home(allowed_users, updates)
    _prompt_telegram_topic(home, updates)
    return token, updates


def _verify_telegram_bot_token(token: str) -> str:
    try:
        response = httpx.get(
            f"https://api.telegram.org/bot{token}/getMe",
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise ValueError("Could not verify the Telegram bot token.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Telegram returned an invalid verification response.")
    result = payload.get("result")
    if payload.get("ok") is not True or not isinstance(result, dict):
        raise ValueError("Telegram rejected the bot token.")
    username = str(result.get("username") or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_]{1,64}", username) is None:
        username = ""
    return f"@{username}" if username else "the configured bot"


def command_telegram_setup(
    state_dir: Path,
    *,
    env_path: Path | None = None,
    settings_path: Path | None = None,
    external_env: frozenset[str] = frozenset(),
) -> int:
    """Configure, verify, and enable Telegram for one profile."""

    lock = ProfileRuntimeLock(state_dir)
    try:
        lock.acquire()
    except RuntimeLockError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        try:
            token, updates = _prompt_telegram_configuration()
        except _SetupCancelled:
            print("Telegram setup cancelled.", file=sys.stderr)
            return 1

        externally_managed = sorted(updates.keys() & external_env)
        if externally_managed:
            print(
                "Cannot update externally supplied environment value(s): "
                + ", ".join(externally_managed)
                + ". Change them at their deployment source.",
                file=sys.stderr,
            )
            return 1
        try:
            bot_name = _verify_telegram_bot_token(token)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
        if updates:
            try:
                update_env_values(env_path or state_dir / ".env", updates)
            except (OSError, ValueError) as exc:
                print(f"Could not save Telegram configuration: {exc}", file=sys.stderr)
                return 1
        selected_settings_path = settings_path or config_path(state_dir)
        if not _save_channel_enabled(selected_settings_path, "telegram", True):
            return 1
        print(f"Telegram {bot_name} is verified, configured, and enabled.")
        return 0
    finally:
        lock.release()


def _read_setup_input(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise _SetupCancelled from exc


def _prompt_yes_no(prompt: str) -> bool:
    try:
        return input(prompt).strip().lower() in {"y", "yes"}
    except (EOFError, KeyboardInterrupt) as exc:
        raise _SetupCancelled from exc


def _normalize_allowed_senders(value: str) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value.replace(";", ",").split(","):
        written = item.strip()
        if not written:
            continue
        domain = ""
        if "@" in written:
            _, _, domain = written.partition("@")
            domain = domain.lower()
            if domain not in {"s.whatsapp.net", "lid", "c.us"}:
                raise ValueError(f"Invalid WhatsApp sender ID: {written!r}")
        local = written.split("@", 1)[0].split(":", 1)[0]
        digits = re.sub(r"[ \t().-]+", "", local).removeprefix("+")
        if not digits.isascii() or not digits.isdigit():
            raise ValueError(
                f"Invalid WhatsApp number {written!r}; use the country code and digits"
            )
        if digits not in seen:
            normalized.append(f"{digits}@lid" if domain == "lid" else digits)
            seen.add(digits)
    if not normalized:
        raise ValueError("At least one allowed WhatsApp number is required")
    return tuple(normalized)


_WhatsAppValue = TypeVar("_WhatsAppValue")


def _prompt_required_whatsapp_value(
    prompt: str,
    normalize: Callable[[str], _WhatsAppValue],
    *,
    existing: _WhatsAppValue | None = None,
    default: str = "",
) -> tuple[_WhatsAppValue, bool]:
    while True:
        written = _read_setup_input(prompt)
        if not written:
            if existing is not None:
                return existing, False
            written = default
        try:
            return normalize(written), True
        except ValueError as exc:
            print(exc, file=sys.stderr)


def _prompt_whatsapp_configuration(
    config: Config,
) -> tuple[tuple[str, ...], dict[str, str]]:
    updates: dict[str, str] = {}
    current_senders = tuple(
        sorted(str(value) for value in getattr(config, "allowed_senders", ()))
    )
    senders_valid = bool(current_senders)
    if current_senders:
        try:
            _normalize_allowed_senders(",".join(current_senders))
        except ValueError as exc:
            print(f"Existing WhatsApp allowlist is invalid: {exc}", file=sys.stderr)
            senders_valid = False
    if senders_valid:
        current = ",".join(current_senders)
        print(f"Allowed WhatsApp numbers: {current}")
        if _prompt_yes_no("Update allowed numbers? [y/N] "):
            allowed_senders, changed = _prompt_required_whatsapp_value(
                "Allowed numbers, comma-separated (blank keeps existing): ",
                _normalize_allowed_senders,
                existing=current_senders,
            )
            if changed:
                updates["PILOTAGE_ALLOWED_SENDERS"] = ",".join(allowed_senders)
        else:
            allowed_senders = current_senders
    else:
        allowed_senders, _ = _prompt_required_whatsapp_value(
            "Allowed WhatsApp numbers, comma-separated: ",
            _normalize_allowed_senders,
        )
        updates["PILOTAGE_ALLOWED_SENDERS"] = ",".join(allowed_senders)

    current_home = str(getattr(config, "home_chat_id", "") or "").strip()
    home_valid = bool(current_home)
    if current_home:
        try:
            normalize_whatsapp_chat_id(current_home)
        except ValueError as exc:
            print(f"Existing WhatsApp home chat is invalid: {exc}", file=sys.stderr)
            home_valid = False
    if home_valid:
        print(f"WhatsApp home chat: {current_home}")
        if _prompt_yes_no("Update the home number/chat? [y/N] "):
            home_chat_id, changed = _prompt_required_whatsapp_value(
                "Home number/chat ID (blank keeps existing): ",
                normalize_whatsapp_chat_id,
                existing=current_home,
            )
            if changed:
                updates["WHATSAPP_HOME_CHANNEL"] = home_chat_id
    else:
        suggested_home = allowed_senders[0]
        home_chat_id, _ = _prompt_required_whatsapp_value(
            f"Home WhatsApp number/chat ID [{suggested_home}]: ",
            normalize_whatsapp_chat_id,
            default=suggested_home,
        )
        updates["WHATSAPP_HOME_CHANNEL"] = home_chat_id

    return allowed_senders, updates


async def _command_run_locked(config: Config, profile_name: str = "default") -> int:
    return await _run_enabled_channels(config, profile_name)


async def _run_enabled_channels(
    config: Config, profile_name: str
) -> int:
    """Run the enabled first-class messaging channels for one profile."""
    whatsapp_enabled = config.settings.flag("whatsapp.enabled", False)
    telegram_config = config.for_channel("telegram")
    telegram_enabled = telegram_config.settings.flag("telegram.enabled", False)
    if not whatsapp_enabled and not telegram_enabled:
        print("No messaging channel is enabled.", file=sys.stderr)
        return 1

    cron_config = config if whatsapp_enabled else telegram_config
    cron_channel_configs = {}
    if whatsapp_enabled:
        cron_channel_configs["whatsapp"] = config
    if telegram_enabled:
        cron_channel_configs["telegram"] = telegram_config
    cron_store = CronStore(
        cron_config.state_dir,
        timezone_name=cron_config.cron_timezone,
        claim_ttl_seconds=cron_config.cron_claim_ttl_seconds,
        output_retention=cron_config.cron_output_retention,
    )
    conversation_store = ConversationStore(cron_config.conversations_path)
    delivery_store = DeliveryStore(cron_config.state_dir / "delivery.db")
    channels = {}
    agents: list[Agent] = []

    async def scheduled_delivery(
        origin: dict[str, str], text: str
    ) -> None:
        channel_name = str(origin.get("channel") or "").lower()
        delivery_channel = channels.get(channel_name)
        chat_id = str(origin.get("chat_id") or "")
        if delivery_channel is None:
            raise ChannelError(
                f"{channel_name or 'Unknown'} delivery adapter is unavailable."
            )
        if not chat_id:
            raise ChannelError("Cron delivery origin has no chat ID.")
        if channel_name == "telegram":
            delivered = await delivery_channel.send(
                chat_id,
                text,
                thread_id=str(origin.get("thread_id") or ""),
            )
        else:
            delivered = await delivery_channel.send(chat_id, text)
        if not delivered:
            raise ChannelError(
                f"{channel_name.title()} rejected the scheduled delivery."
            )

    scheduler = (
        CronScheduler(
            cron_config,
            cron_store,
            deliver=scheduled_delivery,
            channel_configs=cron_channel_configs,
        )
        if cron_config.cron_enabled
        else None
    )
    cron_wake = scheduler.wake if scheduler is not None else None

    if whatsapp_enabled:
        whatsapp_failure_reply = t("runtime.failure", config.language)
        whatsapp_reset_reply = t("runtime.reset", config.language)
        whatsapp_agent = Agent(
            config,
            store=conversation_store,
            cron_store=cron_store,
            cron_wake=cron_wake,
        )
        agents.append(whatsapp_agent)

        async def handle_whatsapp(message: InboundMessage) -> None:
            quoted = message.message_ids[-1] if message.message_ids else ""

            async def notice(text: str) -> None:
                await whatsapp_channel.send(message.chat_id, text, quoted)

            async def approval_notice(text: str) -> None:
                delivered = await whatsapp_channel.send(
                    message.chat_id, text, quoted
                )
                if not delivered:
                    raise ChannelError(
                        "WhatsApp rejected the approval request delivery."
                    )

            try:
                async with whatsapp_channel.typing(message.chat_id):
                    enriched_text, transcripts = await transcription.enrich_message(
                        message.text,
                        message.attachments,
                        config.settings,
                    )
                    logger.info(
                        "WhatsApp inbound (%d chars, %d attachments)",
                        len(enriched_text),
                        len(message.attachments),
                    )
                    if transcripts and transcription.transcript_echo_enabled(
                        config.settings
                    ):
                        for transcript in transcripts:
                            await whatsapp_channel.send(
                                message.chat_id,
                                f'\U0001f399\ufe0f "{transcript}"',
                                quoted,
                                deliver_media=False,
                            )
                    answer = await whatsapp_agent.respond(
                        message.session_id,
                        enriched_text,
                        message.attachments,
                        on_notice=notice,
                        origin={
                            "channel": "whatsapp",
                            "chat_id": message.chat_id,
                        },
                        approval_notify=approval_notice,
                    )
            except ConversationError:
                logger.exception("WhatsApp conversation persistence failed")
                answer = t("runtime.storage_failure", config.language)
            except Exception:
                logger.exception("The WhatsApp model call failed")
                answer = whatsapp_failure_reply
            final_text = answer or whatsapp_failure_reply
            delivered = await deliver_final(
                delivery_store,
                session_key=message.session_id,
                message_ref=quoted,
                platform="whatsapp",
                chat_id=message.chat_id,
                thread_id="",
                content=final_text,
                send=lambda: whatsapp_channel.send(
                    message.chat_id,
                    final_text,
                    quoted,
                ),
            )
            if not delivered:
                logger.error(
                    "WhatsApp final response remains pending for %s",
                    message.chat_id,
                )

        async def manage_whatsapp(
            chat_id: str,
            session_id: str,
            message_id: str,
            invocation: CommandInvocation,
        ) -> None:
            try:
                answer = await execute_command(
                    invocation,
                    agent=whatsapp_agent,
                    config=config,
                    profile_name=profile_name,
                    session_id=session_id,
                    reset_reply=whatsapp_reset_reply,
                )
            except ConversationError:
                logger.exception("WhatsApp conversation reset persistence failed")
                answer = t("runtime.storage_failure", config.language)
            except Exception:
                logger.exception("WhatsApp management command failed")
                answer = whatsapp_failure_reply
            await whatsapp_channel.send(chat_id, answer, message_id)

        whatsapp_channel = WhatsAppChannel(
            config,
            handle_whatsapp,
            manage_whatsapp,
        )
        channels["whatsapp"] = whatsapp_channel

    if telegram_enabled:
        telegram_failure_reply = t(
            "runtime.failure", telegram_config.language
        )
        telegram_reset_reply = t("runtime.reset", telegram_config.language)
        telegram_agent = Agent(
            telegram_config,
            store=conversation_store,
            cron_store=cron_store,
            cron_wake=cron_wake,
        )
        agents.append(telegram_agent)

        async def handle_telegram(message: TelegramInboundMessage) -> None:
            quoted = message.message_ids[-1] if message.message_ids else ""

            async def notice(text: str) -> None:
                await telegram_channel.send(
                    message.chat_id,
                    text,
                    quoted,
                    thread_id=message.thread_id,
                )

            async def approval_notice(text: str) -> None:
                delivered = await telegram_channel.send(
                    message.chat_id,
                    text,
                    quoted,
                    thread_id=message.thread_id,
                )
                if not delivered:
                    raise TelegramChannelError(
                        "Telegram rejected the approval request delivery."
                    )

            try:
                async with telegram_channel.typing(
                    message.chat_id, message.thread_id
                ):
                    enriched_text, transcripts = await transcription.enrich_message(
                        message.text,
                        message.attachments,
                        telegram_config.settings,
                    )
                    logger.info(
                        "Telegram inbound (%d chars, %d attachments)",
                        len(enriched_text),
                        len(message.attachments),
                    )
                    if transcripts and transcription.transcript_echo_enabled(
                        telegram_config.settings
                    ):
                        for transcript in transcripts:
                            await telegram_channel.send(
                                message.chat_id,
                                f'\U0001f399\ufe0f "{transcript}"',
                                quoted,
                                thread_id=message.thread_id,
                                deliver_media=False,
                            )
                    origin = {
                        "channel": "telegram",
                        "chat_id": message.chat_id,
                    }
                    if message.thread_id:
                        origin["thread_id"] = message.thread_id
                    answer = await telegram_agent.respond(
                        message.session_id,
                        enriched_text,
                        message.attachments,
                        on_notice=notice,
                        origin=origin,
                        approval_notify=approval_notice,
                    )
            except ConversationError:
                logger.exception("Telegram conversation persistence failed")
                answer = t(
                    "runtime.storage_failure",
                    telegram_config.language,
                )
            except Exception:
                logger.exception("The Telegram model call failed")
                answer = telegram_failure_reply
            final_text = answer or telegram_failure_reply
            delivered = await deliver_final(
                delivery_store,
                session_key=message.session_id,
                message_ref=quoted,
                platform="telegram",
                chat_id=message.chat_id,
                thread_id=message.thread_id,
                content=final_text,
                send=lambda: telegram_channel.send(
                    message.chat_id,
                    final_text,
                    quoted,
                    thread_id=message.thread_id,
                ),
            )
            if not delivered:
                logger.error(
                    "Telegram final response remains pending for %s",
                    message.chat_id,
                )

        async def manage_telegram(
            chat_id: str,
            session_id: str,
            message_id: str,
            thread_id: str,
            invocation: CommandInvocation,
        ) -> None:
            try:
                answer = await execute_command(
                    invocation,
                    agent=telegram_agent,
                    config=telegram_config,
                    profile_name=profile_name,
                    session_id=session_id,
                    reset_reply=telegram_reset_reply,
                )
            except ConversationError:
                logger.exception("Telegram conversation reset persistence failed")
                answer = t(
                    "runtime.storage_failure",
                    telegram_config.language,
                )
            except Exception:
                logger.exception("Telegram management command failed")
                answer = telegram_failure_reply
            await telegram_channel.send(
                chat_id,
                answer,
                message_id,
                thread_id=thread_id,
            )

        telegram_channel = TelegramChannel(
            telegram_config,
            handle_telegram,
            manage_telegram,
        )
        channels["telegram"] = telegram_channel

    try:
        auth.read_credentials(
            cron_config.credentials_path,
            fallback_path=cron_config.main_credentials_path,
        )
    except auth.AuthError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    start_order = [
        channel
        for name in ("telegram", "whatsapp")
        if (channel := channels.get(name)) is not None
    ]
    started = []
    recovery_task: asyncio.Task[None] | None = None
    try:
        for running_channel in start_order:
            hold_inbound = getattr(running_channel, "hold_inbound", None)
            if callable(hold_inbound):
                hold_inbound()
        for running_channel in start_order:
            await running_channel.start()
            started.append(running_channel)
    except (ChannelError, TelegramChannelError) as exc:
        for running_channel in reversed(started):
            await running_channel.stop()
        await asyncio.gather(
            *(agent.close() for agent in agents),
            return_exceptions=True,
        )
        print(f"{exc}", file=sys.stderr)
        return 1

    claimed_rows = await claim_deliveries(delivery_store, set(channels))
    if claimed_rows:
        async def recover_final_responses() -> None:
            recovered = await redeliver_claimed_deliveries(
                delivery_store,
                channels,
                claimed_rows,
            )
            if recovered:
                logger.info(
                    "Redelivered %d recovered final response(s)",
                    recovered,
                )

        recovery_task = asyncio.create_task(
            recover_final_responses(),
            name="pilotage-delivery-recovery",
        )
        _, pending = await asyncio.wait(
            {recovery_task},
            timeout=STARTUP_RECOVERY_DRAIN_SECONDS,
        )
        if pending:
            logger.info(
                "Releasing inbound while startup delivery recovery continues"
                " in background"
            )
        else:
            await recovery_task

    for running_channel in started:
        release_inbound = getattr(running_channel, "release_inbound", None)
        if callable(release_inbound):
            released = release_inbound()
            if inspect.isawaitable(released):
                await released

    if scheduler is not None:
        try:
            await scheduler.start()
        except (CronError, OSError) as exc:
            if recovery_task is not None and not recovery_task.done():
                recovery_task.cancel()
            if recovery_task is not None:
                await asyncio.gather(
                    recovery_task,
                    return_exceptions=True,
                )
            for running_channel in reversed(started):
                await running_channel.stop()
            await asyncio.gather(
                *(agent.close() for agent in agents),
                return_exceptions=True,
            )
            print(f"Cron scheduler could not start: {exc}", file=sys.stderr)
            return 1

    session_workspace_roots = []
    for agent in agents:
        workspace_root = getattr(agent, "session_workspace_root", None)
        if (
            workspace_root is not None
            and workspace_root not in session_workspace_roots
        ):
            session_workspace_roots.append(workspace_root)

    async def maintain_profile_state() -> None:
        while True:
            if getattr(cron_config, "session_auto_prune", False):
                try:
                    removed = await asyncio.to_thread(
                        conversation_store.prune_old_sessions,
                        cron_config.session_retention_days,
                        workspace_roots=session_workspace_roots,
                    )
                    if removed:
                        logger.info("Pruned %d old conversation session(s)", removed)
                except Exception:
                    logger.exception("Automatic conversation pruning failed")
            try:
                removed_media = await asyncio.to_thread(
                    media.cleanup_cache,
                    cron_config.media_dir,
                )
                if removed_media:
                    logger.info(
                        "Pruned %d stale inbound media file(s)",
                        removed_media,
                    )
            except Exception:
                logger.exception("Inbound media cache pruning failed")
            await asyncio.sleep(SESSION_MAINTENANCE_INTERVAL_SECONDS)

    maintenance_task = asyncio.create_task(
        maintain_profile_state(),
        name="pilotage-profile-maintenance",
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            signal.signal(sig, lambda *_: stop.set())

    channel_labels = {"whatsapp": "WhatsApp", "telegram": "Telegram"}
    logger.info(
        "Listening on %s. Ctrl+C to stop.",
        " and ".join(channel_labels[name] for name in channels),
    )
    waiters = [asyncio.create_task(stop.wait())]
    waiters.extend(
        asyncio.create_task(running_channel.stopped.wait())
        for running_channel in start_order
    )
    if scheduler is not None:
        waiters.append(asyncio.create_task(scheduler.stopped.wait()))
    try:
        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
    except KeyboardInterrupt:
        pass
    finally:
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)
        if maintenance_task is not None:
            maintenance_task.cancel()
            await asyncio.gather(maintenance_task, return_exceptions=True)
        if recovery_task is not None:
            recovery_task.cancel()
            await asyncio.gather(recovery_task, return_exceptions=True)

        # Refuse new work first. Accepted batches are flushed now, so channel
        # turns can finish while an in-flight cron job uses the same transport.
        intake_results = await asyncio.gather(
            *(running_channel.stop_intake() for running_channel in reversed(started)),
            return_exceptions=True,
        )
        for result in intake_results:
            if isinstance(result, BaseException):
                logger.error("Stopping channel intake failed: %s", result)

        deadline = loop.time() + SHUTDOWN_DRAIN_SECONDS
        if scheduler is not None:
            await scheduler.stop(
                drain_timeout_seconds=max(0.0, deadline - loop.time())
            )
        channel_results = await asyncio.gather(
            *(
                running_channel.stop(
                    drain_timeout_seconds=max(0.0, deadline - loop.time())
                )
                for running_channel in reversed(started)
            ),
            return_exceptions=True,
        )
        for result in channel_results:
            if isinstance(result, BaseException):
                logger.error("Stopping a messaging channel failed: %s", result)
        await asyncio.gather(
            *(agent.close() for agent in agents),
            return_exceptions=True,
        )

    if scheduler is not None and scheduler.failure:
        print(scheduler.failure, file=sys.stderr)
        return 1
    for running_channel in start_order:
        if running_channel.failure:
            print(running_channel.failure, file=sys.stderr)
            return 1
    return 0

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pilotage", description="Pilotage Agent")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--profile", help="run one named isolated agent profile")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("login", help="authenticate against ChatGPT")
    ask = subparsers.add_parser("ask", help="ask one question, print the answer")
    ask.add_argument("question", nargs="+")
    subparsers.add_parser("run", help="answer enabled messaging channels until stopped")
    subparsers.add_parser(
        "whatsapp", help="configure allowed numbers, home chat, and QR pairing"
    )
    subparsers.add_parser(
        "telegram", help="configure allowed users, home chat, and bot token"
    )
    subparsers.add_parser("status", help="show the selected agent's essential status")
    subparsers.add_parser(
        "doctor",
        help="run the complete read-only deployment readiness check",
    )
    service = subparsers.add_parser("service", help="control the installed user service")
    service.add_argument("service_action", choices=("start", "stop", "status"))
    add_cron_parser(subparsers)

    profile = subparsers.add_parser("profile", help="manage isolated agent profiles")
    profile_commands = profile.add_subparsers(dest="profile_command", required=True)
    profile_commands.add_parser("list", help="list profiles")
    profile_commands.add_parser("show", help="show the active profile")
    create = profile_commands.add_parser("create", help="create a fresh profile")
    create.add_argument("name")
    use = profile_commands.add_parser("use", help="make a profile the default")
    use.add_argument("name")
    delete = profile_commands.add_parser("delete", help="delete a named profile")
    delete.add_argument("name")
    delete.add_argument("--yes", action="store_true", help="skip typed confirmation")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    if args.command == "profile":
        return _command_profile(args)

    try:
        profile_name, profile_path = profiles.activate_for_process(args.profile)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if args.command == "service":
        return run_service_command(args.service_action, profile_name)

    external_setup_env = frozenset(
        name for name in CHANNEL_SETUP_ENV_KEYS if name in os.environ
    )
    loaded_env_files = load_env_files()
    for path in loaded_env_files:
        logger.info("Read environment from %s", path)
    setup_env_path = (
        loaded_env_files[0] if loaded_env_files else candidate_env_files()[0]
    )
    selected_settings_path = config_path(profile_path)
    if args.command == "telegram":
        return command_telegram_setup(
            profile_path,
            env_path=setup_env_path,
            settings_path=selected_settings_path,
            external_env=external_setup_env,
        )
    try:
        # Parse the exact view that will run while still inside the guarded
        # startup boundary. A malformed channel override must be a clean
        # startup error, not a traceback after the common config passed.
        channel = (
            "whatsapp"
            if args.command in {"run", "status", "doctor"}
            else ""
        )
        config = Config.load(channel=channel)
        if args.command == "status":
            telegram_config = config.for_channel("telegram")
            if (
                not config.settings.flag("whatsapp.enabled", False)
                and telegram_config.settings.flag("telegram.enabled", False)
            ):
                config = telegram_config
    except ConfigError as exc:
        # A broken configuration file stops the agent rather than starting it
        # with defaults: a default can silently re-enable what was switched off.
        logger.error("%s", exc)
        return 1

    if args.command == "login":
        return command_login(config)
    if args.command == "whatsapp":
        return command_whatsapp_pair(
            config,
            env_path=setup_env_path,
            settings_path=selected_settings_path,
            external_env=external_setup_env,
        )
    if args.command == "ask":
        return asyncio.run(command_ask(config, " ".join(args.question)))
    if args.command == "status":
        return command_status(config, profile_name)
    if args.command == "doctor":
        from .doctor import run_doctor

        return asyncio.run(run_doctor(config, profile_name))
    if args.command == "cron":
        return run_cron_command(args, config)
    return asyncio.run(command_run(config, profile_name))


def _command_profile(args: argparse.Namespace) -> int:
    try:
        if args.profile_command == "list":
            for info in profiles.list_profiles():
                marker = "*" if info.is_active else " "
                print(f"{marker} {info.name}\t{info.path}")
            return 0
        if args.profile_command == "show":
            name = profiles.get_active_profile()
            print(f"{name}\t{profiles.get_profile_dir(name)}")
            return 0
        if args.profile_command == "create":
            path = profiles.create_profile(args.name)
            print(f"Created profile at {path}")
            return 0
        if args.profile_command == "use":
            profiles.set_active_profile(args.name)
            print(f"Active profile: {profiles.normalize_profile_name(args.name)}")
            return 0

        canon = profiles.normalize_profile_name(args.name)
        if not args.yes:
            try:
                confirmation = input(f"Type '{canon}' to delete this profile: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("Cancelled.")
                return 1
            if confirmation != canon:
                print("Cancelled.")
                return 1
        path = profiles.delete_profile(canon)
        print(f"Deleted profile at {path}")
        return 0
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
