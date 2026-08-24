"""Command line entry point.

    pilotage login          authenticate against ChatGPT (device code)
    pilotage ask "..."      one question straight to the model, no messaging
    pilotage run            answer enabled messaging channels until stopped
    pilotage doctor         prove deployment readiness
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import shutil
import subprocess
import sys

from . import media, profiles, transcription
from .agent import Agent
from .commands import CommandInvocation, execute_command, status_text
from .channels.whatsapp import (
    ChannelError,
    InboundMessage,
    WhatsAppChannel,
    build_bridge_environment,
)
from .channels.telegram import (
    ChannelError as TelegramChannelError,
    InboundMessage as TelegramInboundMessage,
    TelegramChannel,
)
from .codex import auth
from .config import Config, ConfigError
from .cron.cli import add_cron_parser, run_cron_command
from .cron.jobs import CronError, CronStore
from .cron.scheduler import CronScheduler
from .env import load_env_files
from .history import ConversationStore
from .i18n import t
from .redact import RedactingFormatter
from .runtime_lock import ProfileRuntimeLock, RuntimeLockError
from .service import run_service_command

logger = logging.getLogger("pilotage")

SESSION_MAINTENANCE_INTERVAL_SECONDS = 3600
# Current Hermes gives in-flight cron work 30 seconds. Pilotage gives the same
# bounded window to every already accepted task because it does not carry
# Hermes' much larger interrupted-turn resume subsystem.
SHUTDOWN_DRAIN_SECONDS = 30.0


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


def command_whatsapp_pair(config: Config) -> int:
    """Run Hermes' pair-only bridge flow without starting the agent runtime."""

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
        config.session_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
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
                    allowed_senders=list(
                        getattr(config, "allowed_senders", ())
                    ),
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
        if not (config.session_dir / "creds.json").is_file():
            print("WhatsApp pairing exited without saved credentials.", file=sys.stderr)
            return 1
        print("WhatsApp is connected and its profile credentials are saved.")
        return 0
    finally:
        lock.release()


async def _command_run_locked(config: Config, profile_name: str = "default") -> int:
    return await _run_enabled_channels(config, profile_name)


async def _run_enabled_channels(
    config: Config, profile_name: str
) -> int:
    """Run the enabled first-class messaging channels for one profile."""
    whatsapp_enabled = config.settings.flag("whatsapp.enabled", True)
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
            except Exception:
                logger.exception("The WhatsApp model call failed")
                answer = whatsapp_failure_reply
            await whatsapp_channel.send(
                message.chat_id,
                answer or whatsapp_failure_reply,
                quoted,
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
            except Exception:
                logger.exception("The Telegram model call failed")
                answer = telegram_failure_reply
            await telegram_channel.send(
                message.chat_id,
                answer or telegram_failure_reply,
                quoted,
                thread_id=message.thread_id,
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
    try:
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

    if scheduler is not None:
        try:
            await scheduler.start()
        except (CronError, OSError) as exc:
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
    subparsers.add_parser("whatsapp", help="connect or verify WhatsApp by QR pairing")
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
        profile_name, _ = profiles.activate_for_process(args.profile)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if args.command == "service":
        return run_service_command(args.service_action, profile_name)

    for path in load_env_files():
        logger.info("Read environment from %s", path)
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
                not config.settings.flag("whatsapp.enabled", True)
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
        return command_whatsapp_pair(config)
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
