"""Command line entry point.

    pilotage login          authenticate against ChatGPT (device code)
    pilotage ask "..."      one question straight to the model, no WhatsApp
    pilotage run            answer WhatsApp messages until stopped
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from . import profiles
from .agent import Agent
from .commands import CommandInvocation, execute_command, status_text
from .channels.whatsapp import ChannelError, InboundMessage, WhatsAppChannel
from .codex import auth
from .config import Config, ConfigError
from .cron.cli import add_cron_parser, run_cron_command
from .cron.jobs import CronError, CronStore
from .cron.scheduler import CronScheduler
from .env import load_env_files
from .history import ConversationStore
from .runtime_lock import ProfileRuntimeLock, RuntimeLockError

logger = logging.getLogger("pilotage")

# The only two things the agent says on its own. Written in the language its
# users write in — the model already follows the conversation by itself.
REPLY_ON_FAILURE = "Je n'ai pas pu répondre pour le moment. Réessayez."
REPLY_ON_RESET = "On repart de zéro. J'ai oublié notre conversation."


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
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
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


async def _command_run_locked(config: Config, profile_name: str = "default") -> int:
    channel: WhatsAppChannel
    cron_store = CronStore(
        config.state_dir,
        timezone_name=config.cron_timezone,
        claim_ttl_seconds=config.cron_claim_ttl_seconds,
        output_retention=config.cron_output_retention,
    )

    async def scheduled_delivery(origin: dict[str, str], text: str) -> None:
        await _deliver_scheduled(channel, origin, text)

    scheduler = (
        CronScheduler(config, cron_store, deliver=scheduled_delivery)
        if config.cron_enabled
        else None
    )
    agent = Agent(
        config,
        cron_store=cron_store,
        cron_wake=scheduler.wake if scheduler is not None else None,
    )

    async def handle(message: InboundMessage) -> None:
        logger.info("%s: %s", message.sender_number or message.chat_id, message.text[:120])
        # A batch of messages is one question; quote the last of them, the
        # one the answer arrives under.
        quoted = message.message_ids[-1] if message.message_ids else ""

        async def notice(text: str) -> None:
            await channel.send(message.chat_id, text, quoted)

        try:
            async with channel.typing(message.chat_id):
                answer = await agent.respond(
                    message.session_id,
                    message.text,
                    message.attachments,
                    on_notice=notice,
                    origin={"channel": "whatsapp", "chat_id": message.chat_id},
                )
        except Exception:  # noqa: BLE001 - the user gets an answer either way
            logger.exception("The model call failed")
            answer = REPLY_ON_FAILURE
        if not answer:
            answer = REPLY_ON_FAILURE
        await channel.send(message.chat_id, answer, quoted)

    async def manage(
        chat_id: str,
        session_id: str,
        message_id: str,
        invocation: CommandInvocation,
    ) -> None:
        try:
            answer = await execute_command(
                invocation,
                agent=agent,
                config=config,
                profile_name=profile_name,
                session_id=session_id,
                reset_reply=REPLY_ON_RESET,
            )
        except Exception:  # noqa: BLE001 - commands must always answer
            logger.exception("Management command failed")
            answer = REPLY_ON_FAILURE
        await channel.send(chat_id, answer, message_id)

    channel = WhatsAppChannel(config, handle, manage)

    # Fail before starting the bridge if we are not signed in.
    try:
        auth.read_credentials(
            config.credentials_path,
            fallback_path=config.main_credentials_path,
        )
    except auth.AuthError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    try:
        await channel.start()
    except ChannelError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    if scheduler is not None:
        try:
            await scheduler.start()
        except (CronError, OSError) as exc:
            await channel.stop()
            print(f"Cron scheduler could not start: {exc}", file=sys.stderr)
            return 1

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows: fall back to KeyboardInterrupt below.
            signal.signal(sig, lambda *_: stop.set())

    logger.info("Listening. Ctrl+C to stop.")
    waiters = [asyncio.create_task(stop.wait()), asyncio.create_task(channel.stopped.wait())]
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
        if scheduler is not None:
            await scheduler.stop()
        await channel.stop()

    if scheduler is not None and scheduler.failure:
        print(scheduler.failure, file=sys.stderr)
        return 1
    if channel.failure:
        print(channel.failure, file=sys.stderr)
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
    subparsers.add_parser("run", help="answer WhatsApp messages until stopped")
    subparsers.add_parser("status", help="show the selected agent's essential status")
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

    for path in load_env_files():
        logger.info("Read environment from %s", path)
    try:
        # Parse the exact view that will run while still inside the guarded
        # startup boundary. A malformed channel override must be a clean
        # startup error, not a traceback after the common config passed.
        channel = "whatsapp" if args.command in {"run", "status"} else ""
        config = Config.load(channel=channel)
    except ConfigError as exc:
        # A broken configuration file stops the agent rather than starting it
        # with defaults: a default can silently re-enable what was switched off.
        logger.error("%s", exc)
        return 1

    if args.command == "login":
        return command_login(config)
    if args.command == "ask":
        return asyncio.run(command_ask(config, " ".join(args.question)))
    if args.command == "status":
        return command_status(config, profile_name)
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
