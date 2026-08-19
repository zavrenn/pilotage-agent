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

from .agent import Agent
from .channels.whatsapp import ChannelError, InboundMessage, WhatsAppChannel
from .codex import auth
from .config import Config
from .env import load_env_files
from .history import ConversationStore

logger = logging.getLogger("pilotage")

# The only two things the agent says on its own. Written in the language its
# users write in — the model already follows the conversation by itself.
REPLY_ON_FAILURE = "Je n'ai pas pu répondre pour le moment. Réessayez."
REPLY_ON_RESET = "On repart de zéro. J'ai oublié notre conversation."


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


async def command_run(config: Config) -> int:
    agent = Agent(config)
    channel: WhatsAppChannel

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
                    message.chat_id, message.text, message.attachments, on_notice=notice
                )
        except Exception:  # noqa: BLE001 - the user gets an answer either way
            logger.exception("The model call failed")
            answer = REPLY_ON_FAILURE
        if not answer:
            answer = REPLY_ON_FAILURE
        await channel.send(message.chat_id, answer, quoted)

    async def reset(chat_id: str, message_id: str) -> None:
        await agent.forget(chat_id)
        await channel.send(chat_id, REPLY_ON_RESET, message_id)

    channel = WhatsAppChannel(config, handle, reset)

    # Fail before starting the bridge if we are not signed in.
    try:
        auth.read_credentials(config.credentials_path)
    except auth.AuthError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    try:
        await channel.start()
    except ChannelError as exc:
        print(f"{exc}", file=sys.stderr)
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
    try:
        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
    except KeyboardInterrupt:
        pass
    finally:
        for waiter in waiters:
            waiter.cancel()
        await channel.stop()

    if channel.failure:
        print(channel.failure, file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pilotage", description="Pilotage Agent")
    parser.add_argument("-v", "--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("login", help="authenticate against ChatGPT")
    ask = subparsers.add_parser("ask", help="ask one question, print the answer")
    ask.add_argument("question", nargs="+")
    subparsers.add_parser("run", help="answer WhatsApp messages until stopped")

    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    for path in load_env_files():
        logger.info("Read settings from %s", path)
    config = Config.load()

    if args.command == "login":
        return command_login(config)
    if args.command == "ask":
        return asyncio.run(command_ask(config, " ".join(args.question)))
    return asyncio.run(command_run(config))


if __name__ == "__main__":
    raise SystemExit(main())
