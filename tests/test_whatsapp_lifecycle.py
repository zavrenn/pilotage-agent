from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from pilotage.channels import whatsapp
from pilotage.channels.whatsapp import ChannelError, InboundMessage, WhatsAppChannel
from pilotage.config import Config


async def _handle(_message: InboundMessage) -> None:
    raise AssertionError("no message should run")


async def _command(_chat_id, _session_id, _message_id, _invocation) -> None:
    raise AssertionError("no command should run")


class BridgeOwnershipTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)
        self.channel = WhatsAppChannel(Config.load(), _handle, _command)
        object.__setattr__(
            self.channel._config,
            "allowed_senders",
            frozenset({"212600000000"}),
        )

    def test_spawn_passes_and_persists_the_private_instance_token(self):
        process = mock.Mock(pid=4321)
        with mock.patch.object(whatsapp.subprocess, "Popen", return_value=process) as popen:
            self.channel._spawn_bridge()

        command = popen.call_args.args[0]
        self.assertNotIn("--instance-token", command)
        self.assertEqual(
            popen.call_args.kwargs["env"]["PILOTAGE_BRIDGE_TOKEN"],
            self.channel._bridge_token,
        )
        self.assertEqual(
            popen.call_args.kwargs["env"]["PILOTAGE_ALLOWED_SENDERS"],
            "212600000000",
        )
        self.assertEqual(
            popen.call_args.kwargs["env"]["PILOTAGE_ALLOWED_GROUPS"],
            "",
        )
        self.assertNotIn("212600000000", command)
        group_flag = command.index("--answer-groups")
        self.assertEqual(command[group_flag + 1], "0")
        record = json.loads(self.channel._pidfile.read_text(encoding="utf-8"))
        self.assertEqual(
            record,
            {
                "pid": 4321,
                "port": self.channel._config.bridge_port,
                "token": self.channel._bridge_token,
            },
        )

    def test_spawn_does_not_inherit_agent_secrets(self):
        process = mock.Mock(pid=4321)
        environment = {
            "PATH": "node-path",
            "LANG": "en_US.UTF-8",
            "PILOTAGE_CODEX_BASE_URL": "sentinel-codex",
            "VOICE_TOOLS_OPENAI_KEY": "sentinel-openai",
            "DATABASE_PASSWORD": "sentinel-database",
            "TELEGRAM_BOT_TOKEN": "sentinel-telegram",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(
                whatsapp.subprocess,
                "Popen",
                return_value=process,
            ) as popen,
        ):
            self.channel._spawn_bridge()

        child = popen.call_args.kwargs["env"]
        self.assertEqual(child["PATH"], "node-path")
        self.assertEqual(child["LANG"], "en_US.UTF-8")
        self.assertEqual(
            child["PILOTAGE_BRIDGE_TOKEN"],
            self.channel._bridge_token,
        )
        for secret in (
            "sentinel-codex",
            "sentinel-openai",
            "sentinel-database",
            "sentinel-telegram",
        ):
            self.assertNotIn(secret, child.values())

    def test_spawn_passes_the_group_allowlist_separately_from_dm_senders(self):
        object.__setattr__(self.channel._config, "group_policy", "allowlist")
        object.__setattr__(
            self.channel._config,
            "group_allow_from",
            frozenset({"120363001234567890@g.us"}),
        )
        process = mock.Mock(pid=4322)
        with mock.patch.object(
            whatsapp.subprocess, "Popen", return_value=process
        ) as popen:
            self.channel._spawn_bridge()

        command = popen.call_args.args[0]
        group_flag = command.index("--answer-groups")
        self.assertEqual(command[group_flag + 1], "1")
        self.assertEqual(
            popen.call_args.kwargs["env"]["PILOTAGE_ALLOWED_GROUPS"],
            "120363001234567890@g.us",
        )
        self.assertEqual(
            popen.call_args.kwargs["env"]["PILOTAGE_ALLOWED_SENDERS"],
            "212600000000",
        )

    async def test_wrong_bridge_instance_is_never_accepted_as_ready(self):
        def responder(request: httpx.Request) -> httpx.Response:
            if request.headers.get(whatsapp.BRIDGE_TOKEN_HEADER) != "other-profile":
                return httpx.Response(403, json={"error": "wrong owner"})
            return httpx.Response(200, json={"connected": True})

        self.channel._bridge_token = "this-profile"
        self.channel._process = mock.Mock()
        self.channel._process.poll.return_value = None
        self.channel._http = httpx.AsyncClient(
            transport=httpx.MockTransport(responder),
            headers={whatsapp.BRIDGE_TOKEN_HEADER: self.channel._bridge_token},
        )
        self.addAsyncCleanup(self.channel._http.aclose)

        with (
            mock.patch.object(whatsapp, "BRIDGE_READY_TIMEOUT_SECONDS", 0.01),
            mock.patch.object(whatsapp.asyncio, "sleep", new=mock.AsyncMock()),
            self.assertRaisesRegex(ChannelError, "did not connect"),
        ):
            await self.channel._wait_until_connected()

    async def test_stale_bridge_shutdown_requires_matching_token_and_pid(self):
        record = {"pid": 99, "port": 9876, "token": "old-owner"}
        self.channel._pidfile.write_text(json.dumps(record), encoding="utf-8")
        seen = {}

        class Client:
            def __init__(self, **kwargs):
                seen["headers"] = kwargs["headers"]
                self.stopped = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def get(self, url):
                seen["get"] = url
                if self.stopped:
                    raise httpx.ConnectError(
                        "stopped",
                        request=httpx.Request("GET", url),
                    )
                return httpx.Response(
                    200,
                    json={"pid": 99},
                    request=httpx.Request("GET", url),
                )

            async def post(self, url):
                seen["post"] = url
                self.stopped = True
                return httpx.Response(
                    200,
                    json={"success": True},
                    request=httpx.Request("POST", url),
                )

        with mock.patch.object(whatsapp.httpx, "AsyncClient", Client):
            await self.channel._stop_stale_bridge()

        self.assertEqual(
            seen["headers"],
            {whatsapp.BRIDGE_TOKEN_HEADER: "old-owner"},
        )
        self.assertEqual(seen["post"], "http://127.0.0.1:9876/shutdown")
        self.assertFalse(self.channel._pidfile.exists())

    async def test_failed_start_closes_http_and_terminates_its_child(self):
        client = mock.Mock()
        client.aclose = mock.AsyncMock()

        with (
            mock.patch.object(self.channel, "_preflight"),
            mock.patch.object(
                self.channel,
                "_stop_stale_bridge",
                new=mock.AsyncMock(),
            ),
            mock.patch.object(whatsapp.httpx, "AsyncClient", return_value=client) as factory,
            mock.patch.object(self.channel, "_spawn_bridge"),
            mock.patch.object(

                self.channel,
                "_wait_until_connected",
                new=mock.AsyncMock(side_effect=ChannelError("failed")),
            ),
            mock.patch.object(self.channel, "_terminate_bridge") as terminate,
            self.assertRaisesRegex(ChannelError, "failed"),
        ):
            await self.channel.start()

        self.assertEqual(
            factory.call_args.kwargs["headers"],
            {whatsapp.BRIDGE_TOKEN_HEADER: self.channel._bridge_token},
        )
        client.aclose.assert_awaited_once()
        terminate.assert_called_once()
        self.assertTrue(self.channel.stopped.is_set())

    def test_bridge_allowlist_gate_runs_before_event_and_media_extraction(self):
        source = (Path(__file__).resolve().parent.parent / "bridge" / "bridge.js").read_text(
            encoding="utf-8"
        )
        loop = source.index("sock.ev.on('messages.upsert'")
        sender_gate = source.index("const senderAllowed", loop)
        sender_rejection = source.index("if (!senderAllowed) continue", sender_gate)
        group_gate = source.index("!ANSWER_GROUPS", sender_rejection)
        extraction = source.index("const event = await buildEvent", loop)
        self.assertLess(sender_gate, sender_rejection)
        self.assertLess(sender_rejection, group_gate)
        self.assertLess(group_gate, extraction)



class BatchLifecycleTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _message(text: str, message_id: str) -> InboundMessage:
        return InboundMessage(
            chat_id="212600000000@s.whatsapp.net",
            session_id="212600000000",
            sender_id="212600000000@s.whatsapp.net",
            sender_number="212600000000",
            push_name="User",
            text=text,
            is_group=False,
            message_ids=[message_id],
        )

    async def test_a_steady_stream_flushes_at_the_hard_batch_cap(self):
        delivered = []
        handled = asyncio.Event()

        async def handler(message: InboundMessage) -> None:
            delivered.append(message)
            handled.set()

        config = Config.load()
        object.__setattr__(config, "text_batch_delay_seconds", 1.0)
        object.__setattr__(config, "text_batch_hard_cap_seconds", 0.05)
        channel = WhatsAppChannel(config, handler, _command)
        self.addAsyncCleanup(channel.stop)

        channel._enqueue(self._message("one", "m1"))
        await asyncio.sleep(0.03)
        channel._enqueue(self._message("two", "m2"))
        await asyncio.wait_for(handled.wait(), timeout=0.5)

        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].text, "one\ntwo")
        self.assertEqual(delivered[0].message_ids, ["m1", "m2"])

    async def test_one_active_turn_has_only_one_merged_follow_up(self):
        started = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()
        delivered = []

        async def handler(message: InboundMessage) -> None:
            delivered.append(message.text)
            if len(delivered) == 1:
                started.set()
                await release.wait()
            else:
                finished.set()

        channel = WhatsAppChannel(Config.load(), handler, _command)
        self.addAsyncCleanup(channel.stop)
        channel._queue_turn(self._message("one", "m1"))
        await asyncio.wait_for(started.wait(), timeout=0.5)

        channel._queue_turn(self._message("two", "m2"))
        channel._queue_turn(self._message("three", "m3"))
        self.assertEqual(len(channel._turn_tasks), 1)
        self.assertEqual(channel._queued["212600000000"].text, "two\nthree")

        release.set()
        await asyncio.wait_for(finished.wait(), timeout=0.5)
        await asyncio.sleep(0)
        self.assertEqual(delivered, ["one", "two\nthree"])
        self.assertEqual(channel._turn_tasks, {})

    async def test_approval_command_bypasses_an_active_turn(self):
        started = asyncio.Event()
        release = asyncio.Event()
        command_seen = asyncio.Event()

        async def handler(_message: InboundMessage) -> None:
            started.set()
            await release.wait()

        async def command(_chat_id, session_id, _message_id, invocation) -> None:
            self.assertEqual(session_id, "212600000000")
            self.assertEqual(invocation.command.name, "approve")
            command_seen.set()

        config = Config.load()
        object.__setattr__(config, "allowed_senders", frozenset({"212600000000"}))
        channel = WhatsAppChannel(config, handler, command)
        self.addAsyncCleanup(channel.stop)
        channel._queue_turn(self._message("working", "m1"))
        await asyncio.wait_for(started.wait(), timeout=0.5)

        channel._accept(
            {
                "chatId": "212600000000@s.whatsapp.net",
                "senderId": "212600000000@s.whatsapp.net",
                "senderNumber": "212600000000",
                "messageId": "m2",
                "body": "/approve",
                "isGroup": False,
            }
        )
        await asyncio.wait_for(command_seen.wait(), timeout=0.5)

        self.assertFalse(channel._turn_tasks["212600000000"].done())
        release.set()

    async def test_startup_gate_holds_a_recognized_command_until_release(self):
        command_seen = asyncio.Event()

        async def command(chat_id, session_id, message_id, invocation) -> None:
            self.assertEqual(chat_id, "212600000000@s.whatsapp.net")
            self.assertEqual(session_id, "212600000000")
            self.assertEqual(message_id, "m2")
            self.assertEqual(invocation.command.name, "approve")
            command_seen.set()

        config = Config.load()
        object.__setattr__(config, "allowed_senders", frozenset({"212600000000"}))
        channel = WhatsAppChannel(config, _handle, command)
        self.addAsyncCleanup(channel.stop)

        channel.hold_inbound()
        channel._accept(
            {
                "chatId": "212600000000@s.whatsapp.net",
                "senderId": "212600000000@s.whatsapp.net",
                "senderNumber": "212600000000",
                "messageId": "m2",
                "body": "/approve",
                "isGroup": False,
            }
        )
        await asyncio.sleep(0)
        self.assertFalse(command_seen.is_set())

        channel.release_inbound()
        await asyncio.wait_for(command_seen.wait(), timeout=0.5)

    async def test_stop_cancels_and_awaits_the_active_handler(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def handler(_message: InboundMessage) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        channel = WhatsAppChannel(Config.load(), handler, _command)
        channel._queue_turn(self._message("one", "m1"))
        await asyncio.wait_for(started.wait(), timeout=0.5)
        await channel.stop(drain_timeout_seconds=0)

        self.assertTrue(cancelled.is_set())
        self.assertEqual(channel._turn_tasks, {})

    async def test_stop_drains_every_message_accepted_before_intake_closes(self):
        started = asyncio.Event()
        release = asyncio.Event()
        delivered = []

        async def handler(message: InboundMessage) -> None:
            delivered.append(message.text)
            if len(delivered) == 1:
                started.set()
                await release.wait()

        channel = WhatsAppChannel(Config.load(), handler, _command)
        channel._queue_turn(self._message("one", "m1"))
        await asyncio.wait_for(started.wait(), timeout=0.5)
        channel._queue_turn(self._message("two", "m2"))
        channel._enqueue(self._message("three", "m3"))

        stopping = asyncio.create_task(
            channel.stop(drain_timeout_seconds=0.5)
        )
        await asyncio.sleep(0)
        self.assertFalse(stopping.done())
        release.set()
        await asyncio.wait_for(stopping, timeout=1)

        self.assertEqual(delivered, ["one", "two\nthree"])
        self.assertEqual(channel._turn_tasks, {})


class BridgeSourceTests(unittest.TestCase):
    def test_node_bridge_rejects_requests_without_its_instance_token(self):
        source = (Path(__file__).resolve().parent.parent / "bridge" / "bridge.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("req.get('x-pilotage-bridge-token') !== INSTANCE_TOKEN", source)
        self.assertIn("app.post('/shutdown'", source)


if __name__ == "__main__":
    unittest.main()
