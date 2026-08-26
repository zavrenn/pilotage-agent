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
        self.assertNotIn("PILOTAGE_ALLOWED_GROUPS", popen.call_args.kwargs["env"])
        self.assertNotIn("212600000000", command)
        self.assertNotIn("--answer-groups", command)
        queue_flag = command.index("--inbound-queue")
        self.assertEqual(
            command[queue_flag + 1],
            str(self.channel._config.whatsapp_inbound_queue_dir),
        )
        record = json.loads(self.channel._pidfile.read_text(encoding="utf-8"))
        self.assertEqual(
            record,
            {
                "pid": 4321,
                "port": self.channel._config.bridge_port,
                "token": self.channel._bridge_token,
            },
        )

    def test_preflight_rejects_existing_corrupt_credentials_with_repair_guidance(self):
        session = self.channel._config.session_dir
        session.mkdir(parents=True, exist_ok=True)
        (session / "creds.json").write_text("", encoding="utf-8")
        dependency_dir = self.channel._config.bridge_dir / "node_modules"
        real_exists = Path.exists

        def exists(candidate):
            return candidate == dependency_dir or real_exists(candidate)

        with (
            mock.patch.object(whatsapp.shutil, "which", return_value="node"),
            mock.patch.object(Path, "exists", autospec=True, side_effect=exists),
            self.assertRaisesRegex(ChannelError, "pilotage whatsapp"),
        ):
            self.channel._preflight()

    def test_preflight_preserves_first_connection_pairing_when_creds_are_absent(self):
        dependency_dir = self.channel._config.bridge_dir / "node_modules"
        real_exists = Path.exists

        def exists(candidate):
            return candidate == dependency_dir or real_exists(candidate)

        with (
            mock.patch.object(whatsapp.shutil, "which", return_value="node"),
            mock.patch.object(Path, "exists", autospec=True, side_effect=exists),
        ):
            self.channel._preflight()

        self.assertTrue(self.channel._config.session_dir.is_dir())
        self.assertFalse(
            (self.channel._config.session_dir / "creds.json").exists()
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

    def test_spawn_uses_only_the_person_allowlist_for_dm_and_group_access(self):
        process = mock.Mock(pid=4322)
        with mock.patch.object(
            whatsapp.subprocess, "Popen", return_value=process
        ) as popen:
            self.channel._spawn_bridge()

        command = popen.call_args.args[0]
        self.assertNotIn("--answer-groups", command)
        self.assertNotIn("PILOTAGE_ALLOWED_GROUPS", popen.call_args.kwargs["env"])
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
        media_fence = source.index("item.fenceId = registerMediaFence", sender_rejection)
        extraction = source.index("const event = await buildEvent", loop)
        self.assertLess(sender_gate, sender_rejection)
        self.assertLess(sender_rejection, media_fence)
        self.assertLess(media_fence, extraction)
        self.assertNotIn("PILOTAGE_ALLOWED_GROUPS", source)
        self.assertNotIn("ANSWER_GROUPS", source)
        self.assertIn("res.json({ messages, mediaFences, queue })", source)

    def test_pair_only_waits_for_credentials_before_reporting_success(self):
        source = (Path(__file__).resolve().parent.parent / "bridge" / "bridge.js").read_text(
            encoding="utf-8"
        )
        finalizer = source.index("async function finishPairOnly")
        settle = source.index("setTimeout(resolve, PAIR_SETTLE_MS)", finalizer)
        flush = source.index("await credentialSaves.flush(saveCreds)", finalizer)
        success = source.index("pairing complete; credentials saved", finalizer)
        close = source.index("await pairingSocket?.end?.(undefined)", success)

        self.assertLess(settle, flush)
        self.assertLess(flush, success)
        self.assertLess(success, close)
        self.assertNotIn("setTimeout(() => process.exit(0), 2000)", source)

    def test_runtime_credential_write_failure_stops_the_bridge(self):
        source = (Path(__file__).resolve().parent.parent / "bridge" / "bridge.js").read_text(
            encoding="utf-8"
        )
        callback = source.index("sock.ev.on('creds.update'")
        persistence_failure = source.index(
            "credential persistence failed; stopping the bridge",
            callback,
        )
        fatal_exit = source.index("process.exit(1)", persistence_failure)

        self.assertLess(callback, persistence_failure)
        self.assertLess(persistence_failure, fatal_exit)



class BatchLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(
            os.environ,
            {"PILOTAGE_HOME": str(self.root)},
        )
        environment.start()
        self.addCleanup(environment.stop)

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

    async def test_polling_accelerates_while_a_batch_or_media_fence_is_open(self):
        config = Config.load()
        object.__setattr__(config, "allowed_senders", frozenset({"212600000000"}))
        channel = WhatsAppChannel(config, mock.AsyncMock(), _command)
        self.addAsyncCleanup(channel.stop)

        self.assertEqual(channel._next_poll_interval(), whatsapp.POLL_INTERVAL_SECONDS)
        channel._enqueue(self._message("question", "m1"))
        self.assertEqual(
            channel._next_poll_interval(),
            whatsapp.ACTIVE_BATCH_POLL_INTERVAL_SECONDS,
        )
        channel._flush_pending_now("212600000000")
        channel._reconcile_media_fences(
            [
                {
                    "fenceId": "bridge-1",
                    "chatId": "212600000000@s.whatsapp.net",
                    "senderId": "212600000000@s.whatsapp.net",
                    "senderNumber": "212600000000",
                    "isGroup": False,
                }
            ]
        )
        self.assertEqual(
            channel._next_poll_interval(),
            whatsapp.ACTIVE_BATCH_POLL_INTERVAL_SECONDS,
        )
        channel._reconcile_media_fences([])
        self.assertEqual(channel._next_poll_interval(), whatsapp.POLL_INTERVAL_SECONDS)

    async def test_text_waits_for_bridge_reported_same_session_media(self):
        delivered = []
        handled = asyncio.Event()

        async def handler(message: InboundMessage) -> None:
            delivered.append(message)
            handled.set()

        config = Config.load()
        object.__setattr__(config, "allowed_senders", frozenset({"212600000000"}))
        object.__setattr__(config, "text_batch_delay_seconds", 0.0)
        object.__setattr__(config, "text_batch_hard_cap_seconds", 1.0)
        channel = WhatsAppChannel(config, handler, _command)
        self.addAsyncCleanup(channel.stop)
        fence = {
            "fenceId": "bridge-1",
            "chatId": "212600000000@s.whatsapp.net",
            "senderId": "212600000000@s.whatsapp.net",
            "senderNumber": "212600000000",
            "identities": ["212600000000@s.whatsapp.net"],
            "isGroup": False,
        }

        channel._reconcile_media_fences([fence])
        channel._enqueue(self._message("question", "m1"))
        await asyncio.sleep(0.05)
        self.assertFalse(handled.is_set())

        channel._reconcile_media_fences([])
        channel._enqueue(self._message("[image]", "m2"))
        await asyncio.wait_for(handled.wait(), timeout=0.5)

        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].text, "question\n[image]")
        self.assertEqual(delivered[0].message_ids, ["m1", "m2"])

    async def test_stuck_bridge_media_fence_is_bounded_and_spent_once(self):
        handled = asyncio.Event()
        config = Config.load()
        object.__setattr__(config, "allowed_senders", frozenset({"212600000000"}))
        object.__setattr__(config, "text_batch_delay_seconds", 0.0)
        object.__setattr__(config, "text_batch_hard_cap_seconds", 0.05)
        channel = WhatsAppChannel(
            config,
            mock.AsyncMock(side_effect=lambda _message: handled.set()),
            _command,
        )
        self.addAsyncCleanup(channel.stop)
        fence = {
            "fenceId": "bridge-stuck",
            "chatId": "212600000000@s.whatsapp.net",
            "senderId": "212600000000@s.whatsapp.net",
            "senderNumber": "212600000000",
            "isGroup": False,
        }
        channel._reconcile_media_fences([fence])

        channel._enqueue(self._message("question", "m1"))
        await asyncio.wait_for(handled.wait(), timeout=0.3)

        self.assertIn("bridge-stuck", channel._media_fence_spent)

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

    async def test_durable_claim_is_acked_only_after_handler_completion(self):
        started = asyncio.Event()
        release = asyncio.Event()
        requests = []

        async def handler(_message: InboundMessage) -> None:
            started.set()
            await release.wait()

        async def post(url, *, json, timeout):
            requests.append((url, json, timeout))
            return httpx.Response(
                200,
                json={"success": True},
                request=httpx.Request("POST", url),
            )

        channel = WhatsAppChannel(Config.load(), handler, _command)
        channel._http = mock.Mock()
        channel._http.post = mock.AsyncMock(side_effect=post)
        message = self._message("one", "m1")
        message.claim_ids = ["a" * 64]

        channel._queue_turn(message)
        task = channel._turn_tasks[message.session_id]
        await asyncio.wait_for(started.wait(), timeout=0.5)
        self.assertEqual(requests, [])

        release.set()
        await asyncio.wait_for(task, timeout=0.5)
        self.assertEqual(
            requests,
            [
                (
                    f"{channel._base_url}/messages/ack",
                    {"claims": ["a" * 64]},
                    5.0,
                )
            ],
        )
        channel._http = None

    async def test_failed_handler_releases_durable_claim_for_replay(self):
        requests = []

        async def handler(_message: InboundMessage) -> None:
            raise RuntimeError("transient handler failure")

        async def post(url, *, json, timeout):
            requests.append((url, json, timeout))
            return httpx.Response(
                200,
                json={"success": True},
                request=httpx.Request("POST", url),
            )

        channel = WhatsAppChannel(Config.load(), handler, _command)
        channel._http = mock.Mock()
        channel._http.post = mock.AsyncMock(side_effect=post)
        message = self._message("one", "m1")
        message.claim_ids = ["b" * 64]

        channel._queue_turn(message)
        task = channel._turn_tasks[message.session_id]
        await asyncio.wait_for(task, timeout=0.5)

        self.assertEqual(
            requests,
            [
                (
                    f"{channel._base_url}/messages/release",
                    {"claims": ["b" * 64]},
                    5.0,
                )
            ],
        )
        channel._http = None

    async def test_durable_identity_keeps_equal_message_ids_in_two_chats_distinct(self):
        delivered = []
        finished = asyncio.Event()

        async def handler(message: InboundMessage) -> None:
            delivered.append(message)
            if len(delivered) == 2:
                finished.set()

        config = Config.load()
        object.__setattr__(
            config,
            "allowed_senders",
            frozenset({"212600000000", "212600000001"}),
        )
        object.__setattr__(config, "text_batch_delay_seconds", 0.0)
        channel = WhatsAppChannel(config, handler, _command)
        self.addAsyncCleanup(channel.stop)

        for suffix, claim in (("0", "c" * 64), ("1", "d" * 64)):
            identity = f"21260000000{suffix}"
            channel._accept(
                {
                    "chatId": f"{identity}@s.whatsapp.net",
                    "senderId": f"{identity}@s.whatsapp.net",
                    "senderNumber": identity,
                    "identities": [identity],
                    "messageId": "same-baileys-id",
                    "body": "hello",
                    "_pilotageClaimId": claim,
                }
            )

        await asyncio.wait_for(finished.wait(), timeout=0.5)

        self.assertEqual(len(delivered), 2)
        self.assertEqual(
            {message.message_ids[0] for message in delivered},
            {"same-baileys-id"},
        )
        self.assertEqual(
            {message.dedup_ids[0] for message in delivered},
            {"c" * 64, "d" * 64},
        )

    async def test_completed_claim_replay_only_retries_acknowledgement(self):
        claim_id = "e" * 64
        config = Config.load()
        object.__setattr__(config, "allowed_senders", frozenset({"212600000000"}))
        completed_channel = WhatsAppChannel(config, _handle, _command)
        completed_channel._mark_claims_completed([claim_id])

        # A fresh channel instance represents a full Python-process restart.
        channel = WhatsAppChannel(config, _handle, _command)
        self.addAsyncCleanup(channel.stop)
        channel._http = mock.Mock()
        channel._http.post = mock.AsyncMock(
            return_value=httpx.Response(
                200,
                json={"success": True, "settled": 1},
                request=httpx.Request("POST", f"{channel._base_url}/messages/ack"),
            )
        )

        channel._accept(
            {
                "chatId": "212600000000@s.whatsapp.net",
                "senderId": "212600000000@s.whatsapp.net",
                "senderNumber": "212600000000",
                "identities": ["212600000000"],
                "messageId": "m-replayed",
                "body": "must not run again",
                "_pilotageClaimId": claim_id,
            }
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual(channel._pending, {})
        self.assertEqual(channel._turn_tasks, {})
        channel._http.post.assert_awaited_once_with(
            f"{channel._base_url}/messages/ack",
            json={"claims": [claim_id]},
            timeout=5.0,
        )
        channel._http = None

    def test_completed_claim_ledger_is_bounded(self):
        ledger = whatsapp._CompletedClaimStore(
            self.root / "bounded-completed.db",
            max_entries=2,
        )
        ledger.mark(["a" * 64])
        ledger.mark(["b" * 64])
        ledger.mark(["c" * 64])

        self.assertFalse(ledger.contains("a" * 64))
        self.assertTrue(ledger.contains("b" * 64))
        self.assertTrue(ledger.contains("c" * 64))


class BridgeSourceTests(unittest.TestCase):
    def test_node_bridge_rejects_requests_without_its_instance_token(self):
        source = (Path(__file__).resolve().parent.parent / "bridge" / "bridge.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("req.get('x-pilotage-bridge-token') !== INSTANCE_TOKEN", source)
        self.assertIn("app.post('/shutdown'", source)


if __name__ == "__main__":
    unittest.main()
