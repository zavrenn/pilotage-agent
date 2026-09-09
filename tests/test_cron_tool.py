"""Contract for the model-facing, profile-local cronjob tool."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from pilotage.agent import TurnResult
from pilotage.config import Config
from pilotage.cron.jobs import CronStore
from pilotage.cron.scheduler import CronScheduler
from pilotage.i18n import t
from pilotage.settings import Settings
from pilotage.tools import ToolContext, build_registry, enabled_groups


class CronToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now = datetime(2026, 1, 2, 8, 30, tzinfo=timezone.utc)
        self.store = CronStore(
            self.root, timezone_name="UTC", now=lambda: self.now
        )
        self.wakes = 0
        self.context = ToolContext(
            chat_id="session-1",
            config=SimpleNamespace(
                cron_enabled=True,
                approval_cron=True,
                language="fr",
                settings=Settings({"whatsapp": {"enabled": True}, "telegram": {"enabled": True}}),
            ),
            cron_store=self.store,
            origin={"channel": "whatsapp", "chat_id": "123@c.us"},
            cron_wake=self._wake,
            approval_request=AsyncMock(
                side_effect=AssertionError("Scheduling must not request client approval")
            ),
        )
        self.registry = build_registry()

    def tearDown(self):
        self.context.approval_request.assert_not_awaited()

    def _wake(self):
        self.wakes += 1

    async def call(self, **arguments):
        raw = await self.registry.dispatch(
            "cronjob",
            json.dumps(arguments),
            self.context,
            allowed_groups=enabled_groups(self.context.config.settings, self.registry),
        )
        return json.loads(raw)

    async def create(self, **arguments):
        values = {
            "action": "create",
            "prompt": "Send the morning status",
            "schedule": "1h",
        }
        values.update(arguments)
        return await self.call(**values)

    async def test_registry_exposes_one_cron_tool_in_the_cron_group(self):
        self.assertEqual(self.registry.names(["cron"]), ["cronjob"])
        for arguments in (
            {"action": "list"}, {"action": "get", "job_id": "missing"},
        ):
            blocked = json.loads(await self.registry.dispatch(
                "cronjob", json.dumps(arguments), self.context,
                allowed_groups=["memory"],
            ))
            self.assertEqual(blocked["error"], t("capability.unavailable", "fr"))
            self.assertIs(blocked["disabled"], True)

    async def test_enabled_create_ignores_legacy_approval_switch(self):
        result = await self.create()

        self.assertTrue(result["success"])
        self.assertNotIn("approval", result)
        self.assertEqual(len(self.store.list_jobs(include_disabled=True)), 1)
        self.assertEqual(self.wakes, 1)

    async def test_disabled_create_does_not_write_or_wake(self):
        self.context.config.cron_enabled = False
        for language in ("en", "fr", "ar"):
            with self.subTest(language=language):
                self.context.config.language = language
                result = await self.create()
                self.assertFalse(result["success"])
                self.assertEqual(result["error"], t("cron.unavailable", language))
                self.assertNotIn("cron", result["error"].lower())
                self.assertNotIn("config", result["error"].lower())
                self.assertNotIn("approval", result)
        self.assertFalse(self.store.jobs_path.exists())
        self.assertEqual(self.store.list_jobs(include_disabled=True), [])
        self.assertEqual(self.wakes, 0)

    async def test_disabled_activation_preserves_existing_jobs(self):
        active = self.store.create_job(prompt="Send the status", schedule="every 1h")
        paused = self.store.create_job(prompt="Send the status", schedule="every 1h")
        self.store.pause_job(paused["id"])
        completed = self.store.create_job(
            prompt="Send the status", schedule=self.now.isoformat()
        )
        claimed = self.store.claim_due_jobs()[0]
        self.assertEqual(claimed["id"], completed["id"])
        self.store.finish_job(
            completed["id"], owner=claimed["claim"]["by"], success=True
        )
        self.assertEqual(self.store.resolve_job(completed["id"])["state"], "completed")
        before = self.store.jobs_path.read_bytes()
        self.context.config.cron_enabled = False

        for arguments in (
            {"action": "resume", "job_id": paused["id"]},
            {"action": "run", "job_id": active["id"]},
            {"action": "update", "job_id": completed["id"], "schedule": "every 1h"},
        ):
            with self.subTest(arguments=arguments):
                result = await self.call(**arguments)
                self.assertFalse(result["success"])
                self.assertEqual(result["error"], t("cron.unavailable", "fr"))
                self.assertEqual(self.store.jobs_path.read_bytes(), before)
                self.assertEqual(self.wakes, 0)

    async def test_disabled_scheduling_keeps_nonactivating_management_available(self):
        job = self.store.create_job(prompt="Send the status", schedule="every 1h")
        self.context.config.cron_enabled = False
        self.assertEqual((await self.call(action="list"))["count"], 1)
        inspected = await self.call(action="get", job_id=job["id"])
        self.assertEqual(inspected["job"]["prompt"], "Send the status")
        renamed = await self.call(action="update", job_id=job["id"], name="Renamed")
        self.assertTrue(renamed["success"])
        paused = await self.call(action="pause", job_id=job["id"])
        self.assertEqual(paused["job"]["state"], "paused")
        removed = await self.call(action="remove", job_id=job["id"])
        self.assertTrue(removed["success"])

    def channel_configs(self, enabled_channel):
        settings = Settings({
            "whatsapp": {"enabled": True}, "telegram": {"enabled": True},
            "channels": {
                channel: {"cron": {"enabled": channel == enabled_channel}}
                for channel in ("whatsapp", "telegram")
            },
        })
        return {
            channel: SimpleNamespace(
                cron_enabled=channel == enabled_channel,
                channel=channel,
                settings=settings.for_channel(channel),
                language="fr",
                home_origin={"channel": channel, "chat_id": "home"},
            )
            for channel in ("whatsapp", "telegram")
        }

    async def test_delivery_changes_cannot_enable_a_disabled_origins_job(self):
        for enabled_channel, disabled_channel in (("whatsapp", "telegram"), ("telegram", "whatsapp")):
            configs = self.channel_configs(enabled_channel)
            self.context.config = configs[disabled_channel]
            scheduler = CronScheduler(configs["whatsapp"], self.store, channel_configs=configs)
            job = self.store.create_job(
                prompt="Send the status", schedule="0m",
                origin={"channel": disabled_channel, "chat_id": "42"},
            )
            for destination in ("origin", "local", "whatsapp", "telegram"):
                with self.subTest(origin=disabled_channel, deliver=destination):
                    updated = await self.call(action="update", job_id=job["id"], deliver=destination)
                    self.assertTrue(updated["success"])
                    before = self.store.jobs_path.read_bytes()
                    self.assertFalse(scheduler._can_run_job(self.store.resolve_job(job["id"])))
                    # Scope the claim to this case; previous cases remain disabled.
                    self.assertEqual(self.store.claim_due_jobs(
                        can_run=lambda value: value["id"] == job["id"] and scheduler._can_run_job(value)
                    ), [])
                    self.assertEqual(self.store.jobs_path.read_bytes(), before)

    async def test_enabled_jobs_keep_the_origin_policy_for_every_destination(self):
        for source, disabled in (("whatsapp", "telegram"), ("telegram", "whatsapp")):
            configs = self.channel_configs(source)
            self.context.config = configs[source]
            self.context.origin = {"channel": source, "chat_id": "42"}
            for destination in ("origin", "local", "whatsapp", "telegram"):
                with self.subTest(origin=source, deliver=destination):
                    created = await self.create(schedule="0m", deliver=destination)
                    self.assertTrue(created["success"], created)
                    agent = SimpleNamespace(
                        _allow_persistence_writes=False, _scheduled_run=True,
                        respond_result=AsyncMock(return_value=TurnResult(text="All good", terminal_completed=True)),
                    )
                    factory, deliver = Mock(return_value=agent), AsyncMock()
                    scheduler = CronScheduler(
                        configs["whatsapp"], self.store, channel_configs=configs,
                        agent_factory=factory, deliver=deliver,
                    )
                    jobs = self.store.claim_due_jobs(can_run=scheduler._can_run_job)
                    self.assertEqual([job["id"] for job in jobs], [created["job"]["job_id"]])
                    await scheduler._run_claimed(jobs[0])
                    factory.assert_called_once_with(configs[source])
                    self.assertEqual(self.store.resolve_job(jobs[0]["id"])["state"], "completed")
                    if destination == "local":
                        deliver.assert_not_awaited()
                    else:
                        expected = self.context.origin if destination == "origin" else configs[destination].home_origin
                        self.assertEqual(deliver.await_args.args[:2], (expected, "All good"))

    async def test_activation_from_an_enabled_channel_respects_the_job_origin(self):
        configs = self.channel_configs("whatsapp")
        self.context.config = configs["whatsapp"]
        job = self.store.create_job(
            prompt="Send the status", schedule="every 1h",
            origin={"channel": "telegram", "chat_id": "42"},
        )
        before = self.store.jobs_path.read_bytes()
        for arguments in (
            {"action": "run"}, {"action": "resume"},
            {"action": "update", "schedule": "0m"},
            {"action": "update", "prompt": "Run a different report"},
            {"action": "update", "repeat": 3},
            {"action": "update", "skills": ["different-report"]},
            {"action": "update", "enabled_toolsets": []},
            {"action": "update", "workdir": str(self.root)},
        ):
            with self.subTest(arguments=arguments):
                result = await self.call(job_id=job["id"], **arguments)
                self.assertFalse(result["success"])
                self.assertEqual(result["error"], t("cron.unavailable", "fr"))
                self.assertEqual(self.store.jobs_path.read_bytes(), before)
        self.assertEqual(self.wakes, 0)

    async def test_disabled_caller_cannot_replace_work_in_an_enabled_channels_job(self):
        configs = self.channel_configs("whatsapp")
        self.context.config = configs["telegram"]
        job = self.store.create_job(
            prompt="Send the status", schedule="every 1h",
            origin={"channel": "whatsapp", "chat_id": "42"},
        )
        before = self.store.jobs_path.read_bytes()
        result = await self.call(action="update", job_id=job["id"], prompt="Run a different report")
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], t("cron.unavailable", "fr"))
        self.assertEqual(self.store.jobs_path.read_bytes(), before)
        self.assertEqual(self.wakes, 0)

    async def test_inactive_origins_are_rejected_by_admission_and_execution(self):
        with patch.dict(os.environ, {
            "PILOTAGE_HOME": str(self.root),
            "TELEGRAM_BOT_TOKEN": "123456:test-token",
            "TELEGRAM_ALLOWED_USERS": "42",
        }):
            for active, inactive in (("whatsapp", "telegram"), ("telegram", "whatsapp")):
                for mode in ("disabled", "removed", "overridden"):
                    with self.subTest(active=active, inactive=inactive, mode=mode):
                        written = f"cron:\n  enabled: true\n{active}:\n  enabled: true\n"
                        if mode == "disabled":
                            written += f"{inactive}:\n  enabled: false\n"
                        elif mode == "overridden":
                            written += (
                                f"{inactive}:\n  enabled: true\nchannels:\n"
                                f"  {inactive}:\n    {inactive}:\n      enabled: false\n"
                            )
                        config_path = self.root / "config.yaml"
                        config_path.write_text(written, encoding="utf-8")
                        active_config = Config.load(channel=active)
                        inactive_config = Config.load(channel=inactive)
                        self.assertTrue(active_config.cron_enabled)
                        self.assertTrue(inactive_config.cron_enabled)
                        job = self.store.create_job(
                            prompt="Send the status", schedule="0m", deliver="local",
                            origin={"channel": inactive, "chat_id": "42"},
                        )
                        self.store.pause_job(job["id"])
                        before = self.store.jobs_path.read_bytes()
                        wakes = self.wakes
                        self.context.config = active_config
                        self.context.origin = {"channel": active, "chat_id": "42"}
                        scheduler = CronScheduler(
                            active_config, self.store, channel_configs={active: active_config},
                            default_job_config=Config.load(),
                        )
                        self.assertFalse(scheduler._can_run_job(job))
                        for arguments in (
                            {"action": "run"}, {"action": "resume"},
                            {"action": "update", "schedule": "0m"},
                            {"action": "update", "prompt": "A different report"},
                            {"action": "update", "repeat": 3},
                            {"action": "update", "skills": ["report"]},
                            {"action": "update", "enabled_toolsets": []},
                            {"action": "update", "workdir": str(self.root)},
                        ):
                            result = await self.call(job_id=job["id"], **arguments)
                            self.assertFalse(result["success"], (arguments, result))
                            self.assertEqual(result["error"], t("cron.unavailable", active_config.language))
                            self.assertEqual(self.store.jobs_path.read_bytes(), before)
                        self.context.config = inactive_config
                        self.context.origin = job["origin"]
                        self.assertFalse((await self.create())["success"])
                        self.assertEqual(self.store.jobs_path.read_bytes(), before)
                        self.assertEqual(self.wakes, wakes)

                        self.context.config = active_config
                        self.context.origin = {"channel": active, "chat_id": "42"}
                        self.assertTrue((await self.call(action="list", include_disabled=True))["success"])
                        self.assertTrue((await self.call(
                            action="update", job_id=job["id"], name="Renamed", deliver="origin",
                        ))["success"])
                        self.assertTrue((await self.call(action="pause", job_id=job["id"]))["success"])

                        # Re-enabling the origin restores both admission and execution.
                        config_path.write_text(
                            "cron:\n  enabled: true\nwhatsapp:\n  enabled: true\n"
                            "telegram:\n  enabled: true\n", encoding="utf-8",
                        )
                        configs = {name: Config.load(channel=name) for name in (active, inactive)}
                        self.context.config = configs[active]
                        scheduler = CronScheduler(configs[active], self.store, channel_configs=configs)
                        resumed = await self.call(action="resume", job_id=job["id"])
                        self.assertTrue(resumed["success"], resumed)
                        queued = await self.call(action="run", job_id=job["id"])
                        self.assertTrue(queued["success"], queued)
                        claimed = self.store.claim_due_jobs(can_run=scheduler._can_run_job)
                        self.assertEqual([value["id"] for value in claimed], [job["id"]])
                        self.store.finish_job(job["id"], owner=claimed[0]["claim"]["by"], success=True)

    def configs_with_restricted_channel(self, channel="whatsapp"):
        environment = patch.dict(os.environ, {
            "PILOTAGE_HOME": str(self.root),
            "TELEGRAM_BOT_TOKEN": "123456:test-token",
            "TELEGRAM_ALLOWED_USERS": "42",
        })
        environment.start()
        self.addCleanup(environment.stop)
        (self.root / "config.yaml").write_text(
            "whatsapp:\n  enabled: true\ntelegram:\n  enabled: true\n"
            "tools:\n  enabled: [cron, terminal, web]\n"
            f"channels:\n  {channel}:\n    tools:\n      disabled: [terminal]\n",
            encoding="utf-8",
        )
        configs = {name: Config.load(channel=name) for name in ("", "whatsapp", "telegram")}
        self.context.config = configs["whatsapp"]
        return configs

    async def test_restricted_caller_cannot_edit_or_activate_broader_jobs(self):
        self.configs_with_restricted_channel()
        for origin in ({"channel": "telegram", "chat_id": "42"}, None):
            for toolsets in (None, [], ["terminal"]):
                with self.subTest(origin=origin, toolsets=toolsets):
                    job = self.store.create_job(
                        prompt="Original report", schedule="every 1h",
                        origin=origin, enabled_toolsets=toolsets,
                    )
                    self.store.pause_job(job["id"])
                    before, wakes = self.store.jobs_path.read_bytes(), self.wakes
                    for arguments in (
                        {"action": "update", "prompt": "Different work"},
                        {"action": "update", "schedule": "0m"},
                        {"action": "update", "repeat": 2},
                        {"action": "update", "skills": ["report"]},
                        {"action": "update", "workdir": str(self.root)},
                        {"action": "update", "enabled_toolsets": None},
                        {"action": "update", "enabled_toolsets": []},
                        {"action": "update", "enabled_toolsets": [""]},
                        {"action": "run"}, {"action": "resume"},
                    ):
                        result = await self.call(job_id=job["id"], **arguments)
                        self.assertFalse(result["success"], (arguments, result))
                        self.assertEqual(result["error"], t("capability.unavailable", "fr"))
                        self.assertEqual(self.store.jobs_path.read_bytes(), before)
                        self.assertEqual(self.wakes, wakes)

    async def test_broader_jobs_keep_nonexecuting_management_available(self):
        self.configs_with_restricted_channel()
        job = self.store.create_job(
            prompt="Original report", schedule="every 1h",
            origin={"channel": "telegram", "chat_id": "42"},
        )
        self.assertTrue((await self.call(action="list"))["success"])
        updated = await self.call(action="update", job_id=job["id"], name="Renamed", deliver="local")
        self.assertTrue(updated["success"], updated)
        self.assertEqual(self.store.resolve_job(job["id"])["prompt"], job["prompt"])
        self.assertTrue((await self.call(action="pause", job_id=job["id"]))["success"])
        self.assertTrue((await self.call(action="remove", job_id=job["id"]))["success"])

    async def test_explicitly_narrowed_jobs_run_with_compatible_origin_tools(self):
        configs = self.configs_with_restricted_channel()
        scheduler = CronScheduler(
            configs["whatsapp"], self.store, channel_configs=configs,
            default_job_config=configs[""],
        )
        for origin in ({"channel": "telegram", "chat_id": "42"}, None):
            with self.subTest(origin=origin):
                job = self.store.create_job(prompt="Original", schedule="every 1h", origin=origin)
                result = await self.call(
                    action="update", job_id=job["id"], prompt="Web report",
                    enabled_toolsets=["web"],
                )
                self.assertTrue(result["success"], result)
                self.assertTrue((await self.call(action="pause", job_id=job["id"]))["success"])
                self.assertTrue((await self.call(action="resume", job_id=job["id"]))["success"])
                self.assertTrue((await self.call(action="run", job_id=job["id"]))["success"])
                stored = self.store.resolve_job(job["id"])
                self.assertEqual(stored["origin"], origin)
                agent = scheduler._fresh_agent(scheduler._config_for_job(stored), stored)
                try:
                    self.assertEqual(agent._tool_groups, ["web"])
                    self.assertFalse(agent._persistence_writes_enabled)
                finally:
                    await agent.close()
                # An empty list clears the limit; it must not restore broader tools.
                before = self.store.jobs_path.read_bytes()
                result = await self.call(action="update", job_id=job["id"], enabled_toolsets=[])
                self.assertFalse(result["success"])
                self.assertEqual(self.store.jobs_path.read_bytes(), before)

    async def test_requested_and_stored_tools_must_be_available_to_the_origin(self):
        self.configs_with_restricted_channel("telegram")
        origin = {"channel": "telegram", "chat_id": "42"}
        for toolsets in (["web"], ["terminal"]):
            job = self.store.create_job(
                prompt="Original", schedule="every 1h", origin=origin,
                enabled_toolsets=toolsets,
            )
            self.store.pause_job(job["id"])
            before = self.store.jobs_path.read_bytes()
            actions = [{"action": "update", "enabled_toolsets": ["terminal"]}]
            if toolsets == ["terminal"]:
                actions += [{"action": "run"}, {"action": "resume"}, {"action": "update", "prompt": "Changed"}]
            for arguments in actions:
                result = await self.call(job_id=job["id"], **arguments)
                self.assertFalse(result["success"], result)
                self.assertEqual(result["error"], t("capability.unavailable", "fr"))
                self.assertEqual(self.store.jobs_path.read_bytes(), before)
        self.assertEqual(self.wakes, 0)

    async def test_creation_cannot_escape_the_callers_effective_tool_limit(self):
        configs = self.configs_with_restricted_channel()
        self.context.config = configs["telegram"]
        self.context.origin = {"channel": "telegram", "chat_id": "42"}
        self.context.allowed_tool_groups = frozenset({"cron", "web"})
        for arguments in ({}, {"enabled_toolsets": []}, {"enabled_toolsets": ["terminal"]}):
            result = await self.create(**arguments)
            self.assertFalse(result["success"], result)
            self.assertEqual(result["error"], t("capability.unavailable", "fr"))
            self.assertFalse(self.store.jobs_path.exists())
        self.assertEqual(self.wakes, 0)
        created = await self.create(enabled_toolsets=["web"])
        self.assertTrue(created["success"], created)

    async def test_create_captures_origin_and_list_is_bounded(self):
        created = await self.create(name="Morning", skills=["report"])
        self.assertTrue(created["success"])
        job_id = created["job"]["job_id"]
        persisted = self.store.resolve_job(job_id)
        self.assertEqual(
            persisted["origin"], {"channel": "whatsapp", "chat_id": "123@c.us"}
        )
        self.assertEqual(persisted["deliver"], "origin")
        self.store.save_output(job_id, "x" * 700)
        listed = await self.call(action="list")
        self.assertEqual(listed["count"], 1)
        self.assertEqual(listed["current_origin"], self.context.origin)
        self.assertEqual(listed["jobs"][0]["origin"], persisted["origin"])
        self.assertLessEqual(len(listed["jobs"][0]["last_output_preview"]), 503)
        self.assertEqual(self.wakes, 1)

    async def test_get_reads_the_full_task_and_origin_without_mutating(self):
        prompt = "Summarize the daily report. " * 30 + "Keep amounts in MAD."
        workdir = self.root / "project"
        workdir.mkdir()
        for origin in (
            self.context.origin,
            {"channel": "telegram", "chat_id": "42", "thread_id": "7"},
            None,
        ):
            with self.subTest(origin=origin):
                job = self.store.create_job(
                    name="Detailed report", prompt=prompt, schedule="every 2h",
                    repeat=5, skills=["report"], enabled_toolsets=["web"],
                    workdir=str(workdir), origin=origin,
                )
                self.store.pause_job(job["id"], reason="Wait for the operator")
                before = self.store.jobs_path.read_bytes()
                for reference in (job["id"], job["name"]):
                    result = await self.call(action="get", job_id=reference)
                    self.assertTrue(result["success"], result)
                    self.assertEqual(result["current_origin"], self.context.origin)
                    details = result["job"]
                    self.assertEqual(details["job_id"], job["id"])
                    self.assertEqual(details["prompt"], prompt)
                    self.assertEqual(details["origin"], origin)
                    self.assertEqual(details["deliver"], "origin" if origin else "local")
                    self.assertEqual(details["schedule"], job["schedule_display"])
                    self.assertEqual(details["repeat"], "5 times")
                    self.assertEqual(details["skills"], ["report"])
                    self.assertEqual(details["enabled_toolsets"], ["web"])
                    self.assertEqual(details["workdir"], str(workdir.resolve()))
                    self.assertEqual(details["state"], "paused")
                    self.assertEqual(details["paused_reason"], "Wait for the operator")
                    self.assertEqual(self.store.jobs_path.read_bytes(), before)
                    self.assertEqual(self.wakes, 0)
                listed = await self.call(action="list", include_disabled=True)
                self.assertEqual(listed["current_origin"], self.context.origin)
                self.assertEqual(listed["jobs"][0]["origin"], origin)
                self.assertNotIn("prompt", listed["jobs"][0])
                self.assertEqual(listed["jobs"][0]["prompt_preview"], prompt[:100] + "...")
                self.store.remove_job(job["id"])

    async def test_get_rejects_missing_or_ambiguous_references_without_mutating(self):
        for name in ("same", "SAME"):
            self.store.create_job(name=name, prompt="Send the status", schedule="1h")
        before = self.store.jobs_path.read_bytes()
        for reference in ("missing", "same"):
            result = await self.call(action="get", job_id=reference)
            self.assertFalse(result["success"], result)
            if reference == "same":
                self.assertEqual(len(result["matches"]), 2)
            self.assertEqual(self.store.jobs_path.read_bytes(), before)
            self.assertEqual(self.wakes, 0)

    async def test_create_persists_workdir_and_tool_limits(self):
        workdir = self.root / "project"
        workdir.mkdir()
        created = await self.create(
            enabled_toolsets=["web", "file"],
            workdir=str(workdir),
        )
        self.assertTrue(created["success"])
        stored = self.store.resolve_job(created["job"]["job_id"])
        self.assertEqual(stored["enabled_toolsets"], ["web", "file"])
        self.assertEqual(stored["workdir"], str(workdir.resolve()))

    async def test_invalid_cron_scope_is_rejected_before_persistence(self):
        for values in (
            {"enabled_toolsets": ["unknown"]},
            {"enabled_toolsets": ["cron"]},
            {"workdir": "relative/path"},
        ):
            with self.subTest(values=values):
                result = await self.create(**values)
                self.assertFalse(result["success"])
                self.assertFalse(self.store.jobs_path.exists())
                self.assertEqual(self.wakes, 0)

    async def test_scheduled_toolsets_cannot_enable_a_disabled_group(self):
        self.context.config.settings = Settings({
            "whatsapp": {"enabled": True}, "tools": {"disabled": ["web"]},
        })
        result = await self.create(enabled_toolsets=["web"])
        self.assertFalse(result["success"])
        self.assertIn("disabled", result["error"])
        self.assertFalse(self.store.jobs_path.exists())
        self.assertEqual(self.wakes, 0)

    async def test_cli_origin_creates_a_local_job(self):
        self.context.origin = None
        created = await self.create()
        self.assertEqual(created["job"]["deliver"], "local")

    async def test_update_pause_resume_run_and_remove(self):
        created = await self.create(schedule="every 1h")
        job_id = created["job"]["job_id"]
        updated = await self.call(
            action="update", job_id=job_id, name="Changed", skills=[]
        )
        self.assertEqual(updated["job"]["name"], "Changed")
        self.assertEqual(updated["job"]["skills"], [])

        paused = await self.call(action="pause", job_id=job_id, reason="operator")
        self.assertEqual(paused["job"]["state"], "paused")
        refused = await self.call(action="run", job_id=job_id)
        self.assertFalse(refused["success"])
        self.assertIn("resume", refused["error"])

        resumed = await self.call(action="resume", job_id=job_id)
        self.assertEqual(resumed["job"]["state"], "scheduled")
        queued = await self.call(action="run", job_id=job_id)
        self.assertTrue(queued["success"])
        self.assertEqual(
            self.store.resolve_job(job_id)["next_run_at"], self.now.isoformat()
        )
        removed = await self.call(action="remove", job_id=job_id)
        self.assertTrue(removed["success"])

    async def test_ambiguous_name_returns_ids_instead_of_guessing(self):
        await self.create(name="same")
        await self.create(name="SAME")
        result = await self.call(action="pause", job_id="same")
        self.assertFalse(result["success"])
        self.assertEqual(len(result["matches"]), 2)

    async def test_repeat_limit_can_be_cleared_without_replacing_the_job(self):
        for schedule, expected_times in (("every 1h", None), ("1h", 1)):
            with self.subTest(schedule=schedule):
                created = await self.create(schedule=schedule, repeat=3)
                job_id = created["job"]["job_id"]
                before = self.store.resolve_job(job_id)
                result = await self.call(action="update", job_id=job_id, repeat=None)
                self.assertTrue(result["success"], result)
                self.assertEqual(
                    self.store.resolve_job(job_id),
                    dict(before, repeat={"times": expected_times, "completed": 0}),
                )

    async def test_bad_shapes_and_unsupported_delivery_are_rejected(self):
        for arguments in (
            {"action": "create", "prompt": "x"},
            {"action": "create", "schedule": "1h", "skills": {}},
            {"action": "create", "schedule": "1h", "prompt": "x", "deliver": "all"},
            {"action": "create", "schedule": "1h", "prompt": "x", "job_id": "ignored"},
            {"action": "list", "prompt": "must not be ignored"},
            {"action": "get"},
            {"action": "get", "job_id": "missing", "prompt": "must not be ignored"},
            {"action": "update", "job_id": "missing"},
        ):
            with self.subTest(arguments=arguments):
                result = await self.call(**arguments)
                self.assertFalse(result.get("success", False))
                self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
