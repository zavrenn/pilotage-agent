"""Contract for the model-facing, profile-local cronjob tool."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from pilotage.cron.jobs import CronStore
from pilotage.tools import ToolContext, build_registry


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
            config=None,
            cron_store=self.store,
            origin={"channel": "whatsapp", "chat_id": "123@c.us"},
            cron_wake=self._wake,
        )
        self.registry = build_registry()

    def _wake(self):
        self.wakes += 1

    async def call(self, **arguments):
        raw = await self.registry.dispatch(
            "cronjob",
            json.dumps(arguments),
            self.context,
            allowed_groups=["cron"],
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
        blocked = json.loads(await self.registry.dispatch(
            "cronjob", '{"action":"list"}', self.context,
            allowed_groups=["memory"],
        ))
        self.assertIn("disabled", blocked["error"])

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
        self.assertLessEqual(len(listed["jobs"][0]["last_output_preview"]), 503)
        self.assertEqual(self.wakes, 1)

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

    async def test_bad_shapes_and_unsupported_delivery_are_rejected(self):
        for arguments in (
            {"action": "create", "prompt": "x"},
            {"action": "create", "schedule": "1h", "skills": {}},
            {"action": "create", "schedule": "1h", "prompt": "x", "deliver": "all"},
            {"action": "create", "schedule": "1h", "prompt": "x", "job_id": "ignored"},
            {"action": "list", "prompt": "must not be ignored"},
            {"action": "update", "job_id": "missing"},
        ):
            with self.subTest(arguments=arguments):
                result = await self.call(**arguments)
                self.assertFalse(result.get("success", False))
                self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
