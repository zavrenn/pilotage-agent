from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pilotage import main
from pilotage.config import Config
from pilotage.runtime_lock import ProfileRuntimeLock, RuntimeAlreadyRunning


class RuntimeLockTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_only_one_runtime_can_own_a_profile(self):
        first = ProfileRuntimeLock(self.root)
        second = ProfileRuntimeLock(self.root)
        first.acquire()
        self.addCleanup(first.release)

        with self.assertRaises(RuntimeAlreadyRunning):
            second.acquire()

        first.release()
        second.acquire()
        second.release()

    def test_lock_record_identifies_the_owner(self):
        lock = ProfileRuntimeLock(self.root)
        lock.acquire()
        self.addCleanup(lock.release)

        content = lock.path.read_text(encoding="utf-8")
        self.assertIn('"pid":', content)
        self.assertIn('"state_dir":', content)


class RuntimeEntryPointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        environment = mock.patch.dict(os.environ, {"PILOTAGE_HOME": str(self.root)})
        environment.start()
        self.addCleanup(environment.stop)

    async def test_second_run_stops_before_touching_whatsapp(self):
        config = Config.load(channel="whatsapp")
        owner = ProfileRuntimeLock(self.root)
        owner.acquire()
        self.addCleanup(owner.release)

        with (
            mock.patch.object(main, "WhatsAppChannel") as channel,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = await main.command_run(config)

        self.assertEqual(result, 1)
        channel.assert_not_called()

    async def test_unexpected_startup_failure_releases_the_lock(self):
        config = Config.load(channel="whatsapp")
        with (
            mock.patch.object(
                main,
                "_command_run_locked",
                new=mock.AsyncMock(side_effect=RuntimeError("boom")),
            ),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            await main.command_run(config)

        lock = ProfileRuntimeLock(self.root)
        lock.acquire()
        lock.release()


if __name__ == "__main__":
    unittest.main()
