"""Hermes file-state coordination retained by Pilotage."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest

from pilotage.tools import file_state


class FileStateRegistryTests(unittest.TestCase):
    def setUp(self):
        file_state.get_registry().clear()
        self.files: list[str] = []

    def tearDown(self):
        for path in self.files:
            try:
                os.unlink(path)
            except OSError:
                pass
        file_state.get_registry().clear()

    def make_file(self) -> str:
        descriptor, path = tempfile.mkstemp(prefix="pilotage-file-state-")
        os.close(descriptor)
        self.files.append(path)
        return path

    def test_sibling_write_marks_an_older_read_stale(self):
        path = self.make_file()
        file_state.record_read("A", path)
        time.sleep(0.01)
        file_state.note_write("B", path)
        warning = file_state.check_stale("A", path)
        self.assertIn("sibling", warning.lower())
        self.assertIn("B", warning)

    def test_same_path_locks_serialize(self):
        path = self.make_file()
        events: list[tuple[str, int]] = []
        events_lock = threading.Lock()

        def worker(number: int) -> None:
            with file_state.lock_path(path):
                with events_lock:
                    events.append(("enter", number))
                time.sleep(0.01)
                with events_lock:
                    events.append(("exit", number))

        threads = [threading.Thread(target=worker, args=(number,)) for number in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for index in range(0, len(events), 2):
            self.assertEqual(events[index][0], "enter")
            self.assertEqual(events[index + 1][0], "exit")
            self.assertEqual(events[index][1], events[index + 1][1])

    def test_different_paths_do_not_share_one_global_lock(self):
        first = self.make_file()
        second = self.make_file()
        second_entered = threading.Event()

        def hold_first() -> None:
            with file_state.lock_path(first):
                second_entered.wait(timeout=2)

        def enter_second() -> None:
            with file_state.lock_path(second):
                second_entered.set()

        one = threading.Thread(target=hold_first)
        two = threading.Thread(target=enter_second)
        one.start()
        two.start()
        self.assertTrue(second_entered.wait(timeout=2))
        one.join(timeout=2)
        two.join(timeout=2)

if __name__ == "__main__":
    unittest.main()
