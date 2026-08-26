"""Focused contracts for the deployment readiness command."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilotage import doctor
from pilotage.runtime_lock import ProfileRuntimeLock


class ReportTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_failed_probe_does_not_stop_later_probes(self):
        report = doctor.DoctorReport("work")

        def fail():
            raise RuntimeError("missing")

        await doctor._probe(report, "first", fail)
        await doctor._probe(report, "second", lambda: "ready")

        self.assertFalse(report.ok)
        self.assertEqual(
            [(item.name, item.ok) for item in report.checks],
            [("first", False), ("second", True)],
        )

    async def test_run_doctor_returns_nonzero_and_prints_all_failures(self):
        report = doctor.DoctorReport(
            "work",
            [
                doctor.CheckResult("Python", True, "ready"),
                doctor.CheckResult("SQL", False, "not connected"),
            ],
        )
        lines: list[str] = []
        with mock.patch.object(
            doctor,
            "collect_report",
            mock.AsyncMock(return_value=report),
        ):
            result = await doctor.run_doctor(
                object(),
                "work",
                print_fn=lines.append,
            )

        self.assertEqual(result, 1)
        self.assertIn("PASS Python - ready", lines)
        self.assertIn("FAIL SQL - not connected", lines)
        self.assertIn("Not ready: 1 of 2 checks failed.", lines)

    async def test_run_doctor_returns_zero_only_when_every_check_passes(self):
        report = doctor.DoctorReport(
            "work",
            [doctor.CheckResult("Everything", True, "ready")],
        )
        with mock.patch.object(
            doctor,
            "collect_report",
            mock.AsyncMock(return_value=report),
        ):
            result = await doctor.run_doctor(
                object(),
                "work",
                print_fn=lambda _line: None,
            )
        self.assertEqual(result, 0)


class SecretRenderingTests(unittest.TestCase):
    def test_known_environment_secrets_are_removed_from_failures(self):
        token = "123456:telegram-secret-value"
        password = "database-password-value"
        with mock.patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": token,
                "MSSQL_PASSWORD": password,
            },
            clear=False,
        ):
            rendered = doctor._safe_detail(
                f"request {token} failed with password {password}"
            )

        self.assertNotIn(token, rendered)
        self.assertNotIn(password, rendered)
        self.assertIn("[REDACTED]", rendered)


class DeliveryReadinessTests(unittest.TestCase):
    def test_delivery_probe_uses_the_profile_database(self):
        root = Path("/profile/state")
        store = mock.Mock()
        with mock.patch.object(doctor, "DeliveryStore", return_value=store) as cls:
            detail = doctor._check_delivery_store(SimpleNamespace(state_dir=root))

        cls.assert_called_once_with(root / "delivery.db")
        store.verify_writable.assert_called_once_with()
        self.assertEqual(detail, "delivery write path ready")

    def test_delivery_probe_reports_an_unwritable_database(self):
        store = mock.Mock()
        store.verify_writable.side_effect = sqlite3.OperationalError(
            "attempt to write a readonly database"
        )
        with (
            mock.patch.object(doctor, "DeliveryStore", return_value=store),
            self.assertRaisesRegex(doctor.DoctorError, "not writable"),
        ):
            doctor._check_delivery_store(SimpleNamespace(state_dir=Path("/state")))


class SqlReadinessTests(unittest.TestCase):
    def test_sql_probe_uses_the_production_connection_contract(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout="pilotage_ready\n1\n",
            stderr="",
        )
        environment = {
            "MSSQL_HOST": "db.example,1433",
            "MSSQL_USER": "reader",
            "MSSQL_PASSWORD": "top-secret",
            "MSSQL_DB": "reports",
            "MSSQL_ENCRYPT": "true",
            "MSSQL_TRUST_CERT": "true",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(
                doctor,
                "_sqlcmd_path",
                return_value="/opt/mssql-tools18/bin/sqlcmd",
            ),
            mock.patch.object(
                doctor,
                "_run",
                return_value=completed,
            ) as run,
        ):
            detail = doctor._check_sql_connection()

        command = run.call_args.args[0]
        self.assertEqual(detail, "query completed")
        self.assertEqual(command[0], "/opt/mssql-tools18/bin/sqlcmd")
        self.assertEqual(command[command.index("-S") + 1], "db.example,1433")
        self.assertEqual(command[command.index("-U") + 1], "reader")
        self.assertNotIn("-P", command)
        self.assertNotIn("top-secret", command)
        self.assertEqual(command[command.index("-d") + 1], "reports")
        self.assertIn("-N", command)
        self.assertIn("-C", command)
        self.assertLess(command.index("-N"), command.index("-Q"))
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(child_env["SQLCMDPASSWORD"], "top-secret")
        self.assertNotIn("MSSQL_PASSWORD", child_env)

    def test_sql_probe_names_missing_settings_without_values(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(
                doctor.DoctorError,
                "MSSQL_HOST.*MSSQL_USER.*MSSQL_PASSWORD.*MSSQL_DB",
            ),
        ):
            doctor._check_sql_connection()


class PreparedEnvironmentTests(unittest.TestCase):
    def test_smoke_probe_requires_every_expected_artifact(self):
        def execute(_code, environment, _context, *, workspace):
            for name in doctor._SMOKE_ARTIFACTS[environment]:
                payload = (
                    b"%PDF-valid"
                    if name == "bilingual.pdf"
                    else b"artifact"
                )
                (workspace / name).write_bytes(payload)
            return json.dumps({"status": "success"})

        config = SimpleNamespace()
        with mock.patch.object(doctor, "_execute", side_effect=execute):
            detail = doctor._check_prepared_environment(config, "pdf")

        self.assertEqual(detail, "1 artifact(s)")

    def test_smoke_probe_surfaces_child_failure(self):
        with (
            mock.patch.object(
                doctor,
                "_execute",
                return_value=json.dumps({
                    "status": "error",
                    "error": "missing weasyprint",
                }),
            ),
            self.assertRaisesRegex(doctor.DoctorError, "missing weasyprint"),
        ):
            doctor._check_prepared_environment(
                SimpleNamespace(),
                "pdf",
            )


class WhatsAppReadinessTests(unittest.TestCase):
    def test_access_requires_an_explicit_person_allowlist(self):
        config = SimpleNamespace(
            allowed_senders=frozenset(),
        )

        with self.assertRaisesRegex(
            doctor.DoctorError,
            "no explicit allowed senders",
        ):
            doctor._check_whatsapp_policy(config)

        config.allowed_senders = frozenset({"212600000000"})
        self.assertEqual(
            doctor._check_whatsapp_policy(config),
            "explicit access policy",
        )

    def test_linked_session_requires_completed_pairing_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = root / "whatsapp"
            session.mkdir()
            path = session / "creds.json"
            config = SimpleNamespace(session_dir=session)

            path.write_text(
                json.dumps({"registered": False}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                doctor.DoctorError,
                "credentials are incomplete",
            ):
                doctor._check_whatsapp_session(config)

            path.write_text(
                json.dumps({"registered": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                doctor.DoctorError,
                "credentials are incomplete",
            ):
                doctor._check_whatsapp_session(config)

            path.write_text(
                json.dumps({
                    "registered": False,
                    "noiseKey": {"private": "x", "public": "y"},
                    "signedIdentityKey": {"private": "a", "public": "b"},
                    "me": {"id": "212600000000:1@s.whatsapp.net"},
                    "account": {"details": "signed-device-identity"},
                    "signalIdentities": [{"identifier": "linked-device"}],
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                doctor._check_whatsapp_session(config),
                "linked device credentials ready",
            )

    def test_bridge_health_must_match_owner_and_be_connected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bridge.pid").write_text(
                json.dumps({
                    "pid": 42,
                    "port": 8765,
                    "token": "owner-secret",
                }),
                encoding="utf-8",
            )
            config = SimpleNamespace(state_dir=root)
            response = mock.Mock()
            response.json.return_value = {
                "pid": 42,
                "connected": True,
            }
            with mock.patch.object(
                doctor.httpx,
                "get",
                return_value=response,
            ) as get:
                detail = doctor._check_whatsapp_bridge(config)

        self.assertEqual(detail, "connected")
        self.assertEqual(
            get.call_args.kwargs["headers"],
            {"x-pilotage-bridge-token": "owner-secret"},
        )
        response.raise_for_status.assert_called_once_with()

    def test_bridge_health_rejects_a_durable_queue_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bridge.pid").write_text(
                json.dumps({
                    "pid": 42,
                    "port": 8765,
                    "token": "owner-secret",
                }),
                encoding="utf-8",
            )
            config = SimpleNamespace(state_dir=root)
            response = mock.Mock()
            response.json.return_value = {
                "pid": 42,
                "connected": True,
                "status": "unhealthy",
                "queue": {
                    "storageHealthy": True,
                    "overflowed": True,
                },
            }
            with mock.patch.object(doctor.httpx, "get", return_value=response):
                with self.assertRaisesRegex(doctor.DoctorError, "high-water"):
                    doctor._check_whatsapp_bridge(config)


class HomeChannelReadinessTests(unittest.TestCase):
    def test_at_least_one_enabled_channel_must_declare_a_home(self):
        with self.assertRaisesRegex(doctor.DoctorError, "no enabled channel"):
            doctor._check_home_channel(
                SimpleNamespace(home_origin=None),
                SimpleNamespace(home_origin=None),
                whatsapp_enabled=True,
                telegram_enabled=True,
            )

        self.assertEqual(
            doctor._check_home_channel(
                SimpleNamespace(
                    home_origin={
                        "channel": "whatsapp",
                        "chat_id": "212600000000@s.whatsapp.net",
                    }
                ),
                SimpleNamespace(home_origin=None),
                whatsapp_enabled=True,
                telegram_enabled=False,
            ),
            "configured for WhatsApp",
        )

    def test_invalid_whatsapp_home_is_not_reported_ready(self):
        with self.assertRaisesRegex(doctor.DoctorError, "home chat is invalid"):
            doctor._check_home_channel(
                SimpleNamespace(
                    home_origin={"channel": "whatsapp", "chat_id": "owner"}
                ),
                SimpleNamespace(home_origin=None),
                whatsapp_enabled=True,
                telegram_enabled=False,
            )


class RuntimeTests(unittest.TestCase):
    def test_runtime_lock_must_name_this_live_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            lock = ProfileRuntimeLock(state)
            lock.acquire()
            try:
                config = SimpleNamespace(state_dir=state)
                detail = doctor._check_runtime(config)
            finally:
                lock.release()

        self.assertEqual(detail, f"pid {os.getpid()}")

    def test_stale_live_pid_record_is_not_runtime_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory).resolve()
            (state / ".runtime.lock").write_text(
                json.dumps({"pid": os.getpid(), "state_dir": str(state)}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                doctor.DoctorError,
                "does not own",
            ):
                doctor._check_runtime(SimpleNamespace(state_dir=state))


if __name__ == "__main__":
    unittest.main()
