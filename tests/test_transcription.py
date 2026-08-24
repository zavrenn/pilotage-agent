"""Contract for Hermes-derived OpenAI voice-message transcription."""

from __future__ import annotations

import os
import asyncio
import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
from openai import BadRequestError

from pilotage import media, transcription
from pilotage.settings import ConfigError, Settings


def _settings(extra: dict | None = None) -> Settings:
    data = {
        "stt": {
            "enabled": True,
            "provider": "openai",
            "echo_transcripts": True,
            "cloud_trim_silence": False,
            "openai": {"model": "whisper-1"},
        }
    }
    if extra:
        data["stt"].update(extra)
    return Settings(data)


def _attachment(path: Path, media_type: str = "ptt") -> media.Attachment:
    return media.Attachment(
        path=path,
        mime="audio/ogg",
        media_type=media_type,
    )


class ConfigurationTests(unittest.TestCase):
    def test_defaults_are_the_production_openai_path(self):
        transcription.validate_settings(Settings())
        self.assertTrue(transcription.transcript_echo_enabled(Settings()))

    def test_another_provider_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "stt.provider"):
            transcription.validate_settings(
                Settings({"stt": {"provider": "local"}})
            )

    def test_unknown_openai_model_is_refused(self):
        with self.assertRaisesRegex(ConfigError, "stt.openai.model"):
            transcription.validate_settings(
                Settings({"stt": {"openai": {"model": "future-model"}}})
            )

    def test_api_key_is_refused_in_behavioral_configuration(self):
        with self.assertRaisesRegex(ConfigError, "VOICE_TOOLS_OPENAI_KEY"):
            transcription.validate_settings(
                Settings({"stt": {"openai": {"api_key": "secret"}}})
            )

    def test_voice_key_precedes_the_general_openai_key(self):
        with mock.patch.dict(
            os.environ,
            {
                "VOICE_TOOLS_OPENAI_KEY": "voice-key",
                "OPENAI_API_KEY": "general-key",
            },
        ):
            self.assertEqual(
                transcription._resolve_openai_audio_api_key(),
                "voice-key",
            )


class RequestTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.audio = self.root / "voice.ogg"
        self.audio.write_bytes(b"OggS voice")

    def test_whisper_request_uses_text_response_and_language(self):
        captured = {}

        class Endpoint:
            def create(self, **kwargs):
                captured.update(kwargs)
                return "bonjour"

        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=Endpoint())
        )
        result = transcription._create_transcription(
            client,
            str(self.audio),
            "whisper-1",
            "fr",
            "Food Link",
        )

        self.assertEqual(result, "bonjour")
        self.assertEqual(captured["model"], "whisper-1")
        self.assertEqual(captured["response_format"], "text")
        self.assertEqual(captured["language"], "fr")
        self.assertEqual(captured["prompt"], "Food Link")

    def test_gpt_transcribe_uses_the_languages_list(self):
        captured = {}

        class Endpoint:
            def create(self, **kwargs):
                captured.update(kwargs)
                return {"text": "bonjour"}

        client = SimpleNamespace(
            audio=SimpleNamespace(transcriptions=Endpoint())
        )
        transcription._create_transcription(
            client,
            str(self.audio),
            "gpt-transcribe",
            "fr",
            None,
        )

        self.assertEqual(captured["response_format"], "json")
        self.assertEqual(captured["extra_body"], {"languages": ["fr"]})
        self.assertNotIn("language", captured)

    def test_missing_key_fails_without_constructing_a_client(self):
        with (
            mock.patch.dict(
                os.environ,
                {"VOICE_TOOLS_OPENAI_KEY": "", "OPENAI_API_KEY": ""},
            ),
            mock.patch("openai.OpenAI") as client,
        ):
            result = transcription.transcribe_audio(
                str(self.audio), _settings()
            )

        self.assertFalse(result["success"])
        self.assertIn("VOICE_TOOLS_OPENAI_KEY", result["error"])
        client.assert_not_called()

    def test_openai_transcription_returns_normalized_text_and_closes_client(self):
        client = mock.Mock()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "VOICE_TOOLS_OPENAI_KEY": "voice-key",
                    "OPENAI_API_KEY": "",
                },
            ),
            mock.patch("openai.OpenAI", return_value=client) as factory,
            mock.patch.object(
                transcription,
                "_create_transcription",
                return_value=SimpleNamespace(text="  Bonjour  "),
            ),
        ):
            result = transcription.transcribe_audio(
                str(self.audio), _settings()
            )

        self.assertEqual(
            result,
            {"success": True, "transcript": "Bonjour", "provider": "openai"},
        )
        self.assertEqual(factory.call_args.kwargs["api_key"], "voice-key")
        self.assertEqual(
            str(factory.call_args.kwargs["base_url"]),
            transcription.OPENAI_BASE_URL,
        )
        client.close.assert_called_once_with()

    def test_provider_rejected_ogg_is_transcoded_and_retried_once(self):
        request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
        response = httpx.Response(400, request=request)
        rejected = BadRequestError(
            "Invalid file format",
            response=response,
            body={"error": {"message": "Invalid file format"}},
        )
        converted = self.root / "converted.m4a"
        converted.write_bytes(b"m4a")
        client = mock.Mock()

        with (
            mock.patch.dict(
                os.environ,
                {"VOICE_TOOLS_OPENAI_KEY": "voice-key"},
            ),
            mock.patch("openai.OpenAI", return_value=client),
            mock.patch.object(
                transcription,
                "_create_transcription",
                side_effect=[rejected, SimpleNamespace(text="Bonjour")],
            ) as create,
            mock.patch.object(
                transcription,
                "_transcode_audio_for_stt",
                return_value=(str(converted), None),
            ) as transcode,
        ):
            result = transcription.transcribe_audio(
                str(self.audio), _settings()
            )

        self.assertTrue(result["success"])
        self.assertEqual(create.call_count, 2)
        transcode.assert_called_once()

    def test_unsupported_file_never_reaches_openai(self):
        unsupported = self.root / "voice.txt"
        unsupported.write_text("not audio", encoding="utf-8")
        with mock.patch("openai.OpenAI") as client:
            result = transcription.transcribe_audio(
                str(unsupported), _settings()
            )
        self.assertFalse(result["success"])
        self.assertIn("Unsupported audio format", result["error"])
        client.assert_not_called()

    def test_remote_upload_limit_is_enforced_before_openai(self):
        with (
            mock.patch.object(transcription, "MAX_FILE_SIZE", 1),
            mock.patch("openai.OpenAI") as client,
        ):
            result = transcription.transcribe_audio(
                str(self.audio), _settings()
            )
        self.assertFalse(result["success"])
        self.assertIn("too large", result["error"])
        client.assert_not_called()
    def test_useful_silence_trim_is_kept_for_upload(self):
        work = self.root / "trim-work"

        def make_work_dir(**_kwargs):
            work.mkdir()
            return str(work)

        def encode(_ffmpeg, _input, output, **_kwargs):
            Path(output).write_bytes(b"trimmed")

        with (
            mock.patch.object(transcription, "_find_binary", return_value="ffmpeg"),
            mock.patch.object(
                transcription,
                "_probe_audio_duration",
                side_effect=[20.0, 10.0],
            ),
            mock.patch.object(
                transcription,
                "_run_ffmpeg_stt_encode",
                side_effect=encode,
            ),
            mock.patch.object(
                transcription.tempfile,
                "mkdtemp",
                side_effect=make_work_dir,
            ),
        ):
            trimmed = transcription._trim_silence_for_cloud_stt(
                str(self.audio),
                _settings({"cloud_trim_silence": True}),
            )

        self.assertIsNotNone(trimmed)
        self.assertTrue(Path(str(trimmed)).is_file())

    def test_unhelpful_silence_trim_is_removed(self):
        work = self.root / "trim-work"

        def make_work_dir(**_kwargs):
            work.mkdir()
            return str(work)

        def encode(_ffmpeg, _input, output, **_kwargs):
            Path(output).write_bytes(b"trimmed")

        with (
            mock.patch.object(transcription, "_find_binary", return_value="ffmpeg"),
            mock.patch.object(
                transcription,
                "_probe_audio_duration",
                side_effect=[20.0, 19.0],
            ),
            mock.patch.object(transcription, "_run_ffmpeg_stt_encode", side_effect=encode),
            mock.patch.object(transcription.tempfile, "mkdtemp", side_effect=make_work_dir),
        ):
            trimmed = transcription._trim_silence_for_cloud_stt(
                str(self.audio),
                _settings({"cloud_trim_silence": True}),
            )

        self.assertIsNone(trimmed)
        self.assertFalse(work.exists())



class MessageEnrichmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.audio = self.root / "voice.ogg"
        self.audio.write_bytes(b"OggS voice")
        self.voice = _attachment(self.audio)

    async def test_transcript_is_plain_quoted_content_before_the_caption(self):
        with mock.patch.object(
            transcription,
            "transcribe_audio",
            return_value={
                "success": True,
                "transcript": "Montre les ventes",
                "provider": "openai",
            },
        ):
            text, transcripts = await transcription.enrich_message(
                "pour ce mois",
                [self.voice],
                _settings(),
            )

        self.assertEqual(text, '"Montre les ventes"\n\npour ce mois')
        self.assertEqual(transcripts, ["Montre les ventes"])

    async def test_a_duplicate_cached_path_is_transcribed_once(self):
        with mock.patch.object(
            transcription,
            "transcribe_audio",
            return_value={
                "success": True,
                "transcript": "Bonjour",
                "provider": "openai",
            },
        ) as transcribe:
            text, transcripts = await transcription.enrich_message(
                "",
                [self.voice, self.voice],
                _settings(),
            )

        self.assertEqual(text, '"Bonjour"')
        self.assertEqual(transcripts, ["Bonjour"])
        transcribe.assert_called_once()

    async def test_audio_upload_is_not_mistaken_for_a_voice_note(self):
        uploaded_audio = _attachment(self.audio, media_type="audio")
        with mock.patch.object(transcription, "transcribe_audio") as transcribe:
            text, transcripts = await transcription.enrich_message(
                "listen to this",
                [uploaded_audio],
                _settings(),
            )

        self.assertEqual(text, "listen to this")
        self.assertEqual(transcripts, [])
        transcribe.assert_not_called()

    async def test_disabled_stt_keeps_a_model_visible_voice_handle(self):
        with mock.patch.object(transcription, "transcribe_audio") as transcribe:
            text, transcripts = await transcription.enrich_message(
                "",
                [self.voice],
                _settings({"enabled": False}),
            )

        self.assertIn("The user sent a voice message", text)
        self.assertIn(str(self.audio.resolve()), text)
        self.assertEqual(transcripts, [])
        transcribe.assert_not_called()

    async def test_failure_is_neutral_and_keeps_the_audio_handle(self):
        with mock.patch.object(
            transcription,
            "transcribe_audio",
            return_value={
                "success": False,
                "transcript": "",
                "error": "no key",
            },
        ):
            text, transcripts = await transcription.enrich_message(
                "",
                [self.voice],
                _settings(),
            )

        self.assertIn("could not be transcribed automatically", text)
        self.assertIn(str(self.audio.resolve()), text)
        self.assertNotIn("no key", text)
        self.assertEqual(transcripts, [])

    async def test_empty_success_tells_the_model_not_to_guess(self):
        with mock.patch.object(
            transcription,
            "transcribe_audio",
            return_value={
                "success": True,
                "transcript": " ",
                "provider": "openai",
            },
        ):
            text, transcripts = await transcription.enrich_message(
                "",
                [self.voice],
                _settings(),
            )

        self.assertIn("empty or inaudible", text)
        self.assertIn("Do not guess", text)
        self.assertEqual(transcripts, [])

    def test_media_fallback_skips_ptt_but_still_names_uploaded_audio(self):
        self.assertEqual(media.describe_unreadable([self.voice]), "")
        note = media.describe_unreadable(
            [_attachment(self.audio, media_type="audio")]
        )
        self.assertIn("audio file", note)


class RuntimeWiringTests(unittest.IsolatedAsyncioTestCase):
    async def test_successful_transcript_is_echoed_and_given_to_the_agent(self):
        from pilotage import main
        from pilotage.channels.whatsapp import InboundMessage
        from pilotage.config import Config

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        with mock.patch.dict(
            os.environ,
            {
                "PILOTAGE_HOME": temp.name,
                "PILOTAGE_ALLOWED_SENDERS": "123",
            },
        ):
            config = Config.load(channel="whatsapp")
            object.__setattr__(config, "cron_enabled", False)
            seen = {}
            sent = []

            class FakeAgent:
                def __init__(self, _config, **_runtime_dependencies):
                    pass

                async def close(self):
                    pass

                async def respond(
                    self,
                    _session_id,
                    text,
                    _attachments,
                    *,
                    on_notice,
                    origin,
                    approval_notify,
                ):
                    seen["text"] = text
                    return "answer"

            class FakeChannel:
                def __init__(self, _config, handler, _manage):
                    self.handler = handler
                    self.stopped = asyncio.Event()
                    self.failure = None

                @contextlib.asynccontextmanager
                async def typing(self, _chat_id):
                    yield

                async def send(self, *args, **kwargs):
                    sent.append((args, kwargs))
                    return True

                async def start(self):
                    await self.handler(
                        InboundMessage(
                            chat_id="123@c.us",
                            session_id="123",
                            sender_id="123@s.whatsapp.net",
                            sender_number="123",
                            push_name="User",
                            text="",
                            is_group=False,
                            message_ids=["m1"],
                        )
                    )
                    self.stopped.set()

                async def stop_intake(self):
                    pass

                async def stop(self, *, drain_timeout_seconds=0):
                    pass

            with (
                mock.patch.object(main, "Agent", FakeAgent),
                mock.patch.object(main, "WhatsAppChannel", FakeChannel),
                mock.patch.object(main.auth, "read_credentials"),
                mock.patch.object(
                    main.transcription,
                    "enrich_message",
                    new=mock.AsyncMock(return_value=('"spoken"', ["spoken"])),
                ),
            ):
                self.assertEqual(await main.command_run(config), 0)

        self.assertEqual(seen["text"], '"spoken"')
        self.assertEqual(
            sent[0],
            (
                ("123@c.us", '🎙️ "spoken"', "m1"),
                {"deliver_media": False},
            ),
        )
        self.assertEqual(sent[1], (("123@c.us", "answer", "m1"), {}))


if __name__ == "__main__":
    unittest.main()
