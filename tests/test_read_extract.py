"""Focused contracts for Hermes-backed structured-document reads."""

from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilotage.i18n import t
from pilotage.settings import Settings
from pilotage.tools import read_extract
from pilotage.tools.files import _read
from pilotage.tools.read_extract import (
    DocumentTooLarge,
    ExtractionError,
    extract_document_bytes,
    extract_document_text,
    is_extractable_document,
)
from pilotage.tools.registry import ToolContext


WORD_NAMESPACE = (
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
)
SHEET_NAMESPACE = (
    "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
)
RELATIONSHIP_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    body = "".join(
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>"
        for paragraph in paragraphs
    )
    document = (
        f'<w:document xmlns:w="{WORD_NAMESPACE}">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)


def _write_xlsx(path: Path) -> None:
    workbook = (
        f'<workbook xmlns="{SHEET_NAMESPACE}" '
        f'xmlns:r="{RELATIONSHIP_NAMESPACE}"><sheets>'
        '<sheet name="Data" sheetId="1" r:id="rId1"/>'
        '<sheet name="Hidden" sheetId="2" state="hidden" r:id="rId2"/>'
        "</sheets></workbook>"
    )
    relationships = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/'
        'package/2006/relationships">'
        '<Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="x"/>'
        '<Relationship Id="rId2" Target="worksheets/sheet2.xml" Type="x"/>'
        "</Relationships>"
    )
    shared = (
        f'<sst xmlns="{SHEET_NAMESPACE}">'
        "<si><t>Name</t></si><si><t>Alice</t></si></sst>"
    )
    visible = (
        f'<worksheet xmlns="{SHEET_NAMESPACE}"><sheetData>'
        '<row r="1"><c r="A1" t="s"><v>0</v></c>'
        '<c r="B1"><v>95</v></c></row>'
        '<row r="2"><c r="A2" t="s"><v>1</v></c></row>'
        "</sheetData></worksheet>"
    )
    hidden = (
        f'<worksheet xmlns="{SHEET_NAMESPACE}"><sheetData>'
        '<row r="1"><c r="A1" t="inlineStr">'
        "<is><t>SECRET-HIDDEN</t></is></c></row>"
        "</sheetData></worksheet>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", visible)
        archive.writestr("xl/worksheets/sheet2.xml", hidden)


def _write_forged_office_archive(
    path: Path, parts: dict[str, bytes], padding: dict[str, int],
    compression: int = zipfile.ZIP_DEFLATED,
) -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        for name, content in parts.items():
            archive.writestr(name, content + b" " * padding.get(name, 0))
    data = bytearray(stream.getvalue())
    with zipfile.ZipFile(stream) as archive:
        central = archive.start_dir
        for info in archive.infolist():
            # A valid XML prefix has its own forged size and CRC, hiding
            # padding that the actual ZIP member still contains.
            for start, crc_offset, size_offset in (
                (info.header_offset, 14, 22), (central, 16, 24),
            ):
                struct.pack_into("<I", data, start + crc_offset, zlib.crc32(parts[info.filename]))
                struct.pack_into("<I", data, start + size_offset, len(parts[info.filename]))
            name_size, extra_size, comment_size = struct.unpack_from("<HHH", data, central + 28)
            central += 46 + name_size + extra_size + comment_size
    path.write_bytes(data)


class _Config:
    def __init__(self, workspace: Path):
        self.settings = Settings({"terminal": {"cwd": str(workspace)}})
        self.workspace_dir = workspace
        self.state_dir = workspace
        self.max_tool_result_chars = 100_000


class ExtractorTests(unittest.TestCase):
    def test_recognizes_current_core_formats(self):
        for name in ("book.ipynb", "report.docx", "data.xlsx", "scan.pdf"):
            self.assertTrue(is_extractable_document(name))
        self.assertFalse(is_extractable_document("notes.txt"))

    def test_notebook_keeps_cells_and_useful_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis.ipynb"
            path.write_text(
                json.dumps({
                    "cells": [
                        {"cell_type": "markdown", "source": ["# Result\n"]},
                        {
                            "cell_type": "code",
                            "source": "print('done')",
                            "outputs": [{
                                "output_type": "stream",
                                "text": ["half\rcomplete\n"],
                            }],
                        },
                    ],
                    "metadata": {},
                    "nbformat": 4,
                }),
                encoding="utf-8",
            )
            text = extract_document_text(str(path))

        self.assertLess(text.index("Result"), text.index("print('done')"))
        self.assertIn("complete", text)
        self.assertNotIn("half", text)
        self.assertNotIn("output_type", text)

    def test_docx_extracts_paragraphs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            _write_docx(path, ["First paragraph", "Second paragraph"])
            text = extract_document_text(str(path))

        self.assertEqual(text, "First paragraph\nSecond paragraph\n")

    def test_xlsx_reads_visible_sheet_and_omits_hidden_sheet(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workbook.xlsx"
            _write_xlsx(path)
            text = extract_document_text(str(path))

        self.assertIn("Sheet: Data", text)
        self.assertIn("Name\t95", text)
        self.assertIn("Alice", text)
        self.assertNotIn("SECRET-HIDDEN", text)

    def test_malformed_docx_has_a_specific_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.docx"
            path.write_bytes(b"not a zip")
            with self.assertRaisesRegex(ExtractionError, "valid DOCX"):
                extract_document_text(str(path))

    def test_compressed_docx_is_rejected_before_xml_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            _write_docx(path, ["A" * 8192])
            self.assertLess(path.stat().st_size, 1024)
            with (
                mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", 1024),
                mock.patch.object(read_extract.ET, "fromstring") as parse,
                self.assertRaises(DocumentTooLarge),
            ):
                extract_document_text(str(path))
            parse.assert_not_called()

    def test_docx_expansion_boundary_counts_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            paragraph = "Été مرحبا" * 100
            _write_docx(path, [paragraph])
            with zipfile.ZipFile(path) as archive:
                expanded = archive.getinfo("word/document.xml").file_size
            for budget, accepted in ((expanded, True), (expanded - 1, False)):
                with (
                    self.subTest(budget=budget),
                    mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", budget),
                ):
                    if accepted:
                        self.assertEqual(extract_document_text(str(path)), paragraph + "\n")
                    else:
                        with self.assertRaises(DocumentTooLarge):
                            extract_document_text(str(path))

    def test_stored_and_deflated_office_xml_remain_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            for extension, write, expected in (
                ("docx", lambda path: _write_docx(path, ["Ready"]), "Ready"),
                ("xlsx", _write_xlsx, "Name\t95"),
            ):
                path = Path(directory) / f"report.{extension}"
                write(path)
                with zipfile.ZipFile(path) as archive:
                    parts = {name: archive.read(name) for name in archive.namelist()}
                for method in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    with self.subTest(extension=extension, method=method):
                        with zipfile.ZipFile(path, "w", compression=method) as archive:
                            for name, content in parts.items():
                                archive.writestr(name, content)
                        self.assertIn(expected, extract_document_text(str(path)))

    def test_unbounded_codecs_are_rejected_before_decompression_with_forged_sizes(self):
        document = (
            f'<w:document xmlns:w="{WORD_NAMESPACE}"><w:body>'
            '<w:p><w:r><w:t>Ready</w:t></w:r></w:p>'
            '</w:body></w:document>'
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            for method in (zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA):
                with self.subTest(method=method):
                    stream = io.BytesIO()
                    with zipfile.ZipFile(stream, "w", compression=method) as archive:
                        archive.writestr("word/document.xml", document + b" " * 65_536)
                    data = bytearray(stream.getvalue())
                    # ZIP's declared size and CRC can describe only the valid XML
                    # prefix while the codec still expands the entire padded body.
                    for marker, crc_offset, size_offset in (
                        (b"PK\x03\x04", 14, 22),
                        (b"PK\x01\x02", 16, 24),
                    ):
                        start = data.index(marker)
                        struct.pack_into("<I", data, start + crc_offset, zlib.crc32(document))
                        struct.pack_into("<I", data, start + size_offset, len(document))
                    path.write_bytes(data)
                    with zipfile.ZipFile(path) as archive:
                        self.assertLess(archive.getinfo("word/document.xml").file_size, 1024)
                    with (
                        mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", 1024),
                        mock.patch.object(zipfile, "_get_decompressor", wraps=zipfile._get_decompressor) as codec,
                        self.assertRaises(read_extract.UnsupportedDocumentCompression),
                    ):
                        extract_document_text(str(path))
                    codec.assert_not_called()

    def test_xlsx_budget_is_shared_across_consumed_parts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.xlsx"
            _write_xlsx(path)
            with zipfile.ZipFile(path) as archive:
                consumed = [
                    item.file_size for item in archive.infolist()
                    if item.filename != "xl/worksheets/sheet2.xml"
                ]
            # Every member fits separately; their combined expansion does not.
            self.assertLess(max(consumed), sum(consumed) - 1)
            with mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", sum(consumed)):
                self.assertIn("Name\t95", extract_document_text(str(path)))
            with (
                mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", sum(consumed) - 1),
                self.assertRaises(DocumentTooLarge),
            ):
                extract_document_text(str(path))

    def test_all_xlsx_xml_read_paths_enforce_expansion_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.xlsx"
            _write_xlsx(path)
            with zipfile.ZipFile(path) as archive:
                parts = {name: archive.read(name) for name in archive.namelist()}
            for target in (
                "xl/sharedStrings.xml", "xl/workbook.xml",
                "xl/_rels/workbook.xml.rels", "xl/worksheets/sheet1.xml",
            ):
                with self.subTest(part=target):
                    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                        for name, content in parts.items():
                            archive.writestr(name, content + (b" " * 8192 if name == target else b""))
                    self.assertLess(path.stat().st_size, 8192)
                    with (
                        mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", 8192),
                        self.assertRaises(DocumentTooLarge),
                    ):
                        extract_document_text(str(path))

    def test_unread_office_members_do_not_consume_expansion_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.xlsx"
            _write_xlsx(path)
            with zipfile.ZipFile(path) as archive:
                parts = {name: archive.read(name) for name in archive.namelist()}
            consumed = sum(len(content) for name, content in parts.items()
                           if name != "xl/worksheets/sheet2.xml")
            parts["xl/worksheets/sheet2.xml"] += b" " * 8192
            parts["xl/media/image1.bin"] = b"X" * 8192
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, content in parts.items():
                    archive.writestr(name, content)
            with mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", consumed):
                text = extract_document_text(str(path))
            self.assertIn("Alice", text)
            self.assertNotIn("SECRET-HIDDEN", text)

    def test_later_sheet_over_budget_does_not_return_a_partial_workbook(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.xlsx"
            _write_xlsx(path)
            with zipfile.ZipFile(path) as archive:
                parts = {name: archive.read(name) for name in archive.namelist()}
            parts["xl/workbook.xml"] = parts["xl/workbook.xml"].replace(b' state="hidden"', b"")
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, content in parts.items():
                    archive.writestr(name, content)
            with (
                mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES",
                                  sum(map(len, parts.values())) - 1),
                self.assertRaises(DocumentTooLarge),
            ):
                extract_document_text(str(path))

    def test_forged_deflate_size_cannot_hide_docx_expansion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            _write_docx(path, ["Ready"])
            with zipfile.ZipFile(path) as archive:
                document = archive.read("word/document.xml")
            _write_forged_office_archive(path, {"word/document.xml": document},
                                        {"word/document.xml": 65_536})
            with zipfile.ZipFile(path) as archive:
                self.assertLess(archive.getinfo("word/document.xml").file_size, 1024)
            with (
                mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", 1024),
                mock.patch.object(read_extract.ET, "fromstring") as parse,
                self.assertRaises(DocumentTooLarge),
            ):
                extract_document_text(str(path))
            parse.assert_not_called()

    def test_forged_stored_size_is_rejected_before_reading_the_member(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            _write_docx(path, ["Ready"])
            with zipfile.ZipFile(path) as archive:
                document = archive.read("word/document.xml")
            _write_forged_office_archive(path, {"word/document.xml": document},
                                        {"word/document.xml": 8192}, zipfile.ZIP_STORED)
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo("word/document.xml")
                self.assertEqual(info.file_size, len(document))
                self.assertEqual(info.compress_size, len(document) + 8192)
            with (
                mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", 1024),
                mock.patch.object(zipfile.ZipFile, "open") as member_open,
                self.assertRaisesRegex(ExtractionError, "Inconsistent stored document size"),
            ):
                extract_document_text(str(path))
            member_open.assert_not_called()

    def test_forged_xlsx_parts_charge_actual_inflation_including_skipped_xml(self):
        decoded: list[int] = []
        get_decompressor = zipfile._get_decompressor

        class CountingInflater:
            def __init__(self, method):
                self._inner = get_decompressor(method)

            @property
            def eof(self):
                return self._inner.eof

            @property
            def unconsumed_tail(self):
                return self._inner.unconsumed_tail

            def decompress(self, data, max_length=0):
                output = self._inner.decompress(data, max_length)
                decoded.append(len(output))
                return output

            def flush(self):
                output = self._inner.flush()
                decoded.append(len(output))
                return output

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.xlsx"
            _write_xlsx(path)
            with zipfile.ZipFile(path) as archive:
                original_parts = {name: archive.read(name) for name in archive.namelist()}
            for skipped_sheet in (False, True):
                with self.subTest(skipped_sheet=skipped_sheet):
                    parts = dict(original_parts)
                    if skipped_sheet:
                        parts["xl/workbook.xml"] = parts["xl/workbook.xml"].replace(b' state="hidden"', b"")
                        parts["xl/worksheets/sheet1.xml"] = b"<malformed>"
                        padding = {"xl/worksheets/sheet1.xml": 2200,
                                   "xl/worksheets/sheet2.xml": 2200}
                    else:
                        padding = {name: 2000 for name in parts}
                    _write_forged_office_archive(path, parts, padding)
                    decoded.clear()
                    with (
                        mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", 4096),
                        mock.patch.object(zipfile, "_get_decompressor", side_effect=CountingInflater),
                        self.assertRaises(DocumentTooLarge),
                    ):
                        extract_document_text(str(path))
                    self.assertGreater(len(decoded), 1)
                    self.assertEqual(sum(decoded), 4097)

    def test_incomplete_deflate_is_rejected_without_an_unbounded_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            _write_docx(path, ["Ready"])
            with zipfile.ZipFile(path) as archive:
                document = archive.read("word/document.xml")
            _write_forged_office_archive(path, {"word/document.xml": document},
                                        {"word/document.xml": 100})
            data = bytearray(path.read_bytes())
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo("word/document.xml")
                for start, offset in ((info.header_offset, 18), (archive.start_dir, 20)):
                    struct.pack_into("<I", data, start + offset, info.compress_size - 1)
            path.write_bytes(data)
            with self.assertRaisesRegex(ExtractionError, "Incomplete compressed document content"):
                extract_document_text(str(path))

    def test_metered_deflate_preserves_zip_crc_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            _write_docx(path, ["Ready"])
            data = bytearray(path.read_bytes())
            with zipfile.ZipFile(path) as archive:
                central = archive.start_dir
                for info in archive.infolist():
                    if info.filename == "word/document.xml":
                        struct.pack_into("<I", data, central + 16, info.CRC ^ 1)
                    name_size, extra_size, comment_size = struct.unpack_from("<HHH", data, central + 28)
                    central += 46 + name_size + extra_size + comment_size
            path.write_bytes(data)
            with self.assertRaisesRegex(ExtractionError, "Bad CRC-32"):
                extract_document_text(str(path))

    def test_byte_entry_point_enforces_expanded_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.docx"
            _write_docx(path, ["A" * 8192])
            data = path.read_bytes()
        with (
            mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", 1024),
            self.assertRaises(DocumentTooLarge),
        ):
            extract_document_bytes(data, "report.docx")

    def test_pdf_uses_preinstalled_pdftotext_without_installing(self):
        from pilotage.tools import read_extract

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.pdf"
            path.write_bytes(b"%PDF-1.4")
            process = SimpleNamespace(
                returncode=0,
                stdout=(
                    b"Section one has enough readable text.\f"
                    b"\f"
                    b"\f"
                    b"Section four has enough readable text.\f"
                ),
            )
            with (
                mock.patch.object(read_extract, "_anydoc_module", None),
                mock.patch.object(
                    read_extract.shutil,
                    "which",
                    return_value="/usr/bin/pdftotext",
                ),
                mock.patch.object(
                    read_extract.subprocess,
                    "run",
                    return_value=process,
                ) as run,
            ):
                text = extract_document_text(str(path))

        self.assertIn("Section one", text)
        self.assertIn("2 of 4 pages", text)
        self.assertIn("pages 2-3", text)
        run.assert_called_once()

    def test_pdf_fails_loudly_when_deployment_dependency_is_missing(self):
        from pilotage.tools import read_extract

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.pdf"
            path.write_bytes(b"%PDF-1.4")
            with (
                mock.patch.object(read_extract, "_anydoc_module", None),
                mock.patch.object(read_extract.shutil, "which", return_value=None),
                self.assertRaisesRegex(ExtractionError, "pdftotext"),
            ):
                extract_document_text(str(path))


class ReadFileIntegrationTests(unittest.TestCase):
    def _read_document(
        self,
        workspace: Path,
        name: str,
        *,
        offset: int = 1,
        limit: int = 2000,
        language: str = "en",
    ) -> dict:
        context = ToolContext("chat", _Config(workspace))
        context.config.language = language
        shell = SimpleNamespace(cwd=str(workspace))

        def add_line_numbers(content: str, start_line: int = 1) -> str:
            return "\n".join(
                f"{start_line + index}|{line}"
                for index, line in enumerate(content.splitlines())
            )

        operations = SimpleNamespace(
            _add_line_numbers=add_line_numbers,
            read_file=mock.Mock(
                side_effect=AssertionError(
                    "structured documents must not use the raw reader"
                )
            ),
        )
        with mock.patch("pilotage.tools.files.file_state.record_read"):
            return json.loads(
                _read(
                    {"path": name, "offset": offset, "limit": limit},
                    context,
                    shell,
                    operations,
                )
            )

    def test_document_read_is_line_numbered_and_paginated(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_docx(
                workspace / "report.docx",
                ["First", "Second", "Third"],
            )
            result = self._read_document(
                workspace,
                "report.docx",
                offset=2,
                limit=1,
            )

        self.assertTrue(result["extracted_document"])
        self.assertEqual(result["content"], "2|Second")
        self.assertEqual(result["total_lines"], 3)
        self.assertTrue(result["truncated"])

    def test_corrupt_document_surfaces_extraction_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "bad.docx").write_bytes(b"not a zip")
            result = self._read_document(workspace, "bad.docx")

        self.assertIn("document extraction failed", result["error"])
        self.assertIn("valid DOCX", result["error"])

    def test_document_limit_returns_plain_localized_failure_even_for_small_page(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            _write_docx(workspace / "private-client-report.docx", ["A" * 8192])
            for language in ("en", "fr", "ar"):
                with (
                    self.subTest(language=language),
                    mock.patch.object(read_extract, "_MAX_OFFICE_XML_BYTES", 1024),
                ):
                    result = self._read_document(
                        workspace, "private-client-report.docx", limit=1, language=language,
                    )
                    self.assertEqual(result["error"], t("document.too_large", language))
                    self.assertNotIn("content", result)
                    self.assertNotIn("private-client-report", result["error"])

    def test_unsupported_compression_returns_plain_localized_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            path = workspace / "private-client-report.docx"
            _write_docx(path, ["Ready"])
            with zipfile.ZipFile(path) as archive:
                document = archive.read("word/document.xml")
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_BZIP2) as archive:
                archive.writestr("word/document.xml", document)
            for language in ("en", "fr", "ar"):
                with self.subTest(language=language):
                    result = self._read_document(workspace, path.name, language=language)
                    self.assertEqual(result["error"], t("document.unreadable", language))
                    self.assertNotIn("content", result)
                    self.assertNotIn("private-client-report", result["error"])


if __name__ == "__main__":
    unittest.main()
