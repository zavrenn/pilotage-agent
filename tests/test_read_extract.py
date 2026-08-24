"""Focused contracts for Hermes-backed structured-document reads."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from pilotage.settings import Settings
from pilotage.tools.files import _read
from pilotage.tools.read_extract import (
    ExtractionError,
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
    with zipfile.ZipFile(path, "w") as archive:
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
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", visible)
        archive.writestr("xl/worksheets/sheet2.xml", hidden)


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
    ) -> dict:
        context = ToolContext("chat", _Config(workspace))
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


if __name__ == "__main__":
    unittest.main()
