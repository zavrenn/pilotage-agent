"""Structured-document text extraction used by read_file.

This is the current Hermes stdlib extractor narrowed to Pilotage's local
runtime.  It supports notebooks, DOCX, and XLSX without Python dependencies.
PDFs use a preinstalled anydoc binding when available, otherwise the
deployment-provided pdftotext executable.  Nothing is installed at request
time.
"""

from __future__ import annotations

import importlib
import json
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

from .ansi_strip import strip_ansi

__all__ = [
    "ANYDOC_EXTENSIONS",
    "EXTRACTABLE_EXTENSIONS",
    "MAX_DOCUMENT_BYTES",
    "ExtractionError",
    "extract_document_bytes",
    "extract_document_text",
    "is_extractable_document",
]


EXTRACTABLE_EXTENSIONS = frozenset({".ipynb", ".docx", ".xlsx", ".pdf"})
ANYDOC_EXTENSIONS = frozenset({
    ".doc", ".docm",
    ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".xls", ".xlsm", ".xlsb",
    ".odt", ".ods", ".odp",
    ".rtf", ".epub",
})
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
_MAX_XLSX_ROWS_PER_SHEET = 5000
_MAX_XLSX_COLS = 256
_MAX_OUTPUT_CHARS = 20_000

_NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_NS_S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

PDF_EMPTY_PAGE_CHARS = 20
PDF_COVERAGE_MIN_EMPTY = 2
PDF_COVERAGE_MIN_RATIO = 0.2
PDF_COVERAGE_ABSOLUTE_EMPTY = 10
PDF_PAGE_SCAN_TIMEOUT = 20.0
PDF_GAP_MAP_MAX_ENTRIES = 20
_GAP_CONTEXT_CHARS = 60


class ExtractionError(Exception):
    """A supported-looking document could not be rendered as text."""


_ANYDOC_UNSET = object()
_anydoc_module: Any = _ANYDOC_UNSET
_anydoc_lock = threading.Lock()


def _anydoc() -> Optional[Any]:
    """Load a converter already installed by deployment; never install it."""
    global _anydoc_module
    if _anydoc_module is not _ANYDOC_UNSET:
        return _anydoc_module
    with _anydoc_lock:
        if _anydoc_module is _ANYDOC_UNSET:
            try:
                _anydoc_module = importlib.import_module("anydoc")
            except Exception:
                _anydoc_module = None
    return _anydoc_module


def _extension(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext in EXTRACTABLE_EXTENSIONS:
        return ext
    if ext in ANYDOC_EXTENSIONS and _anydoc() is not None:
        return ext
    return ""


def is_extractable_document(path: str) -> bool:
    return bool(_extension(path))


def _check_size(path: str) -> int:
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    if size > MAX_DOCUMENT_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({size:,} bytes, "
            f"limit is {MAX_DOCUMENT_BYTES:,})"
        )
    return size


def extract_document_text(path: str) -> str:
    _check_size(path)
    ext = _extension(path)
    if ext == ".ipynb":
        return _extract_notebook(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".xlsx":
        return _extract_xlsx(path)
    if ext == ".pdf":
        return _extract_pdf(path)
    if ext in ANYDOC_EXTENSIONS:
        return _extract_anydoc(path)
    raise ExtractionError(f"Unsupported document type: {path!r}")


def extract_document_bytes(data: bytes, path: str) -> str:
    """Extract document bytes while preserving the filename extension."""
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ExtractionError(
            f"Document too large to convert ({len(data):,} bytes, "
            f"limit is {MAX_DOCUMENT_BYTES:,})"
        )
    ext = Path(path).suffix.lower()
    if ext not in EXTRACTABLE_EXTENSIONS | ANYDOC_EXTENSIONS:
        raise ExtractionError(f"Unsupported document type: {path!r}")
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as stream:
            stream.write(data)
            temp_path = stream.name
        return extract_document_text(temp_path)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _extract_anydoc(path: str) -> str:
    converter = _anydoc()
    if converter is None:
        raise ExtractionError(
            "The document converter is not installed in the prepared runtime"
        )
    try:
        text = converter.to_markdown(path)
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    except Exception as exc:
        raise ExtractionError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Document contains no extractable text")
    return text.rstrip("\n") + "\n"


def _pdf_page_texts(path: str) -> Optional[list[str]]:
    if shutil.which("pdftotext") is None:
        return None
    try:
        process = subprocess.run(
            ["pdftotext", path, "-"],
            capture_output=True,
            timeout=PDF_PAGE_SCAN_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if process.returncode != 0:
        return None
    pages = process.stdout.decode("utf-8", errors="replace").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()
    return pages or None


def _extract_pdf(path: str) -> str:
    converter = _anydoc()
    if converter is not None:
        text = _extract_anydoc(path)
        note = _pdf_coverage_note(path)
        return note + text if note else text
    if shutil.which("pdftotext") is None:
        raise ExtractionError(
            "PDF extraction requires the deployment-provided pdftotext command"
        )
    pages = _pdf_page_texts(path)
    if pages is None:
        raise ExtractionError("pdftotext could not extract this PDF")
    text = "\n\n".join(page.strip() for page in pages).strip()
    if not text:
        raise ExtractionError(
            "PDF contains no extractable text; inspect its pages with vision or OCR"
        )
    note = _pdf_coverage_note_from_pages(pages, path)
    return note + text.rstrip("\n") + "\n"


def _group_ranges(pages: list[int]) -> list[list[int]]:
    ranges: list[list[int]] = []
    for page in pages:
        if ranges and page == ranges[-1][1] + 1:
            ranges[-1][1] = page
        else:
            ranges.append([page, page])
    return ranges


def _gap_map(counts: list[int], texts: list[str], empty: list[int]) -> str:
    ranges = _group_ranges(empty)
    lines: list[str] = []
    for first, last in ranges[:PDF_GAP_MAP_MAX_ENTRIES]:
        label = ""
        for previous in range(first - 2, -1, -1):
            if counts[previous] >= PDF_EMPTY_PAGE_CHARS:
                snippet = " ".join(texts[previous].split())[:_GAP_CONTEXT_CHARS]
                label = f' - after "{snippet}" (p{previous + 1})'
                break
        span = f"page {first}" if first == last else f"pages {first}-{last}"
        count = last - first + 1
        lines.append(
            f"  {span} ({count} page{'s' if count != 1 else ''}){label}"
        )
    if len(ranges) > PDF_GAP_MAP_MAX_ENTRIES:
        remainder = ranges[PDF_GAP_MAP_MAX_ENTRIES:]
        page_count = sum(last - first + 1 for first, last in remainder)
        lines.append(f"  ... {len(remainder)} more gaps ({page_count} pages)")
    return "\n".join(lines)


def _pdf_coverage_note_from_pages(
    texts: list[str],
    display_path: str,
) -> str:
    if len(texts) < 2:
        return ""
    counts = [len(page.strip()) for page in texts]
    empty = [
        index + 1
        for index, count in enumerate(counts)
        if count < PDF_EMPTY_PAGE_CHARS
    ]
    total = len(counts)
    if len(empty) < PDF_COVERAGE_MIN_EMPTY:
        return ""
    if (
        len(empty) / total < PDF_COVERAGE_MIN_RATIO
        and len(empty) < PDF_COVERAGE_ABSOLUTE_EMPTY
    ):
        return ""
    return (
        "[EXTRACTION COVERAGE WARNING: "
        f"{len(empty)} of {total} pages in this PDF yielded no text. "
        "Those pages are likely scanned images or blank, so their content "
        "is missing from the extracted text below. Unreadable gaps:\n"
        f"{_gap_map(counts, texts, empty)}\n"
        "Inspect only the needed gaps. Render a range with pdftoppm and "
        f"inspect the images with vision_analyze. Source: {display_path}]\n"
    )


def _pdf_coverage_note(path: str) -> str:
    pages = _pdf_page_texts(path)
    return "" if not pages else _pdf_coverage_note_from_pages(pages, path)


def _source_text(source: Any) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(item for item in source if isinstance(item, str))
    return ""


def _human_size(byte_count: int) -> str:
    return (
        f"{round(byte_count / 1024)} KB"
        if byte_count >= 1024
        else f"{byte_count} B"
    )


def _base64_bytes(payload: str) -> int:
    clean = re.sub(r"[^0-9+/=A-Za-z]", "", payload)
    padding = min(2, len(clean) - len(clean.rstrip("=")))
    return max(0, (len(clean) * 3) // 4 - padding)


def _clean_stream_text(text: str) -> str:
    cleaned = strip_ansi(text).replace("\r\n", "\n")
    lines = []
    for line in cleaned.split("\n"):
        frames = [frame for frame in line.split("\r") if frame]
        lines.append(frames[-1] if frames else "")
    return "\n".join(lines)


def _notebook_output_text(output: Any) -> str:
    if not isinstance(output, dict):
        return ""
    output_type = output.get("output_type")
    if output_type == "stream":
        body = _clean_stream_text(_source_text(output.get("text", "")))
        return body if body.strip() else ""
    if output_type in {"error", "pyerr"}:
        traceback = output.get("traceback")
        traceback_text = ""
        if isinstance(traceback, list):
            traceback_text = _clean_stream_text(
                "\n".join(line for line in traceback if isinstance(line, str))
            )
        header = (
            f"Error: {output.get('ename', '')}: "
            f"{output.get('evalue', '')}"
        ).rstrip(": ")
        return f"{header}\n{traceback_text}".rstrip()
    if output_type not in {"execute_result", "display_data", "pyout"}:
        return ""

    data = output.get("data")
    if not isinstance(data, dict):
        data = {}
        if isinstance(output.get("text"), (str, list)):
            data["text/plain"] = output["text"]
        for old_key, mime in (
            ("png", "image/png"),
            ("jpeg", "image/jpeg"),
            ("svg", "image/svg+xml"),
            ("html", "text/html"),
        ):
            if old_key in output:
                data[mime] = output[old_key]

    if "application/vnd.jupyter.widget-view+json" in data:
        return "[interactive widget - omitted]"
    for mime in ("text/plain", "text/markdown"):
        if mime in data:
            body = _clean_stream_text(_source_text(data[mime]))
            if body.strip():
                return body
    for mime, value in data.items():
        if isinstance(mime, str) and mime.startswith("image/"):
            size = _base64_bytes(_source_text(value))
            return f"[{mime} output - {_human_size(size)}, omitted]"
    if "text/html" in data:
        html = _source_text(data["text/html"])
        return f"[text/html output - {len(html):,} chars, omitted]"
    mime_names = ", ".join(str(mime) for mime in data) or "unknown"
    return f"[{mime_names} output - omitted]"


def _notebook_outputs(
    cell: dict[str, Any],
    jq_pointer: str = "",
    filename: str = "",
) -> str:
    outputs = cell.get("outputs")
    if not isinstance(outputs, list):
        return ""
    blocks = [
        text
        for text in (_notebook_output_text(output) for output in outputs)
        if text
    ]
    if not blocks:
        return ""
    joined = "\n".join(blocks)
    if len(joined) > _MAX_OUTPUT_CHARS:
        omitted = len(joined) - _MAX_OUTPUT_CHARS
        hint = ""
        if jq_pointer and filename:
            hint = f" - full output: jq -r '{jq_pointer}' {filename}"
        joined = (
            joined[:_MAX_OUTPUT_CHARS]
            + f"\n... [{omitted:,} output chars truncated{hint}]"
        )
    return joined


def _extract_notebook(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            notebook = json.load(stream)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ExtractionError(f"Not a valid notebook: {exc}") from exc
    if not isinstance(notebook, dict):
        raise ExtractionError("Notebook root is not an object")

    raw_cells = notebook.get("cells")
    if isinstance(raw_cells, list):
        cells = [
            (f".cells[{index}].outputs", cell)
            for index, cell in enumerate(raw_cells)
        ]
    else:
        cells = [
            (
                f".worksheets[{workbook_index}].cells[{cell_index}].outputs",
                cell,
            )
            for workbook_index, worksheet in enumerate(
                notebook.get("worksheets", [])
            )
            if isinstance(worksheet, dict)
            for cell_index, cell in enumerate(worksheet.get("cells", []))
        ]
    if not cells:
        raise ExtractionError("Notebook contains no cells")

    notebook_name = os.path.basename(path)
    counts = {"markdown": 0, "code": 0, "raw": 0}
    labels = {"markdown": "Markdown", "code": "Code", "raw": "Raw"}
    output: list[str] = []
    for jq_pointer, cell in cells:
        if not isinstance(cell, dict):
            continue
        cell_type = cell.get("cell_type")
        if cell_type not in labels:
            continue
        counts[cell_type] += 1
        suffix = f" {counts[cell_type]}" if cell_type != "raw" else ""
        output.extend((
            f"# -- {labels[cell_type]} cell{suffix} --",
            _source_text(cell.get("source", "")).rstrip("\n"),
            "",
        ))
        if cell_type == "code":
            rendered = _notebook_outputs(cell, jq_pointer, notebook_name)
            if rendered:
                output.extend((
                    f"# -- Output (cell {counts[cell_type]}) --",
                    rendered.rstrip("\n"),
                    "",
                ))
    if not output:
        raise ExtractionError("Notebook contains no readable cells")
    return "\n".join(output).rstrip("\n") + "\n"


def _zip_xml(archive: zipfile.ZipFile, name: str) -> ET.Element:
    try:
        return ET.fromstring(archive.read(name))
    except KeyError as exc:
        raise ExtractionError(f"Missing {name}") from exc
    except ET.ParseError as exc:
        raise ExtractionError(f"Malformed XML in {name}: {exc}") from exc


def _extract_docx(path: str) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            root = _zip_xml(archive, "word/document.xml")
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Not a valid DOCX: {exc}") from exc
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc

    namespace = f"{{{_NS_W}}}"
    lines: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        buffer: list[str] = []
        for node in paragraph.iter():
            if node.tag == f"{namespace}t":
                buffer.append(node.text or "")
            elif node.tag == f"{namespace}tab":
                buffer.append("\t")
            elif node.tag in {f"{namespace}br", f"{namespace}cr"}:
                buffer.append("\n")
        lines.extend("".join(buffer).split("\n"))
    if not any(line.strip() for line in lines):
        raise ExtractionError("DOCX contains no extractable text")
    return "\n".join(lines).rstrip("\n") + "\n"


def _extract_xlsx(path: str) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            shared = _shared_strings(archive, names)
            sheets = _workbook_sheets(archive)
            relationships = _workbook_rels(archive, names)
            output: list[str] = []
            for name, state, relationship_id in sheets:
                if state in {"hidden", "veryHidden"}:
                    continue
                part = _sheet_part(relationships.get(relationship_id, ""))
                if part not in names:
                    continue
                try:
                    rows = _sheet_rows(archive.read(part), shared)
                except ET.ParseError:
                    continue
                output.append(f"# -- Sheet: {name} --")
                output.extend("\t".join(row) for row in rows)
                if not rows:
                    output.append("(empty)")
                output.append("")
    except zipfile.BadZipFile as exc:
        raise ExtractionError(f"Not a valid XLSX: {exc}") from exc
    except OSError as exc:
        raise ExtractionError(str(exc)) from exc
    if not output:
        raise ExtractionError("XLSX has no visible sheets with content")
    return "\n".join(output).rstrip("\n") + "\n"


def _shared_strings(
    archive: zipfile.ZipFile,
    names: set[str],
) -> list[str]:
    if "xl/sharedStrings.xml" not in names:
        return []
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except ET.ParseError:
        return []
    namespace = f"{{{_NS_S}}}"
    return [
        "".join(text.text or "" for text in item.iter(f"{namespace}t"))
        for item in root.iter(f"{namespace}si")
    ]


def _workbook_sheets(
    archive: zipfile.ZipFile,
) -> list[tuple[str, str, str]]:
    root = _zip_xml(archive, "xl/workbook.xml")
    sheet_namespace = f"{{{_NS_S}}}"
    relationship_namespace = f"{{{_NS_REL}}}"
    return [
        (
            sheet.get("name", "Sheet"),
            sheet.get("state", "visible"),
            sheet.get(f"{relationship_namespace}id", ""),
        )
        for sheet in root.iter(f"{sheet_namespace}sheet")
    ]


def _workbook_rels(
    archive: zipfile.ZipFile,
    names: set[str],
) -> dict[str, str]:
    relationships_path = "xl/_rels/workbook.xml.rels"
    if relationships_path not in names:
        return {}
    try:
        root = ET.fromstring(archive.read(relationships_path))
    except ET.ParseError:
        return {}
    relationship_tag = f"{{{_NS_PKG_REL}}}Relationship"
    return {
        relationship.get("Id", ""): relationship.get("Target", "")
        for relationship in root.iter(relationship_tag)
        if relationship.get("Id")
    }


def _sheet_part(target: str) -> str:
    target = target.lstrip("/")
    return posixpath.normpath(
        target if target.startswith("xl/") else f"xl/{target}"
    )


def _column_index(reference: str) -> int:
    index = 0
    for character in reference:
        if not character.isalpha():
            break
        index = index * 26 + ord(character.upper()) - ord("A") + 1
    return max(index - 1, 0)


def _sheet_rows(xml_bytes: bytes, shared: list[str]) -> list[list[str]]:
    root = ET.fromstring(xml_bytes)
    namespace = f"{{{_NS_S}}}"
    rows: list[list[str]] = []
    for row in root.iter(f"{namespace}row"):
        if len(rows) >= _MAX_XLSX_ROWS_PER_SHEET:
            break
        cells: dict[int, str] = {}
        maximum_column = -1
        for cell in row.iter(f"{namespace}c"):
            column = (
                _column_index(cell.get("r", ""))
                if cell.get("r")
                else maximum_column + 1
            )
            if column >= _MAX_XLSX_COLS:
                continue
            cells[column] = _cell_value(cell, shared, namespace)
            maximum_column = max(maximum_column, column)
        rows.append(
            [cells.get(index, "") for index in range(maximum_column + 1)]
            if maximum_column >= 0
            else []
        )
    while rows and not any(value.strip() for value in rows[-1]):
        rows.pop()
    return rows


def _cell_value(
    cell: ET.Element,
    shared: list[str],
    namespace: str,
) -> str:
    value = cell.findtext(f"{namespace}v") or ""
    cell_type = cell.get("t", "")
    if cell_type == "s":
        try:
            return shared[int(value)]
        except (ValueError, IndexError):
            return ""
    if cell_type == "inlineStr":
        inline = cell.find(f"{namespace}is")
        if inline is None:
            return ""
        return "".join(
            text.text or "" for text in inline.iter(f"{namespace}t")
        )
    if cell_type == "b":
        return "TRUE" if value.strip() in {"1", "true", "TRUE"} else "FALSE"
    if cell_type == "e":
        return value or "#ERROR"
    return value
