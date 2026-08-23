"""Pilotage's local file tools, extracted from Hermes' proven file stack.

Hermes owns the difficult file behavior in ``file_operations``,
``patch_parser``, ``fuzzy_match`` and ``file_state``. This module is only the
Pilotage boundary: it shares the chat's terminal session, resolves paths
against its live cwd, applies credential guards, and registers four tools.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from . import file_state
from .binary_extensions import BINARY_EXTENSIONS, OPAQUE_DOCUMENT_EXTENSIONS
from .file_operations import (
    ShellFileOperations,
    normalize_read_pagination,
    normalize_search_pagination,
)
from .file_safety import get_read_block_error, get_write_denied_error
from .patch_parser import OperationType, apply_v4a_operations, parse_v4a_patch
from .registry import Tool, ToolContext, tool_error
from .shell import DEFAULT_TIMEOUT_SECONDS, Shell
from .terminal import TerminalSession, get_terminal_session, shell_cwd, shell_env

STATE_KEY = "file"
MAX_JSON_CHARS = 95_000
MAX_READ_CHARS = 85_000


@dataclass
class FileSession:
    shell: Optional[Shell] = None
    operations: Optional[ShellFileOperations] = None


@dataclass
class _PathBackup:
    path: Path
    existed: bool
    backup: Optional[Path] = None


def _backup_paths(paths: Iterable[Path]) -> list[_PathBackup]:
    """Snapshot mutation targets with same-filesystem links where possible."""
    snapshots: list[_PathBackup] = []
    try:
        for path in dict.fromkeys(paths):
            if not path.exists():
                snapshots.append(_PathBackup(path=path, existed=False))
                continue
            if not path.is_file():
                raise ValueError(f"Refusing to modify non-regular path: {path}")
            descriptor, raw_backup = tempfile.mkstemp(
                prefix=".pilotage-v4a-backup-", dir=str(path.parent)
            )
            os.close(descriptor)
            backup = Path(raw_backup)
            backup.unlink()
            try:
                os.link(path, backup)
            except OSError:
                shutil.copy2(path, backup)
            snapshots.append(_PathBackup(path=path, existed=True, backup=backup))
        return snapshots
    except Exception:
        _discard_backups(snapshots)
        raise


def _discard_backups(snapshots: Iterable[_PathBackup]) -> None:
    for snapshot in snapshots:
        if snapshot.backup is not None:
            try:
                snapshot.backup.unlink(missing_ok=True)
            except OSError:
                pass


def _restore_backups(snapshots: Iterable[_PathBackup]) -> Optional[str]:
    failures: list[str] = []
    for snapshot in reversed(list(snapshots)):
        try:
            if snapshot.existed:
                if snapshot.backup is None:
                    raise OSError("backup is missing")
                snapshot.path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(snapshot.backup, snapshot.path)
                snapshot.backup = None
            elif snapshot.path.exists() or snapshot.path.is_symlink():
                if not snapshot.path.is_file() and not snapshot.path.is_symlink():
                    raise OSError("failed operation created a non-file path")
                snapshot.path.unlink()
        except OSError as exc:
            failures.append(f"{snapshot.path}: {exc}")
    return " | ".join(failures) if failures else None


def _file_session(context: ToolContext, terminal: TerminalSession) -> FileSession:
    session = context.state.get(STATE_KEY)
    if not isinstance(session, FileSession) or session.shell is not terminal.shell:
        session = FileSession(shell=terminal.shell)
        context.state[STATE_KEY] = session
    if session.operations is None and terminal.shell is not None:
        session.operations = ShellFileOperations(terminal.shell)
    return session


def _setting(context: ToolContext, name: str, default: Any) -> Any:
    settings = getattr(context.config, "settings", None)
    if settings is None:
        return default
    if isinstance(default, int):
        return settings.count(name, default)
    return settings.text(name, default)


def _new_shell(context: ToolContext) -> Shell:
    timeout = int(_setting(context, "terminal.timeout", DEFAULT_TIMEOUT_SECONDS))
    if timeout <= 0:
        raise ValueError("terminal.timeout must be greater than zero")
    return Shell(cwd=shell_cwd(context), timeout=timeout, env=shell_env(context))


def _requested_path(path: Any, shell: Shell) -> Path:
    """Return the absolute path as requested, without following symlinks."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    candidate = Path(os.path.expanduser(path))
    if not candidate.is_absolute():
        candidate = Path(shell.cwd) / candidate
    return candidate


def _resolve(path: Any, shell: Shell) -> Path:
    return _requested_path(path, shell).resolve(strict=False)


def _read_guard(path: Path) -> Optional[str]:
    try:
        return get_read_block_error(str(path))
    except Exception:  # defense in depth must not become an availability risk
        return None


def _write_guard(path: Path, verb: str = "Write") -> Optional[str]:
    return get_write_denied_error(str(path), verb=verb)


def _binary_document_write_error(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix in OPAQUE_DOCUMENT_EXTENSIONS:
        return (
            f"Refusing to write '{path}': {suffix} is an opaque document format. "
            "Writing extracted text would destroy the document."
        )
    if suffix in BINARY_EXTENSIONS:
        return f"Refusing to write '{path}': binary files are read-only to the file tools."
    if suffix == ".pdf" and path.exists():
        return f"Refusing to overwrite existing PDF '{path}' as plain text."
    if path.exists() and path.is_file():
        try:
            with path.open("rb") as stream:
                sample = stream.read(1000)
        except OSError as exc:
            return f"Refusing to overwrite '{path}': existing bytes could not be inspected ({exc})."
        if ShellFileOperations._is_likely_binary_bytes(sample):
            return f"Refusing to overwrite '{path}': existing content is binary."
    return None


def _looks_line_numbered(content: str) -> bool:
    """Hermes guard against writing read_file's display gutter as source."""
    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    numbered: list[int] = []
    for line in lines:
        prefix, separator, _ = line.lstrip().partition("|")
        if separator and prefix.isdigit():
            numbered.append(int(prefix))
    if len(numbered) < 2 or len(numbered) / len(lines) < 0.6:
        return False
    consecutive = sum(
        current == previous + 1
        for previous, current in zip(numbered, numbered[1:])
    )
    return consecutive >= len(numbered) - 1


def _truncate_read(content: str, max_chars: int) -> tuple[str, int, bool]:
    """Hermes' complete-line read truncation with a progressing cursor."""
    if len(content) <= max_chars:
        return content, content.count("\n") + 1 if content else 0, False
    lines = content.split("\n")
    kept: list[str] = []
    running = 0
    for line in lines:
        addition = len(line) + (1 if kept else 0)
        if running + addition > max_chars:
            break
        kept.append(line)
        running += addition
    if not kept:
        kept.append(lines[0][:max_chars])
    return "\n".join(kept), len(kept), True


def _json(data: Dict[str, Any]) -> str:
    """Serialize without ever returning sliced, invalid JSON."""
    encoded = json.dumps(data, ensure_ascii=False)
    if len(encoded) <= MAX_JSON_CHARS:
        return encoded
    return json.dumps(
        {
            "error": "The tool result exceeded its context budget. Narrow the request.",
            "truncated": True,
        },
        ensure_ascii=False,
    )


def _fit_read_json(data: Dict[str, Any], offset: int) -> Dict[str, Any]:
    """Keep the largest complete-line read page whose encoded JSON fits."""
    if len(json.dumps(data, ensure_ascii=False)) <= MAX_JSON_CHARS:
        return data
    content = data.get("content")
    if not isinstance(content, str) or not content:
        return data
    lines = content.split("\n")
    low, high = 1, len(lines)
    best: Optional[Dict[str, Any]] = None
    while low <= high:
        middle = (low + high) // 2
        candidate = dict(data)
        candidate["content"] = "\n".join(lines[:middle])
        candidate["truncated"] = True
        candidate["truncated_by"] = "encoded characters"
        candidate["next_offset"] = offset + middle
        candidate["hint"] = (
            f"Encoded output reached its context budget. Use "
            f"offset={offset + middle} to continue."
        )
        if len(json.dumps(candidate, ensure_ascii=False)) <= MAX_JSON_CHARS:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best if best is not None else data


def _read(args: Dict[str, Any], context: ToolContext, shell: Shell,
          operations: ShellFileOperations) -> str:
    path = _resolve(args.get("path"), shell)
    blocked = _read_guard(path)
    if blocked:
        return tool_error(blocked)
    if path.suffix.lower() in BINARY_EXTENSIONS | OPAQUE_DOCUMENT_EXTENSIONS:
        return tool_error(
            f"Cannot read '{path}': binary and opaque document formats are not "
            "part of this text-file tool group."
        )
    offset, limit = normalize_read_pagination(args.get("offset", 1), args.get("limit", 2000))
    result = operations.read_file(str(path), offset, limit)
    data = result.to_dict()
    content = str(data.get("content") or "")
    trimmed, lines_kept, truncated = _truncate_read(content, MAX_READ_CHARS)
    if truncated:
        next_offset = offset + lines_kept
        data.update(
            content=trimmed,
            truncated=True,
            truncated_by="characters",
            next_offset=next_offset,
            hint=(
                f"Output reached the {MAX_READ_CHARS:,}-character budget. "
                f"Use offset={next_offset} to continue."
            ),
        )
    if not data.get("error"):
        file_state.record_read(
            context.chat_id,
            path,
            partial=offset > 1 or bool(data.get("truncated")),
        )
    data["resolved_path"] = str(path)
    return _json(_fit_read_json(data, offset))


def _write(args: Dict[str, Any], context: ToolContext, shell: Shell,
           operations: ShellFileOperations) -> str:
    if "content" not in args:
        return tool_error("write_file: missing required field 'content'")
    content = args.get("content")
    if not isinstance(content, str):
        return tool_error(f"write_file: 'content' must be a string, got {type(content).__name__}")
    if _looks_line_numbered(content):
        return tool_error(
            "Refusing to write read_file's line-numbered display text. "
            "Remove the '<line>|' prefixes first."
        )
    requested_path = _requested_path(args.get("path"), shell)
    path = requested_path.resolve(strict=False)
    denied = (
        _read_guard(path)
        or _write_guard(requested_path)
        or _binary_document_write_error(path)
    )
    if denied:
        return tool_error(denied)
    with file_state.lock_path(path):
        warning = file_state.check_stale(context.chat_id, path)
        snapshots = _backup_paths([path])
        try:
            try:
                result = operations.write_file(str(path), content)
            except Exception:
                rollback_error = _restore_backups(snapshots)
                if rollback_error:
                    raise RuntimeError(f"write failed and rollback failed: {rollback_error}")
                raise
            data = result.to_dict()
            if data.get("error"):
                rollback_error = _restore_backups(snapshots)
                if rollback_error:
                    data["error"] += f" Rollback also failed: {rollback_error}"
                else:
                    data["restored"] = True
        finally:
            _discard_backups(snapshots)
        data["resolved_path"] = str(path)
        if warning:
            data["_warning"] = warning
        if not data.get("error"):
            data["files_modified"] = [str(path)]
            file_state.note_write(context.chat_id, path)
    return _json(data)


def _v4a_paths(operations: Iterable[Any], shell: Shell) -> list[Path]:
    paths: list[Path] = []
    for operation in operations:
        raw_paths = [operation.file_path]
        if operation.operation == OperationType.MOVE and operation.new_path:
            raw_paths.append(operation.new_path)
        for raw in raw_paths:
            if ".." in Path(raw).parts:
                raise ValueError(
                    f"V4A patch header contains '..' traversal: {raw!r}. "
                    "Use a cwd-relative path without '..' or an absolute path."
                )
            resolved = _resolve(raw, shell)
            if resolved not in paths:
                paths.append(resolved)
    return paths


def _rewrite_v4a_paths(operations: Iterable[Any], shell: Shell) -> None:
    for operation in operations:
        operation.file_path = str(_resolve(operation.file_path, shell))
        if operation.operation == OperationType.MOVE and operation.new_path:
            operation.new_path = str(_resolve(operation.new_path, shell))


def _patch(args: Dict[str, Any], context: ToolContext, shell: Shell,
           operations: ShellFileOperations) -> str:
    mode = args.get("mode", "replace")
    if mode == "replace":
        if args.get("old_string") is None or args.get("new_string") is None:
            return tool_error("patch replace mode requires old_string and new_string")
        requested_path = _requested_path(args.get("path"), shell)
        path = requested_path.resolve(strict=False)
        denied = (
            _read_guard(path)
            or _write_guard(requested_path)
            or _binary_document_write_error(path)
        )
        if denied:
            return tool_error(denied)
        with file_state.lock_path(path):
            warning = file_state.check_stale(context.chat_id, path)
            snapshots = _backup_paths([path])
            try:
                try:
                    result = operations.patch_replace(
                        str(path),
                        args["old_string"],
                        args["new_string"],
                        bool(args.get("replace_all", False)),
                    )
                except Exception:
                    rollback_error = _restore_backups(snapshots)
                    if rollback_error:
                        raise RuntimeError(
                            f"patch failed and rollback failed: {rollback_error}"
                        )
                    raise
                data = result.to_dict()
                if data.get("error"):
                    rollback_error = _restore_backups(snapshots)
                    if rollback_error:
                        data["error"] += f" Rollback also failed: {rollback_error}"
                    else:
                        data["restored"] = True
            finally:
                _discard_backups(snapshots)
            data["resolved_path"] = str(path)
            if warning:
                data["_warning"] = warning
            if not data.get("error"):
                file_state.note_write(context.chat_id, path)
        return _json(data)

    if mode != "patch":
        return tool_error(f"Unknown patch mode: {mode}")
    patch_content = args.get("patch")
    if not isinstance(patch_content, str) or not patch_content.strip():
        return tool_error("patch mode requires patch content")
    parsed, error = parse_v4a_patch(patch_content)
    if error:
        return tool_error(f"Failed to parse patch: {error}")
    paths = _v4a_paths(parsed, shell)
    for operation in parsed:
        targets = [operation.file_path]
        if operation.operation == OperationType.MOVE and operation.new_path:
            targets.append(operation.new_path)
        for target in targets:
            requested_target = _requested_path(target, shell)
            resolved_target = requested_target.resolve(strict=False)
            denied = _read_guard(resolved_target) or _write_guard(
                requested_target, "Patch"
            )
            if denied:
                return tool_error(denied)
        if operation.operation in {OperationType.ADD, OperationType.UPDATE}:
            denied = _binary_document_write_error(_resolve(operation.file_path, shell))
            if denied:
                return tool_error(denied)
        if operation.operation == OperationType.ADD:
            destination = _resolve(operation.file_path, shell)
            if destination.exists():
                return tool_error(
                    f"Patch validation failed (no files were modified): "
                    f"{destination} already exists"
                )

    _rewrite_v4a_paths(parsed, shell)
    with ExitStack() as locks:
        for path in sorted(paths, key=str):
            locks.enter_context(file_state.lock_path(path))
        warnings = [
            warning
            for path in paths
            if (warning := file_state.check_stale(context.chat_id, path))
        ]
        snapshots = _backup_paths(paths)
        try:
            try:
                result = apply_v4a_operations(parsed, operations)
            except Exception:
                rollback_error = _restore_backups(snapshots)
                if rollback_error:
                    raise RuntimeError(
                        f"multi-file patch failed and rollback failed: {rollback_error}"
                    )
                raise
            data = result.to_dict()
            if data.get("error"):
                rollback_error = _restore_backups(snapshots)
                if rollback_error:
                    data["error"] += f" Rollback also failed: {rollback_error}"
                else:
                    data["restored"] = True
        finally:
            _discard_backups(snapshots)
        if warnings:
            data["_warning"] = " | ".join(warnings)
        if not data.get("error"):
            for path in paths:
                file_state.note_write(context.chat_id, path)
    return _json(data)


def _result_path(raw: str, shell: Shell) -> Path:
    return _resolve(raw, shell)


def _filter_blocked_search_results(result: Any, shell: Shell) -> int:
    omitted = 0
    if getattr(result, "matches", None):
        kept = []
        for match in result.matches:
            if _read_guard(_result_path(match.path, shell)):
                omitted += 1
            else:
                kept.append(match)
        result.matches = kept
    if getattr(result, "files", None):
        kept_files = []
        for raw in result.files:
            if _read_guard(_result_path(raw, shell)):
                omitted += 1
            else:
                kept_files.append(raw)
        result.files = kept_files
    if getattr(result, "counts", None):
        kept_counts = {}
        for raw, count in result.counts.items():
            if _read_guard(_result_path(raw, shell)):
                omitted += 1
            else:
                kept_counts[raw] = count
        result.counts = kept_counts
    return omitted


def _bounded_search_dict(result: Any, offset: int) -> Dict[str, Any]:
    data = result.to_dict(densify=True)
    if len(json.dumps(data, ensure_ascii=False)) <= MAX_JSON_CHARS:
        return data

    collection = None
    if result.matches:
        collection = result.matches
    elif result.files:
        collection = result.files
    elif result.counts:
        collection = result.counts
    original = len(collection) if collection is not None else 0
    while collection and len(json.dumps(result.to_dict(densify=True), ensure_ascii=False)) > MAX_JSON_CHARS:
        keep = max(0, len(collection) // 2)
        if isinstance(collection, dict):
            collection = dict(list(collection.items())[:keep])
            result.counts = collection
        else:
            del collection[keep:]
    data = result.to_dict(densify=True)
    kept = len(collection) if collection is not None else 0
    advance = max(1, kept) if original else 0
    data["truncated"] = True
    data["next_offset"] = offset + advance
    data["hint"] = (
        f"Result was reduced from {original} to {kept} item(s) to fit the context. "
        f"Use offset={offset + advance} to continue or narrow the pattern."
    )
    return data


def _search(args: Dict[str, Any], context: ToolContext, shell: Shell,
            operations: ShellFileOperations) -> str:
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return tool_error("pattern must be a non-empty string")
    target = args.get("target", "content")
    if target not in {"content", "files"}:
        return tool_error("target must be 'content' or 'files'")
    output_mode = args.get("output_mode", "content")
    if output_mode not in {"content", "files_only", "count"}:
        return tool_error("output_mode must be 'content', 'files_only', or 'count'")
    offset, limit = normalize_search_pagination(args.get("offset", 0), args.get("limit", 50))
    try:
        context_lines = max(0, int(args.get("context", 0)))
    except (TypeError, ValueError):
        return tool_error("context must be a non-negative integer")
    root = _resolve(args.get("path", "."), shell)
    blocked = _read_guard(root)
    if blocked:
        return tool_error(blocked)
    effective_offset = offset
    omitted = 0
    for _ in range(20):
        result = operations.search(
            pattern=pattern,
            path=str(root),
            target=target,
            file_glob=args.get("file_glob"),
            limit=limit,
            offset=effective_offset,
            output_mode=output_mode,
            context=context_lines,
        )
        omitted += _filter_blocked_search_results(result, shell)
        has_payload = bool(result.matches or result.files or result.counts)
        if has_payload or result.error or not result.truncated:
            break
        # A blocked file can occupy the whole requested page. Advance instead
        # of returning an empty, non-progressing cursor.
        effective_offset += limit
    data = _bounded_search_dict(result, effective_offset)
    if data.get("truncated") and "next_offset" not in data:
        data["next_offset"] = effective_offset + limit
    if omitted:
        data["_omitted"] = f"{omitted} secret-bearing result(s) omitted."
    return _json(data)


async def _run(handler: Any, args: Dict[str, Any], context: ToolContext) -> str:
    terminal = get_terminal_session(context)
    async with terminal.lock:
        try:
            if terminal.shell is None:
                terminal.shell = await asyncio.to_thread(_new_shell, context)
            session = _file_session(context, terminal)
            if session.operations is None:
                raise RuntimeError("file operations could not initialize")
            return await asyncio.to_thread(
                handler, args, context, terminal.shell, session.operations
            )
        except Exception as exc:
            return tool_error(f"{type(exc).__name__}: {exc}")


async def handle_read_file(args: Dict[str, Any], context: ToolContext) -> str:
    return await _run(_read, args, context)


async def handle_write_file(args: Dict[str, Any], context: ToolContext) -> str:
    return await _run(_write, args, context)


async def handle_patch(args: Dict[str, Any], context: ToolContext) -> str:
    return await _run(_patch, args, context)


async def handle_search_files(args: Dict[str, Any], context: ToolContext) -> str:
    return await _run(_search, args, context)


READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "Read a text file with compact line numbers and pagination. Use this instead of cat/head/tail. Output lines are '<line>|<content>'.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute, relative, or ~/ path."},
            "offset": {"type": "integer", "minimum": 1, "default": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 2000, "default": 2000},
        },
        "required": ["path"],
    },
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "Replace a text file completely, creating parents when needed. Use patch for targeted edits. verified:true means the on-disk hash matched.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
}

PATCH_SCHEMA = {
    "name": "patch",
    "description": "Targeted edits using Hermes' fuzzy matching. replace mode requires path, old_string and new_string; patch mode accepts a V4A multi-file patch.",
    "parameters": {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["replace", "patch"], "default": "replace"},
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False},
            "patch": {"type": "string"},
        },
        "required": ["mode"],
    },
}

SEARCH_FILES_SCHEMA = {
    "name": "search_files",
    "description": "Search text with a regex or find files with a glob. Uses ripgrep when available and grep/find fallbacks otherwise.",
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "target": {"type": "string", "enum": ["content", "files"], "default": "content"},
            "path": {"type": "string", "default": "."},
            "file_glob": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "default": 50},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "output_mode": {"type": "string", "enum": ["content", "files_only", "count"], "default": "content"},
            "context": {"type": "integer", "minimum": 0, "default": 0},
        },
        "required": ["pattern"],
    },
}

FILE_TOOLS = (
    Tool("read_file", "file", READ_FILE_SCHEMA, handle_read_file, "📖"),
    Tool("write_file", "file", WRITE_FILE_SCHEMA, handle_write_file, "✍️"),
    Tool("patch", "file", PATCH_SCHEMA, handle_patch, "🔧"),
    Tool("search_files", "file", SEARCH_FILES_SCHEMA, handle_search_files, "🔎"),
)

__all__ = [
    "FILE_TOOLS",
    "PATCH_SCHEMA",
    "READ_FILE_SCHEMA",
    "SEARCH_FILES_SCHEMA",
    "WRITE_FILE_SCHEMA",
    "handle_patch",
    "handle_read_file",
    "handle_search_files",
    "handle_write_file",
]
