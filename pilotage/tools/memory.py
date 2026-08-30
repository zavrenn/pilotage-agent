"""Bounded, profile-scoped curated memory.

The store and tool behavior are extracted from Hermes' production memory
implementation. Pilotage keeps only the built-in file store: no external
providers. The small approval queue lives outside the store so direct storage
integrity tests remain deterministic; the model-facing handler applies the
per-profile gate. Each Agent supplies the selected profile's memory directory
explicitly.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..approvals import approval_error, approval_required
from .registry import Tool, ToolContext, tool_error
from .threat_patterns import first_threat_message as _first_threat_message

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

logger = logging.getLogger(__name__)


def _atomic_write_text(path: Path, content: str) -> None:
    """Write a complete memory file through fsync and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".mem_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


# Stable header prefixes for the system-prompt memory blocks rendered by
# MemoryStore._render_block. Exported so compression's prompt-retention check
# (agent/conversation_compression.py) can detect a leftover block for a
# target whose entries have since been emptied — keep in lockstep with
# _render_block below.
MEMORY_BLOCK_HEADERS = {
    "memory": "MEMORY (your personal notes)",
    "user": "USER PROFILE (who the user is)",
}

ENTRY_DELIMITER = "\n§\n"


# ---------------------------------------------------------------------------
# Memory content scanning — lightweight check for injection/exfiltration
# in content that gets injected into the system prompt.
#
# Patterns live in ``tools/threat_patterns.py`` — the single source of truth
# shared with the context-file scanner and the tool-result delimiter system.
# Memory uses the "strict" scope (broadest pattern set) because:
#  - memory entries are user-curated; the user can rewrite a flagged entry
#  - memory enters the system prompt as a FROZEN snapshot, so a poisoned
#    entry persists for the entire session and across sessions until
#    explicitly removed.
# ---------------------------------------------------------------------------

def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns. Returns error string if blocked."""
    return _first_threat_message(content, scope="strict")


def _drift_error(path: "Path") -> Dict[str, Any]:
    """Build the error dict returned when external drift is detected.

    The on-disk memory file contains content that wouldn't round-trip
    through the tool's parser/serializer — flushing would discard the
    appended/edited content from a patch tool, shell append, manual edit,
    or sister-session write. We refuse the mutation and leave the canonical
    bytes untouched for operator inspection.
    """
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: file on disk has content that "
            f"wouldn't round-trip through the memory tool (likely added by "
            f"the patch tool, a shell append, a manual edit, or a "
            f"concurrent session). The file was left untouched. "
            f"Resolve the drift first — either rewrite the file as a clean "
            f"§-delimited list of entries, or move the extra content out — "
            f"then retry. This guard exists to prevent silent data loss "
            f"(issue #26045)."
        ),
        "remediation": (
            "Inspect the original file, integrate the missing entries into the "
            "memory tool one at a time via memory(action=add, content=...), "
            "then remove or rewrite the original file to a clean state."
        ),
    }


# Sentinel returned by ``_reload_target`` when the target file EXISTS but could
# not be read. Distinct from the boolean drift result: the caller must abort the
# mutation rather than persist over an unreadable file.
_READ_FAILED = object()


def _read_failed_error(path: "Path") -> Dict[str, Any]:
    """Build the error dict returned when the on-disk memory file is unreadable.

    A file that exists but cannot be read is NOT an empty store. Reading it as
    ``[]`` and then persisting would rewrite the whole file from an empty entry
    list — wiping the user's memory. We refuse the write so nothing is lost.
    """
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: the file exists on disk but could "
            f"not be read right now (temporarily locked by another program, a "
            f"permission change, invalid/corrupt text encoding, or a filesystem "
            f"error). Treating an unreadable file as empty and saving would wipe "
            f"existing memory, so the write is refused. Nothing was changed — "
            f"retry in a moment."
        ),
    }


class MemoryStore:
    """
    Bounded curated memory with file persistence. One instance per AIAgent.

    Maintains two parallel states:
      - _system_prompt_snapshot: frozen at load time, used for system prompt injection.
        Never mutated mid-session. Keeps prefix cache stable.
      - memory_entries / user_entries: live state, mutated by tool calls, persisted to disk.
        Tool responses always reflect this live state.
    """

    # After this many failed targeting/validation attempts in one turn, return
    # a terminal result so a malformed replace/remove cannot loop to budget
    # exhaustion and suppress the user's reply (issue #42405).
    _MAX_CONSOLIDATION_FAILURES_PER_TURN = 3

    def __init__(self, memory_dir: Path, memory_char_limit: int = 2200, user_char_limit: int = 1375):
        self.memory_dir = Path(memory_dir)
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        # Frozen snapshot for system prompt -- set once at load_from_disk()
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}
        # Per-turn counter of failed targeting/validation attempts; reset at
        # each turn boundary by reset_consolidation_failures() (#42405).
        self._consolidation_failures: ContextVar[Optional[List[int]]] = ContextVar(
            f"pilotage_memory_failures_{id(self)}", default=None
        )

    def _failure_counter(self) -> List[int]:
        counter = self._consolidation_failures.get()
        if counter is None:
            counter = [0]
            self._consolidation_failures.set(counter)
        return counter

    def reset_consolidation_failures(self) -> None:
        """Reset the per-turn consolidation-failure counter (call at turn start)."""
        self._consolidation_failures.set([0])

    def _consolidation_failure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Count a recoverable targeting/validation failure and stop loops.

        Under the per-turn cap, return ``response`` unchanged (it already tells
        the model how to self-correct + retry in this turn). Once the cap is
        exceeded, drop the retry instruction and return a TERMINAL result so the
        model stops looping memory calls and proceeds to answer the user — a
        failed memory side effect must never block the turn's reply (#42405).
        """
        counter = self._failure_counter()
        counter[0] += 1
        failure_count = counter[0]
        if failure_count <= self._MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        return {
            "success": False,
            "done": True,
            "error": (
                f"Memory consolidation failed {failure_count} times "
                "this turn. Stop retrying memory calls — leave memory unchanged for "
                "now and continue with your reply to the user. The fact can be saved "
                "in a later turn."
            ),
        }

    def _capacity_failure(self, target: str, detail: str) -> Dict[str, Any]:
        """Reject an over-limit mutation without pressuring unrelated eviction."""
        current = self._char_count(target)
        limit = self._char_limit(target)
        return {
            "success": False,
            "done": True,
            "error": (
                f"{detail} Nothing was changed. Never delete or shorten unrelated "
                "entries merely to make room. Consolidate only entries that durable "
                "evidence shows are overlapping, obsolete, or superseded; otherwise "
                "leave memory unchanged."
            ),
            "current_entries": self._entries_for(target),
            "usage": f"{current:,}/{limit:,}",
        }

    def load_from_disk(self):
        """Load entries from MEMORY.md and USER.md, capture system prompt snapshot.

        The frozen snapshot is what enters the system prompt. We scan each
        entry for injection/promptware patterns at snapshot-build time —
        ANY hit replaces the entry text in the snapshot with a placeholder
        like ``[BLOCKED: …]``, so a poisoned-on-disk memory file (supply
        chain, compromised tool, sister-session write) cannot inject into
        the system prompt.

        The live ``memory_entries`` / ``user_entries`` lists keep the
        original text so it remains explicitly inspectable through the live tool.

        Scanning is deterministic from disk bytes, so the snapshot remains
        stable for the entire session (prefix-cache invariant holds).
        """
        mem_dir = self.memory_dir
        mem_dir.mkdir(parents=True, exist_ok=True)

        self.memory_entries = self._read_file(mem_dir / "MEMORY.md")
        self.user_entries = self._read_file(mem_dir / "USER.md")

        # Deduplicate entries (preserves order, keeps first occurrence)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # Sanitize entries for the system-prompt snapshot only.  Live state
        # (memory_entries / user_entries) keeps the raw text for explicit
        # inspection and removal through the tool.
        sanitized_memory = self._sanitize_entries_for_snapshot(self.memory_entries, "MEMORY.md")
        sanitized_user = self._sanitize_entries_for_snapshot(self.user_entries, "USER.md")

        # Capture frozen snapshot for system prompt injection
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", sanitized_memory),
            "user": self._render_block("user", sanitized_user),
        }

    def refresh_live_from_disk(
        self, targets: tuple[str, ...] = ("memory", "user")
    ) -> None:
        """Refresh mutation state without changing the session's prompt snapshot."""

        for target in targets:
            if target not in {"memory", "user"}:
                raise ValueError(f"Invalid memory refresh target: {target}")
            path = self._path_for(target)
            with self._file_lock(path):
                entries, read_ok = self._read_entries_checked(path)
                if not read_ok:
                    raise RuntimeError(f"Could not refresh live memory from {path}")
                self._set_entries(target, list(dict.fromkeys(entries)))

    def list_live(self, target: str) -> Dict[str, Any]:
        """Return the current canonical entries, including sister-session writes."""

        path = self._path_for(target)
        with self._file_lock(path):
            entries, read_ok = self._read_entries_checked(path)
            if not read_ok:
                return _read_failed_error(path)
            entries = list(dict.fromkeys(entries))
            self._set_entries(target, entries)
            current = self._char_count(target)
            limit = self._char_limit(target)
            return {
                "success": True,
                "done": True,
                "target": target,
                "entries": entries,
                "usage": f"{current:,}/{limit:,}",
            }

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: List[str], filename: str) -> List[str]:
        """Return ``entries`` with any threat-matching entry replaced by a placeholder.

        Each entry is scanned with the shared threat-pattern library at the
        ``"strict"`` scope (same as memory writes).  On match, the entry is
        replaced in the returned list with ``"[BLOCKED: <filename> entry
        contained threat pattern: <ids>. Removed from system prompt.]"`` —
        the placeholder enters the snapshot, the original entry stays in
        live state for the user to inspect and delete.

        Empty entries pass through unchanged.
        """
        from .threat_patterns import scan_for_threats

        sanitized: List[str] = []
        for entry in entries:
            if not entry:
                sanitized.append(entry)
                continue
            findings = scan_for_threats(entry, scope="strict")
            if findings:
                logger.warning(
                    "Memory entry from %s blocked at load time: %s",
                    filename, ", ".join(findings),
                )
                sanitized.append(
                    f"[BLOCKED: {filename} entry contained threat pattern(s): "
                    f"{', '.join(findings)}. Removed from system prompt; "
                    "inspect the live memory target before removing it.]"
                )
            else:
                sanitized.append(entry)
        return sanitized

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Acquire an exclusive file lock for read-modify-write safety.

        Uses a separate .lock file so the memory file itself can still be
        atomically replaced via os.replace().
        """
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        if fcntl is None and msvcrt is None:
            yield
            return

        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            # msvcrt locks a byte range, so an empty lock file has no byte to
            # lock. This is the same Windows guard used by Codex auth.
            if msvcrt:
                fd.seek(0, os.SEEK_END)
                if fd.tell() == 0:
                    fd.write(" ")
                    fd.flush()
                fd.seek(0)
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            if fcntl:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except (OSError, IOError):
                    pass
            elif msvcrt:
                try:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    pass
            fd.close()

    def _path_for(self, target: str) -> Path:
        mem_dir = self.memory_dir
        if target == "user":
            return mem_dir / "USER.md"
        return mem_dir / "MEMORY.md"

    def _reload_target(self, target: str, *, skip_drift: bool = False):
        """Re-read entries from disk into in-memory state.

        Called under file lock to get the latest state before mutating.
        Returns ``True`` if external drift was detected (the on-disk
        file contains content that wouldn't round-trip through our
        parser/serializer, OR an entry larger than the store's char limit).
        When drift is detected the caller must abort the mutation —
        flushing would discard the un-roundtrippable content.
        Returns ``False`` on clean reload.

        Returns the ``_READ_FAILED`` sentinel when the file EXISTS but could not
        be read. The caller MUST abort: the on-disk entries are unknown, so
        overwriting from an assumed-empty view would wipe them. This is the real
        exposure behind ``add`` — it skips the drift guard because appending is
        safe, but that reasoning only holds when the reload actually saw the
        file. A failed read reported as ``[]`` turned ``add`` into a full-file
        rewrite down to a single entry.

        When *skip_drift* is True the round-trip / entry-size check is
        bypassed.  Used by the ``add`` action which appends without
        rewriting, so existing content is never clobbered.
        """
        path = self._path_for(target)
        raw, read_ok = self._read_raw_checked(path)
        if not read_ok:
            # Leave in-memory entries untouched and tell the caller to abort;
            # persisting over an unreadable file would destroy it.
            return _READ_FAILED
        # Derive BOTH the drift check and the entry parse from the same raw
        # snapshot. The drift guard used to re-read the file itself and treat
        # a failed second read as "no drift" — so a read failure between the
        # checked reload and the drift check let replace/remove/apply_batch
        # rewrite the file from a stale view, silently discarding whatever an
        # external writer had just added. One read, one snapshot, no window.
        drift_detected = False if skip_drift else self._detect_external_drift(target, raw)
        fresh = self._parse_entries(raw)
        fresh = list(dict.fromkeys(fresh))  # deduplicate
        self._set_entries(target, fresh)
        return drift_detected

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def add(
        self, target: str, content: str, *, preview: bool = False
    ) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        # Scan for injection/exfiltration before accepting
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            # Re-read from disk under lock to pick up writes from other sessions.
            # For add (append-only), we skip the drift guard — appending never
            # clobbers existing content, so round-trip mismatches from prior
            # tool-written entries in the same session are harmless.  The drift
            # guard remains active for replace/remove where full-file rewrite
            # would discard un-roundtrippable content (issue #26045).
            #
            # But "append never clobbers" only holds when the reload actually
            # read the file. add rewrites the WHOLE file from the parsed
            # entries, so a file that exists but read as empty (transient lock,
            # permission blip, I/O error) would be rewritten down to just the
            # new entry — wiping every prior memory. Refuse instead.
            if self._reload_target(target, skip_drift=True) is _READ_FAILED:
                return _read_failed_error(self._path_for(target))

            entries = self._entries_for(target)
            limit = self._char_limit(target)

            # Reject exact duplicates
            if content in entries:
                response = self._success_response(
                    target, "Entry already exists (no duplicate added)."
                )
                response["no_change"] = True
                return response

            # Calculate what the new total would be
            new_entries = entries + [content]
            new_total = len(ENTRY_DELIMITER.join(new_entries))

            if new_total > limit:
                return self._capacity_failure(
                    target,
                    f"Adding this entry would exceed the {limit:,}-character limit.",
                )

            if preview:
                return {"success": True, "done": True, "would_change": True}

            entries.append(content)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry added.")

    def replace(
        self,
        target: str,
        old_text: str,
        new_content: str,
        *,
        preview: bool = False,
    ) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        old_text = (old_text or "").strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}

        # Scan replacement content for injection/exfiltration
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}

        with self._file_lock(self._path_for(target)):
            drift_detected = self._reload_target(target)
            if drift_detected is _READ_FAILED:
                return _read_failed_error(self._path_for(target))
            if drift_detected:
                return _drift_error(self._path_for(target))

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return self._consolidation_failure({
                    "success": False,
                    "error": f"No entry matched '{old_text}'. Check current_entries below and retry with the exact text of the entry you want to replace.",
                    "current_entries": entries,
                })

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), operate on the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = self._previews([e for _, e in matches])
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to replace just the first

            idx = matches[0][0]
            if entries[idx] == new_content:
                response = self._success_response(
                    target, "Entry already contained the intended text."
                )
                response["no_change"] = True
                return response
            limit = self._char_limit(target)

            # Check that replacement doesn't blow the budget
            test_entries = entries.copy()
            test_entries[idx] = new_content
            new_total = len(ENTRY_DELIMITER.join(test_entries))

            if new_total > limit:
                return self._capacity_failure(
                    target,
                    f"This replacement would exceed the {limit:,}-character limit.",
                )

            if preview:
                return {"success": True, "done": True, "would_change": True}

            entries[idx] = new_content
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry replaced.")

    def remove(
        self, target: str, old_text: str, *, preview: bool = False
    ) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = (old_text or "").strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        with self._file_lock(self._path_for(target)):
            drift_detected = self._reload_target(target)
            if drift_detected is _READ_FAILED:
                return _read_failed_error(self._path_for(target))
            if drift_detected:
                return _drift_error(self._path_for(target))

            entries = self._entries_for(target)
            matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

            if not matches:
                return self._consolidation_failure({
                    "success": False,
                    "error": f"No entry matched '{old_text}'. Check current_entries below and retry with the exact text of the entry you want to remove.",
                    "current_entries": entries,
                })

            if len(matches) > 1:
                # If all matches are identical (exact duplicates), remove the first one
                unique_texts = {e for _, e in matches}
                if len(unique_texts) > 1:
                    previews = self._previews([e for _, e in matches])
                    return {
                        "success": False,
                        "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                        "matches": previews,
                    }
                # All identical -- safe to remove just the first

            idx = matches[0][0]
            if preview:
                return {"success": True, "done": True, "would_change": True}
            entries.pop(idx)
            self._set_entries(target, entries)
            self.save_to_disk(target)

        return self._success_response(target, "Entry removed.")

    def apply_batch(
        self,
        target: str,
        operations: List[Dict[str, Any]],
        *,
        preview: bool = False,
    ) -> Dict[str, Any]:
        """Apply a sequence of add/replace/remove ops to one target atomically.

        All operations are validated and applied against the final budget;
        intermediate state is irrelevant. This keeps an evidence-backed set of
        related changes atomic.

        Semantics: all-or-nothing. If any op is malformed, doesn't match, or
        the net result would exceed the char limit, NOTHING is written and an
        error is returned describing the first failure plus the live state.
        """
        if not operations:
            return {"success": False, "error": "operations list is empty."}

        # Scan every add/replace content for injection/exfil BEFORE touching
        # disk -- a single poisoned op rejects the whole batch.
        for i, op in enumerate(operations):
            act = (op or {}).get("action")
            new_content = (op or {}).get("content") or (op or {}).get("new_text")
            if act in {"add", "replace"} and new_content:
                scan_error = _scan_memory_content(new_content)
                if scan_error:
                    return {"success": False, "error": f"Operation {i + 1}: {scan_error}"}

        with self._file_lock(self._path_for(target)):
            drift_detected = self._reload_target(target)
            if drift_detected is _READ_FAILED:
                return _read_failed_error(self._path_for(target))
            if drift_detected:
                return _drift_error(self._path_for(target))

            # Work on a copy; only commit if the whole batch validates.
            original = list(self._entries_for(target))
            working: List[str] = list(original)
            limit = self._char_limit(target)

            for i, op in enumerate(operations):
                op = op or {}
                act = op.get("action")
                content = (op.get("content") or op.get("new_text") or "").strip()
                old_text = (op.get("old_text") or "").strip()
                pos = f"Operation {i + 1} ({act or 'unknown'})"

                if act == "add":
                    if not content:
                        return self._batch_error(target, f"{pos}: content is required.")
                    if content in working:
                        continue  # idempotent -- skip duplicate, don't fail the batch
                    working.append(content)

                elif act == "replace":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    if not content:
                        return self._batch_error(
                            target,
                            f"{pos}: content is required (use action='remove' to delete).",
                        )
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(target, f"{pos}: no entry matched '{old_text}'.")
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific.",
                        )
                    working[matches[0]] = content

                elif act == "remove":
                    if not old_text:
                        return self._batch_error(target, f"{pos}: old_text is required.")
                    matches = [j for j, e in enumerate(working) if old_text in e]
                    if not matches:
                        return self._batch_error(target, f"{pos}: no entry matched '{old_text}'.")
                    if len({working[j] for j in matches}) > 1:
                        return self._batch_error(
                            target,
                            f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific.",
                        )
                    working.pop(matches[0])

                else:
                    return self._batch_error(
                        target,
                        f"{pos}: unknown action. Use add, replace, or remove.",
                    )

            # Budget check against the FINAL state only.
            new_total = len(ENTRY_DELIMITER.join(working)) if working else 0
            if new_total > limit:
                return self._capacity_failure(
                    target,
                    f"This batch would exceed the {limit:,}-character limit.",
                )

            # Commit.
            if working == original:
                response = self._success_response(
                    target, "The requested memory state was already current."
                )
                response["no_change"] = True
                return response
            if preview:
                return {"success": True, "done": True, "would_change": True}
            self._set_entries(target, working)
            self.save_to_disk(target)

        return self._success_response(target, f"Applied {len(operations)} operation(s).")

    def _batch_error(self, target: str, message: str) -> Dict[str, Any]:
        """Build a batch-abort error that reports live (uncommitted) state."""
        current = self._char_count(target)
        limit = self._char_limit(target)
        return self._consolidation_failure({
            "success": False,
            "error": message + " No operations were applied (batch is all-or-nothing).",
            "current_entries": self._entries_for(target),
            "usage": f"{current:,}/{limit:,}",
        })

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """
        Return the frozen snapshot for system prompt injection.

        This returns the state captured at load_from_disk() time, NOT the live
        state. Mid-session writes do not affect this. This keeps the system
        prompt stable across all turns, preserving the prefix cache.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # -- Internal helpers --

    @staticmethod
    def _previews(entries: List[str], width: int = 80) -> List[str]:
        """Truncated one-line previews of entries for error feedback."""
        return [e[:width] + ("..." if len(e) > width else "") for e in entries]

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        # A successful write means the consolidation loop made progress, so the
        # per-turn failure budget resets (the cap counts consecutive failures,
        # not lifetime ones within a turn) (#42405).
        self._failure_counter()[0] = 0
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        # The success response is intentionally TERMINAL: it confirms the write
        # landed and tells the model to stop. We do NOT echo the full entries
        # list here -- dumping it invites the model to "find more to fix" and
        # re-issue the same operations (observed thrash: the correct batch on
        # call 1, then 5 redundant repeats). Entries are only shown on the
        # error/over-budget paths, where the model genuinely needs them to
        # decide what to consolidate.
        resp = {
            "success": True,
            "done": True,
            "target": target,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
        }
        if message:
            resp["message"] = message
        resp["note"] = "Write saved. This update is complete — do not repeat it."
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""

        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        if target == "user":
            header = f"{MEMORY_BLOCK_HEADERS['user']} [{pct}% — {current:,}/{limit:,} chars]"
        else:
            header = f"{MEMORY_BLOCK_HEADERS['memory']} [{pct}% — {current:,}/{limit:,} chars]"

        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_raw_checked(path: Path) -> Tuple[str, bool]:
        """Read a memory file's raw text, distinguishing unreadable from empty.

        Returns ``(raw, read_ok)``. ``read_ok`` is False ONLY when the file
        EXISTS but could not be read — an absent file is a clean ``("", True)``.
        Invalid UTF-8 counts as unreadable too: the bytes on disk hold content
        we cannot faithfully round-trip, so a rewrite would corrupt or discard
        it just like a failed read. Read-modify-write callers must treat
        ``read_ok=False`` as "abort" rather than "empty store", or a transient
        read failure would let them persist over — and wipe — the on-disk
        memory (issue #26045 is about the same class: never rewrite a file
        from a view that isn't the real one).

        No file locking needed: _write_file uses atomic rename, so readers
        always see either the previous complete file or the new complete file.
        """
        if not path.exists():
            return "", True
        try:
            # utf-8-sig strips a leading UTF-8 BOM (Notepad-edited memory
            # files on Windows) and is byte-identical to utf-8 otherwise.
            # Plain utf-8 kept U+FEFF glued to the first entry, corrupting
            # matching/dedup for that entry forever (#10878 / PR #10888).
            # Decode errors stay STRICT on purpose: errors="replace" would
            # hand read-modify-write callers a lossy view that a subsequent
            # save persists over the real bytes — the wipe class documented
            # above. Undecodable bytes must surface as read_ok=False.
            return path.read_text(encoding="utf-8-sig"), True
        except (OSError, IOError, UnicodeDecodeError):
            return "", False

    @staticmethod
    def _parse_entries(raw: str) -> List[str]:
        """Split raw memory-file text into stripped, non-empty entries."""
        if not raw.strip():
            return []
        # Use ENTRY_DELIMITER for consistency with _write_file. Splitting by "§"
        # alone would incorrectly split entries that contain "§" in their content.
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _read_entries_checked(path: Path) -> Tuple[List[str], bool]:
        """Read + parse a memory file, distinguishing unreadable from empty.

        Returns ``(entries, read_ok)`` — see ``_read_raw_checked`` for the
        ``read_ok`` contract.
        """
        raw, read_ok = MemoryStore._read_raw_checked(path)
        if not read_ok:
            return [], False
        return MemoryStore._parse_entries(raw), True

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file and split into entries (empty list on any error).

        Retained for read-only callers (``load_from_disk``) that build in-memory
        state without persisting; a failed read degrading to ``[]`` there is
        harmless because nothing is written back. Read-modify-write paths use
        ``_read_raw_checked`` so they can refuse to overwrite an unreadable
        file — see ``_reload_target``.
        """
        return MemoryStore._read_entries_checked(path)[0]

    def _detect_external_drift(self, target: str, raw: str) -> bool:
        """Return whether on-disk content shows external drift.

        *raw* is the file content already read by the caller's checked read
        (``_read_raw_checked``). Drift detection MUST operate on that same
        snapshot — an earlier version re-read the file here and treated a
        failed second read as "no drift", which let a mutation proceed from a
        stale first snapshot and rewrite away content an external writer added
        between the two reads.

        The memory file is supposed to be a list of small entries the tool
        wrote, joined by §. Detect drift via two signals:

        1. Round-trip mismatch — re-parsing and re-serializing the file
           doesn't produce identical bytes (rare; would catch oddly-encoded
           delimiters).
        2. Entry-size overflow — any single parsed entry exceeds the
           store's whole-file char limit. The tool budgets the ENTIRE store
           against that limit; no single tool-written entry can exceed it.
           When we see one entry larger than the limit, an external writer
           (patch tool, shell append, manual edit, sister session) appended
           free-form content into what the tool will treat as one entry.
           Flushing would then truncate that entry to the model's new
           content, discarding the appended bytes — issue #26045.

        Note: this is an INSTANCE method (not static) because we need the
        per-target char_limit for signal #2.
        """
        if not raw.strip():
            return False

        parsed = [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]
        roundtrip = ENTRY_DELIMITER.join(parsed)

        char_limit = self._char_limit(target)
        max_entry_len = max((len(e) for e in parsed), default=0)

        drift_detected = (raw.strip() != roundtrip) or (max_entry_len > char_limit)
        return drift_detected

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Write entries to a memory file using atomic temp-file + rename.

        Previous implementation used open("w") + flock, but "w" truncates the
        file *before* the lock is acquired, creating a race window where
        concurrent readers see an empty file. Atomic rename avoids this:
        readers always see either the old complete file or the new one.
        """
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            if path.is_file() and path.read_text(encoding="utf-8-sig") == content:
                return
            _atomic_write_text(path, content)
        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")


def _missing_old_text_error(store: "MemoryStore", target: str, action: str) -> str:
    """Build a recoverable error for a replace/remove call that arrived without
    ``old_text``.

    ``replace``/``remove`` are inherently targeted -- without ``old_text`` there
    is no entry to act on, so we cannot fulfil the call. But returning a bare
    "old_text is required" is a dead-end: some structured-output clients omit the
    optional ``old_text`` field (it isn't, and can't be, schema-required without
    a top-level combinator the Codex backend rejects -- see
    tests/tools/test_memory_tool_schema.py). So instead we return the current
    entry inventory plus an explicit retry instruction, letting the model reissue
    the call with ``old_text`` set to a unique substring of the entry it means.
    Mirrors the batch path's ``_batch_error`` shape. (issues #43412, #49466)
    """
    entries = store._entries_for(target)
    current = store._char_count(target)
    limit = store._char_limit(target)
    return json.dumps(
        {
            "success": False,
            "error": (
                f"'{action}' needs old_text -- a short unique substring of the entry "
                f"to {action}. None was provided. Reissue the {action} with old_text "
                f"set to part of one of the current_entries below."
            ),
            "current_entries": entries,
            "usage": f"{current:,}/{limit:,}",
        },
        ensure_ascii=False,
    )


def memory_tool(
    action: str = None,
    target: str = "memory",
    content: str = None,
    old_text: str = None,
    new_text: str = None,
    operations: Optional[List[Dict[str, Any]]] = None,
    store: Optional[MemoryStore] = None,
    *,
    preview: bool = False,
) -> str:
    """
    Single entry point for the memory tool. Dispatches to MemoryStore methods.

    Two shapes:
      - Single op: action + (content / old_text).
      - Batch:     operations=[{action, content?, old_text?}, ...] applied
                   atomically against the final char budget in ONE call.

    ``new_text`` is accepted as an alias for ``content`` on both shapes. The
    replace/remove ops target by ``old_text`` and supply the replacement via
    ``content``; callers naturally reach for ``new_text`` to mirror
    ``old_text`` (it's the patch tool's ``old_string``/``new_string`` shape),
    which silently left ``content`` empty and errored. Coalescing here removes
    that trap.

    Returns JSON string with results.
    """
    if store is None:
        return tool_error("Memory is not available. It may be disabled in config or this environment.", success=False)

    # Accept new_text as an alias for content (single-op path). See docstring.
    if content is None and new_text is not None:
        content = new_text

    # Some strict providers fill optional schema fields with JSON null rather
    # than omitting them.  Treat ``target: null`` as omitted so memory writes
    # still use the documented default store instead of failing validation.
    if target is None:
        target = "memory"

    if target not in {"memory", "user"}:
        return tool_error(f"Invalid target '{target}'. Use 'memory' or 'user'.", success=False)

    if action is not None and operations:
        if action == "list":
            return tool_error(
                "action='list' cannot be combined with mutation operations.",
                success=False,
            )
        return tool_error(
            "Use either top-level action or operations, not both.",
            success=False,
        )

    if action == "list":
        return json.dumps(store.list_live(target), ensure_ascii=False)

    # --- Batch path -------------------------------------------------------
    if operations:
        if not isinstance(operations, list):
            return tool_error("operations must be a list of {action, content?, old_text?} objects.", success=False)
        result = store.apply_batch(target, operations, preview=preview)
        return json.dumps(result, ensure_ascii=False)

    # --- Single-op path ---------------------------------------------------
    # Validate required params BEFORE the gate so an invalid write is rejected
    # immediately instead of being staged and only failing at approve time.
    if action == "add" and not content:
        return tool_error("Content is required for 'add' action.", success=False)
    if action == "replace" and (not old_text or not content):
        missing = "old_text" if not old_text else "content"
        if not old_text:
            # The client/model omitted old_text. Replace is inherently targeted
            # -- we can't guess which entry. Return the current inventory plus a
            # retry instruction so the model can reissue with old_text set,
            # instead of hitting a dead-end error. (issues #43412, #49466)
            return _missing_old_text_error(store, target, "replace")
        return tool_error(f"{missing} is required for 'replace' action.", success=False)
    if action == "remove" and not old_text:
        return _missing_old_text_error(store, target, "remove")

    if action == "add":
        result = store.add(target, content, preview=preview)

    elif action == "replace":
        result = store.replace(target, old_text, content, preview=preview)

    elif action == "remove":
        result = store.remove(target, old_text, preview=preview)

    else:
        return tool_error(f"Unknown action '{action}'. Use: add, replace, remove", success=False)

    return json.dumps(result, ensure_ascii=False)


MEMORY_SCHEMA = {
    "name": "memory",
    "description": (
        "Inspect and maintain bounded cross-session memory. Use action='list' to "
        "read the live target. Use top-level fields for one change or operations "
        "for one atomic batch; the final state must fit the target limit. A full "
        "target remains unchanged. Consolidate or remove an entry only when durable "
        "evidence shows it is overlapping, obsolete, or superseded—never merely to "
        "make room. Targets: 'user' stores durable identity, preferences, and "
        "communication style; 'memory' stores other durable user-specific context "
        "and personal constraints."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "add", "replace", "remove"],
                "description": "List the live target or perform one mutation. Omit when using 'operations'."
            },
            "target": {
                "type": "string",
                "enum": ["memory", "user"],
                "description": "Which memory store: 'memory' for personal notes, 'user' for user profile."
            },
            "content": {
                "type": "string",
                "description": "The entry content. Required for 'add' and 'replace' (single-op shape). Alias: 'new_text' is also accepted (mirrors old_text)."
            },
            "old_text": {
                "type": "string",
                "description": "REQUIRED for 'replace' and 'remove' (single-op shape): a short unique substring identifying the existing entry to modify. Omit only for 'add'."
            },
            "new_text": {
                "type": "string",
                "description": "Alias for 'content' (single-op shape). Provided so the replace/remove old_text/new_text pairing works; if both are set, 'content' wins."
            },
            "change_reason": {
                "type": "string",
                "maxLength": 240,
                "description": (
                    "Required for any mutation: one short factual sentence naming "
                    "the durable evidence. Omit for action='list'. Never include "
                    "private reasoning or a raw transcript."
                ),
            },
            "operations": {
                "type": "array",
                "description": (
                    "Atomic mutation list. The final state must fit the target limit; "
                    "each item uses action with content and/or old_text as required."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                        "content": {"type": "string", "description": "Entry content for add/replace. Alias: 'new_text'."},
                        "new_text": {"type": "string", "description": "Alias for 'content' in a batch op."},
                        "old_text": {"type": "string", "description": "Substring identifying the entry for replace/remove."},
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["target"],
    },
}


def _memory_approval_summary(args: Dict[str, Any]) -> Optional[str]:
    """Describe one valid-shaped memory mutation without changing the store."""

    target = args.get("target", "memory")
    if target is None:
        target = "memory"
    if target not in {"memory", "user"}:
        return None
    label = "user profile" if target == "user" else "memory"
    operations = args.get("operations")
    if operations:
        if args.get("action") is not None or not isinstance(operations, list):
            return None
        detail = json.dumps(operations, ensure_ascii=False)
        return (
            f"Apply {len(operations)} change(s) to {label}:\n"
            f"{detail[:1600]}{'…' if len(detail) > 1600 else ''}"
        )

    action = args.get("action")
    if action not in {"add", "replace", "remove"}:
        return None
    content = args.get("content")
    if content is None:
        content = args.get("new_text")
    old_text = args.get("old_text")
    if action == "add" and not content:
        return None
    if action == "replace" and (not old_text or not content):
        return None
    if action == "remove" and not old_text:
        return None
    if action == "add":
        detail = str(content)
    elif action == "replace":
        detail = f"Old: {old_text}\nNew: {content}"
    else:
        detail = str(old_text)
    detail = detail[:1600] + ("…" if len(detail) > 1600 else "")
    return f"{action.title()} in {label}:\n{detail}"


def _invoke_memory(
    args: Dict[str, Any], store: Optional[MemoryStore], *, preview: bool = False
) -> str:
    return memory_tool(
        action=args.get("action"),
        target=args.get("target", "memory"),
        content=args.get("content"),
        old_text=args.get("old_text"),
        new_text=args.get("new_text"),
        operations=args.get("operations"),
        store=store,
        preview=preview,
    )


async def preauthorize_memory_mutation(
    args: Dict[str, Any], context: ToolContext
) -> Optional[str]:
    """Validate and resolve required memory approval before audit locking."""

    preview = _invoke_memory(args, context.memory_store, preview=True)
    try:
        payload = json.loads(preview)
    except (TypeError, ValueError):
        return preview
    if not isinstance(payload, dict) or payload.get("would_change") is not True:
        return preview

    summary = _memory_approval_summary(args)
    if summary is None:
        return tool_error("The memory mutation is not valid for approval.")
    outcome = await context.authorize("memory", summary)
    if not outcome.approved:
        return tool_error(
            approval_error(outcome),
            success=False,
            approval=outcome.status,
        )
    return None


async def handle_memory(args: Dict[str, Any], context: ToolContext) -> str:
    if (
        "memory" not in context.persistence_approved_categories
        and approval_required(context.config, "memory")
    ):
        final = await preauthorize_memory_mutation(args, context)
        if final is not None:
            return final
    return _invoke_memory(args, context.memory_store)


MEMORY_TOOL = Tool(
    "memory",
    "memory",
    MEMORY_SCHEMA,
    handle_memory,
    emoji="🧠",
)


__all__ = [
    "ENTRY_DELIMITER",
    "MEMORY_BLOCK_HEADERS",
    "MEMORY_SCHEMA",
    "MEMORY_TOOL",
    "MemoryStore",
    "handle_memory",
    "memory_tool",
    "preauthorize_memory_mutation",
]
