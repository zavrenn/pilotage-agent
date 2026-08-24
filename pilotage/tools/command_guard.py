"""Unconditional model-controlled command guards derived from Hermes.

This is deliberately not an approval framework. It contains only Hermes's
small hardline floor and the resident-agent self-lifecycle guard.
"""

from __future__ import annotations

import ast
import fnmatch
import os
import posixpath
import re
import shlex
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Sequence

from .ansi_strip import strip_ansi
from .shell_heredoc import strip_inert_heredoc_bodies

_MAX_COMMAND_CHARS = 128_000
_MAX_SEPARATOR_FREE_CHARS = 4_096
_MAX_SEGMENTS = 25_000
_MAX_RECURSION = 8
_MAX_SCRIPT_BYTES = 1024 * 1024
_CONTROL_TOKENS = frozenset(";&|()")
_SHELLS = frozenset({"sh", "bash", "dash", "ksh", "zsh"})
_PYTHON_EXECUTABLE = re.compile(r"(?:py|python(?:[23](?:\.\d+)*)?)", re.I)
_BLOCK_DEVICE = re.compile(r"^/dev/(?:sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*$", re.I)
_FORK_BOMB = re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PILOTAGE_EMBEDDED = re.compile(
    r"(?i)(?<![/\w.\-])pilotage\b(?P<flags>[^\n;|&]{0,256}?)"
    r"\bservice\s+(?P<action>stop|restart)\b"
)

_TRANSPARENT_PREFIXES = frozenset(
    {
        "sudo",
        "doas",
        "env",
        "nohup",
        "setsid",
        "nice",
        "ionice",
        "stdbuf",
        "timeout",
        "exec",
        "command",
        "builtin",
        "time",
        "pkexec",
    }
)
_PREFIX_VALUE_OPTIONS = {
    "sudo": {"-u", "-g", "-U", "-C", "-p", "--user", "--group", "--prompt"},
    "doas": {"-u", "-C"},
    "env": {"-u", "--unset", "-S", "--split-string", "-C", "--chdir"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p", "--class", "--classdata"},
    "stdbuf": {"-i", "-o", "-e", "--input", "--output", "--error"},
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
    "pkexec": {"--user"},
}


@dataclass(frozen=True)
class CommandGuardFinding:
    category: str
    description: str

    @property
    def message(self) -> str:
        if self.category == "catastrophic":
            return f"Blocked catastrophic command: {self.description}."
        return f"Blocked Pilotage self-lifecycle command: {self.description}."


def profile_name_for_state_dir(state_dir: Any) -> str:
    """Derive the selected profile from Pilotage's existing path contract."""
    try:
        path = Path(state_dir).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return "default"
    if path.parent.name == "profiles" and path.name:
        return path.name.lower()
    return "default"


def _normalize(text: str) -> str:
    text = strip_ansi(str(text or "")).replace("\x00", "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\\\r?\n[ \t]*", "", text)
    text = re.sub(r"\$\{IFS\b[^}]*\}|\$IFS\b", " ", text)
    return text


def _parser_limit_exceeded(text: str) -> bool:
    if len(text) > _MAX_COMMAND_CHARS:
        return True
    if len(text) > _MAX_SEPARATOR_FREE_CHARS and not any(
        char in text for char in ";&|\n"
    ):
        return True
    separators = 0
    for char in text:
        if char in ";&|\n":
            separators += 1
            if separators >= _MAX_SEGMENTS:
                return True
    return False


def _split_logical_lines(text: str) -> list[str]:
    lines: list[str] = []
    current: list[str] = []
    single = False
    double = False
    escaped = False
    for char in text:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\" and not single:
            current.append(char)
            escaped = True
            continue
        if char == "'" and not double:
            single = not single
            current.append(char)
            continue
        if char == '"' and not single:
            double = not double
            current.append(char)
            continue
        if char == "\n" and not single and not double:
            lines.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        lines.append("".join(current))
    return lines


def _tokenize(line: str) -> Optional[list[str]]:
    try:
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        return list(lexer)
    except ValueError:
        return None


def _iter_segments(text: str) -> Iterator[list[str]]:
    for logical_line in _split_logical_lines(text):
        token_sets: list[list[str]] = []
        tokens = _tokenize(logical_line)
        if tokens is not None:
            token_sets.append(tokens)
        else:
            for physical_line in logical_line.splitlines():
                fallback = _tokenize(physical_line)
                if fallback is not None:
                    token_sets.append(fallback)
        for tokens in token_sets:
            segment: list[str] = []
            for token in tokens:
                if token and set(token) <= _CONTROL_TOKENS:
                    if segment:
                        yield segment
                        segment = []
                    continue
                segment.append(token)
            if segment:
                yield segment


def _executable_name(token: str) -> str:
    name = Path(token).name or token
    name = name.lower()
    return name[:-4] if name.endswith(".exe") else name


def _command_index(segment: Sequence[str]) -> Optional[int]:
    for index, token in enumerate(segment):
        if token in {"{", "}"} or _ENV_ASSIGNMENT.match(token):
            continue
        return index
    return None


def _peel_one_prefix(segment: Sequence[str], index: int) -> int:
    if index >= len(segment):
        return index
    name = _executable_name(segment[index])
    if name not in _TRANSPARENT_PREFIXES:
        return index
    index += 1
    value_options = _PREFIX_VALUE_OPTIONS.get(name, set())
    while index < len(segment):
        token = segment[index]
        if token == "--":
            index += 1
            break
        if token in value_options:
            index += 2
            continue
        if token.startswith("-") or _ENV_ASSIGNMENT.match(token):
            index += 1
            continue
        break
    if name == "timeout" and index < len(segment):
        index += 1
    return index


def _peel_prefixes(segment: Sequence[str], index: int) -> int:
    for _ in range(_MAX_RECURSION):
        next_index = _peel_one_prefix(segment, index)
        if next_index == index:
            return index
        index = next_index
    return index


def _env_split_string_command(
    segment: Sequence[str], index: int
) -> Optional[str]:
    """Return the command reconstructed from GNU ``env -S`` syntax."""
    if _executable_name(segment[index]) != "env":
        return None
    position = index + 1
    value_options = _PREFIX_VALUE_OPTIONS["env"] - {
        "-S",
        "--split-string",
    }
    while position < len(segment):
        token = segment[position]
        if token == "--":
            return None
        payload: Optional[str] = None
        remainder_index = position + 1
        if token in {"-S", "--split-string"}:
            if remainder_index >= len(segment):
                return None
            payload = segment[remainder_index]
            remainder_index += 1
        elif token.startswith("--split-string="):
            payload = token.split("=", 1)[1]
        elif token.startswith("-S") and token != "-S":
            payload = token[2:]
        if payload is not None:
            remainder = shlex.join(list(segment[remainder_index:]))
            return " ".join(part for part in (payload, remainder) if part)
        if token in value_options:
            position += 2
            continue
        if token.startswith("-") or _ENV_ASSIGNMENT.match(token):
            position += 1
            continue
        return None
    return None


def _root_collapsing_path(path: str) -> bool:
    candidate = path.rstrip("*")
    candidate = re.sub(r"/+", "/", candidate)
    return candidate.startswith("/") and posixpath.normpath(candidate) == "/"


def _protected_rm_target(target: str) -> Optional[str]:
    if _root_collapsing_path(target):
        return "recursive delete of root filesystem"
    normalized = target.rstrip("/")
    protected = {"/home", "/root", "/etc", "/usr", "/var", "/bin", "/sbin", "/boot", "/lib"}
    if normalized in protected or (
        normalized.endswith("/*") and normalized[:-2].rstrip("/") in protected
    ):
        return "recursive delete of system directory"
    if re.fullmatch(r"(?:~|\$\{?HOME\}?)(?:/?|/\*)?", target, re.I):
        return "recursive delete of home directory"
    try:
        home = Path.home().as_posix().rstrip("/")
    except (OSError, RuntimeError):
        home = ""
    if home and normalized in {home, home + "/*"}:
        return "recursive delete of home directory"
    return None


def _hardline_finding(segment: Sequence[str], index: int) -> Optional[CommandGuardFinding]:
    name = _executable_name(segment[index])
    arguments = list(segment[index + 1 :])
    if name == "rm":
        for target in arguments:
            if target == "--" or target.startswith("-"):
                continue
            description = _protected_rm_target(target)
            if description:
                return CommandGuardFinding("catastrophic", description)
    if name == "mkfs" or name.startswith("mkfs."):
        return CommandGuardFinding("catastrophic", "format filesystem (mkfs)")
    if name == "dd" and any(
        token.lower().startswith("of=")
        and _BLOCK_DEVICE.fullmatch(token[3:])
        for token in arguments
    ):
        return CommandGuardFinding("catastrophic", "dd to raw block device")
    for position, token in enumerate(arguments[:-1]):
        if token.startswith(">") and _BLOCK_DEVICE.fullmatch(arguments[position + 1]):
            return CommandGuardFinding("catastrophic", "redirect to raw block device")
    if name == "kill" and "-1" in arguments:
        return CommandGuardFinding("catastrophic", "kill all processes")
    if name in {"shutdown", "reboot", "halt", "poweroff"}:
        return CommandGuardFinding("catastrophic", "system shutdown/reboot")
    if name in {"init", "telinit"} and any(arg in {"0", "6"} for arg in arguments):
        return CommandGuardFinding("catastrophic", f"{name} 0/6 (shutdown/reboot)")
    if name == "systemctl":
        action, _ = _systemctl_action_and_units(arguments)
        if action in {"poweroff", "reboot", "halt", "kexec"}:
            return CommandGuardFinding("catastrophic", "systemctl poweroff/reboot")
    return None


def _systemctl_action_and_units(arguments: Sequence[str]) -> tuple[str, list[str]]:
    value_options = {"-H", "--host", "-M", "--machine", "--root", "--image"}
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token == "--":
            index += 1
            break
        if token in value_options:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        break
    if index >= len(arguments):
        return "", []
    return arguments[index].lower(), list(arguments[index + 1 :])


def _unit_targets_current(unit: str, current_profile: Optional[str]) -> bool:
    if not current_profile:
        return False
    written = unit.lower()
    expected = f"pilotage-agent@{current_profile.lower()}.service"
    if not written.endswith(".service"):
        written += ".service"
    return written == expected or fnmatch.fnmatchcase(expected, written)


def _pilotage_lifecycle(
    arguments: Sequence[str], current_profile: Optional[str]
) -> bool:
    selected: Optional[str] = None
    positional: list[str] = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if token in {"-p", "--profile"}:
            if index + 1 >= len(arguments):
                return False
            selected = arguments[index + 1].strip("\"'").lower()
            index += 2
            continue
        if token.startswith("--profile="):
            selected = token.split("=", 1)[1].strip("\"'").lower()
            index += 1
            continue
        if token.startswith("-") and not positional:
            index += 1
            continue
        positional.append(token.lower())
        index += 1
    if len(positional) < 2 or positional[:1] != ["service"]:
        return False
    if positional[1] not in {"stop", "restart"}:
        return False
    return selected is None or (
        current_profile is not None and selected == current_profile.lower()
    )


def _self_lifecycle_finding(
    segment: Sequence[str], index: int, current_profile: Optional[str]
) -> Optional[CommandGuardFinding]:
    name = _executable_name(segment[index])
    arguments = list(segment[index + 1 :])
    if name == "pilotage" and _pilotage_lifecycle(arguments, current_profile):
        return CommandGuardFinding(
            "self_lifecycle",
            "the agent cannot stop or restart its own service",
        )
    if name == "systemctl":
        action, units = _systemctl_action_and_units(arguments)
        if action in {"start", "stop", "restart"} and any(
            _unit_targets_current(unit, current_profile) for unit in units
        ):
            return CommandGuardFinding(
                "self_lifecycle",
                "systemctl targets the active Pilotage service",
            )
    if name in {"kill", "pkill"}:
        joined = " ".join(arguments).lower()
        if "pilotage" in joined and "agent" in joined:
            return CommandGuardFinding("self_lifecycle", "process kill targets the Pilotage agent")
    return None


def _mask_quoted_data(text: str) -> str:
    output: list[str] = []
    quote: Optional[str] = None
    escaped = False
    for char in text:
        if escaped:
            output.append(" " if quote else char)
            escaped = False
            continue
        if char == "\\" and quote != "'":
            output.append(" " if quote else char)
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = None
                output.append(char)
            else:
                output.append(" ")
            continue
        if char in {"'", '"'}:
            quote = char
        output.append(char)
    return "".join(output)


def _substitution_end(text: str, start: int) -> Optional[int]:
    depth = 1
    quote: Optional[str] = None
    escaped = False
    index = start + 2
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif char == "'" and quote != '"':
            quote = None if quote == "'" else "'"
        elif char == '"' and quote != "'":
            quote = None if quote == '"' else '"'
        elif quote != "'" and char == "(":
            depth += 1
        elif quote != "'" and char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _iter_substitution_payloads(text: str) -> Iterator[str]:
    quote: Optional[str] = None
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            index += 1
            continue
        if char == "'" and quote != '"':
            quote = None if quote == "'" else "'"
            index += 1
            continue
        if char == '"' and quote != "'":
            quote = None if quote == '"' else '"'
            index += 1
            continue
        if quote != "'" and text.startswith("$(", index):
            end = _substitution_end(text, index)
            if end is not None:
                yield text[index + 2 : end]
                index = end + 1
                continue
        if quote != "'" and char == "`":
            end = text.find("`", index + 1)
            if end != -1:
                yield text[index + 1 : end]
                index = end + 1
                continue
        index += 1


def _resolve_script(candidate: str, cwd: Optional[str]) -> Optional[Path]:
    if not candidate or "\x00" in candidate:
        return None
    try:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = Path(cwd or Path.cwd()) / path
        return path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _scan_script(
    candidate: str,
    *,
    cwd: Optional[str],
    current_profile: Optional[str],
    depth: int,
    visited: set[Path],
) -> Optional[CommandGuardFinding]:
    path = _resolve_script(candidate, cwd)
    if path is None or path in visited or not path.exists():
        return None
    try:
        metadata = path.stat()
    except OSError:
        return CommandGuardFinding("self_lifecycle", "referenced script could not be inspected")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_SCRIPT_BYTES:
        return CommandGuardFinding(
            "self_lifecycle",
            "referenced script could not be inspected safely",
        )
    try:
        data = path.read_bytes()
    except OSError:
        return CommandGuardFinding("self_lifecycle", "referenced script could not be inspected")
    if data.startswith((b"\x7fELF", b"MZ", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf")):
        return None
    visited.add(path)
    text = data.decode("utf-8-sig", errors="replace").replace("\x00", "")
    return _find_blocked_command(
        text,
        cwd=str(path.parent),
        current_profile=current_profile,
        scan_referenced_scripts=True,
        hardline=False,
        depth=depth + 1,
        visited=visited,
    )


def _python_source_payload(arguments: Sequence[str]) -> Optional[str]:
    """Return code passed to a Python interpreter's ``-c`` option."""
    position = 0
    while position < len(arguments):
        token = arguments[position]
        if token == "--":
            return None
        if token == "-c":
            return arguments[position + 1] if position + 1 < len(arguments) else None
        if token.startswith("-c") and token != "-c":
            return token[2:]
        if token in {"-W", "-X", "--check-hash-based-pycs"}:
            position += 2
            continue
        if token.startswith(("-W", "-X", "--check-hash-based-pycs=")):
            position += 1
            continue
        if token.startswith("-"):
            position += 1
            continue
        return None
    return None


def _carrier_or_script_finding(
    segment: Sequence[str],
    index: int,
    *,
    cwd: Optional[str],
    current_profile: Optional[str],
    scan_referenced_scripts: bool,
    hardline: bool,
    depth: int,
    visited: set[Path],
) -> Optional[CommandGuardFinding]:
    name = _executable_name(segment[index])
    arguments = list(segment[index + 1 :])
    if name in _SHELLS:
        position = 0
        while position < len(arguments):
            token = arguments[position]
            if token in {"-O", "+O", "-o", "+o"}:
                position += 2
                continue
            if token == "-c" or (
                token.startswith("-") and "c" in token[1:] and token != "--"
            ):
                if position + 1 < len(arguments):
                    return _find_blocked_command(
                        arguments[position + 1],
                        cwd=cwd,
                        current_profile=current_profile,
                        scan_referenced_scripts=scan_referenced_scripts,
                        hardline=hardline,
                        depth=depth + 1,
                        visited=visited,
                    )
                return None
            if token == "--":
                position += 1
                break
            if token.startswith("-"):
                position += 1
                continue
            break
        if scan_referenced_scripts and position < len(arguments):
            return _scan_script(
                arguments[position],
                cwd=cwd,
                current_profile=current_profile,
                depth=depth,
                visited=visited,
            )
    if _PYTHON_EXECUTABLE.fullmatch(name):
        payload = _python_source_payload(arguments)
        if payload is not None:
            return _find_blocked_python_source(
                payload,
                cwd=cwd,
                current_profile=current_profile,
                hardline=hardline,
            )
    if name == "eval" and arguments:
        return _find_blocked_command(
            " ".join(arguments),
            cwd=cwd,
            current_profile=current_profile,
            scan_referenced_scripts=scan_referenced_scripts,
            hardline=hardline,
            depth=depth + 1,
            visited=visited,
        )
    if scan_referenced_scripts and name in {"source", "."} and arguments:
        return _scan_script(
            arguments[0],
            cwd=cwd,
            current_profile=current_profile,
            depth=depth,
            visited=visited,
        )
    if scan_referenced_scripts and ("/" in segment[index] or name.endswith(".sh")):
        return _scan_script(
            segment[index],
            cwd=cwd,
            current_profile=current_profile,
            depth=depth,
            visited=visited,
        )
    return None


def _find_blocked_command(
    text: str,
    *,
    cwd: Optional[str],
    current_profile: Optional[str],
    scan_referenced_scripts: bool,
    hardline: bool,
    depth: int,
    visited: set[Path],
) -> Optional[CommandGuardFinding]:
    if depth > _MAX_RECURSION:
        category = "catastrophic" if hardline else "self_lifecycle"
        return CommandGuardFinding(category, "command parser limit exceeded")
    normalized = _normalize(strip_inert_heredoc_bodies(text))
    if hardline and _parser_limit_exceeded(normalized):
        return CommandGuardFinding("catastrophic", "command parser limit exceeded")
    if hardline and _FORK_BOMB.search(_mask_quoted_data(normalized)):
        return CommandGuardFinding("catastrophic", "fork bomb")
    for payload in _iter_substitution_payloads(normalized):
        finding = _find_blocked_command(
            payload,
            cwd=cwd,
            current_profile=current_profile,
            scan_referenced_scripts=scan_referenced_scripts,
            hardline=hardline,
            depth=depth + 1,
            visited=visited,
        )
        if finding:
            return finding
    for segment in _iter_segments(normalized):
        index = _command_index(segment)
        if index is None:
            continue
        peeled_index = _peel_prefixes(segment, index)
        prefix_index = index
        while prefix_index < peeled_index:
            split_command = _env_split_string_command(segment, prefix_index)
            if split_command:
                finding = _find_blocked_command(
                    split_command,
                    cwd=cwd,
                    current_profile=current_profile,
                    scan_referenced_scripts=scan_referenced_scripts,
                    hardline=hardline,
                    depth=depth + 1,
                    visited=visited,
                )
                if finding:
                    return finding
            next_index = _peel_one_prefix(segment, prefix_index)
            if next_index == prefix_index:
                break
            prefix_index = next_index
        index = peeled_index
        if index >= len(segment):
            continue
        if hardline:
            finding = _hardline_finding(segment, index)
            if finding:
                return finding
        finding = _self_lifecycle_finding(segment, index, current_profile)
        if finding:
            return finding
        finding = _carrier_or_script_finding(
            segment,
            index,
            cwd=cwd,
            current_profile=current_profile,
            scan_referenced_scripts=scan_referenced_scripts,
            hardline=hardline,
            depth=depth,
            visited=visited,
        )
        if finding:
            return finding
    return None


def find_blocked_command(
    text: str,
    *,
    cwd: Optional[str] = None,
    current_profile: Optional[str] = "default",
    scan_referenced_scripts: bool = True,
) -> Optional[CommandGuardFinding]:
    """Return the first unconditional terminal guard finding, if any."""
    return _find_blocked_command(
        text,
        cwd=cwd,
        current_profile=current_profile,
        scan_referenced_scripts=scan_referenced_scripts,
        hardline=True,
        depth=0,
        visited=set(),
    )


def find_embedded_self_lifecycle(
    text: str, *, current_profile: Optional[str] = "default"
) -> Optional[CommandGuardFinding]:
    """Scan a cron prompt for concrete Pilotage lifecycle command shapes."""
    normalized = _normalize(text)
    direct = _find_blocked_command(
        normalized,
        cwd=None,
        current_profile=current_profile,
        scan_referenced_scripts=False,
        hardline=False,
        depth=0,
        visited=set(),
    )
    if direct:
        return direct
    for match in _PILOTAGE_EMBEDDED.finditer(normalized):
        tokens = _tokenize(match.group(0)) or []
        if tokens and _pilotage_lifecycle(tokens[1:], current_profile):
            return CommandGuardFinding(
                "self_lifecycle",
                "the scheduled agent cannot stop or restart its own service",
            )
    for match in re.finditer(
        r"(?i)(?<![/\w.\-])(?:systemctl|p?kill)\b[^\n;|&]{0,512}", normalized
    ):
        for segment in _iter_segments(match.group(0)):
            index = _command_index(segment)
            if index is not None:
                finding = _self_lifecycle_finding(segment, index, current_profile)
                if finding:
                    return finding
    return None


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _literal_value(node: ast.AST, constants: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_literal_value(item, constants) for item in node.elts]
        return values if all(isinstance(item, str) for item in values) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal_value(node.left, constants)
        right = _literal_value(node.right, constants)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
        if isinstance(left, list) and isinstance(right, list):
            return left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for item in node.values:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                parts.append(item.value)
                continue
            if (
                isinstance(item, ast.FormattedValue)
                and item.conversion == -1
                and item.format_spec is None
            ):
                value = _literal_value(item.value, constants)
                if isinstance(value, str):
                    parts.append(value)
                    continue
            return None
        return "".join(parts)
    return None


def _command_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return shlex.join(value)
    return None


_UNKNOWN = object()
_SHELL_PROCESS_APIS = frozenset(
    {
        "os.popen",
        "os.system",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.popen",
        "subprocess.run",
    }
)
_SHELL_CREATION_APIS = frozenset({"asyncio.create_subprocess_shell"})
_EXEC_PROCESS_APIS = frozenset(
    {
        "asyncio.create_subprocess_exec",
        "os.execl",
        "os.execlp",
    }
)
_VECTOR_PROCESS_APIS = frozenset(
    {
        "os.execv",
        "os.execve",
        "os.execvp",
        "os.execvpe",
    }
)


@dataclass
class _PythonBindings:
    constants: Dict[str, Any]
    aliases: Dict[str, str]


class _PythonCommandVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        cwd: Optional[str],
        current_profile: Optional[str],
        hardline: bool,
    ) -> None:
        self.cwd = cwd
        self.current_profile = current_profile
        self.hardline = hardline
        self.finding: Optional[CommandGuardFinding] = None
        self.scopes = [_PythonBindings(constants={}, aliases={})]

    def _visible_constants(self) -> Dict[str, Any]:
        visible: Dict[str, Any] = {}
        for scope in self.scopes:
            for name in scope.aliases:
                visible.pop(name, None)
            visible.update(scope.constants)
        return visible

    def _bind(self, name: str, value: Any = _UNKNOWN) -> None:
        scope = self.scopes[-1]
        scope.aliases.pop(name, None)
        scope.constants[name] = value

    def _bind_target(self, target: ast.AST, value: Any = _UNKNOWN) -> None:
        if isinstance(target, ast.Name):
            self._bind(target.id, value)
        elif isinstance(target, (ast.List, ast.Tuple)):
            for item in target.elts:
                self._bind_target(item)

    def _bind_alias(self, name: str, canonical: str) -> None:
        scope = self.scopes[-1]
        scope.constants.pop(name, None)
        scope.aliases[name] = canonical

    def _canonical_call_name(self, node: ast.AST) -> str:
        written = _call_name(node)
        if not written:
            return ""
        root, separator, suffix = written.partition(".")
        for scope in reversed(self.scopes):
            if root in scope.constants:
                return ""
            canonical = scope.aliases.get(root)
            if canonical:
                return f"{canonical}{separator}{suffix}".lower()
        return written.lower()

    @staticmethod
    def _argument_names(arguments: ast.arguments) -> Iterator[str]:
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            yield argument.arg
        if arguments.vararg:
            yield arguments.vararg.arg
        if arguments.kwarg:
            yield arguments.kwarg.arg

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".", 1)[0]
            canonical = alias.name if alias.asname else bound
            self._bind_alias(bound, canonical)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.module:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            self._bind_alias(
                alias.asname or alias.name,
                f"{node.module}.{alias.name}",
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = _literal_value(node.value, self._visible_constants())
        for target in node.targets:
            self._bind_target(target, value if value is not None else _UNKNOWN)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            value = _literal_value(node.value, self._visible_constants())
        else:
            value = None
        self._bind_target(node.target, value if value is not None else _UNKNOWN)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._bind_target(node.target)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self._bind(node.name)
        self.scopes.append(_PythonBindings(constants={}, aliases={}))
        try:
            for name in self._argument_names(node.args):
                self._bind(name)
            for statement in node.body:
                self.visit(statement)
                if self.finding:
                    return
        finally:
            self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self.scopes.append(_PythonBindings(constants={}, aliases={}))
        try:
            for name in self._argument_names(node.args):
                self._bind(name)
            self.visit(node.body)
        finally:
            self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        self._bind(node.name)
        self.scopes.append(_PythonBindings(constants={}, aliases={}))
        try:
            for statement in node.body:
                self.visit(statement)
                if self.finding:
                    return
        finally:
            self.scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if self.finding:
            return
        name = self._canonical_call_name(node.func)
        constants = self._visible_constants()
        command: Optional[str] = None
        if name in _SHELL_PROCESS_APIS:
            argument = node.args[0] if node.args else next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"args", "cmd", "command"}
                ),
                None,
            )
            if argument is not None:
                command = _command_text(_literal_value(argument, constants))
        elif name in _SHELL_CREATION_APIS:
            argument = node.args[0] if node.args else next(
                (
                    keyword.value
                    for keyword in node.keywords
                    if keyword.arg in {"args", "cmd"}
                ),
                None,
            )
            if argument is not None:
                command = _command_text(_literal_value(argument, constants))
        elif name in _EXEC_PROCESS_APIS and node.args:
            values = [_literal_value(item, constants) for item in node.args]
            if all(isinstance(item, str) for item in values):
                command = shlex.join(values)
        elif name in _VECTOR_PROCESS_APIS and len(node.args) >= 2:
            command = _command_text(_literal_value(node.args[1], constants))
        if command:
            self.finding = _find_blocked_command(
                command,
                cwd=self.cwd,
                current_profile=self.current_profile,
                scan_referenced_scripts=False,
                hardline=self.hardline,
                depth=0,
                visited=set(),
            )
        if not self.finding:
            self.generic_visit(node)


def _find_blocked_python_source(
    code: str,
    *,
    cwd: Optional[str],
    current_profile: Optional[str],
    hardline: bool,
) -> Optional[CommandGuardFinding]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    visitor = _PythonCommandVisitor(
        cwd=cwd,
        current_profile=current_profile,
        hardline=hardline,
    )
    visitor.visit(tree)
    return visitor.finding


def find_blocked_python_source(
    code: str,
    *,
    cwd: Optional[str] = None,
    current_profile: Optional[str] = "default",
) -> Optional[CommandGuardFinding]:
    """Inspect literal commands handed to Python process-launching APIs."""
    return _find_blocked_python_source(
        code,
        cwd=cwd,
        current_profile=current_profile,
        hardline=True,
    )


__all__ = [
    "CommandGuardFinding",
    "find_blocked_command",
    "find_blocked_python_source",
    "find_embedded_self_lifecycle",
    "profile_name_for_state_dir",
]
