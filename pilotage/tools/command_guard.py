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
        if self.category == "persistence":
            return f"Blocked direct persistent-store access: {self.description}."
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


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def find_persistence_store_reference(
    text: str,
    *,
    cwd: Optional[str],
    state_dir: Any,
    source_kind: str = "shell",
) -> Optional[CommandGuardFinding]:
    """Fence direct mutation of validated learning stores.

    This is defense in depth, not a same-UID sandbox. The model is separately
    instructed to use the canonical memory and file tools, which provide the
    semantic policy, validation, provenance, and rollback boundary. Reads and
    execution are intentionally allowed because skills are application code.
    """

    if state_dir is None:
        # Direct handler tests and embedders may omit profile state entirely.
        # Production Config always supplies it; without a root there is no
        # canonical store path this guard can resolve.
        return None
    try:
        state = Path(state_dir).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return CommandGuardFinding(
            "persistence", "the active profile state path is unavailable"
        )

    protected_roots = tuple(
        root.resolve(strict=False)
        for root in (state / "memories", state / "skills")
    )
    try:
        current = Path(cwd).expanduser().resolve(strict=False) if cwd else None
    except (OSError, RuntimeError, TypeError, ValueError):
        current = None

    normalized = _normalize(text).replace("\\", "/").casefold()
    normalized = re.sub(r"/+", "/", normalized)
    audit_text = str(state / "persistence-audit.db").replace("\\", "/").casefold()
    if (
        audit_text in normalized
        or "persistence-audit.db" in normalized
    ):
        return CommandGuardFinding(
            "persistence", "the private provenance journal is operator-only"
        )

    mutates = (
        _python_mutates_protected(text, current, protected_roots)
        if source_kind == "python"
        else _shell_mutates_protected(text, current, protected_roots)
    )
    if mutates:
        return CommandGuardFinding(
            "persistence", "memory or skill mutations require their canonical tools"
        )
    return None


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


_SHELL_PATH_MUTATORS = frozenset(
    {"rm", "unlink", "rmdir", "mkdir", "touch", "truncate", "mv", "rename"}
)
_SHELL_DESTINATION_MUTATORS = frozenset({"cp", "install", "ln"})


def _target_is_protected(
    target: str,
    cwd: Optional[Path],
    roots: Sequence[Path],
) -> bool:
    written = str(target or "").strip().strip("\"'`[],{}()")
    if not written or written.startswith("-"):
        return False
    normalized = re.sub(r"/+", "/", written.replace("\\", "/")).casefold()
    root_text = [
        str(root).replace("\\", "/").casefold().rstrip("/") for root in roots
    ]
    if any(
        root and (normalized == root or normalized.startswith(root + "/"))
        for root in root_text
    ):
        return True
    home_reference = re.search(
        r"(?:pilotage_home|hermes_home|\.pilotage-agent)", normalized
    )
    store_reference = re.search(
        r"(?<![a-z0-9_.-])(?:memories|skills)(?![a-z0-9_.-])", normalized
    )
    if home_reference and store_reference:
        return True
    try:
        candidate = Path(written).expanduser()
        if not candidate.is_absolute():
            if cwd is None:
                return False
            candidate = cwd / candidate
        candidate = candidate.resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return any(candidate == root or _path_is_within(candidate, root) for root in roots)


def _shell_source_payload(arguments: Sequence[str]) -> Optional[str]:
    position = 0
    while position < len(arguments):
        token = arguments[position]
        if token in {"-O", "+O", "-o", "+o"}:
            position += 2
            continue
        if token == "-c" or (
            token.startswith("-") and "c" in token[1:] and token != "--"
        ):
            return arguments[position + 1] if position + 1 < len(arguments) else None
        if token == "--":
            return None
        if token.startswith("-"):
            position += 1
            continue
        return None
    return None


_SHELL_VARIABLE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|"
    r"(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)


def _expand_shell_bindings(value: str, bindings: Dict[str, str]) -> str:
    return _SHELL_VARIABLE.sub(
        lambda match: bindings.get(
            match.group("braced") or match.group("plain"),
            match.group(0),
        ),
        value,
    )


def _shell_mutates_protected(
    text: str,
    cwd: Optional[Path],
    roots: Sequence[Path],
    *,
    depth: int = 0,
) -> bool:
    current = cwd
    bindings: Dict[str, str] = {}
    for segment in _iter_segments(_normalize(text)):
        index = _command_index(segment)
        if index is None:
            for token in segment:
                if _ENV_ASSIGNMENT.match(token):
                    name, value = token.split("=", 1)
                    bindings[name] = _expand_shell_bindings(value, bindings)
            continue
        index = _peel_prefixes(segment, index)
        if index >= len(segment):
            continue
        name = _executable_name(segment[index])
        arguments = list(segment[index + 1 :])
        if name in _SHELLS and depth < _MAX_RECURSION:
            payload = _shell_source_payload(arguments)
            if payload and _shell_mutates_protected(
                payload,
                current,
                roots,
                depth=depth + 1,
            ):
                return True
        if name in {"cd", "chdir", "pushd"}:
            operands = [token for token in arguments if not token.startswith("-")]
            if operands:
                try:
                    destination = Path(
                        _expand_shell_bindings(operands[-1], bindings)
                    ).expanduser()
                    if not destination.is_absolute() and current is not None:
                        destination = current / destination
                    current = destination.resolve(strict=False)
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
            continue
        if _PYTHON_EXECUTABLE.fullmatch(name):
            payload = _python_source_payload(arguments)
            if payload and _python_mutates_protected(payload, current, roots):
                return True
        targets = [
            arguments[position + 1]
            for position, token in enumerate(arguments[:-1])
            if token in {">", ">>"}
        ]
        operands = [
            token
            for token in arguments
            if token not in {"--", ">", ">>"} and not token.startswith("-")
        ]
        if name in _SHELL_PATH_MUTATORS or name == "tee":
            targets.extend(operands)
        elif name in _SHELL_DESTINATION_MUTATORS and operands:
            targets.append(operands[-1])
        elif name in {"chmod", "chown", "chgrp"} and len(operands) > 1:
            targets.extend(operands[1:])
        elif (
            name in {"sed", "perl"}
            and any(token == "-i" or token.startswith("-i") for token in arguments)
            and operands
        ):
            targets.append(operands[-1])
        elif name == "dd":
            targets.extend(token[3:] for token in arguments if token.startswith("of="))
        for target in targets:
            expanded = _expand_shell_bindings(target, bindings)
            if _target_is_protected(expanded, current, roots):
                return True
    return False


class _PythonPersistenceVisitor(ast.NodeVisitor):
    """Track simple path bindings and inspect only actual mutation targets."""

    _PATH_METHODS = frozenset(
        {"write_text", "write_bytes", "unlink", "rmdir", "mkdir", "touch", "chmod"}
    )
    _ONE_PATH_CALLS = frozenset(
        {
            "os.remove",
            "os.unlink",
            "os.rmdir",
            "os.mkdir",
            "os.makedirs",
            "os.chmod",
            "os.chown",
            "os.truncate",
            "shutil.rmtree",
        }
    )
    _TWO_PATH_CALLS = frozenset({"os.rename", "os.replace", "shutil.move"})
    _DESTINATION_CALLS = frozenset(
        {"shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree"}
    )

    def __init__(self, code: str, cwd: Optional[Path], roots: Sequence[Path]):
        self.code = code
        self.current = cwd
        self.roots = roots
        self.bindings: dict[str, tuple[str, bool]] = {}
        self.path_constructors = {"path", "pathlib.path"}
        self.call_aliases: dict[str, str] = {}
        self.mutates = False

    def _value(self, node: ast.AST) -> Optional[tuple[str, bool]]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value, False
        if isinstance(node, ast.Name):
            return self.bindings.get(node.id)
        if isinstance(node, ast.Call):
            name = _call_name(node.func).casefold()
            if name in self.path_constructors and node.args:
                value = self._value(node.args[0])
                source = ast.get_source_segment(self.code, node) or ""
                return (value[0] if value else source), True
            if name == "os.path.join" and node.args:
                parts = [self._value(item) for item in node.args]
                if all(part is not None for part in parts):
                    return os.path.join(*(part[0] for part in parts if part)), False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left = self._value(node.left)
            right = self._value(node.right)
            if left and right:
                return str(Path(left[0]) / right[0]), True
            source = ast.get_source_segment(self.code, node) or ""
            return (source, True) if source else None
        constants = {name: value for name, (value, _is_path) in self.bindings.items()}
        literal = _literal_value(node, constants)
        return (literal, False) if isinstance(literal, str) else None

    def _protected(self, node: ast.AST) -> bool:
        value = self._value(node)
        if value and _target_is_protected(value[0], self.current, self.roots):
            return True
        source = ast.get_source_segment(self.code, node) or ""
        return bool(source) and _target_is_protected(source, self.current, self.roots)

    def _mode_mutates(self, node: ast.Call, position: int) -> bool:
        mode = node.args[position] if len(node.args) > position else next(
            (keyword.value for keyword in node.keywords if keyword.arg == "mode"),
            None,
        )
        value = self._value(mode) if mode is not None else None
        return bool(value and any(marker in value[0] for marker in "wax+"))

    @staticmethod
    def _argument(
        node: ast.Call, position: int, *keyword_names: str
    ) -> Optional[ast.AST]:
        if len(node.args) > position:
            return node.args[position]
        return next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg in keyword_names
            ),
            None,
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name.casefold() == "pathlib":
                self.path_constructors.add(
                    f"{(alias.asname or alias.name).casefold()}.path"
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = (node.module or "").casefold()
        if module == "pathlib":
            for alias in node.names:
                if alias.name.casefold() == "path":
                    self.path_constructors.add(
                        (alias.asname or alias.name).casefold()
                    )
        if module in {"os", "shutil"}:
            supported = (
                self._ONE_PATH_CALLS
                | self._TWO_PATH_CALLS
                | self._DESTINATION_CALLS
            )
            for alias in node.names:
                canonical = f"{module}.{alias.name.casefold()}"
                if canonical in supported:
                    self.call_aliases[
                        (alias.asname or alias.name).casefold()
                    ] = canonical

    @staticmethod
    def _parameter_names(node: ast.AST) -> Iterator[str]:
        arguments = getattr(node, "args", None)
        if not isinstance(arguments, ast.arguments):
            return
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

    def _visit_function_scope(self, node: ast.AST) -> None:
        outer_bindings = self.bindings
        outer_constructors = self.path_constructors
        outer_call_aliases = self.call_aliases
        outer_current = self.current
        self.bindings = dict(outer_bindings)
        self.path_constructors = set(outer_constructors)
        self.call_aliases = dict(outer_call_aliases)
        for parameter in self._parameter_names(node):
            self.bindings.pop(parameter, None)
            folded = parameter.casefold()
            self.path_constructors = {
                constructor
                for constructor in self.path_constructors
                if constructor.split(".", 1)[0] != folded
            }
            self.call_aliases.pop(folded, None)
        try:
            self.generic_visit(node)
        finally:
            self.bindings = outer_bindings
            self.path_constructors = outer_constructors
            self.call_aliases = outer_call_aliases
            self.current = outer_current

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        outer_bindings = self.bindings
        outer_constructors = self.path_constructors
        outer_call_aliases = self.call_aliases
        self.bindings = dict(outer_bindings)
        self.path_constructors = set(outer_constructors)
        self.call_aliases = dict(outer_call_aliases)
        try:
            self.generic_visit(node)
        finally:
            self.bindings = outer_bindings
            self.path_constructors = outer_constructors
            self.call_aliases = outer_call_aliases

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        value = self._value(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                folded = target.id.casefold()
                self.path_constructors.discard(folded)
                self.call_aliases.pop(folded, None)
                if value is None:
                    self.bindings.pop(target.id, None)
                else:
                    self.bindings[target.id] = value

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        if isinstance(node.target, ast.Name):
            folded = node.target.id.casefold()
            self.path_constructors.discard(folded)
            self.call_aliases.pop(folded, None)
            value = self._value(node.value) if node.value is not None else None
            if value is None:
                self.bindings.pop(node.target.id, None)
            else:
                self.bindings[node.target.id] = value

    def visit_Call(self, node: ast.Call) -> None:
        if self.mutates:
            return
        written_name = _call_name(node.func).casefold()
        name = self.call_aliases.get(written_name, written_name)
        if name == "os.chdir" and node.args:
            value = self._value(node.args[0])
            if value:
                try:
                    destination = Path(value[0]).expanduser()
                    if not destination.is_absolute() and self.current is not None:
                        destination = self.current / destination
                    self.current = destination.resolve(strict=False)
                except (OSError, RuntimeError, TypeError, ValueError):
                    pass
            return

        targets: list[ast.AST] = []
        file_target = self._argument(node, 0, "file")
        if (
            name in {"open", "io.open"}
            and file_target is not None
            and self._mode_mutates(node, 1)
        ):
            targets.append(file_target)
        elif isinstance(node.func, ast.Attribute):
            method = node.func.attr.casefold()
            base = self._value(node.func.value)
            if base and base[1] and method in self._PATH_METHODS:
                targets.append(node.func.value)
            elif base and base[1] and method == "open" and self._mode_mutates(node, 0):
                targets.append(node.func.value)
            elif base and base[1] and method in {"rename", "replace"}:
                targets.append(node.func.value)
                target = self._argument(node, 0, "target")
                if target is not None:
                    targets.append(target)

        if name in self._ONE_PATH_CALLS:
            target = self._argument(node, 0, "path", "name")
            if target is not None:
                targets.append(target)
        elif name in self._TWO_PATH_CALLS:
            source = self._argument(node, 0, "src")
            destination = self._argument(node, 1, "dst")
            targets.extend(
                target for target in (source, destination) if target is not None
            )
        elif name in self._DESTINATION_CALLS:
            destination = self._argument(node, 1, "dst")
            if destination is not None:
                targets.append(destination)
        if any(self._protected(target) for target in targets):
            self.mutates = True
            return
        self.generic_visit(node)


def _python_mutates_protected(
    code: str,
    cwd: Optional[Path],
    roots: Sequence[Path],
) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    visitor = _PythonPersistenceVisitor(code, cwd, roots)
    visitor.visit(tree)
    return visitor.mutates


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
    "find_persistence_store_reference",
    "profile_name_for_state_dir",
]
