"""The task list the model keeps while it works.

Copied from the Hermes agent (``tools/todo_tool.py``), which has run this in
production. One tool does both jobs: called with a list it writes, called with
nothing it reads, and either way it answers with the whole list — so the model
never has to remember what it last wrote.

Two things from Hermes are deliberately left behind. The list is not
re-injected after context compression, because we do not compress context; and
it is not rebuilt from replayed history, because our history lives in our own
database rather than being handed back to us by a gateway. What remains is the
list itself and the bounds that keep it from growing into the conversation.

The list belongs to the chat and to the process. It is a plan for the work in
front of the model, not a record worth keeping: a restart clears it, which is
correct, because after a restart the work it described is no longer under way.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .registry import Tool, ToolContext, tool_error

VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

# A todo item is a short task description and a real list is a handful of
# items. These caps are generous against that, and exist so one oversized item
# cannot quietly take over the conversation the plan is meant to serve.
MAX_TODO_CONTENT_CHARS = 4000
MAX_TODO_ITEMS = 256
TRUNCATION_MARKER = "… [truncated]"

# Where the store lives on the chat's scratch space.
STATE_KEY = "todo"


class TodoStore:
    """One chat's list.

    Items are ordered — position is priority. Each item is an id the model
    chose, a description, and a status.
    """

    def __init__(self) -> None:
        self._items: List[Dict[str, str]] = []

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """Write items and return the whole list as it now stands.

        Replacing is the default: the model re-states its plan and the plan is
        what it says. Merging updates items by id and appends the rest, for
        when only one step's status has changed.
        """
        if not merge:
            self._items = self._normalize_order(
                [self._validate(item) for item in self._dedupe_by_id(todos)]
            )
        else:
            existing = {item["id"]: item for item in self._items}
            for todo in self._dedupe_by_id(todos):
                item_id = str(todo.get("id", "")).strip() if isinstance(todo, dict) else ""
                if not item_id:
                    continue  # Nothing to merge against.

                if item_id in existing:
                    # Only the fields the model actually sent are touched.
                    if todo.get("content"):
                        existing[item_id]["content"] = self._cap_content(
                            str(todo["content"]).strip()
                        )
                    if todo.get("status"):
                        status = str(todo["status"]).strip().lower()
                        if status in VALID_STATUSES:
                            existing[item_id]["status"] = status
                else:
                    validated = self._validate(todo)
                    existing[validated["id"]] = validated
                    self._items.append(validated)

            seen = set()
            rebuilt: List[Dict[str, str]] = []
            for item in self._items:
                current = existing.get(item["id"], item)
                if current["id"] not in seen:
                    rebuilt.append(current)
                    seen.add(current["id"])
            self._items = self._normalize_order(rebuilt)

        # Keep the head: list order is priority.
        if len(self._items) > MAX_TODO_ITEMS:
            self._items = self._items[:MAX_TODO_ITEMS]
        return self.read()

    def read(self) -> List[Dict[str, str]]:
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        return bool(self._items)

    @staticmethod
    def _cap_content(content: str) -> str:
        if len(content) > MAX_TODO_CONTENT_CHARS:
            keep = MAX_TODO_CONTENT_CHARS - len(TRUNCATION_MARKER)
            return content[:keep] + TRUNCATION_MARKER
        return content

    @staticmethod
    def _validate(item: Any) -> Dict[str, str]:
        """Make one item into something the list can hold.

        Nothing is rejected. A malformed item becomes a visible placeholder
        rather than an error, because the model's next move should be to fix
        its plan, not to argue with the tool about it.
        """
        if not isinstance(item, dict):
            return {"id": "?", "content": "(invalid item)", "status": "pending"}

        item_id = str(item.get("id", "")).strip() or "?"

        content = str(item.get("content", "")).strip()
        content = TodoStore._cap_content(content) if content else "(no description)"

        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"

        return {"id": item_id, "content": content, "status": status}

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse repeated ids, keeping the last one in its own position."""
        last_index: Dict[str, int] = {}
        for index, item in enumerate(todos):
            if not isinstance(item, dict):
                # Keep it, under a key of its own, for _validate to deal with.
                last_index[f"__invalid_{index}"] = index
                continue
            item_id = str(item.get("id", "")).strip() or "?"
            last_index[item_id] = index
        return [todos[index] for index in sorted(last_index.values())]

    @staticmethod
    def _normalize_order(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Lift the step being worked on ahead of earlier untouched ones."""
        active_index = next(
            (i for i, item in enumerate(items) if item["status"] == "in_progress"),
            None,
        )
        if active_index is None:
            return items

        pending_index = next(
            (i for i, item in enumerate(items[:active_index]) if item["status"] == "pending"),
            None,
        )
        if pending_index is None:
            return items

        normalized = items.copy()
        active_item = normalized.pop(active_index)
        normalized.insert(pending_index, active_item)
        return normalized


def _store(context: ToolContext) -> TodoStore:
    store = context.state.get(STATE_KEY)
    if not isinstance(store, TodoStore):
        store = TodoStore()
        context.state[STATE_KEY] = store
    return store


def handle(args: Dict[str, Any], context: ToolContext) -> str:
    todos: Optional[Any] = args.get("todos")
    merge = bool(args.get("merge", False))
    store = _store(context)

    if todos is not None:
        # The model sometimes sends the list as a JSON string.
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except ValueError:
                return tool_error("todos must be a list of objects, got unparseable string")
        if not isinstance(todos, list):
            return tool_error(f"todos must be a list, got {type(todos).__name__}")
        items = store.write(todos, merge)
    else:
        items = store.read()

    counts = {status: 0 for status in sorted(VALID_STATUSES)}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1

    return json.dumps(
        {"todos": items, "summary": {"total": len(items), **counts}},
        ensure_ascii=False,
    )


# The whole behavioural guidance lives in the description: it is part of the
# static schema, so it is cached and never changes under a conversation.
TODO_SCHEMA = {
    "name": "todo",
    "description": (
        "Manage your task list for the current session. Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "Call with no parameters to read the current list.\n\n"
        "Writing:\n"
        "- Provide 'todos' array to create/update items\n"
        "- merge=false (default): replace the entire list with a fresh plan\n"
        "- merge=true: update existing items by id, add any new ones\n\n"
        "Each item: {id: string, content: string, "
        "status: pending|in_progress|completed|cancelled}\n"
        "List order is priority. Only ONE item in_progress at a time.\n"
        "Mark items completed immediately when done. If something fails, "
        "cancel it and add a revised item.\n\n"
        "Always returns the full current list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Task items to write. Omit to read current list.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Unique item identifier"},
                        "content": {"type": "string", "description": "Task description"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                            "description": "Current status",
                        },
                    },
                    "required": ["id", "content", "status"],
                },
            },
            "merge": {
                "type": "boolean",
                "description": (
                    "true: update existing items by id, add new ones. "
                    "false (default): replace the entire list."
                ),
                "default": False,
            },
        },
        "required": [],
    },
}

TODO_TOOL = Tool(
    name="todo",
    group="todo",
    schema=TODO_SCHEMA,
    handler=handle,
    emoji="📋",
)
