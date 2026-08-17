"""Context-local state for delegate_task child execution.

delegate_task children run inside the parent Pilotage process. This module lets
code paths that resolve tool schemas or spawn subprocesses tell a delegated
child apart from its parent without mutating global ``os.environ``, and carries
that lineage across a fork via an env marker.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Mapping, MutableMapping

_DELEGATED_CHILD_CONTEXT: ContextVar[bool] = ContextVar(
    "pilotage_delegated_child_context",
    default=False,
)

DELEGATED_CHILD_ENV_MARKER = "PILOTAGE_DELEGATED_CHILD_CONTEXT"


@contextmanager
def delegated_child_context(session_id: str | None = None) -> Iterator[None]:
    """Mark child execution and isolate its task-local session identity.

    Child construction calls ``set_current_session_id`` internally, so even a
    context entered without an id must restore the parent's ContextVar.  Child
    execution passes its explicit id and receives it only for this scope.
    """
    token = _DELEGATED_CHILD_CONTEXT.set(True)
    try:
        # Import lazily: session_context calls is_delegated_child_context() when
        # deciding whether the compatibility os.environ mirror is safe.
        from gateway.session_context import scoped_current_session_id

        with scoped_current_session_id(session_id):
            yield
    finally:
        _DELEGATED_CHILD_CONTEXT.reset(token)


def is_delegated_child_context() -> bool:
    """Return True while code is running for a delegate_task child."""
    return bool(_DELEGATED_CHILD_CONTEXT.get())


def is_delegated_child_process_context() -> bool:
    """Return True in this process or a subprocess spawned by a child."""
    import os

    return bool(_DELEGATED_CHILD_CONTEXT.get()) or bool(
        os.environ.get(DELEGATED_CHILD_ENV_MARKER)
    )


def scrub_delegated_child_env(env: Mapping[str, str] | MutableMapping[str, str]) -> dict[str, str]:
    """Return *env* tagged as a delegated-child environment."""
    cleaned = dict(env)
    cleaned[DELEGATED_CHILD_ENV_MARKER] = "1"
    return cleaned


def delegated_child_subprocess_env(
    env: Mapping[str, str] | MutableMapping[str, str] | None = None,
) -> dict[str, str] | None:
    """Return an env override only when delegated-child lineage must cross fork.

    Most subprocess call sites historically used ``env=None`` to inherit the
    process environment.  That loses the ContextVar in the new process, so this
    helper preserves normal ``env=None`` semantics for non-delegated calls and
    only materializes a tagged env when the lineage marker must be propagated
    across a child-process boundary.
    """
    if not is_delegated_child_process_context():
        return None if env is None else dict(env)

    if env is None:
        import os

        env = os.environ
    return scrub_delegated_child_env(env)
