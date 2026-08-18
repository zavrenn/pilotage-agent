"""Unified removal contract for every credential source Pilotage reads from.

Pilotage seeds its credential pool from many places:

    env:<VAR>     — os.environ / ~/.pilotage/.env
    device_code   — auth.json providers.<provider> (openai-codex, ...)
    config:<name> — custom_providers config entry
    model_config  — model.api_key when model.provider == "custom"
    manual        — user ran `pilotage auth add`

Each source has its own reader inside ``agent.credential_pool._seed_from_*``
(which keep their existing shape — we haven't restructured them).  What we
unify here is **removal**:

    ``pilotage auth remove <provider> <N>`` must make the pool entry stay gone.

Before this module, every source had an ad-hoc removal branch in
``auth_remove_command``, and several sources had no branch at all — so
``auth remove`` silently reverted on the next ``load_pool()`` call for
custom-config sources.

Now every source registers a ``RemovalStep`` that does exactly three things
in the same shape:

    1. Clean up whatever externally-readable state the source reads from
       (.env line, auth.json block, OAuth file, etc.)
    2. Suppress the ``(provider, source_id)`` in auth.json so the
       corresponding ``_seed_from_*`` branch skips the upsert on re-load
    3. Return ``RemovalResult`` describing what was cleaned and any
       diagnostic hints the user should see (shell-exported env vars,
       external credential files we deliberately don't delete, etc.)

Adding a new credential source is:
    - wire up a reader branch in ``_seed_from_*`` (existing pattern)
    - gate that reader behind ``is_source_suppressed(provider, source_id)``
    - register a ``RemovalStep`` here

No more per-source if/elif chain in ``auth_remove_command``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class RemovalResult:
    """Outcome of removing a credential source.

    Attributes:
        cleaned: Short strings describing external state that was actually
            mutated (``"Cleared OPENAI_API_KEY from .env"``,
            ``"Cleared openai-codex OAuth tokens from auth store"``).
            Printed as plain lines to the user.
        hints: Diagnostic lines ABOUT state the user may need to clean up
            themselves or is deliberately left intact (shell-exported env
            var, external credential file we don't delete, etc.).
            Printed as plain lines to the user.  Always non-destructive.
        suppress: Whether to call ``suppress_credential_source`` after
            cleanup so future ``load_pool`` calls skip this source.
            Default True — almost every source needs this to stay sticky.
            The only legitimate False is ``manual`` entries, which aren't
            seeded from anywhere external.
    """

    cleaned: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    suppress: bool = True


@dataclass
class RemovalStep:
    """How to remove one specific credential source cleanly.

    Attributes:
        provider: Provider pool key (``"openai"``, ``"openai-codex"``, ...).
            Special value ``"*"`` means "matches any provider" — used for
            sources like ``manual`` that aren't provider-specific.
        source_id: Source identifier as it appears in
            ``PooledCredential.source``.  May be a literal (``"device_code"``)
            or a prefix pattern matched via ``match_fn``.
        match_fn: Optional predicate overriding literal ``source_id``
            matching.  Gets the removed entry's source string.  Used for
            ``env:*`` (any env-seeded key), ``config:*`` (any custom
            pool), and ``manual:*`` (any manual-source variant).
        remove_fn: ``(provider, removed_entry) -> RemovalResult``.  Does the
            actual cleanup and returns what happened for the user.
        description: One-line human-readable description for docs / tests.
    """

    provider: str
    source_id: str
    remove_fn: Callable[..., RemovalResult]
    match_fn: Optional[Callable[[str], bool]] = None
    description: str = ""

    def matches(self, provider: str, source: str) -> bool:
        if self.provider != "*" and self.provider != provider:
            return False
        if self.match_fn is not None:
            return self.match_fn(source)
        return source == self.source_id


_REGISTRY: List[RemovalStep] = []


def register(step: RemovalStep) -> RemovalStep:
    _REGISTRY.append(step)
    return step


def find_removal_step(provider: str, source: str) -> Optional[RemovalStep]:
    """Return the first matching RemovalStep, or None if unregistered.

    Unregistered sources fall through to the default remove path in
    ``auth_remove_command``: the pool entry is already gone (that happens
    before dispatch), no external cleanup, no suppression.  This is the
    correct behaviour for ``manual`` entries — they were only ever stored
    in the pool, nothing external to clean up.
    """
    for step in _REGISTRY:
        if step.matches(provider, source):
            return step
    return None


# ---------------------------------------------------------------------------
# Individual RemovalStep implementations — one per source.
# ---------------------------------------------------------------------------
# Each remove_fn is intentionally small and single-purpose.  Adding a new
# credential source means adding ONE entry here — no other changes to
# auth_remove_command.


def _remove_env_source(provider: str, removed) -> RemovalResult:
    """env:<VAR> — the most common case.

    Handles three user situations:
      1. Var lives only in ~/.pilotage/.env  → clear it
      2. Var lives only in the user's shell (shell profile, systemd
         EnvironmentFile, launchd plist) → hint them where to unset it
      3. Var lives in both → clear from .env, hint about shell
    """
    from pilotage_cli.config import get_env_path, remove_env_value

    result = RemovalResult()
    env_var = removed.source[len("env:"):]
    if not env_var:
        return result

    # Detect shell vs .env BEFORE remove_env_value pops os.environ.
    env_in_process = bool(os.getenv(env_var))
    env_in_dotenv = False
    try:
        env_path = get_env_path()
        if env_path.exists():
            # Read the .env as UTF-8 with BOM tolerance, matching the
            # canonical reader in pilotage_cli/config.py. read_text() with no
            # encoding falls back to the system locale (cp1252/GBK on Windows)
            # and never strips a BOM, so a Notepad-edited .env (BOM + non-ASCII
            # values) would make the first line fail the startswith() check —
            # misreporting a .env-backed var as a shell export.
            env_in_dotenv = any(
                line.strip().startswith(f"{env_var}=")
                for line in env_path.read_text(
                    encoding="utf-8-sig", errors="replace"
                ).splitlines()
            )
    except OSError:
        pass
    shell_exported = env_in_process and not env_in_dotenv

    cleared = remove_env_value(env_var)
    if cleared:
        result.cleaned.append(f"Cleared {env_var} from .env")

    if shell_exported:
        result.hints.extend([
            f"Note: {env_var} is still set in your shell environment "
            f"(not in ~/.pilotage/.env).",
            "  Unset it there (shell profile, systemd EnvironmentFile, "
            "launchd plist, etc.) or it will keep being visible to Pilotage.",
            f"  The pool entry is now suppressed — Pilotage will ignore "
            f"{env_var} until you run `pilotage auth add {provider}`.",
        ])
    else:
        result.hints.append(
            f"Suppressed env:{env_var} — it will not be re-seeded even "
            f"if the variable is re-exported later."
        )
    return result


def _clear_auth_store_provider(provider: str) -> bool:
    """Delete auth_store.providers[provider].  Returns True if deleted."""
    from pilotage_cli.auth import (
        _auth_store_lock,
        _load_auth_store,
        _save_auth_store,
    )

    with _auth_store_lock():
        auth_store = _load_auth_store()
        providers_dict = auth_store.get("providers")
        if isinstance(providers_dict, dict) and provider in providers_dict:
            del providers_dict[provider]
            _save_auth_store(auth_store)
            return True
    return False




def _remove_codex_device_code(provider: str, removed) -> RemovalResult:
    """Codex tokens live in TWO places: our auth store AND ~/.codex/auth.json.

    refresh_codex_oauth_pure() writes both every time, so clearing only
    the Pilotage auth store is not enough — _seed_from_singletons() would
    re-import from ~/.codex/auth.json on the next load_pool() call and
    the removal would be instantly undone.  We suppress instead of
    deleting Codex CLI's file, so the Codex CLI itself keeps working.

    The canonical source name in ``_seed_from_singletons`` is
    ``"device_code"`` (no prefix).  Entries may show up in the pool as
    either ``"device_code"`` (seeded) or ``"manual:device_code"`` (added
    via ``pilotage auth add openai-codex``), but in both cases the re-seed
    gate lives at the ``"device_code"`` suppression key.  We suppress
    that canonical key here; the central dispatcher also suppresses
    ``removed.source`` which is fine — belt-and-suspenders, idempotent.
    """
    from pilotage_cli.auth import suppress_credential_source

    result = RemovalResult()
    if _clear_auth_store_provider(provider):
        result.cleaned.append(f"Cleared {provider} OAuth tokens from auth store")
    # Suppress the canonical re-seed source, not just whatever source the
    # removed entry had.  Otherwise `manual:device_code` removals wouldn't
    # block the `device_code` re-seed path.
    suppress_credential_source(provider, "device_code")
    result.hints.extend([
        "Suppressed openai-codex device_code source — it will not be re-seeded.",
        "Note: Codex CLI credentials still live in ~/.codex/auth.json",
        "Run `pilotage auth add openai-codex` to re-enable if needed.",
    ])
    return result




def _remove_custom_config(provider: str, removed) -> RemovalResult:
    """Custom provider pools are seeded from custom_providers config or
    model.api_key.  Both are in config.yaml — modifying that from here
    is more invasive than suppression.  We suppress; the user can edit
    config.yaml if they want to remove the key from disk entirely.
    """
    source_label = removed.source
    return RemovalResult(hints=[
        f"Suppressed {source_label} — it will not be re-seeded.",
        "Note: The underlying value in config.yaml is unchanged.  Edit it "
        "directly if you want to remove the credential from disk.",
    ])


def _register_all_sources() -> None:
    """Called once on module import.

    ORDER MATTERS — ``find_removal_step`` returns the first match.  Put
    provider-specific steps before the generic ``env:*`` step.
    """
    register(RemovalStep(
        provider="*", source_id="env:",
        match_fn=lambda src: src.startswith("env:"),
        remove_fn=_remove_env_source,
        description="Any env-seeded credential (OPENAI_API_KEY, etc.)",
    ))
    register(RemovalStep(
        provider="openai-codex", source_id="device_code",
        match_fn=lambda src: src == "device_code" or src.endswith(":device_code"),
        remove_fn=_remove_codex_device_code,
        description="auth.json providers.openai-codex + ~/.codex/auth.json",
    ))
    register(RemovalStep(
        provider="*", source_id="config:",
        match_fn=lambda src: src.startswith("config:") or src == "model_config",
        remove_fn=_remove_custom_config,
        description="Custom provider config.yaml api_key field",
    ))


_register_all_sources()
