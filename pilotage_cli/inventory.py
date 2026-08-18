"""Provider/model inventory context for the ``/model``-family pickers.

Two jobs, one entry point each:

- :func:`load_picker_context` takes the config-slice every picker needs
  (``model.{default,name,provider,base_url}``, ``providers:``,
  ``model_catalog.excluded_providers``) out of ``load_config()`` so no
  caller re-derives it and drops a field.
- :func:`build_aux_picker_rows` calls ``list_authenticated_providers``
  with the kwargs that make an auxiliary picker match ``/model``, and
  :func:`format_aux_picker_entries` renders the result.

Substrate fact (verified May 2026): ``list_authenticated_providers``
already populates each row's ``models`` from the curated catalog (same
source as the picker). Do NOT call ``provider_model_ids()`` per row to
"freshen" — that bypasses curation and pulls in non-agentic models (a bare
/models can return ~400 IDs including TTS, embeddings, rerankers,
image/video generators).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


# ─── Public types ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConfigContext:
    """Snapshot of the model + provider config every inventory caller
    needs. Built once via ``load_picker_context()``; the TUI overlays
    live agent state via ``with_overrides()`` before passing through.
    """

    current_provider: str
    current_model: str
    current_base_url: str
    user_providers: dict
    excluded_providers: list = None

    def with_overrides(
        self,
        *,
        current_provider: Optional[str] = None,
        current_model: Optional[str] = None,
        current_base_url: Optional[str] = None,
    ) -> "ConfigContext":
        """Return a copy with truthy overrides applied.

        Truthy-only because the TUI reads agent attributes that may be
        empty strings before an agent is spawned — empties must NOT
        clobber the disk-config values.
        """
        kw: dict = {}
        if current_provider:
            kw["current_provider"] = current_provider
        if current_model:
            kw["current_model"] = current_model
        if current_base_url:
            kw["current_base_url"] = current_base_url
        return replace(self, **kw) if kw else self


def load_picker_context() -> ConfigContext:
    """Load the disk-config snapshot every consumer needs.

    Replaces the inline 17-LOC config-slice that ``web_server.py`` and
    ``tui_gateway/server.py`` (×2 sites) used to do.
    """
    from pilotage_cli.config import load_config

    cfg = load_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        current_model = model_cfg.get("default", model_cfg.get("name", "")) or ""
        current_provider = model_cfg.get("provider", "") or ""
        current_base_url = model_cfg.get("base_url", "") or ""
    else:
        # config.model can be a bare string in older configs.
        current_model = str(model_cfg) if model_cfg else ""
        current_provider = ""
        current_base_url = ""
    raw = cfg.get("providers")
    excluded = cfg.get("model_catalog", {}).get("excluded_providers") or []
    return ConfigContext(
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
        user_providers=raw if isinstance(raw, dict) else {},
        excluded_providers=excluded if isinstance(excluded, list) else [],
    )


# ─── Public: auxiliary-task pickers ─────────────────────────────────────


def build_aux_picker_rows(
    *,
    current_provider: str = "",
    current_model: str = "",
    current_base_url: str = "",
    max_models: int | None = None,
) -> list[dict]:
    """Provider rows for any auxiliary-task picker (vision, compression, …).

    THE entry point for every aux picker — present and future. Call this
    instead of ``list_authenticated_providers()`` directly.

    Aux pickers kept re-deriving their own kwargs and each one silently
    dropped a different slice of the user's configuration (user ``providers:``
    entries never appeared; providers with an exhausted credential pool were
    hidden). Both fixes were per-site kwarg patches, so the next aux picker
    would have reintroduced the same gap. Routing through one function makes
    the correct behaviour the default that a new caller cannot forget:

    - user-defined ``providers:`` entries
    - ``model_catalog.excluded_providers`` honoured, matching ``/model``
    - exhausted-credential-pool providers stay visible (``for_picker``)

    Rows are the standard ``list_authenticated_providers`` shape. Pair with
    :func:`format_aux_picker_entries` to render them.
    """
    from pilotage_cli.model_switch import list_authenticated_providers

    ctx = load_picker_context().with_overrides(
        current_provider=current_provider,
        current_model=current_model,
        current_base_url=current_base_url,
    )
    return list_authenticated_providers(
        current_provider=ctx.current_provider,
        current_base_url=ctx.current_base_url,
        current_model=ctx.current_model,
        user_providers=ctx.user_providers,
        max_models=max_models,
        for_picker=True,
        excluded_providers=ctx.excluded_providers or [],
    )


def format_aux_picker_entries(
    rows: list[dict],
    *,
    current_provider: str = "",
    current_base_url: str = "",
) -> list[tuple[str, str, list[str]]]:
    """Render aux-picker rows as ``(slug, label, models)`` menu entries.

    Owns the label text and the ``← current`` marker so every aux picker
    presents providers identically. Callers add their own leading/trailing
    entries (``auto``, ``Custom endpoint``, ``Back``) around this list.

    A custom endpoint set via a raw ``base_url`` is "current" only through
    that URL — never through a provider slug — so when ``current_base_url``
    is set no provider row is marked, matching the pre-existing behaviour of
    both call sites.
    """
    entries: list[tuple[str, str, list[str]]] = []
    current_slug = str(current_provider or "").strip().lower()
    has_base_url = bool(str(current_base_url or "").strip())
    for row in rows:
        slug = str(row.get("slug") or "")
        name = row.get("name") or slug
        total = row.get("total_models") or len(row.get("models") or [])
        model_hint = f" — {total} models" if total else ""
        marker = (
            "  ← current"
            if slug.lower() == current_slug and current_slug and not has_base_url
            else ""
        )
        entries.append((slug, f"{name}{model_hint}{marker}", list(row.get("models") or [])))
    return entries
